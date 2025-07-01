import json
import logging
import math
import os
import time
import gc
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
try:
    from torch._six import inf
except: 
    from torch import inf
import torch.nn.functional as F
import torch.distributed as dist
from llm2vec import LLM2Vec
try:
    import wandb
except ImportError:
    wandb = None
from eva_clip import ClipLoss, get_cast_dtype, get_tokenizer
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .zeroshot_retrieval import retrieval_eval
from .precision import get_autocast
from .utils import save_file
from .loss import SigLipLoss

# Create a nuclear logging filter to block ALL DeepSpeed messages
class DeepSpeedFilter(logging.Filter):
    def filter(self, record):
        # Block anything from deepspeed or containing specific timing keywords
        if (hasattr(record, 'name') and 'deepspeed' in record.name.lower()) or \
           (hasattr(record, 'msg') and isinstance(record.msg, str) and 
            ('time (ms)' in record.msg or 'optimizer_' in record.msg or 
             'bwd_' in record.msg or '_microstep' in record.msg)):
            return False
        return True

# Apply the filter to the root logger to catch ALL log messages
root_logger = logging.getLogger()
root_logger.addFilter(DeepSpeedFilter())

# Also silence specific loggers for good measure
for logger_name in ['deepspeed', 'deepspeed.comm', 'deepspeed.runtime', 
                   'deepspeed.runtime.engine', 'deepspeed.runtime.zero', 'deepspeed.utils']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False  # Prevent propagation to parent loggers

# Completely monkey-patch the DeepSpeed logging function as a last resort
import deepspeed
def completely_silent_log_dist(*args, **kwargs):
    return None
deepspeed.utils.logging.log_dist = completely_silent_log_dist

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    loss_scale = None
    if hasattr(optimizer, 'loss_scale'):
        loss_scale = optimizer.loss_scale
    elif hasattr(optimizer, 'cur_scale'):
        loss_scale = optimizer.cur_scale
    return loss_scale, optimizer._global_grad_norm

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm.to(dtype=torch.float32)

def train_one_epoch(model, tokenizer, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    ## Should update to SigLIP Loss
    loss = SigLipLoss(
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        bidir=True,
        use_horovod=False,
    )
    loss2 = nn.BCEWithLogitsLoss()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    accumulate_count = 0
    
    for i, batch in tqdm(
        enumerate(dataloader), 
        total=num_batches_per_epoch,
        desc=f"Epoch {epoch}/{args.epochs}",
        unit="batch",
        bar_format="{l_bar}{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        colour="green",
        leave=False,
        dynamic_ncols=True
    ):
        # if i>3:
        #     break
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        prev_images, cur_images, captions, oids, labels = batch

        prev_images = prev_images.to(device=device, dtype=cast_dtype, non_blocking=True)
        cur_images = cur_images.to(device=device, dtype=cast_dtype, non_blocking=True)
        
        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()
    
        with autocast():
            image_features = model.visual(cur_images, prev_images).projected_global_embedding
            
            captions = captions.to(device)
            # Get text features and ensure consistent dtype
            text_features = model.text.model(captions)
            # Explicitly cast to the same dtype as image_features
            text_features = model.text.projection(text_features.to(dtype=cast_dtype))
            # Just use the loss value without capturing accuracy metrics
            siglip_loss = loss(image_features, text_features, model.logit_scale, model.logit_bias)

            if args.bce_loss:
                # Calculate similarity between image_features and back_features
                # Normalize features for cosine similarity
                back_features = model.visual(prev_images, cur_images).projected_global_embedding
                norm_image_features = F.normalize(image_features, p=2, dim=1)
                norm_back_features = F.normalize(back_features, p=2, dim=1)
                
                # Compute cosine similarity (dot product of normalized vectors)
                similarity = torch.sum(norm_image_features * norm_back_features, dim=1)
                
                # Apply BCE loss - when labels=0, similarity should be high (close to 1)
                # when labels=1, similarity should be low (close to 0)
                # Convert labels to tensor if it's a list
                if isinstance(labels, list):
                    bce_target = torch.tensor(labels, device=device).float()
                else:
                    bce_target = labels.float().to(device)  # Convert labels to float
                
                bce_loss = loss2(similarity, 1.0 - bce_target)  # Invert target if needed
                bce_loss_value = bce_loss.clone().detach()  # Save for logging
                
                # Add BCE loss to total loss with a weight parameter
                bce_weight = 0.3  # Adjust this weight as needed
                total_loss = siglip_loss + bce_weight * bce_loss
            else:
                total_loss = siglip_loss

        # If using distributed training, gather the loss from all GPUs
        if args.world_size > 1:
            loss_list = [torch.zeros_like(total_loss) for _ in range(args.world_size)]
            dist.all_gather(loss_list, total_loss)
            loss_list = torch.stack(loss_list)
        else:
            loss_list = torch.tensor([total_loss])
 
        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            
            scaler.step(optimizer)
            scaler.update()
        elif args.enable_deepspeed:
            model.backward(total_loss)
            model.step()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(cur_images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            # For SigLip, we don't have separate clip_loss or accuracy metrics
            # so we'll use the same total_loss for both
            loss_clip_m.update(total_loss.item(), batch_size)

            loss_scale_value = model.logit_scale.item()
            grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            if args.bce_loss:
                logging.info(
                    f"Global Steps: {step + 1} "
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(SigLip): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Loss(BCE): {bce_loss_value.item():#.5g} "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {model.logit_scale.item():.3f} "
                f"Logit Bias: {model.logit_bias.item():.3f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
                )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                log_data = {
                    "loss": loss_m.val,
                    "loss_siglip": siglip_loss.item(),  # Use the separate siglip loss
                    "loss_bce": bce_loss_value.item(),  # Add BCE loss to logging
                    "loss_scaler": loss_scaler.val,
                    "grad_nrom": grad_norm_m.val,
                    "scale": model.logit_scale.item(),
                    "bias": model.logit_bias.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "lr_visual": optimizer.param_groups[index_visual]["lr"],
                    "lr_text": optimizer.param_groups[index_text]["lr"],
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                }
            else:
                logging.info(
                    f"Global Steps: {step + 1} "
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(SigLip): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {model.logit_scale.item():.3f} "
                f"Logit Bias: {model.logit_bias.item():.3f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
                )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                log_data = {
                    "loss": loss_m.val,
                    "loss_siglip": siglip_loss.item(),  # Use the separate siglip loss
                    "loss_scaler": loss_scaler.val,
                    "grad_nrom": grad_norm_m.val,
                    "scale": model.logit_scale.item(),
                    "bias": model.logit_bias.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "lr_visual": optimizer.param_groups[index_visual]["lr"],
                    "lr_text": optimizer.param_groups[index_text]["lr"],
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
        
        eval_point = int(num_batches_per_epoch/5)
        if step>0 and step%eval_point ==0:
            if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
                # evaluate_iter(model, tokenizer, data, step ,epoch, args, tb_writer)
                torch.cuda.empty_cache()
                model.train()
    # end for

def evaluate_iter(model, tokenizer, data, iter_nums, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()
    # l2v = LLM2Vec(model.text.model, tokenizer, pooling_mode="latent_attention", max_length=512) #TODO: modify this
    print('evaluating retrieval')

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    if 'val' in data:
        
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        logit_scale = model.logit_scale
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                prev_images, cur_images, captions, oids, labels = batch
                prev_images = prev_images.to(device=device, dtype=cast_dtype, non_blocking=True)
                cur_images = cur_images.to(device=device, dtype=cast_dtype, non_blocking=True)
                
                # Check if captions is a dictionary (tokenizer output) or a tensor
                if isinstance(captions, dict):
                    # Move all tensors in the dictionary to the device
                    for key in captions:
                        if isinstance(captions[key], torch.Tensor):
                            captions[key] = captions[key].to(device)
                elif hasattr(captions, 'device') and captions.device != device:
                    captions = captions.to(device)
                
                with autocast():
                    image_features = model.visual(cur_images, prev_images).projected_global_embedding
                    text_features = model.text.model.forward(captions)
                    text_features = model.text.projection(text_features.to(dtype=cast_dtype))
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = cur_images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
                epoch=epoch,
                oids=None,
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "iter": iter_nums, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch, 'iter_nums':iter_nums})

    return metrics


def evaluate(model, tokenizer, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()
    
    # l2v = LLM2Vec(model.text.model, tokenizer, pooling_mode="latent_attention", max_length=512) #TODO: modify this
    
    # Handle retrieval evaluation
    retrieval_zero_shot_metrics = retrieval_eval(model, data, epoch, args)
    metrics.update(retrieval_zero_shot_metrics)
    print('wawa retrieval--------------------------')
    
    # Handle zero-shot evaluation
    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    
    # Log each category of metrics separately
    if 'accuracies' in zero_shot_metrics:
        logging.info(f"Zero-shot accuracies: {json.dumps(zero_shot_metrics['accuracies'], indent=2)}")
        # Add overall accuracies with proper prefix
        for finding, acc in zero_shot_metrics['accuracies'].items():
            metrics[f"overall_accuracy/{finding}"] = acc
    
    if 'improving_accuracy' in zero_shot_metrics:
        logging.info(f"Zero-shot improving accuracies: {json.dumps(zero_shot_metrics['improving_accuracy'], indent=2)}")
        # Add improving accuracies with proper prefix
        for finding, acc in zero_shot_metrics['improving_accuracy'].items():
            metrics[f"improving_accuracy/{finding}"] = acc
    
    if 'stable_accuracy' in zero_shot_metrics:
        logging.info(f"Zero-shot stable accuracies: {json.dumps(zero_shot_metrics['stable_accuracy'], indent=2)}")
        # Add stable accuracies with proper prefix
        for finding, acc in zero_shot_metrics['stable_accuracy'].items():
            metrics[f"stable_accuracy/{finding}"] = acc
    
    if 'worsened_accuracy' in zero_shot_metrics:
        logging.info(f"Zero-shot worsened accuracies: {json.dumps(zero_shot_metrics['worsened_accuracy'], indent=2)}")
        # Add worsened accuracies with proper prefix
        for finding, acc in zero_shot_metrics['worsened_accuracy'].items():
            metrics[f"worsened_accuracy/{finding}"] = acc
    
    # Rest of the function remains unchanged
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 \
        or epoch==-1 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        logit_scale = model.logit_scale
        total_oids = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                prev_images, cur_images, captions, oids, labels = batch
                prev_images = prev_images.to(device=device, dtype=cast_dtype, non_blocking=True)
                cur_images = cur_images.to(device=device, dtype=cast_dtype, non_blocking=True)
                
                # Check if captions is a dictionary (tokenizer output) or a tensor
                captions = captions.to(device)
                
                with autocast():
                    image_features = model.visual(cur_images, prev_images).projected_global_embedding
                    text_features = model.text.model.forward(captions)
                    text_features = model.text.projection(text_features.to(dtype=cast_dtype))
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    
                    # Create SigLip loss for evaluation
                    siglip_eval_loss = SigLipLoss(
                        cache_labels=True,
                        rank=0,  # For evaluation
                        world_size=1,  # For evaluation
                        bidir=True,
                        use_horovod=False,
                    )
                    
                    # Calculate SigLip loss
                    batch_size = cur_images.shape[0]
                    total_loss = siglip_eval_loss(image_features, text_features, logit_scale.mean(), model.logit_bias)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")
                total_oids.extend(oids)

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
                epoch=epoch,
                oids=total_oids,
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )


    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics

def save_model(model, epoch, args):
    if args.logs and args.logs.lower() != 'none' and args.enable_deepspeed:
        deepspeed_checkpoint_path = os.path.join('/model/llm2clip', "checkpoints")
        client_state = {'epoch': epoch}
        # Ensure all ranks use the same tag by removing rank-specific information
        checkpoint_tag = f"epoch_{epoch}"
        if dist.is_initialized():
            # Synchronize all processes before saving
            dist.barrier()
        model.save_checkpoint(save_dir=deepspeed_checkpoint_path, tag=checkpoint_tag, client_state=client_state)
        if dist.is_initialized():
            # Synchronize all processes after saving
            dist.barrier()
        return deepspeed_checkpoint_path

def load_model(pth, model):
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    # Load the DeepSpeed checkpoint into regular PyTorch state dict
    checkpoint_path = pth  # Directory containing the checkpoint files
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    return model


def get_metrics(image_features, text_features, logit_scale, epoch, oids=None):
    import json
    import os
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics

def extract_features(model, data, args, device):
    
    img_emb_folder = args.img_emb_path
    text_emb_folder = args.text_emb_path

    save_interval = args.save_interval if args.save_interval else 100
    all_features = []
    feature_info = {}

    model.eval()
    # autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    if 'val' in data:
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples
        
        all_image_features = []
        all_text_features = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                idx = i+1

                images, texts, oids = batch

                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                image_features, text_features = model(images, texts)

                all_image_features.append(image_features)
                all_text_features.append(text_features)

                batch_size = images.shape[0]
                num_samples += batch_size
                logging.info(
                    f"Extract RANK: {args.rank} [{num_samples} / {samples_per_val}]"
                )

                if idx % save_interval == 0:

                    img_feat = np.concatenate(all_image_features)
                    text_feat = np.concatenate(all_text_features)
                    

                    split = "%08d" % (idx//save_interval)
                    out_img_feat_file = (
                        f"{img_emb_folder}/rank{args.rank}_img_emb_{split}.npy"
                    )
                    out_text_feat_file = (
                        f"{text_emb_folder}/rank{args.rank}_text_emb_{split}.npy"
                    )

                    save_file(img_feat, out_img_feat_file)
                    save_file(text_feat, out_text_feat_file)

                    
                    all_image_features = []
                    all_text_features = []

            if len(all_image_features) > 0:
                img_feat = np.concatenate(all_image_features)
                text_feat = np.concatenate(all_text_features)

                split = "%08d" % ((idx//save_interval)+1)
                out_img_feat_file = (
                    f"{img_emb_folder}/rank{args.rank}_img_emb_{split}.npy"
                )
                out_text_feat_file = (
                    f"{text_emb_folder}/rank{args.rank}_text_emb_{split}.npy"
                )

                save_file(img_feat, out_img_feat_file)
                save_file(text_feat, out_text_feat_file)
    torch.distributed.barrier()
