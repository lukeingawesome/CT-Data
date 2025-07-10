#!/usr/bin/env python3
"""
Evaluation script for image-text retrieval using trained model.
Loads a CSV file with img_path and report columns, performs retrieval evaluation,
and saves results including top-1 retrieved reports for each image.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the current directory to path for imports
sys.path.append(os.getcwd())

from transformers import AutoTokenizer
from merlin import Merlin
from training.llm2vec_wrapper import LLM2VecWrapper as LLM2Vec
from peft import LoraConfig, get_peft_model, TaskType
from training.data import CustomCSVDataset
from training.ct_transform import get_val_transform

class ModelWithCustomVisual(nn.Module):
    """Combines a custom visual model (Merlin) with a text model for CLIP-style training."""
    
    def __init__(self, visual_model, text_model, vision_projection=None):
        super().__init__()
        self.visual = visual_model
        self.text = text_model
        self.vision_projection = vision_projection
        
        # Initialize learnable logit_scale and logit_bias
        self.logit_scale = nn.Parameter(torch.tensor(10.0))   # linear scale = 10
        self.logit_bias  = nn.Parameter(torch.tensor(-10.0))
        
    def encode_image(self, image):
        # Handle both 2D (CXR) and 3D (CT) images
        if len(image.shape) == 5:  # 3D CT: (B, C, D, H, W)
            features = self.visual(image)
        elif len(image.shape) == 4:  # 2D CXR: (B, C, H, W)  
            features = self.visual(image)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Ensure features are in the right shape
        if len(features.shape) > 2:
            features = features.squeeze()
        
        # Apply vision projection layer if provided
        if self.vision_projection is not None:
            features = self.vision_projection(features)
        
        # Normalize projected features
        return features / features.norm(dim=-1, keepdim=True)
        
    def encode_text(self, text):
        features = self.text(text)
        return features / features.norm(dim=-1, keepdim=True)
    
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # Compute similarity using the learnable logit_scale and logit_bias
        logits = self.logit_scale * image_features @ text_features.t() + self.logit_bias
            
        return logits

class LLM2VecWithProjection(nn.Module):
    def __init__(self, llm2vec_model, projection):
        super().__init__()
        self.model = llm2vec_model
        self.projection = projection
        
        # Ensure the LLM2Vec model is in the same dtype as the projection
        self.model.to(next(self.projection.parameters()).dtype)
            
        # Freeze the base LLM model parameters but keep LoRA parameters trainable
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True  # Keep LoRA parameters trainable
            else:
                param.requires_grad = False  # Freeze base model parameters
            
        # Ensure projection layer is trainable
        for param in self.projection.parameters():
            param.requires_grad = True

    def forward(self, text):
        # Ensure input is in the correct dtype if it's a tensor
        if isinstance(text, torch.Tensor):
            text = text.to(next(self.model.parameters()).dtype)
        
        embeddings = self.model(text)
        # Ensure consistent dtype
        if embeddings.dtype != next(self.projection.parameters()).dtype:
            embeddings = embeddings.to(next(self.projection.parameters()).dtype)
        return self.projection(embeddings)

def load_model(checkpoint_path, text_base_model="microsoft/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned", device="cuda", precision="bf16"):
    """Load the trained model from checkpoint."""
    
    logging.info("Loading model components...")
    
    # Set up precision
    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # Load visual model (Merlin)
    visual_model = Merlin(ImageEmbedding=True)
    visual_model.to(dtype)
    
    # Load text model (LLM2Vec)
    text_model = LLM2Vec.from_pretrained(
        base_model_name_or_path=text_base_model,
        enable_bidirectional=True,
        pooling_mode="latent_attention",
        max_length=512,
        torch_dtype=dtype,
    )
    text_model.to(device)
    # Ensure all parameters are in correct dtype
    text_model.to(dtype)
    
    # Add LoRA configuration to the text model
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=32,  # rank
        lora_alpha=32,  # scaling parameter
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # target attention modules
        bias="none",
    )
    
    # Apply LoRA only to the underlying transformer model
    base_transformer = text_model.model
    base_transformer = get_peft_model(base_transformer, lora_config)
    text_model.model = base_transformer
    # Ensure the model is still in correct dtype after LoRA
    text_model.to(dtype)

    # Create projection layers
    hidden_size = text_model.config.hidden_size
    text_projection_layer = nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, 1280)  # Project to 1280 dimensions
    ).to(device).to(dtype)
    
    vision_projection_layer = nn.Sequential(
        nn.LayerNorm(2048),  # Merlin outputs 2048 features
        nn.Linear(2048, 1280)  # Project to 1280 dimensions
    ).to(device).to(dtype)

    # Create wrapped text model
    text_model = LLM2VecWithProjection(text_model, text_projection_layer)
    
    # Ensure all parameters and buffers are in correct dtype
    for name, param in text_model.named_parameters():
        if param.dtype != dtype:
            param.data = param.data.to(dtype)
    
    for name, buffer in text_model.named_buffers():
        if buffer.dtype != dtype:
            buffer.data = buffer.data.to(dtype)
    
    # Create combined model
    model = ModelWithCustomVisual(visual_model, text_model, vision_projection_layer)
    
    # Load checkpoint
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DistributedDataParallel)
    if next(iter(state_dict.items()))[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing)}  |  Unexpected: {len(unexpected)}")
    
    # Print first few missing keys to debug the issue
    if len(missing) > 0:
        print("First 10 missing keys:")
        for key in missing[:10]:
            print(f"  - {key}")
    
    if len(unexpected) > 0:
        print("First 10 unexpected keys:")
        for key in unexpected[:10]:
            print(f"  - {key}")
    
    # Check if missing keys are LoRA-related
    lora_missing = [key for key in missing if 'lora_' in key]
    non_lora_missing = [key for key in missing if 'lora_' not in key]
    
    print(f"LoRA missing keys: {len(lora_missing)}")
    print(f"Non-LoRA missing keys: {len(non_lora_missing)}")
    
    # Only assert if LoRA keys are missing (non-LoRA keys are expected - they're base model weights)
    assert len(lora_missing)==0, f"LoRA weights were not restored: {lora_missing[:5]}"
    
    if len(lora_missing) == 0 and len(missing) > 0:
        print("✅ All LoRA weights loaded successfully!")
        print(f"ℹ️  {len(non_lora_missing)} base model weights missing (expected - loaded from pretrained model)")
    model.to(device)
    model.to(dtype)  # Ensure final model is in correct dtype
    model.eval()
    
    # Log model parameters for debugging
    logging.info(f"Model loaded successfully!")
    logging.info(f"Logit scale: {model.logit_scale.item():.4f}")
    logging.info(f"Logit bias: {model.logit_bias.item():.4f}")
    
    return model

def compute_retrieval_metrics(image_features, text_features, image_paths, texts, logit_scale=None, logit_bias=None):
    """Compute retrieval metrics and return detailed results."""
    
    # Normalize features (convert to float32 for numpy compatibility)
    image_features = F.normalize(image_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)
    
    # Compute similarity matrices - USE LOGIT_SCALE AND LOGIT_BIAS like in training!
    logits_per_image = (image_features @ text_features.t()).float()  # [N_images, N_texts]
    
    if logit_scale is not None:
        # Use .mean() like in training validation
        logits_per_image = logit_scale.mean().float() * logits_per_image
    
    if logit_bias is not None:
        logits_per_image = logits_per_image + logit_bias.float()
        
    logits_per_text = logits_per_image.t()  # [N_texts, N_images]
    
    metrics = {}
    detailed_results = []
    
    # Image-to-text retrieval
    i2t_rankings = torch.argsort(logits_per_image, descending=True)  # [N_images, N_texts]
    ground_truth_i2t = torch.arange(len(image_features)).view(-1, 1)  # [N_images, 1]
    i2t_ranks = torch.where(i2t_rankings == ground_truth_i2t)[1].detach().cpu().numpy()
    
    metrics['i2t_mean_rank'] = i2t_ranks.mean() + 1
    metrics['i2t_median_rank'] = np.median(i2t_ranks) + 1
    for k in [1, 5, 10]:
        metrics[f'i2t_R@{k}'] = np.mean(i2t_ranks < k)
    
    # Text-to-image retrieval  
    t2i_rankings = torch.argsort(logits_per_text, descending=True)  # [N_texts, N_images]
    ground_truth_t2i = torch.arange(len(text_features)).view(-1, 1)  # [N_texts, 1]
    t2i_ranks = torch.where(t2i_rankings == ground_truth_t2i)[1].detach().cpu().numpy()
    
    metrics['t2i_mean_rank'] = t2i_ranks.mean() + 1
    metrics['t2i_median_rank'] = np.median(t2i_ranks) + 1
    for k in [1, 5, 10]:
        metrics[f't2i_R@{k}'] = np.mean(t2i_ranks < k)
    
    # Collect detailed results for each image
    for i in range(len(image_features)):
        # Get top-1 retrieved text for this image
        top1_text_idx = i2t_rankings[i, 0].item()
        top1_retrieved_text = texts[top1_text_idx]
        
        # Get similarity scores for top retrievals
        i2t_top_scores = logits_per_image[i, i2t_rankings[i, :10]].detach().cpu().numpy()
        
        result = {
            'image_path': image_paths[i],
            'ground_truth_text': texts[i],
            'top1_retrieved_text': top1_retrieved_text,
            'i2t_rank': i2t_ranks[i] + 1,
            'i2t_top1_score': float(i2t_top_scores[0]),
            'i2t_top5_scores': i2t_top_scores[:5].tolist(),
            'i2t_top10_scores': i2t_top_scores[:10].tolist(),
        }
        detailed_results.append(result)
    
    return metrics, detailed_results

def evaluate_csv(model, csv_path, output_dir, batch_size=8, device="cuda", 
                img_column="img_path", text_column="report", dataset_mode="ct", use_3channel=False, 
                text_separator="!@#$%^&*()", split=None, split_column="split"):
    """Evaluate retrieval performance on a CSV file."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned", padding_side="left")
    
    # Get transforms
    if dataset_mode == "ct":
        transform = get_val_transform()
    else:
        assert False, "Dataset mode not supported"
    
    # Create dataset
    dataset = CustomCSVDataset(
        csv_file=csv_path,
        transform=transform,
        img_key=img_column,
        caption_key=text_column,
        tokenizer=tokenizer,
        is_train=False,
        dataset_mode=dataset_mode,
        use_3channel=use_3channel,
        separator=text_separator,
        split=split,
        split_column=split_column
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    
    logging.info(f"Evaluating {len(dataset)} samples...")
    
    # Extract features
    all_image_features = []
    all_text_features = []
    all_image_paths = []
    all_texts = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            if len(batch) == 2:  # CT format: images, texts
                images, texts = batch
                images = images.to(device, dtype=torch.bfloat16)
                texts = texts.to(device)
                
                # Get current batch info
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(dataset))
                batch_paths = [dataset.data_frame.iloc[i][img_column] for i in range(start_idx, end_idx)]
                batch_texts = [dataset.data_frame.iloc[i][text_column] for i in range(start_idx, end_idx)]
                
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} items")
            
            # Use the same embedding pipeline as training
            image_features = model.encode_image(images)   # already includes projection & normalise
            text_features  = model.encode_text(texts)     # idem
            
            # Store features and metadata
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
            all_image_paths.extend(batch_paths)
            all_texts.extend(batch_texts)
    
    # Concatenate all features
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    
    logging.info("Computing retrieval metrics...")
    
    # Compute metrics
    logit_scale = model.logit_scale.cpu() if hasattr(model, 'logit_scale') else None
    logit_bias = model.logit_bias.cpu() if hasattr(model, 'logit_bias') else None
    
    logging.info(f"Using logit_scale: {logit_scale.item() if logit_scale is not None else 'None'}")
    logging.info(f"Using logit_bias: {logit_bias.item() if logit_bias is not None else 'None'}")
    
    metrics, detailed_results = compute_retrieval_metrics(
        all_image_features, all_text_features, all_image_paths, all_texts, 
        logit_scale=logit_scale,
        logit_bias=logit_bias
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics summary
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(output_dir, "retrieval_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    detailed_path = os.path.join(output_dir, "detailed_retrieval_results.csv")
    detailed_df.to_csv(detailed_path, index=False)
    
    # Print metrics
    logging.info("Retrieval Metrics:")
    logging.info("-" * 50)
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    logging.info(f"\nResults saved to:")
    logging.info(f"  Metrics: {metrics_path}")
    logging.info(f"  Detailed: {detailed_path}")
    
    return metrics, detailed_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate image-text retrieval model")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to model checkpoint")
    parser.add_argument("--csv_file", type=str, required=True,
                       help="Path to CSV file with image paths and reports")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--img_column", type=str, default="img_path",
                       help="Name of image path column in CSV")
    parser.add_argument("--text_column", type=str, default="findings", 
                       help="Name of text/report column in CSV")
    parser.add_argument("--dataset_mode", type=str, default="ct", choices=["ct", "cxr"],
                       help="Dataset mode: 'ct' for CT scans, 'cxr' for chest X-rays")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for inference")
    parser.add_argument("--text_base", type=str, 
                       default="microsoft/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned",
                       help="Base text model name")
    parser.add_argument("--use_3channel", action="store_true", default=False,
                       help="Use 3-channel CT processing (lung/mediastinum/bone windows)")
    parser.add_argument("--text_separator", type=str, default="!@#$%^&*()",
                       help="Text separator used for splitting text (should match training)")
    parser.add_argument("--precision", type=str, default="bf16", 
                       choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
                       help="Floating point precision (should match training)")
    parser.add_argument("--split", type=str, default=None,
                       help="Data split to filter by (e.g., 'train', 'val', 'test'). If None, uses all data.")
    parser.add_argument("--split_column", type=str, default="split",
                       help="Column name for data splits")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Check if files exist
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(f"CSV file not found: {args.csv_file}")
    
    # Load model
    model = load_model(args.checkpoint, args.text_base, args.device, args.precision)
    
    # Run evaluation
    metrics, detailed_results = evaluate_csv(
        model=model,
        csv_path=args.csv_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        img_column=args.img_column,
        text_column=args.text_column,
        dataset_mode=args.dataset_mode,
        use_3channel=args.use_3channel,
        text_separator=args.text_separator,
        split=args.split,
        split_column=args.split_column
    )
    
    print(f"\nEvaluation completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 