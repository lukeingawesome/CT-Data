import open_clip
import torch
from torch import nn
import torch.nn.functional as F
import os
from PIL import Image

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from torch.utils.data import Dataset
import json
from pathlib import Path
from precision import get_autocast
from tqdm import tqdm
from torch.utils.data import default_collate

def evaluate(model, dataloader, tokenizer,  device, precision, distributed=False,recall_k_list=[1, 5, 10]):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------
    
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    device: cpu/cuda

    precision: floating point precision

    recall_k_list: list of int
        recall@k k's to use
    
    Returns
    -------
    
    dict of retrieval metrics
    """
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    num_batches = dataloader.num_batches
    dataloader = dataloader_with_indices(dataloader)
    autocast = get_autocast(precision)
    pbar = tqdm(total=num_batches)
    # for batch_images, batch_texts, inds in tqdm(dataloader):
    for batch_images, batch_texts, inds in dataloader:
        batch_images = batch_images.to(device)
        # tokenize all texts in the batch
        if tokenizer:
            batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        else:
            # batch_texts_tok = batch_texts.view(-1, 4096).to(device)
            batch_texts_tok = torch.tensor([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        # store the index of image for each text
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]
        # compute the embedding of images and texts
        # with torch.no_grad(), autocast():
        with torch.no_grad(), torch.cuda.amp.autocast():
            if distributed:
                batch_images_emb = F.normalize(model.module.encode_image(batch_images), dim=-1)
                batch_texts_emb = F.normalize(model.module.encode_text(batch_texts_tok), dim=-1)
            else:
                # batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1)
                # batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok), dim=-1)
                batch_images_emb = F.normalize(model.vis_proj(batch_images), dim=-1)
                batch_texts_emb = F.normalize(model.text_proj(batch_texts_tok), dim=-1)
        
        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)
        
        pbar.update(1)
        
    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    # get the score for each text and image pair
    scores  = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item()

    return metrics

def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end

def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)



class RetrievalDataset(Dataset):
    def __init__(self, input_filename, transforms, img_root=None, img_feature_path=None,
                text_feature_path=None, tokenizer=None):
        
        self.meta = json.load(open(input_filename, 'r'))
        self.img_features, self.text_features = None, None
        if img_feature_path:
            if Path(img_feature_path).suffix == '.npy':
                self.img_features = np.memmap(img_feature_path, dtype='float32', mode='r', 
                                        shape=(len(self.meta), 1024))  
            elif Path(img_feature_path).suffix == '.dpt':
                self.img_features = torch.load(img_feature_path)
        if text_feature_path:
            if Path(text_feature_path).suffix == '.npy':
                self.text_features = np.memmap(text_feature_path, dtype='float32', mode='r', 
                                        shape=(len(self.meta), 5, 4096))
            elif Path(text_feature_path).suffix == '.dpt':
                self.text_features = torch.load(text_feature_path)
        self.img_root = img_root
        self.transforms = transforms
        self.tokenize = tokenizer

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        images, texts = None, None
        if self.img_features is not None:
            images = self.img_features[idx]
        if self.text_features is not None:
            texts = np.array(self.text_features[idx])
        if images is None:
            image_path = os.path.join(self.img_root, self.meta[idx]['image'])
            images = self.transforms(Image.open(image_path))
        if texts is None:
            texts = self.meta[idx]['caption']
            if self.tokenize:
                texts = self.tokenize([texts])[0]     
        return images, texts


import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class FeaturePairRetrievalDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_feat_path, text_feat_path):
        """
        vis_feat_path (string): path to tensor dict. file containing visual features
        text_feat_path (string): path to tensor dict. file containing text features
        """
        self.vis_feats = torch.load(vis_feat_path)
        self.text_feats = torch.load(text_feat_path)
        assert "txt2img" in self.text_feats.keys(), "missing mapping from text to visual features, ensure feature extraction was done correctly."
        assert "img2txt" in self.text_feats.keys(), "missing mapping from visual to text features, ensure feature extraction was done correctly."
        self.txt2img = self.text_feats.pop("txt2img")
        self.img2txt = self.text_feats.pop("img2txt")
        assert len(self.vis_feats) == len(self.img2txt), "length mismatch: there are {} visual features and {} visual mappings.".format(len(self.vis_feats), len(self.img2txt))
        assert len(self.text_feats) == len(self.txt2img), "length mismatch: there are {} text features and {} text mappings.".format(len(self.text_feats), len(self.txt2img))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        
        self.text_feats_list = []
        for i in range(len(self.text_feats)):
            text_embed = self.text_feats[str(i)]
            self.text_feats_list.append(text_embed)

    def __len__(self):
        return len(self.vis_feats)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor 

    def __getitem__(self, index):
        vis_embed = self.vis_feats[str(index)]
        # vis_embed = self.vis_processor(vis_embed)

        return {
            "vis_embed": vis_embed,
            "instance_id": str(index)
        }


class Block(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Linear(dim, int(expansion_factor * dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(expansion_factor * dim), dim),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.fn(self.ln(x))


class MLPContrastiveFusion(nn.Module):
    def __init__(self, vis_embed_dim=768, text_embed_dim=4096, proj_embed_dim=512,
                 proj_bias=True, num_layers_vis=4, num_layers_text=4,
                 expansion_factor=4, dropout=0., unimodal_loss_coeff=1.0):
        super().__init__()
        self.vis_embed_dim = vis_embed_dim
        self.text_embed_dim = text_embed_dim
        self.proj_embed_dim = proj_embed_dim
        self.unimodal_loss_coeff = unimodal_loss_coeff
        self.mixup_alpha = -1
        
        self.vis_proj =  nn.Linear(vis_embed_dim, proj_embed_dim)
        expansion_factor = 2
        self.text_proj = nn.Sequential(
            *[Block(text_embed_dim, expansion_factor, dropout) for _ in range(num_layers_text)],
            nn.LayerNorm(text_embed_dim),
            nn.Linear(text_embed_dim, proj_embed_dim, bias=proj_bias),
        )

        self.temp = nn.Parameter(0.07 * torch.ones([]))

    def proj_vis(self, vis_embed):
        return self.vis_proj(vis_embed)

    def proj_text(self, text_embed):
        return self.text_proj(text_embed)

    def load_checkpoint(self, url_or_filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """
        if os.path.isfile(url_or_filename):
            print('loading')
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=True)
        return msg
    
def image_captions_collate_fn(batch):
    transposed = list(zip(*batch))
    imgs = default_collate(transposed[0])
    texts = transposed[1]

    return imgs, texts
 
def main():
    device = 'cuda' 

    ckpt_path = '/home/aiscuser/fusemix/lavis/output/fusion/mlp_contrastive_fusion/pretrain/pre_clip_vitL14_336_llm3-vec_sharegpt4v_coco_vg_sbu_cap_cc3m_Rwds/1280proj_Texp2_06drop_0noise_1e6warm100_1e4to1e-5lr_cosine_500ep_01wd_40kX3bs/20240723102/checkpoint_best.pth'
    fusion_model = MLPContrastiveFusion(vis_embed_dim=1024,proj_embed_dim=1280).to(device)
    fusion_model.eval()
    fusion_model.load_checkpoint(ckpt_path)
   
   
    dataset = RetrievalDataset('/home/aiscuser/data/cache/coco/annotations/coco_karpathy_test.json',
                               None, None,'/home/aiscuser/fusemix/EVA-CLIP/data/coco_ret_vitl336.dpt',
                               '/home/aiscuser/fusemix/EVA-CLIP/data/coco_test_llm_features.dpt')
    # dataset = RetrievalDataset('/home/aiscuser/data/cache/flickr30k/annotations/test.json',
    #                            None, None,
    #                            '/home/aiscuser/fusemix/EVA-CLIP/data/flickr_ret_vitl336.dpt',
    #                             "/home/aiscuser/fusemix/EVA-CLIP/data/flickr30k_test_llm_features.dpt")
                            #    '/home/aiscuser/fusemix/EVA-CLIP/data/flickr30k_test_llm_features.dpt')
                            #    "/home/aiscuser/data/features_npy/clip_feature_extractor/vitl14_336_pre_logits/flickr30k/test.npy",
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        num_workers=0,
        sampler=None,
        shuffle=False,
        drop_last=False,
        collate_fn=image_captions_collate_fn
    )
    dataloader.num_batches = len(dataset)//1024
    metrics = evaluate(fusion_model, dataloader, None, 'cuda', precision='float32')
    print(metrics)

@torch.no_grad()
def compute_sim_matrix(model, data_loader, device, **kwargs):

    # start_time = time.time()
    model.eval()
    vis_embeds_proj_norm = []
    for samples in data_loader:
        vis_embed = samples["vis_embed"].to(device)
        vis_embed_proj_norm = F.normalize(model.proj_vis(vis_embed), dim=-1)
        vis_embeds_proj_norm.append(vis_embed_proj_norm)
    vis_embeds_proj_norm = torch.cat(vis_embeds_proj_norm, dim=0)

    text_feats = data_loader.dataset.text_feats_list
    num_text = len(text_feats)
    text_bs = 1024
    text_embeds_proj_norm = []
    for i in range(0, num_text, text_bs):
        text_embed = torch.stack(text_feats[i: min(num_text, i + text_bs)]).to(device)
        text_embed_proj_norm = F.normalize(model.proj_text(text_embed), dim=-1)
        text_embeds_proj_norm.append(text_embed_proj_norm)
    text_embeds_proj_norm = torch.cat(text_embeds_proj_norm, dim=0)

    sim_v2t = vis_embeds_proj_norm @ text_embeds_proj_norm.T
    sim_t2v = sim_v2t.T

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    return sim_v2t.cpu().numpy(), sim_t2v.cpu().numpy()

def _report_metrics(scores_i2t, scores_t2i, txt2img, img2txt):

        # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    agg_metrics = (tr1 + tr5 + tr10) / 3

    eval_result = {
        "txt_r1": tr1,
        "txt_r5": tr5,
        "txt_r10": tr10,
        "txt_r_mean": tr_mean,
        "img_r1": ir1,
        "img_r5": ir5,
        "img_r10": ir10,
        "img_r_mean": ir_mean,
        "r_mean": r_mean,
        "agg_metrics": agg_metrics,
    }
    return eval_result
        
def main_2():
    device = 'cuda' 
    ckpt_path = '/home/aiscuser/fusemix/lavis/output/fusion/mlp_contrastive_fusion/pretrain/pre_clip_vitL14_336_llm3-vec_sharegpt4v_coco_vg_sbu_cap_cc3m_Rwds/1280proj_Texp2_06drop_0noise_1e6warm100_1e4to1e-5lr_cosine_500ep_01wd_40kX3bs/20240723102/checkpoint_best.pth'
    fusion_model = MLPContrastiveFusion(vis_embed_dim=1024,proj_embed_dim=1280).to(device)
    fusion_model.eval()
    fusion_model.load_checkpoint(ckpt_path)
    fusion_model.to(device)
    # vis_feat_path ='/home/aiscuser/data/cache/features/clip_feature_extractor/vitl14_336_pre_logits/flickr30k/test.dpt'
    # text_feat_path = '/home/aiscuser/data/cache/features/Llama_feature_extractor/8b/flickr30k/test.dpt'
    
    vis_feat_path ='/home/aiscuser/data/cache/features/clip_feature_extractor/vitl14_336_pre_logits/coco_retrieval/test.dpt'
    text_feat_path = '/home/aiscuser/data/cache/features/Llama_feature_extractor/8b/coco_retrieval/test.dpt'
    
    dataset = FeaturePairRetrievalDataset(None,None,vis_feat_path,text_feat_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        num_workers=0,
        sampler=None,
        shuffle=False,
        drop_last=False,
        # collate_fn=image_captions_collate_fn
    )
    score_i2t, score_t2i = compute_sim_matrix(fusion_model,dataloader, device)
    metrics = _report_metrics(score_i2t, score_t2i,dataloader.dataset.txt2img,
                dataloader.dataset.img2txt,)
    import ipdb; ipdb.set_trace()
    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # os.environ["TIMM_FUSED_ATTN"] = "0"
    # from timm.layers.config import set_fused_attn
    # set_fused_attn(False)
    
    main()
    # main_2()