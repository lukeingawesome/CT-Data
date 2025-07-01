import random
import re
from dataclasses import dataclass
from multiprocessing import Value
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from health_multimodal.image.data.io import load_image
from .distributed import is_master

import warnings
warnings.filterwarnings("ignore")
# from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # Truncated File Read
Image.MAX_IMAGE_PIXELS = None # DecompressionBombWarning
ImageFile.MAX_IMAGE_PIXELS = None

def shuffle_sentences(text, probability=0.5):
    # Split the text into sentences using a regex to account for periods that end sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Shuffle the sentences
    if random.random() < probability:
        random.shuffle(sentences)
    
    # Join the shuffled sentences back into a single string
    shuffled_text = ' '.join(sentences)
    return shuffled_text

class CustomCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None, img_key='image_path', caption_key='caption', 
                 tokenizer=None, is_train=True, dataset_mode='cxr', split=None, split_column='split'):
        """
        Flexible CSV dataset that supports both CXR temporal data and CT single scan data.
        
        Args:
            csv_file (string): Path to the csv file
            transform (callable, optional): Optional transform to be applied on images
            img_key (string): Column name for image paths
            caption_key (string): Column name for captions/findings
            tokenizer (callable, optional): Optional tokenizer for processing captions
            is_train (bool): Whether this is training data (affects data augmentation)
            dataset_mode (string): 'cxr' for temporal CXR data, 'ct' for single CT scans
            split (string, optional): Data split to filter by (e.g., 'train', 'val', 'test')
            split_column (string): Column name for data splits
        """
        self.data_frame = pd.read_csv(csv_file)
        
        # Filter by split if specified
        if split and split_column in self.data_frame.columns:
            self.data_frame = self.data_frame[self.data_frame[split_column] == split].reset_index(drop=True)
            print(f"Loaded {len(self.data_frame)} samples for {split} split from {csv_file}")
        else:
            print(f"Loaded {len(self.data_frame)} samples from {csv_file}")
        
        self.transform = transform
        self.img_key = img_key
        self.caption_key = caption_key
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.dataset_mode = dataset_mode.lower()
        self.max_length = 256
        
        # Validate required columns based on dataset mode
        if self.dataset_mode == 'cxr':
            # CXR mode requires temporal columns
            required_cols = ['object_id', 'img_path', 'previous_img_path', 'caption', 'label']
            missing_cols = [col for col in required_cols if col not in self.data_frame.columns]
            if missing_cols:
                print(f"Warning: Missing CXR columns {missing_cols}, will use available columns")
        elif self.dataset_mode == 'ct':
            # CT mode requires single image and findings
            required_cols = [img_key, caption_key]
            missing_cols = [col for col in required_cols if col not in self.data_frame.columns]
            if missing_cols:
                raise ValueError(f"Missing required CT columns: {missing_cols}")
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        """Returns one sample of data"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.data_frame.iloc[idx]
        
        if self.dataset_mode == 'cxr':
            return self._get_cxr_item(row, idx)
        elif self.dataset_mode == 'ct':
            return self._get_ct_item(row, idx)
        else:
            raise ValueError(f"Unsupported dataset mode: {self.dataset_mode}")
    
    def _get_cxr_item(self, row, idx):
        """Get CXR temporal data item"""
        # Get image paths and caption
        oid = row.get('object_id', f'sample_{idx}')
        cur_path = Path(row['img_path'])
        prev_path = Path(row['previous_img_path'])
        caption = str(row.get('caption', ''))
        hard_caption = ''  # not yet implemented
        label = row.get('label', 0)
        
        if self.is_train:
            caption = shuffle_sentences(caption, probability=0.2)
        
        # Load and process images
        prev_image = load_image(prev_path)
        cur_image = load_image(cur_path)
        if self.transform:
            prev_image = self.transform(prev_image)
            cur_image = self.transform(cur_image)
        
        return prev_image, cur_image, caption, hard_caption, oid, label
    
    def _get_ct_item(self, row, idx):
        """Get CT single scan item"""
        img_path = row[self.img_key]
        caption = row[self.caption_key]
        
        # Load image from NPZ file (for CT scans)
        try:
            if str(img_path).endswith('.npz'):
                # Load NPZ file (CT scan format)
                arr = np.load(img_path)["image"]  # (C, D, H, W) float16
                image = torch.from_numpy(arr).float()  # cast to float32 for gradients
            else:
                # Load regular image file
                image = load_image(img_path)
                if self.transform:
                    image = self.transform(image)
        except Exception as e:
            raise RuntimeError(f"Failed to load image from {img_path}: {e}")
        
        # Handle missing or NaN findings
        if pd.isna(caption) or not isinstance(caption, str):
            caption = "Medical scan findings"  # Default placeholder
        
        if self.is_train:
            caption = shuffle_sentences(caption, probability=0.2)
        
        # Create metadata
        meta = {
            "id": Path(img_path).stem,
            "img_path": img_path,
            "idx": idx
        }
        
        # Return format compatible with CT training
        return {
            "image": image,
            "text": str(caption),
            "meta": meta
        }
    
    def collate_fn(self, batch):
        """Collate function that handles both CXR and CT data formats"""
        if self.dataset_mode == 'cxr':
            return self._collate_cxr(batch)
        elif self.dataset_mode == 'ct':
            return self._collate_ct(batch)
    
    def _collate_cxr(self, batch):
        """Collate function for CXR temporal data"""
        prev_images, cur_images, captions, hard_captions, oids, labels = zip(*batch)
        prev_images = torch.stack(prev_images)
        cur_images = torch.stack(cur_images)
        
        if self.tokenizer:
            captions = self.tokenizer(
                captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            hard_captions = self.tokenizer(
                hard_captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
        return prev_images, cur_images, captions, hard_captions, oids, labels
    
    def _collate_ct(self, batch):
        """Collate function for CT single scan data"""
        images = []
        texts = []
        metas = []
        
        for item in batch:
            images.append(item["image"])
            texts.append(item["text"])
            metas.append(item["meta"])
        
        # Stack images
        images = torch.stack(images)
        
        # Tokenize texts if tokenizer is provided
        if self.tokenizer:
            texts = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
        
        return {
            "image": images,
            "text": texts,
            "meta": metas
        }

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_cxr_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CustomCSVDataset(
        csv_file=input_filename,
        transform=preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        tokenizer=tokenizer,
        is_train=is_train)
    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=dataset.collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_retrieval_dataset(args, preprocess_fn, is_train=False, tokenizer=None,
                          input_filename=None, dataset_mode='cxr'):
    # input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    
    # Set up parameters based on dataset mode
    if dataset_mode == 'ct':
        # CT mode parameters
        split = getattr(args, 'train_split', 'train') if is_train else getattr(args, 'val_split', 'val')
        img_key = getattr(args, 'csv_img_key', 'img_path')
        caption_key = getattr(args, 'csv_caption_key', 'findings')
        split_column = getattr(args, 'split_column', 'split')
    else:
        # CXR mode parameters (default)
        split = None
        split_column = 'split'
        img_key = args.csv_img_key
        caption_key = args.csv_caption_key
    
    dataset = CustomCSVDataset(
        csv_file=input_filename,
        transform=preprocess_fn,
        img_key=img_key,
        caption_key=caption_key,
        tokenizer=tokenizer,
        is_train=is_train,
        dataset_mode=dataset_mode,
        split=split,
        split_column=split_column)
    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=dataset.collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "cxr":
        return get_cxr_dataset
    elif dataset_type == "ct":
        return get_ct_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns ##TODO: Modify this
    data = {}

    if args.train_data or args.dataset_type in ["cxr", "ct"]:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    return data
