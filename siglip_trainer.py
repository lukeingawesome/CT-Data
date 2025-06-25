import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import monai
from typing import List, Dict, Any, Optional, Tuple
import argparse
import logging
import wandb
from tqdm import tqdm
import os
import json
import pandas as pd
from datetime import datetime

# Import modules from existing codebase
import sys
sys.path.append('./llm2vec4cxr')
from llm2vec_wrapper import LLM2VecWrapper as LLM2Vec
from merlin import Merlin


class CSVDataset(monai.data.Dataset):
    """
    Dataset for loading NPZ files from CSV with img_path, findings, and split columns.
    
    Returns:
        dict(
            image = FloatTensor (C=2, D, H, W) – float32 for training,
            text = str (radiology findings),
            meta = { 'id': <str>, 'img_path': <str> }
        )
    """
    def __init__(self, csv_file: str, split: str = 'train', findings_column: str = 'findings', 
                 img_path_column: str = 'img_path', split_column: str = 'split'):
        self.csv_file = csv_file
        self.split = split
        self.findings_column = findings_column
        self.img_path_column = img_path_column
        self.split_column = split_column
        
        # Load CSV and filter by split
        self.df = pd.read_csv(csv_file)
        if split_column in self.df.columns:
            self.df = self.df[self.df[split_column] == split].reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} samples for {split} split from {csv_file}")
        
        # Validate required columns
        required_cols = [img_path_column, findings_column]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        
        super().__init__(data=list(range(len(self.df))))

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = row[self.img_path_column]
        findings = row[self.findings_column]
        
        # Load image from NPZ file
        try:
            arr = np.load(img_path)["image"]       # (C, D, H, W) float16
            arr = torch.from_numpy(arr).float()    # cast to float32 for gradients
        except Exception as e:
            raise RuntimeError(f"Failed to load image from {img_path}: {e}")
        
        # Handle missing or NaN findings
        if pd.isna(findings) or not isinstance(findings, str):
            findings = "Chest CT scan findings"  # Default placeholder
        
        result = {
            "image": arr, 
            "text": str(findings),
            "meta": {
                "id": Path(img_path).stem,
                "img_path": img_path,
                "idx": idx
            }
        }
        
        return result


class NPZDataset(monai.data.Dataset):
    """
    Legacy dataset for loading NPZ files from directory (kept for backward compatibility).
    
    Returns:
        dict(
            image = FloatTensor (C=2, D, H, W) – float32 for training,
            meta  = { 'id': <str> },
            text  = str (optional)
        )
    """
    def __init__(self, files: List[Path], text_reports: Optional[Dict[str, str]] = None):
        self.files = files
        self.text_reports = text_reports or {}
        super().__init__(data=files)

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fname = self.files[idx]
        arr = np.load(fname)["image"]       # (C, D, H, W) float16
        arr = torch.from_numpy(arr).float() # cast to float32 for gradients
        
        result = {"image": arr, "meta": {"id": fname.stem}}
        
        # Add text report if available
        if self.text_reports and fname.stem in self.text_reports:
            result["text"] = self.text_reports[fname.stem]
        else:
            # Use a default placeholder text if no report is available
            result["text"] = "Chest CT scan findings"
        
        return result


def get_csv_loader(csv_file: str,
                   split: str = 'train',
                   findings_column: str = 'findings',
                   img_path_column: str = 'img_path', 
                   split_column: str = 'split',
                   batchsize: int = 2,
                   shuffle: bool = True,
                   num_workers: int = 8):
    """Create a DataLoader from CSV file."""
    ds = CSVDataset(
        csv_file=csv_file,
        split=split,
        findings_column=findings_column,
        img_path_column=img_path_column,
        split_column=split_column
    )
    return monai.data.DataLoader(ds,
                                 batch_size=batchsize,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 persistent_workers=num_workers > 0)


def get_loader(npz_root: str,
               text_reports: Optional[Dict[str, str]] = None,
               batchsize: int = 2,
               shuffle: bool = True,
               num_workers: int = 8):
    """Create a DataLoader for NPZ files (legacy function)."""
    files = sorted(Path(npz_root).glob("*.npz"))
    ds = NPZDataset(files, text_reports)
    return monai.data.DataLoader(ds,
                                 batch_size=batchsize,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 persistent_workers=num_workers > 0)


class TextEncoder(nn.Module):
    """Text encoder using LLM2Vec from tutorial.py"""
    
    def __init__(self, model_path: Optional[str] = None, max_length: int = 512):
        super().__init__()
        self.max_length = max_length
        
        # Load LLM2Vec model similar to tutorial.py
        self.text_model = LLM2Vec.from_pretrained(
            base_model_name_or_path='microsoft/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned',
            enable_bidirectional=True,
            pooling_mode="latent_attention" if model_path else "mean",
            max_length=max_length,
            torch_dtype=torch.bfloat16,
        )
        
        # Load checkpoint if provided
        if model_path and os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, weights_only=True)
                self.text_model.load_state_dict(ckpt, strict=False)
                logging.info(f"Loaded text model checkpoint from {model_path}")
            except Exception as e:
                logging.warning(f"Failed to load checkpoint: {e}")
        
        # Configure tokenizer
        self.tokenizer = self.text_model.tokenizer
        self.tokenizer.padding_side = 'left'
        
        # Get the hidden size from the model config
        hidden_size = getattr(self.text_model.config, 'hidden_size', 4096)
        
        # Projection head to match image embedding dimension
        self.projection = nn.Linear(hidden_size, 512)
        
    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize texts with special handling for separator-based splitting."""
        texts_2 = []
        original_texts = []
        separator = '!@#$%^&*()'
        
        for text in texts:
            parts = text.split(separator)
            texts_2.append(parts[1] if len(parts) > 1 else "")
            original_texts.append("".join(parts))

        # Tokenize original texts
        tokenized = self.tokenizer(
            original_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        # Create embedding masks
        embed_mask = None
        for t_i, t in enumerate(texts_2):
            ids = self.tokenizer(
                [t],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            
            e_m = torch.zeros_like(tokenized["attention_mask"][t_i])
            if len(ids["input_ids"][0]) > 0:
                e_m[-len(ids["input_ids"][0]):] = torch.ones(len(ids["input_ids"][0]))
                
            if embed_mask is None:
                embed_mask = e_m.unsqueeze(0)
            else:
                embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)

        tokenized["embed_mask"] = embed_mask
        return tokenized
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Forward pass for text encoding."""
        tokenized = self.tokenize(texts)
        device = next(self.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        # Get embeddings from LLM2Vec
        with torch.cuda.amp.autocast():
            embeddings = self.text_model(tokenized)
        
        # Project to desired dimension
        embeddings = self.projection(embeddings)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings


class ImageEncoder(nn.Module):
    """Image encoder using Merlin model"""
    
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        # Load Merlin model for image embedding
        self.merlin_model = Merlin(ImageEmbedding=True)
        
        # Freeze the base model initially (can be unfrozen later)
        if freeze_backbone:
            for param in self.merlin_model.parameters():
                param.requires_grad = False
            
        # Add a projection head for contrastive learning
        # Merlin model outputs 2048-dimensional features
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512)
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for image encoding."""
        with torch.cuda.amp.autocast():
            # Get features from Merlin (should return shape [batch_size, 2048])
            features = self.merlin_model(images)
            
            # Ensure features are in the right shape
            if len(features.shape) > 2:
                features = features.squeeze()
            
            # Project to desired dimension
            embeddings = self.projection(features)
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
        return embeddings


class SigLIPLoss(nn.Module):
    """
    SigLIP loss implementation
    Paper: https://arxiv.org/abs/2303.15343
    """
    
    def __init__(self, temperature: float = 1.0, bias: float = -10.0):
        super().__init__()
        self.temperature = temperature
        self.bias = bias
        
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute SigLIP loss
        
        Args:
            image_features: [batch_size, embed_dim]
            text_features: [batch_size, embed_dim]
        """
        batch_size = image_features.shape[0]
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.t()) / self.temperature + self.bias
        
        # Create labels (positive pairs are on the diagonal)
        labels = torch.eye(batch_size, device=logits.device, dtype=torch.float32)
        labels = labels * 2 - 1  # Convert to {-1, 1}
        
        # Compute loss using sigmoid and binary cross entropy
        loss = -torch.sum(F.logsigmoid(labels * logits)) / batch_size
        
        return loss


class ChestCTCLIP(nn.Module):
    """Complete Chest CT CLIP model with SigLIP training"""
    
    def __init__(self, text_model_path: Optional[str] = None, 
                 temperature: float = 1.0, freeze_image_backbone: bool = True):
        super().__init__()
        self.image_encoder = ImageEncoder(freeze_backbone=freeze_image_backbone)
        self.text_encoder = TextEncoder(model_path=text_model_path)
        self.loss_fn = SigLIPLoss(temperature=temperature)
        
    def forward(self, images: torch.Tensor, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning image features, text features, and loss."""
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)
        loss = self.loss_fn(image_features, text_features)
        
        return image_features, text_features, loss


class SigLIPTrainer:
    """Trainer class for SigLIP Chest CT CLIP model"""
    
    def __init__(self, 
                 model: ChestCTCLIP,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 warmup_steps: int = 1000,
                 max_steps: int = 100000,
                 save_dir: str = "./checkpoints",
                 log_interval: int = 100,
                 save_interval: int = 5000,
                 use_wandb: bool = False):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=warmup_steps,
            T_mult=2,
            eta_min=learning_rate * 0.01
        )
        
        # Setup scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Training parameters
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Setup logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="chest-ct-clip", config={
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
            })
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_step(self, batch: Dict[str, Any]) -> float:
        """Single training step"""
        self.model.train()
        
        images = batch["image"].to(next(self.model.parameters()).device)
        texts = batch["text"]  # List of strings
        
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            image_features, text_features, loss = self.model(images, texts)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        return loss.item()
    
    def validate(self) -> float:
        """Validation step"""
        if self.val_loader is None:
            return 0.0
            
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(next(self.model.parameters()).device)
                texts = batch["text"]
                
                with torch.cuda.amp.autocast():
                    _, _, loss = self.model(images, texts)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint at step {self.global_step}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        train_iter = iter(self.train_loader)
        
        progress_bar = tqdm(total=self.max_steps, desc="Training")
        
        while self.global_step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Training step
            loss = self.train_step(batch)
            self.global_step += 1
            
            # Logging
            if self.global_step % self.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                self.logger.info(f"Step {self.global_step}: Loss = {loss:.4f}, LR = {lr:.2e}")
                
                if self.use_wandb:
                    wandb.log({
                        "train_loss": loss,
                        "learning_rate": lr,
                        "step": self.global_step
                    })
            
            # Validation and saving
            if self.global_step % self.save_interval == 0:
                val_loss = self.validate()
                
                if val_loss > 0:
                    self.logger.info(f"Validation loss: {val_loss:.4f}")
                    if self.use_wandb:
                        wandb.log({"val_loss": val_loss, "step": self.global_step})
                
                # Save checkpoint
                is_best = val_loss < self.best_val_loss and val_loss > 0
                if is_best:
                    self.best_val_loss = val_loss
                
                self.save_checkpoint(is_best=is_best)
            
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss:.4f}"})
        
        progress_bar.close()
        self.logger.info("Training completed!")


def load_text_reports(reports_path: str) -> Dict[str, str]:
    """Load text reports from JSON file"""
    if os.path.exists(reports_path):
        with open(reports_path, 'r') as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description='Train SigLIP Chest CT CLIP model')
    
    # Data arguments - CSV mode
    parser.add_argument('--csv_file', type=str, default=None,
                        help='Path to CSV file with img_path, findings, and split columns')
    parser.add_argument('--train_split', type=str, default='train',
                        help='Split name for training data in CSV')
    parser.add_argument('--val_split', type=str, default='val',
                        help='Split name for validation data in CSV')
    parser.add_argument('--findings_column', type=str, default='findings',
                        help='Column name for text findings in CSV')
    parser.add_argument('--img_path_column', type=str, default='img_path',
                        help='Column name for image paths in CSV')
    parser.add_argument('--split_column', type=str, default='split',
                        help='Column name for data splits in CSV')
    
    # Data arguments - Legacy directory mode (kept for backward compatibility)
    parser.add_argument('--train_data_dir', type=str, default=None,
                        help='Directory containing training NPZ files (legacy mode)')
    parser.add_argument('--val_data_dir', type=str, default=None,
                        help='Directory containing validation NPZ files (legacy mode)')
    parser.add_argument('--text_reports_path', type=str, default=None,
                        help='Path to JSON file containing text reports (legacy mode)')
    
    # Model arguments
    parser.add_argument('--text_model_path', type=str, default=None,
                        help='Path to pretrained text model checkpoint')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for SigLIP loss')
    parser.add_argument('--freeze_image_backbone', action='store_true',
                        help='Freeze the image encoder backbone')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Maximum number of training steps')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loader workers')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='Checkpoint saving interval')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine data loading mode
    if args.csv_file:
        # CSV mode - preferred method
        if not os.path.exists(args.csv_file):
            raise ValueError(f"CSV file not found: {args.csv_file}")
        
        print(f"Loading data from CSV: {args.csv_file}")
        
        # Create train loader
        train_loader = get_csv_loader(
            csv_file=args.csv_file,
            split=args.train_split,
            findings_column=args.findings_column,
            img_path_column=args.img_path_column,
            split_column=args.split_column,
            batchsize=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        
        # Create validation loader if validation split exists
        val_loader = None
        df_check = pd.read_csv(args.csv_file)
        if args.split_column in df_check.columns and args.val_split in df_check[args.split_column].values:
            print(f"Creating validation loader for split: {args.val_split}")
            val_loader = get_csv_loader(
                csv_file=args.csv_file,
                split=args.val_split,
                findings_column=args.findings_column,
                img_path_column=args.img_path_column,
                split_column=args.split_column,
                batchsize=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )
        else:
            print(f"No validation split '{args.val_split}' found in CSV")
            
    elif args.train_data_dir:
        # Legacy directory mode
        print("Using legacy directory mode")
        
        # Load text reports if provided
        text_reports = load_text_reports(args.text_reports_path) if args.text_reports_path else None
        
        # Create data loaders
        print(f"Loading training data from: {args.train_data_dir}")
        train_loader = get_loader(
            npz_root=args.train_data_dir,
            text_reports=text_reports,
            batchsize=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        
        val_loader = None
        if args.val_data_dir:
            print(f"Loading validation data from: {args.val_data_dir}")
            val_loader = get_loader(
                npz_root=args.val_data_dir,
                text_reports=text_reports,
                batchsize=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )
    else:
        raise ValueError("Must provide either --csv_file or --train_data_dir")
    
    # Create model
    print("Creating model...")
    model = ChestCTCLIP(
        text_model_path=args.text_model_path,
        temperature=args.temperature,
        freeze_image_backbone=args.freeze_image_backbone
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = SigLIPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        use_wandb=args.use_wandb
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main() 