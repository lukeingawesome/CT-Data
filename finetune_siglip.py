#!/usr/bin/env python3
"""
SigLIP Finetuning Script for Chest CT Multilabel Classification

This script loads a pretrained SigLIP model and finetunes it for multilabel classification
on chest CT images. It supports various training configurations, loss functions, and
evaluation metrics.

Author: Medical AI Engineering Team
"""

import argparse
import logging
import os
import random
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from transformers.optimization import get_cosine_schedule_with_warmup

try:
    from torch.optim.swa_utils import AveragedModel, update_bn
    EMA_AVAILABLE = True
except ImportError:
    EMA_AVAILABLE = False
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available, logging disabled")

# Add training modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), "training"))

from training.data import CustomCSVDataset
from training.ct_transform import get_train_transform, get_val_transform
from merlin import Merlin

warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for multilabel classification to handle class imbalance.
    
    Args:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: Specifies the reduction to apply to the output
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BalancedBCELoss(nn.Module):
    """
    Balanced Binary Cross Entropy Loss for multilabel classification.
    
    Args:
        pos_weights: Tensor of positive class weights for each label
    """
    
    def __init__(self, pos_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weights = pos_weights
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weights)


class SigLIPClassifier(nn.Module):
    """
    SigLIP-based multilabel classifier for chest CT images.
    
    Args:
        backbone: Pretrained SigLIP visual encoder
        num_classes: Number of output classes
        dropout_rate: Dropout probability for regularization
        freeze_up_to: Number of layers to freeze from the beginning
    """
    
    def __init__(
        self, 
        backbone: nn.Module, 
        num_classes: int = 18,
        dropout_rate: float = 0.1,
        freeze_up_to: int = 0
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        
        # Freeze specified layers
        if freeze_up_to > 0:
            self._freeze_layers(freeze_up_to)
            
        # Get backbone output dimension
        backbone_dim = self._get_backbone_dim()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_dim // 2, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
        
    def _get_backbone_dim(self) -> int:
        """Determine the output dimension of the backbone."""
        # For Merlin model, the output dimension is typically 2048
        if hasattr(self.backbone, 'output_dim'):
            return self.backbone.output_dim
        else:
            # Fallback: run a dummy forward pass
            dummy_input = torch.randn(1, 1, 160, 224, 224)
            with torch.no_grad():
                dummy_output = self.backbone(dummy_input)
            return dummy_output.shape[-1]
    
    def _freeze_layers(self, freeze_up_to: int):
        """Freeze the first `freeze_up_to` layers of the backbone."""
        layers_frozen = 0
        for name, param in self.backbone.named_parameters():
            if layers_frozen < freeze_up_to:
                param.requires_grad = False
                layers_frozen += 1
            else:
                break
        logger.info(f"Froze {layers_frozen} layers of the backbone")
    
    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, depth, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # This also calls torch.cuda.manual_seed_all()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_pos_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Calculate positive class weights for balanced BCE loss.
    
    Args:
        labels: Array of shape (n_samples, n_classes) with binary labels
        
    Returns:
        Tensor of positive weights for each class
    """
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    pos_weights = neg_counts / (pos_counts + 1e-8)  # Add small epsilon to avoid division by zero
    return torch.from_numpy(pos_weights).float()


class CTMultilabelDataset(Dataset):
    """
    Custom dataset for CT multilabel classification.
    
    Args:
        csv_file: Path to CSV file with data
        label_columns: List of label column names
        transform: Optional transform to be applied on images
        split: Data split to filter by ('train' or 'val')
        use_3channel: Whether to use 3-channel CT windows
    """
    
    def __init__(
        self,
        csv_file: str,
        label_columns: List[str],
        transform=None,
        split: str = 'train',
        use_3channel: bool = True
    ):
        self.data_frame = pd.read_csv(csv_file)
        self.label_columns = label_columns
        self.transform = transform
        self.use_3channel = use_3channel
        
        # Filter by split
        if split:
            self.data_frame = self.data_frame[self.data_frame['split'] == split].reset_index(drop=True)
            logger.info(f"Loaded {len(self.data_frame)} samples for {split} split")
        
        # Verify required columns
        required_cols = ['series_id', 'split', 'img_path'] + label_columns
        missing_cols = [col for col in required_cols if col not in self.data_frame.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.data_frame.iloc[idx]
        
        # Get image path and labels
        img_path = row['img_path']
        labels = row[self.label_columns].values.astype(np.float32)
        
        # Load image from NPZ file
        try:
            if str(img_path).endswith('.npz'):
                with np.load(img_path) as npz_file:
                    if "image" not in npz_file:
                        raise KeyError(f"'image' key not found in NPZ file: {img_path}")
                    
                    arr = npz_file["image"]  # (C, D, H, W) float16/32
                    
                    # Validate array shape
                    if arr.ndim != 4:
                        raise ValueError(f"Expected 4D array (C, D, H, W), got {arr.ndim}D in {img_path}")
                    
                    if self.use_3channel:
                        # Convert to HU if needed
                        if arr.max() <= 1.0:
                            arr = arr * 2500.0 - 1000.0
                        
                        # Use first channel and create 3-channel windows
                        if arr.ndim == 4:
                            arr = arr[0]  # (D, H, W)
                        
                        # Create lung, mediastinum, and bone windows
                        lung = _hu_window_to_unit(arr, -600, 1000)
                        medi = _hu_window_to_unit(arr, 40, 400)
                        bone = _hu_window_to_unit(arr, 700, 1500)
                        
                        image = np.stack([lung, medi, bone], axis=0)  # (3, D, H, W)
                    else:
                        image = arr  # Keep original format
                    
                    # Convert to tensor
                    image = torch.from_numpy(image.copy()).float()
                    
                    # Apply transforms
                    if self.transform:
                        image = self.transform(image)
                        
            else:
                raise ValueError(f"Unsupported file format: {img_path}")
                
        except Exception as e:
            logger.error(f"Failed to load image from {img_path}: {e}")
            # Return dummy data for failed loads
            if self.use_3channel:
                image = torch.zeros(3, 160, 224, 224, dtype=torch.float32)
            else:
                image = torch.zeros(1, 160, 224, 224, dtype=torch.float32)
            labels = np.zeros(len(self.label_columns), dtype=np.float32)
        
        return image, torch.from_numpy(labels)


def _hu_window_to_unit(volume: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Clip a HU volume to the given window and scale to [0,1].
    
    Args:
        volume: Raw HU ndarray, shape (D,H,W)
        center: Window centre in HU
        width: Window width in HU
    """
    lower, upper = center - width / 2.0, center + width / 2.0
    vol = np.clip(volume, lower, upper)
    return (vol - lower) / (upper - lower)


def get_dataloaders(
    csv_path: str,
    label_columns: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    use_3channel: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        csv_path: Path to CSV file with data
        label_columns: List of label column names
        batch_size: Batch size for training
        num_workers: Number of worker processes
        use_3channel: Whether to use 3-channel CT windows
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get transforms
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    # Create datasets
    train_dataset = CTMultilabelDataset(
        csv_file=csv_path,
        label_columns=label_columns,
        transform=train_transform,
        split='train',
        use_3channel=use_3channel
    )
    
    val_dataset = CTMultilabelDataset(
        csv_file=csv_path,
        label_columns=label_columns,
        transform=val_transform,
        split='val',
        use_3channel=use_3channel
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    return train_loader, val_loader


def build_model(
    pretrained_path: str,
    num_classes: int = 18,
    dropout_rate: float = 0.1,
    freeze_up_to: int = 0
) -> SigLIPClassifier:
    """
    Build the SigLIP classifier model.
    
    Args:
        pretrained_path: Path to pretrained SigLIP checkpoint
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        freeze_up_to: Number of layers to freeze
        
    Returns:
        SigLIPClassifier model
    """
    # Create Merlin backbone (this should match the pretrained model)
    backbone = Merlin(ImageEmbedding=True)
    
    # Load pretrained weights
    if os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Extract visual encoder weights (assumes the checkpoint contains the full model)
        visual_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('visual.'):
                # Remove 'visual.' prefix
                new_key = key[7:]
                visual_state_dict[new_key] = value
        
        # Load the visual encoder weights
        backbone.load_state_dict(visual_state_dict, strict=False)
        logger.info("Successfully loaded pretrained visual encoder weights")
    else:
        logger.warning(f"Pretrained weights not found at {pretrained_path}, using random initialization")
    
    # Create classifier
    model = SigLIPClassifier(
        backbone=backbone,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_up_to=freeze_up_to
    )
    
    return model


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    log_interval: int = 50,
    warmup_scheduler=None,
    ema_model=None
) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scaler: Gradient scaler for mixed precision
        device: Device to train on
        epoch: Current epoch number
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        log_interval: Steps between logging
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(
        train_loader, 
        desc=f"Epoch {epoch}",
        leave=False,
        dynamic_ncols=True
    )
    
    for batch_idx, batch in enumerate(progress_bar):
        # Handle batch format - should be (images, labels) for CT classification
        images, labels = batch
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            optimizer.zero_grad()
            
            # Step warmup scheduler if provided
            if warmup_scheduler is not None:
                warmup_scheduler.step()
            
            # Update EMA model if provided
            if ema_model is not None:
                ema_model.update_parameters(model)
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
        
        # Log intermediate results
        if batch_idx % log_interval == 0:
            logger.info(
                f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                f"Loss: {loss.item() * gradient_accumulation_steps:.4f}"
            )
    
    avg_loss = total_loss / num_batches
    return {'train_loss': avg_loss}


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_names: List[str],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate the model on validation data.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on
        label_names: Names of the labels
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            # Handle batch format
            images, labels = batch
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get probabilities and predictions
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > threshold).float()
            
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_probabilities = np.concatenate(all_probabilities, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    
    # Per-label metrics
    per_label_metrics = {}
    for i, label_name in enumerate(label_names):
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels[:, i], all_predictions[:, i], average='binary', zero_division='warn'
        )
        try:
            auc = roc_auc_score(all_labels[:, i], all_probabilities[:, i])
        except ValueError:
            auc = 0.0  # Handle case where all labels are the same
        
        per_label_metrics[f'{label_name}_precision'] = precision
        per_label_metrics[f'{label_name}_recall'] = recall
        per_label_metrics[f'{label_name}_f1'] = f1
        per_label_metrics[f'{label_name}_auc'] = auc
    
    # Overall metrics
    macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division='warn')
    micro_f1 = f1_score(all_labels, all_predictions, average='micro', zero_division='warn')
    
    metrics = {
        'val_loss': avg_loss,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        **per_label_metrics
    }
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: Dict[str, float],
    output_dir: str,
    prefix: str = "checkpoint",
    ema_model=None
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    # Save EMA model state if available
    if ema_model is not None:
        checkpoint['ema_state_dict'] = ema_model.state_dict()
    
    checkpoint_path = os.path.join(output_dir, f"{prefix}_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


class CheckpointManager:
    """Efficient checkpoint management that tracks top-k models in memory."""
    
    def __init__(self, keep_top_k: int = 3):
        self.keep_top_k = keep_top_k
        self.top_checkpoints = []  # list of (macro_f1, path)
    
    def add_checkpoint(self, macro_f1: float, checkpoint_path: str):
        """Add a new checkpoint and maintain top-k."""
        self.top_checkpoints.append((macro_f1, checkpoint_path))
        self.top_checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Remove checkpoints beyond top-k
        for score, path in self.top_checkpoints[self.keep_top_k:]:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed checkpoint: {path}")
        
        self.top_checkpoints = self.top_checkpoints[:self.keep_top_k]
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get the path to the best checkpoint."""
        if self.top_checkpoints:
            return self.top_checkpoints[0][1]
        return None


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    label_names: List[str],
    output_path: str,
    threshold: float = 0.5
):
    """Test the model and save predictions to CSV."""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_ids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing", leave=False)):
            images, _ = batch  # Labels are not needed for testing
            # Generate sample IDs for testing
            batch_size = len(images)
            batch_ids = [f"sample_{batch_idx * batch_size + i}" for i in range(batch_size)]
            all_ids.extend(batch_ids)
            
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > threshold).float()
            
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_probabilities = np.concatenate(all_probabilities, axis=0)
    
    # Create results DataFrame
    results = pd.DataFrame({'id': all_ids})
    
    # Add probabilities
    for i, label_name in enumerate(label_names):
        results[f'{label_name}_prob'] = all_probabilities[:, i]
        results[f'{label_name}_pred'] = all_predictions[:, i].astype(int)
    
    # Save to CSV
    results.to_csv(output_path, index=False)
    logger.info(f"Test results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SigLIP Finetuning for Chest CT Classification")
    
    # Data arguments
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with data")
    parser.add_argument("--pretrained", type=str, required=True, help="Path to pretrained SigLIP model")
    parser.add_argument("--label-columns", nargs="+", required=True, help="List of label column names")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw", help="Optimizer")
    parser.add_argument("--scheduler", choices=["plateau", "cosine", "warmup"], default="plateau", help="Learning rate scheduler")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup steps")
    
    # Model arguments
    parser.add_argument("--freeze-up-to", type=int, default=0, help="Number of layers to freeze")
    parser.add_argument("--dropout-rate", type=float, default=0.1, help="Dropout rate")
    
    # Loss arguments
    parser.add_argument("--loss", choices=["bce", "focal", "balanced"], default="bce", help="Loss function")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument("--focal-alpha", type=float, default=1.0, help="Focal loss alpha parameter")
    
    # Evaluation arguments
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    
    # System arguments
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    
    # Training options
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--use-ema", action="store_true", help="Use exponential moving average of weights")
    
    # Logging arguments
    parser.add_argument("--wandb-project", type=str, default="siglip-ct-classification", help="Wandb project name")
    parser.add_argument("--log-interval", type=int, default=50, help="Logging interval")
    
    # Testing arguments
    parser.add_argument("--test-only", type=str, help="Path to checkpoint for testing only")
    parser.add_argument("--test-output", type=str, default="test_predictions.csv", help="Test output CSV file")
    
    # Data options
    parser.add_argument("--use-3channel", action="store_true", default=True, help="Use 3-channel CT windows")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Arguments: {args}")
    
    # Initialize wandb
    if WANDB_AVAILABLE and not args.test_only:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            save_code=True
        )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(
        csv_path=args.csv,
        label_columns=args.label_columns,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_3channel=args.use_3channel
    )
    
    # Build model
    model = build_model(
        pretrained_path=args.pretrained,
        num_classes=len(args.label_columns),
        dropout_rate=args.dropout_rate,
        freeze_up_to=args.freeze_up_to
    )
    model.to(device)
    
    # Multi-GPU support with DataParallel
    if torch.cuda.device_count() > 1 and not args.test_only:
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    elif torch.cuda.is_available():
        logger.info(f"Using single GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")
    
    # Compile model for better performance (PyTorch 2.0+)
    if hasattr(torch, 'compile') and not args.test_only:
        try:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile for better performance")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")
    
    # Create EMA model if requested
    ema_model = None
    if args.use_ema and EMA_AVAILABLE and not args.test_only:
        ema_model = AveragedModel(model)
        logger.info("EMA model created for smoother convergence")
    elif args.use_ema and not EMA_AVAILABLE:
        logger.warning("EMA requested but torch.optim.swa_utils not available")
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Test-only mode
    if args.test_only:
        logger.info(f"Loading checkpoint from {args.test_only}")
        checkpoint = torch.load(args.test_only, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_model(
            model=model,
            test_loader=val_loader,  # Use validation data for testing
            device=device,
            label_names=args.label_columns,
            output_path=args.test_output,
            threshold=args.threshold
        )
        return
    
    # Create loss function
    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.loss == "balanced":
        # Calculate positive weights from training data
        train_df = pd.read_csv(args.csv)
        train_df = train_df[train_df['split'] == 'train']
        train_labels = train_df[args.label_columns].values
        pos_weights = calculate_pos_weights(train_labels)
        criterion = BalancedBCELoss(pos_weights=pos_weights.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Create optimizer
    if args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    
    # Create scheduler
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        warmup_scheduler = None
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        warmup_scheduler = None
    elif args.scheduler == "warmup":
        total_steps = len(train_loader) // args.gradient_accumulation_steps * args.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup, num_training_steps=total_steps
        )
        warmup_scheduler = scheduler
    else:  # Default to plateau
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        warmup_scheduler = None
    
    # Mixed precision scaler
    scaler = GradScaler() if args.amp else None
    
    # Training loop
    best_macro_f1 = 0.0
    patience_counter = 0
    checkpoint_manager = CheckpointManager(keep_top_k=3)
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Training
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch + 1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            log_interval=args.log_interval,
            warmup_scheduler=warmup_scheduler,
            ema_model=ema_model
        )
        
        # Evaluation (use EMA model if available)
        eval_model = ema_model if ema_model is not None else model
        val_metrics = evaluate(
            model=eval_model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            label_names=args.label_columns,
            threshold=args.threshold
        )
        
        # Update scheduler
        if args.scheduler == "plateau":
            scheduler.step(val_metrics['macro_f1'])
        elif warmup_scheduler is None:
            scheduler.step()
        # Warmup scheduler is stepped after each batch, not after each epoch
        
        # Log metrics
        all_metrics = {**train_metrics, **val_metrics, 'epoch': epoch + 1}
        logger.info(f"Epoch {epoch + 1} - " + " - ".join([f"{k}: {v:.4f}" for k, v in all_metrics.items()]))
        
        if WANDB_AVAILABLE:
            wandb.log(all_metrics)
        
        # Save checkpoint (save EMA model if using it)
        save_model = ema_model if ema_model is not None else model
        checkpoint_path = save_checkpoint(
            model=save_model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics=val_metrics,
            output_dir=args.output_dir,
            ema_model=ema_model
        )
        
        # Add to checkpoint manager
        checkpoint_manager.add_checkpoint(val_metrics['macro_f1'], checkpoint_path)
        
        # Check for improvement
        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            patience_counter = 0
            
            # Create symlink to best model instead of copying
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            if os.path.exists(best_model_path) or os.path.islink(best_model_path):
                os.remove(best_model_path)
            os.symlink(os.path.basename(checkpoint_path), best_model_path)
            logger.info(f"New best model saved with macro F1: {best_macro_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            logger.info(f"Early stopping triggered after {args.early_stopping_patience} epochs without improvement")
            break
    
    # Final evaluation with best model
    logger.info("Loading best model for final evaluation")
    best_checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    final_metrics = evaluate(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        label_names=args.label_columns,
        threshold=args.threshold
    )
    
    logger.info("Final evaluation results:")
    for key, value in final_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    if WANDB_AVAILABLE:
        wandb.log({"final_" + k: v for k, v in final_metrics.items()})
        wandb.finish()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main() 