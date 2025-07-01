import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from health_multimodal.image.data.transforms import get_chest_xray_transforms
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import os
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineScheduler(_LRScheduler):
    """
    Implements a learning rate scheduler with warmup followed by cosine annealing.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, warmup_start_lr=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs_after_warmup = None
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr) 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            if self.base_lrs_after_warmup is None:
                self.base_lrs_after_warmup = list(self.base_lrs)
                
            # Calculate where we are in the cosine cycle
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(1.0, progress)  # Ensure we don't go beyond 1.0
            
            # Apply cosine schedule
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            return [self.min_lr + cosine_factor * (base_lr - self.min_lr) 
                    for base_lr in self.base_lrs_after_warmup]

def parse_args():
    parser = argparse.ArgumentParser(description='3-class classification')
    parser.add_argument('--train_csv', type=str, default='pneumothorax_train_path.csv', help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default='pneumothorax_val_path.csv', help='Path to validation CSV file')
    parser.add_argument('--img_key', type=str, default='img_path', help='Column name for current image paths')
    parser.add_argument('--prev_img_key', type=str, default='previous_img_path', help='Column name for previous image paths')
    parser.add_argument('--label_key', type=str, default='label', help='Column name for labels')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--mode', type=str, default='linear', choices=['linear', 'full'], 
                       help='Training mode: linear probing or full finetuning')
    parser.add_argument('--switch_mode', default=False, help='Enable switching between original and swapped inputs/labels')
    parser.add_argument('--loss_type', type=str, default='original', choices=['original', 'switched', 'TCL'],
                       help='Type of loss to use: original cross-entropy, switched cross-entropy, or TCL loss')
    parser.add_argument('--weighted_loss', default=False, action='store_true', 
                       help='Use class weights inversely proportional to class frequencies')
    parser.add_argument('--output_dir', type=str, default=f'/model/workspace/ptx_test_ours_base_ci', help='Output directory')
    parser.add_argument('--wandb_project', type=str, default='tila-finetune', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity name')
    parser.add_argument('--wandb_name', type=str, default='ptx_test_ours_base_ci', help='W&B run name')
    parser.add_argument('--checkpoint_path', type=str, default="./final/tila_inv_text/pytorch_model.bin", help='Path to pre-trained model checkpoint')
    parser.add_argument('--warmup_proportion', type=float, default=0.05, help='Proportion of training to use for warmup')
    parser.add_argument('--tcl_weight', type=float, default=50.0, help='Weight for TCL loss term')
    return parser.parse_args()

class CXRClassificationDataset(Dataset):
    def __init__(self, csv_path, img_key, prev_img_key, label_key, transform=None, sample=False):
        self.df = pd.read_csv(csv_path)
        self.img_paths = self.df[img_key].tolist()
        self.prev_img_paths = self.df[prev_img_key].tolist()
        self.labels = self.df[label_key].tolist()
        self.transform = transform
        if sample==True:
            self.df= (
                self.df.groupby('label', group_keys=False)
                .apply(lambda x: x.sample(frac=0.5, random_state=42))
                .reset_index(drop=True)
            )
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        prev_img_path = self.prev_img_paths[idx]

        if not os.path.exists(img_path):
            img_path = self.img_paths[idx]
        if not os.path.exists(prev_img_path):
            prev_img_path = self.prev_img_paths[idx]

        label = self.labels[idx]
        
        # Load and transform images
        from health_multimodal.image.data.io import load_image
        from pathlib import Path
        current_image = load_image(Path(img_path))
        previous_image = load_image(Path(prev_img_path))
        
        if self.transform:
            current_image = self.transform(current_image)
            previous_image = self.transform(previous_image)
            
        # Convert to bfloat16
        current_image = current_image.to(torch.bfloat16)
        previous_image = previous_image.to(torch.bfloat16)
            
        return current_image, previous_image, label

class ClassificationModel(nn.Module):
    def __init__(self, image_encoder, num_classes=3, mode='linear'):
        super().__init__()
        self.image_encoder = image_encoder
        self.mode = mode
        
        # Freeze image encoder for linear probing
        if mode == 'linear':
            for param in self.image_encoder.parameters():
                param.requires_grad = False
                
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, current_image, previous_image):
        # Ensure inputs are bfloat16
        current_image = current_image.to(torch.bfloat16)
        previous_image = previous_image.to(torch.bfloat16)
        
        # Get image features from BioVIL-T
        features = self.image_encoder(current_image, previous_image).img_embedding
        # Ensure features are float32 for classification
        features = features.to(torch.float32)
        
        # Classification
        logits = self.classifier(features)
        return logits

def switch_labels(labels):
    """Switch labels: 0->2, 2->0, 1->1"""
    switched_labels = labels.clone()
    switched_labels[labels == 0] = 2
    switched_labels[labels == 2] = 0
    return switched_labels

def switch_scores(scores):
    """Switch scores: score[0] <-> score[2]"""
    switched_scores = scores.clone()
    switched_scores[:, [0, 2]] = switched_scores[:, [2, 0]]
    return switched_scores

def compute_class_weights(dataset):
    """Compute class weights inversely proportional to class frequencies"""
    labels = dataset.labels
    # Count samples per class
    class_counts = {}
    for label in labels:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    # Get number of samples and classes
    n_samples = len(labels)
    n_classes = len(class_counts)
    
    # Compute weights: n_samples / (n_classes * count(c))
    weights = []
    for i in range(n_classes):
        weights.append(n_samples / (n_classes * class_counts[i]))
    
    return torch.tensor(weights)

def switch_weights(weights):
    """Switch weights: weights[0] <-> weights[2]"""
    switched_weights = weights.clone()
    switched_weights[0], switched_weights[2] = switched_weights[2], switched_weights[0]
    return switched_weights

def TCL_loss_3class(forward_probs, backward_probs):
    # Create the transformed version of backward predictions
    transformed_backward = backward_probs.clone()
    transformed_backward[:, 0] = backward_probs[:, 2]
    transformed_backward[:, 2] = backward_probs[:, 0]
    # Class 1 stays the same
    # Direct MSE between distributions
    return F.mse_loss(forward_probs, transformed_backward)

def train_epoch(model, train_loader, criterion, optimizer, device, switch_mode=False, loss_type='original', class_weights=None, current_epoch=0, tcl_weight=50.0):
    model.train()
    total_loss = 0
    total_original_loss = 0
    total_switched_loss = 0
    total_TCL_loss = 0
    all_preds = []
    all_labels = []
    
    # Check if we should use specialized losses yet
    use_specialized_loss = current_epoch >= 20
    effective_loss_type = loss_type
    if not use_specialized_loss and loss_type == 'TCL':
        effective_loss_type = 'switched'  # Fall back to switched loss before epoch 10
    
    for current_images, previous_images, labels in tqdm(train_loader, desc='Training'):
        current_images = current_images.to(device)
        previous_images = previous_images.to(device)
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        
        # Forward pass with original inputs
        outputs = model(current_images, previous_images)
        outputs = outputs.to(torch.float32)
        original_scores = F.softmax(outputs, dim=1)
        
        if effective_loss_type == 'original':
            loss = criterion(outputs, labels)
            preds = torch.argmax(original_scores, dim=1)
            total_original_loss += loss.item()
            
        elif effective_loss_type == 'switched': # Bidirectional Crossentropy Loss
            switched_outputs = model(previous_images, current_images)
            switched_outputs = switched_outputs.to(torch.float32)
            switched_labels = switch_labels(labels)
            original_loss = criterion(outputs, labels)
            if class_weights is not None:
                switched_weights = switch_weights(class_weights)
                switched_criterion = nn.CrossEntropyLoss(weight=switched_weights)
                switched_loss = switched_criterion(switched_outputs, switched_labels)
            else:
                switched_loss = criterion(switched_outputs, switched_labels)
            loss = (original_loss + switched_loss) / 2
            preds = torch.argmax(original_scores, dim=1)
            total_original_loss += original_loss.item()
            total_switched_loss += switched_loss.item()
            
        elif effective_loss_type == 'TCL': # Temporal Consistency Loss
            switched_outputs = model(previous_images, current_images)
            switched_outputs = switched_outputs.to(torch.float32)
            switched_scores = F.softmax(switched_outputs, dim=1)
            switched_labels = switch_labels(labels)
            TCL_loss = TCL_loss_3class(original_scores, switched_scores)
            original_ce_loss = criterion(outputs, labels)
            if class_weights is not None:
                switched_weights = switch_weights(class_weights)
                switched_criterion = nn.CrossEntropyLoss(weight=switched_weights)
                switched_ce_loss = switched_criterion(switched_outputs, switched_labels)
            else:
                switched_ce_loss = criterion(switched_outputs, switched_labels)
            loss = (original_ce_loss + switched_ce_loss)/2 + tcl_weight * TCL_loss
            preds = torch.argmax(original_scores, dim=1)
            total_original_loss += original_ce_loss.item()
            total_switched_loss += switched_ce_loss.item()
            total_TCL_loss += TCL_loss.item()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    avg_original_loss = total_original_loss / len(train_loader) if total_original_loss > 0 else 0
    avg_switched_loss = total_switched_loss / len(train_loader) if total_switched_loss > 0 else 0
    avg_TCL_loss = total_TCL_loss / len(train_loader) if total_TCL_loss > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, f1, avg_original_loss, avg_switched_loss, avg_TCL_loss

def validate(model, val_loader, criterion, device, switch_mode=False, loss_type='original', class_weights=None, current_epoch=0, tcl_weight=50.0):
    model.eval()
    total_loss = 0
    total_original_loss = 0
    total_switched_loss = 0
    total_TCL_loss = 0
    all_preds = []
    all_labels = []
    
    # Check if we should use specialized losses yet
    use_specialized_loss = current_epoch >= 10
    effective_loss_type = loss_type
    if not use_specialized_loss and loss_type == 'TCL':
        effective_loss_type = 'switched'
    
    with torch.no_grad():
        for current_images, previous_images, labels in tqdm(val_loader, desc='Validating'):
            current_images = current_images.to(device)
            previous_images = previous_images.to(device)
            labels = labels.to(device).long()
            
            outputs = model(current_images, previous_images)
            outputs = outputs.to(torch.float32)
            original_scores = F.softmax(outputs, dim=1)
            
            if effective_loss_type == 'original':
                loss = criterion(outputs, labels)
                preds = torch.argmax(original_scores, dim=1)
                total_original_loss += loss.item()
                
            elif effective_loss_type == 'switched':
                switched_outputs = model(previous_images, current_images)
                switched_outputs = switched_outputs.to(torch.float32)
                switched_labels = switch_labels(labels)
                original_loss = criterion(outputs, labels)
                if class_weights is not None:
                    switched_weights = switch_weights(class_weights)
                    switched_criterion = nn.CrossEntropyLoss(weight=switched_weights)
                    switched_loss = switched_criterion(switched_outputs, switched_labels)
                else:
                    switched_loss = criterion(switched_outputs, switched_labels)
                loss = (original_loss + switched_loss) / 2
                preds = torch.argmax(original_scores, dim=1)
                total_original_loss += original_loss.item()
                total_switched_loss += switched_loss.item()
                
            elif effective_loss_type == 'TCL':
                switched_outputs = model(previous_images, current_images)
                switched_outputs = switched_outputs.to(torch.float32)
                switched_scores = F.softmax(switched_outputs, dim=1)
                switched_labels = switch_labels(labels)
                TCL_loss = TCL_loss_3class(original_scores, switched_scores)
                original_ce_loss = criterion(outputs, labels)
                if class_weights is not None:
                    switched_weights = switch_weights(class_weights)
                    switched_criterion = nn.CrossEntropyLoss(weight=switched_weights)
                    switched_ce_loss = switched_criterion(switched_outputs, switched_labels)
                else:
                    switched_ce_loss = criterion(switched_outputs, switched_labels)
                loss = (original_ce_loss + switched_ce_loss)/2 + tcl_weight * TCL_loss
                preds = torch.argmax(original_scores, dim=1)
                total_original_loss += original_ce_loss.item()
                total_switched_loss += switched_ce_loss.item()
                total_TCL_loss += TCL_loss.item()
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    avg_original_loss = total_original_loss / len(val_loader) if total_original_loss > 0 else 0
    avg_switched_loss = total_switched_loss / len(val_loader) if total_switched_loss > 0 else 0
    avg_TCL_loss = total_TCL_loss / len(val_loader) if total_TCL_loss > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()
    
    return avg_loss, accuracy, f1, avg_original_loss, avg_switched_loss, avg_TCL_loss, balanced_acc

def main():
    args = parse_args()
    run_name = args.wandb_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_encoder = get_biovil_t_image_encoder()
    print(f"Loading weights from {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('visual.'):
            new_key = k[len('visual.'):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    image_encoder.load_state_dict(new_state_dict, strict=False)
    print("Successfully loaded pre-trained weights")
    image_encoder = image_encoder.to(torch.bfloat16)
    train_transform, val_transform = get_chest_xray_transforms(size=448, crop_size=448)
    if args.mode == 'linear':
        train_dataset = CXRClassificationDataset(args.train_csv, args.img_key, args.prev_img_key, args.label_key, train_transform, sample=True)
        val_dataset = CXRClassificationDataset(args.val_csv, args.img_key, args.prev_img_key, args.label_key, val_transform, sample=False)
    else:
        train_dataset = CXRClassificationDataset(args.train_csv, args.img_key, args.prev_img_key, args.label_key, train_transform, sample=False)
        val_dataset = CXRClassificationDataset(args.val_csv, args.img_key, args.prev_img_key, args.label_key, val_transform, sample=False)
    class_weights = None
    if args.weighted_loss:
        class_weights = compute_class_weights(train_dataset)
        class_weights = class_weights.to(device)
        print(f"Using calculated class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = ClassificationModel(image_encoder, num_classes=3, mode=args.mode)
    model = model.to(device)
    wandb.watch(model, log="all", log_freq=10)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    warmup_epochs = max(1, int(args.epochs * args.warmup_proportion))
    print(f"Using warmup for first {warmup_epochs} epochs ({args.warmup_proportion:.1%} of training)")
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=warmup_epochs,
        total_epochs=args.epochs,
        min_lr=args.lr/10,
        warmup_start_lr=args.lr/100
    )
    os.makedirs(args.output_dir, exist_ok=True)
    top_checkpoints_acc = []
    best_balanced_acc = -float('inf')
    best_ckpt_path = None
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc, train_f1, train_orig_loss, train_switch_loss, train_TCL_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            args.switch_mode, args.loss_type, class_weights, epoch, args.tcl_weight
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        val_loss, val_acc, val_f1, val_orig_loss, val_switch_loss, val_TCL_loss, val_balanced_acc = validate(
            model, val_loader, criterion, device, 
            args.switch_mode, args.loss_type, class_weights, epoch, args.tcl_weight
        )
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val Balanced Acc: {val_balanced_acc:.4f}")
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        loss_status = "Regular" if epoch < 10 else "Specialized"
        if args.loss_type == 'TCL':
            print(f"Using {loss_status} loss type (Epoch {epoch+1})")
        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "train_original_loss": train_orig_loss,
            "train_switched_loss": train_switch_loss,
            "train_TCL_loss": train_TCL_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "val_balanced_accuracy": val_balanced_acc,
            "val_original_loss": val_orig_loss,
            "val_switched_loss": val_switch_loss,
            "val_TCL_loss": val_TCL_loss,
            "learning_rate": current_lr,
            "epoch": epoch + 1,
            "using_specialized_loss": epoch >= 10,
            "tcl_weight": args.tcl_weight
        })
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_balanced_acc': val_balanced_acc,
        }
        ckpt_path = os.path.join(args.output_dir, f"best_acc_epoch_{epoch+1}_acc_{val_balanced_acc:.4f}.pth")
        if val_balanced_acc > best_balanced_acc:
            if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
                print(f"Removed previous best checkpoint: {best_ckpt_path}")
            torch.save(checkpoint_data, ckpt_path)
            print(f"Saved new best ACC checkpoint: {ckpt_path}")
            best_balanced_acc = val_balanced_acc
            best_ckpt_path = ckpt_path

    wandb.finish()

if __name__ == "__main__":
    main()
