# SigLIP Finetuning for Chest CT Multilabel Classification

This repository contains a production-ready finetuning script for adapting pretrained SigLIP models to chest CT multilabel classification tasks.

## Features

- **Pretrained Model Loading**: Load and adapt SigLIP checkpoints for multilabel classification
- **Flexible Architecture**: Support for layer freezing and custom classification heads
- **Advanced Loss Functions**: Focal Loss, Balanced BCE, and standard BCE for handling class imbalance
- **Warmup Scheduling**: Cosine learning rate schedule with configurable warmup steps
- **Model Compilation**: Automatic torch.compile for 5-15% performance boost on compatible hardware
- **EMA Weights**: Exponential moving average for smoother convergence and better performance
- **Efficient Checkpointing**: In-memory top-k checkpoint management without reloading
- **Comprehensive Metrics**: Per-label accuracy, F1 scores, macro/micro F1, and AUC
- **Early Stopping**: Automated training termination based on macro F1 score
- **Mixed Precision Training**: Automatic mixed precision with gradient scaling
- **Wandb Integration**: Comprehensive logging and experiment tracking
- **Test-Only Mode**: Dedicated inference mode for model evaluation
- **Reproducibility**: Deterministic training with configurable random seeds

## Installation

```bash
# Install required dependencies
pip install torch torchvision torchaudio
pip install monai
pip install scikit-learn pandas numpy
pip install wandb tqdm
pip install transformers

# Install health-multimodal for image processing
pip install health-multimodal
```

## Data Format

The script expects a CSV file with the following structure:

```csv
series_id,split,img_path,Medical material,Arterial wall calcification,Cardiomegaly,Pericardial effusion,Coronary artery wall calcification,Hiatal hernia,Lymphadenopathy,Emphysema,Atelectasis,Lung nodule,Lung opacity,Pulmonary fibrotic sequela,Pleural effusion,Mosaic attenuation pattern,Peribronchial thickening,Consolidation,Bronchiectasis,Interlobular septal thickening
CT001,train,/path/to/ct001.npz,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0
CT002,val,/path/to/ct002.npz,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
...
```

**Required columns:**
- `series_id`: Unique identifier for each CT scan
- `split`: Data split (`train` or `val`)
- `img_path`: Path to the NPZ file containing the CT volume
- Label columns: Binary labels (0 or 1) for each pathology

**CT Data Format:**
- NPZ files should contain an `image` key with CT volume data
- Expected shape: `(C, D, H, W)` where C≥1 (HU windows), D=depth, H=height, W=width
- **Channel-first orientation**: The format assumes channels come first, not last
- Values should be normalized (0-1 range) or in HU units (-1000 to +3000)
- If your data is in different orientations (e.g., D×H×W), please reshape before saving

## Usage Examples

### Basic Training

```bash
python finetune_siglip.py \
    --csv /path/to/all_ct_wlabels.csv \
    --pretrained /opt/project/logs/ckpt/pytorch_model.bin \
    --label-columns "Medical material" "Arterial wall calcification" "Cardiomegaly" "Pericardial effusion" "Coronary artery wall calcification" "Hiatal hernia" "Lymphadenopathy" "Emphysema" "Atelectasis" "Lung nodule" "Lung opacity" "Pulmonary fibrotic sequela" "Pleural effusion" "Mosaic attenuation pattern" "Peribronchial thickening" "Consolidation" "Bronchiectasis" "Interlobular septal thickening" \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4 \
    --output-dir ./outputs/siglip_ct_finetune
```

### Advanced Training with Focal Loss

```bash
python finetune_siglip.py \
    --csv /path/to/all_ct_wlabels.csv \
    --pretrained /opt/project/logs/ckpt/pytorch_model.bin \
    --label-columns "Medical material" "Arterial wall calcification" "Cardiomegaly" "Pericardial effusion" "Coronary artery wall calcification" "Hiatal hernia" "Lymphadenopathy" "Emphysema" "Atelectasis" "Lung nodule" "Lung opacity" "Pulmonary fibrotic sequela" "Pleural effusion" "Mosaic attenuation pattern" "Peribronchial thickening" "Consolidation" "Bronchiectasis" "Interlobular septal thickening" \
    --epochs 100 \
    --batch-size 16 \
    --lr 2e-4 \
    --weight-decay 1e-4 \
    --optimizer adamw \
    --scheduler plateau \
    --loss focal \
    --focal-gamma 2.0 \
    --focal-alpha 1.0 \
    --freeze-up-to 50 \
    --dropout-rate 0.2 \
    --amp \
    --gradient-accumulation-steps 2 \
    --early-stopping-patience 15 \
    --wandb-project "siglip-ct-multilabel" \
    --output-dir ./outputs/siglip_focal_frozen
```

### Training with Balanced Loss

```bash
python finetune_siglip.py \
    --csv /path/to/all_ct_wlabels.csv \
    --pretrained /opt/project/logs/ckpt/pytorch_model.bin \
    --label-columns "Medical material" "Arterial wall calcification" "Cardiomegaly" "Pericardial effusion" "Coronary artery wall calcification" "Hiatal hernia" "Lymphadenopathy" "Emphysema" "Atelectasis" "Lung nodule" "Lung opacity" "Pulmonary fibrotic sequela" "Pleural effusion" "Mosaic attenuation pattern" "Peribronchial thickening" "Consolidation" "Bronchiectasis" "Interlobular septal thickening" \
    --loss balanced \
    --epochs 75 \
    --batch-size 12 \
    --lr 1e-4 \
    --output-dir ./outputs/siglip_balanced
```

### Test-Only Mode

```bash
python finetune_siglip.py \
    --csv /path/to/all_ct_wlabels.csv \
    --pretrained /opt/project/logs/ckpt/pytorch_model.bin \
    --label-columns "Medical material" "Arterial wall calcification" "Cardiomegaly" "Pericardial effusion" "Coronary artery wall calcification" "Hiatal hernia" "Lymphadenopathy" "Emphysema" "Atelectasis" "Lung nodule" "Lung opacity" "Pulmonary fibrotic sequela" "Pleural effusion" "Mosaic attenuation pattern" "Peribronchial thickening" "Consolidation" "Bronchiectasis" "Interlobular septal thickening" \
    --test-only ./outputs/siglip_ct_finetune/best_model.pth \
    --test-output ./predictions.csv \
    --threshold 0.3
```

### Shell Script Usage (Recommended)

For convenience, use the provided shell script with predefined configurations:

```bash
# Basic training with standard BCE loss
./run_finetune.sh basic

# Training with focal loss for class imbalance
./run_finetune.sh focal

# Training with balanced loss weights
./run_finetune.sh balanced

# Inference only with trained model
./run_finetune.sh inference

# Quick debug run with minimal epochs
./run_finetune.sh debug

# Advanced training with custom parameters
./run_finetune.sh advanced

# Show help
./run_finetune.sh help
```

The shell script automatically:
- Sets all 18 pathology labels
- Uses appropriate batch sizes and learning rates for each mode
- Creates organized output directories
- Handles error checking and system information display

**Configuration:**
- Edit the top of `run_finetune.sh` to change paths:
  - `CSV_FILE="all_ct_wlabels.csv"`
  - `PRETRAINED_PATH="/opt/project/logs/ckpt/pytorch_model.bin"`
  - `OUTPUT_BASE_DIR="./outputs"`

## Command Line Arguments

### Required Arguments
- `--csv`: Path to CSV file with data and labels
- `--pretrained`: Path to pretrained SigLIP model checkpoint
- `--label-columns`: List of label column names (space-separated)

### Training Arguments
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 16)
- `--lr`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay for regularization (default: 1e-4)
- `--optimizer`: Optimizer choice ['adamw', 'sgd'] (default: 'adamw')
- `--scheduler`: Learning rate scheduler ['plateau', 'cosine'] (default: 'plateau')
- `--warmup`: Warmup steps (default: 0)

### Model Arguments
- `--freeze-up-to`: Number of layers to freeze from beginning (default: 0)
- `--dropout-rate`: Dropout rate for classification head (default: 0.1)

### Loss Arguments
- `--loss`: Loss function ['bce', 'focal', 'balanced'] (default: 'bce')
- `--focal-gamma`: Focal loss gamma parameter (default: 2.0)
- `--focal-alpha`: Focal loss alpha parameter (default: 1.0)

### Training Options
- `--amp`: Enable automatic mixed precision training
- `--gradient-accumulation-steps`: Steps to accumulate gradients (default: 1)
- `--max-grad-norm`: Maximum gradient norm for clipping (default: 1.0)
- `--early-stopping-patience`: Early stopping patience (default: 10)
- `--use-ema`: Use exponential moving average of weights for smoother convergence

### System Arguments
- `--num-workers`: Number of data loader workers (default: 4)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output-dir`: Output directory for checkpoints and logs (default: './outputs')

### Logging Arguments
- `--wandb-project`: Wandb project name (default: 'siglip-ct-classification')
- `--log-interval`: Logging interval in steps (default: 50)

### Testing Arguments
- `--test-only`: Path to checkpoint for testing only
- `--test-output`: Output CSV file for test predictions (default: 'test_predictions.csv')
- `--threshold`: Classification threshold (default: 0.5)

## Output Files

The script creates the following output files:

```
outputs/
├── training.log                 # Training logs
├── checkpoint_epoch_*.pth       # Epoch checkpoints
├── best_model.pth              # Best model based on macro F1
├── test_predictions.csv        # Test predictions (if --test-only used)
└── results.jsonl              # Evaluation results per epoch
```

### Test Predictions Format

The test predictions CSV contains:
- `id`: Sample identifier
- `{label}_prob`: Probability for each label
- `{label}_pred`: Binary prediction for each label (based on threshold)

## Key Features Explained

### Layer Freezing
Use `--freeze-up-to N` to freeze the first N layers of the backbone, allowing fine-tuning of only the top layers:
```bash
--freeze-up-to 50  # Freeze first 50 layers
```

### Loss Functions
- **BCE**: Standard binary cross-entropy
- **Focal**: Addresses class imbalance by focusing on hard examples
- **Balanced**: Uses class-specific weights to handle imbalance

### Early Stopping
Training automatically stops if macro F1 doesn't improve for `--early-stopping-patience` epochs.

### Checkpoint Management
Only the top-3 models (by macro F1) are kept to save disk space.

### Mixed Precision Training
Enable with `--amp` for faster training and reduced memory usage:
```bash
--amp --gradient-accumulation-steps 2
```

### EMA Weights
Use exponential moving average for smoother convergence and better performance:
```bash
--use-ema
```

### Warmup Scheduling
Use cosine learning rate schedule with warmup for better convergence:
```bash
--scheduler cosine --warmup 1000
```

### Model Compilation
The script automatically uses torch.compile for 5-15% performance boost on compatible hardware (PyTorch 2.0+).

## Monitoring Training

### Wandb Integration
The script automatically logs:
- Training/validation loss
- Per-label precision, recall, F1, and AUC
- Macro and micro F1 scores
- Learning rate and gradient norms

### Local Logs
- `training.log`: Detailed training logs
- Progress bars with real-time metrics
- Checkpoint save notifications

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size and/or enable gradient accumulation
   --batch-size 4 --gradient-accumulation-steps 4
   ```

2. **Slow Training**
   ```bash
   # Enable mixed precision and reduce workers if I/O bound
   --amp --num-workers 2
   ```

3. **Poor Convergence**
   ```bash
   # Try different loss functions and learning rates
   --loss focal --lr 5e-5 --weight-decay 1e-5
   ```

4. **Class Imbalance**
   ```bash
   # Use balanced loss or focal loss
   --loss balanced
   # OR
   --loss focal --focal-gamma 2.0 --focal-alpha 1.0
   ```

### Performance Tips

1. **Use appropriate batch sizes**: Start with 8-16 for CT volumes
2. **Enable mixed precision**: Use `--amp` for faster training
3. **Tune learning rates**: Start with 1e-4, adjust based on convergence
4. **Monitor validation metrics**: Use wandb for comprehensive tracking
5. **Freeze layers gradually**: Start with `--freeze-up-to 0`, increase if overfitting

## Citation

If you use this code, please cite:
```bibtex
@misc{siglip-ct-finetuning,
  title={SigLIP Finetuning for Chest CT Multilabel Classification},
  author={Medical AI Engineering Team},
  year={2024},
  note={Production-ready finetuning framework for medical imaging}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 