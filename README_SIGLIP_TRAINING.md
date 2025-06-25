# SigLIP Training for Chest CT CLIP

This repository contains code for training a Chest CT CLIP model using SigLIP (Sigmoid Loss for Language Image Pre-training) that combines:

- **Text Encoder**: LLM2Vec model from `tutorial.py` for processing radiology reports
- **Image Encoder**: Merlin model for processing chest CT scans
- **Training Objective**: SigLIP loss for contrastive learning

## Overview

The training framework implements:

1. **SigLIP Loss**: An improved version of CLIP loss that uses sigmoid instead of softmax
2. **Mixed Precision Training**: Using automatic mixed precision for memory efficiency
3. **Flexible Text Handling**: Support for both paired text-image data and placeholder text
4. **Modular Architecture**: Separate encoders that can be fine-tuned independently

## File Structure

```
siglip_trainer.py         # Main training script with complete implementation
train_csv_example.py      # Example script for CSV-based training (recommended)
train_example.py          # Legacy example script for directory-based training
README_SIGLIP_TRAINING.md # This documentation file
requirements_siglip.txt   # Python dependencies
```

## Key Components

### 1. TextEncoder
- Uses LLM2Vec from the tutorial.py implementation
- Supports loading pretrained checkpoints
- Handles special separator-based text tokenization
- Projects embeddings to 512-dimensional space

### 2. ImageEncoder  
- Uses Merlin model for CT image processing
- Option to freeze backbone weights initially
- Projection head for contrastive learning
- Handles 3D CT volume inputs

### 3. SigLIPLoss
- Implements SigLIP objective function
- Configurable temperature and bias parameters
- More stable training than standard CLIP loss

## Data Format

### CSV Format (Recommended)
The training expects a CSV file with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `img_path` | Path to NPZ file | `/path/to/image001.npz` |
| `findings` | Radiology report text | `"Normal chest CT scan. No acute findings."` |
| `split` | Data split (train/val/test) | `train` |

Example CSV structure:
```csv
img_path,findings,split
/data/ct_scans/patient001.npz,"Normal chest CT scan. No acute findings.",train
/data/ct_scans/patient002.npz,"Bilateral pleural effusions noted.",train
/data/ct_scans/patient003.npz,"Evidence of consolidation in right lower lobe.",val
```

### NPZ Files
Each NPZ file should contain chest CT images with the format:
```python
{
    'image': numpy.ndarray  # Shape: (C, D, H, W) where C=2, D=depth, H=height, W=width
}
```

### Legacy JSON Format (Directory Mode)
For backward compatibility, you can provide text reports as a JSON file:
```json
{
    "image_id_1": "Chest CT shows normal findings with no acute abnormalities.",
    "image_id_2": "There is evidence of bilateral pleural effusions...",
    "image_id_3": "Consolidation present in the right lower lobe..."
}
```

## Usage

### CSV Mode (Recommended)

The preferred method is to use a CSV file containing image paths, findings, and splits:

#### Basic CSV Training
```bash
python siglip_trainer.py \
    --csv_file /path/to/your/data.csv \
    --batch_size 4 \
    --max_steps 50000 \
    --save_dir ./checkpoints
```

#### CSV with Custom Column Names
```bash
python siglip_trainer.py \
    --csv_file /path/to/your/data.csv \
    --findings_column "report_text" \
    --img_path_column "image_file" \
    --split_column "data_split" \
    --batch_size 4 \
    --max_steps 50000
```

#### Using the CSV Example Script
```bash
# Train with your CSV
python train_csv_example.py --csv_file /path/to/your/data.csv

# Create example CSV and train
python train_csv_example.py \
    --create_example_csv \
    --npz_directory /path/to/npz/files \
    --batch_size 2 \
    --max_steps 5000

# Validate CSV format
python train_csv_example.py \
    --csv_file /path/to/your/data.csv \
    --validate_csv
```

### Legacy Directory Mode

For backward compatibility, you can still use directory-based training:

#### Basic Training
```bash
python siglip_trainer.py \
    --train_data_dir /path/to/npz/files \
    --batch_size 4 \
    --max_steps 50000 \
    --save_dir ./checkpoints
```

#### Training with Text Reports
```bash
python siglip_trainer.py \
    --train_data_dir /path/to/npz/files \
    --text_reports_path /path/to/reports.json \
    --batch_size 4 \
    --max_steps 50000 \
    --save_dir ./checkpoints
```

#### Training with Validation
```bash
python siglip_trainer.py \
    --train_data_dir /path/to/train/npz \
    --val_data_dir /path/to/val/npz \
    --text_reports_path /path/to/reports.json \
    --batch_size 4 \
    --max_steps 50000 \
    --save_dir ./checkpoints
```

## Training Arguments

### CSV Data Arguments (Recommended)
- `--csv_file`: Path to CSV file with img_path, findings, and split columns
- `--train_split`: Split name for training data (default: 'train')
- `--val_split`: Split name for validation data (default: 'val')
- `--findings_column`: Column name for text findings (default: 'findings')
- `--img_path_column`: Column name for image paths (default: 'img_path')
- `--split_column`: Column name for data splits (default: 'split')

### Legacy Data Arguments (Directory Mode)
- `--train_data_dir`: Directory containing training NPZ files
- `--val_data_dir`: Directory containing validation NPZ files (optional)
- `--text_reports_path`: Path to JSON file with text reports (optional)

### Model Arguments
- `--text_model_path`: Path to pretrained text model checkpoint (optional)
- `--temperature`: Temperature parameter for SigLIP loss (default: 1.0)
- `--freeze_image_backbone`: Freeze the image encoder backbone weights

### Training Arguments
- `--batch_size`: Batch size for training (default: 4)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay for optimizer (default: 1e-4)
- `--warmup_steps`: Number of warmup steps (default: 1000)
- `--max_steps`: Maximum training steps (default: 100000)
- `--num_workers`: Number of data loader workers (default: 8)

### Logging and Saving
- `--save_dir`: Directory to save checkpoints (default: ./checkpoints)
- `--log_interval`: Logging interval in steps (default: 100)
- `--save_interval`: Checkpoint saving interval in steps (default: 5000)

## Training Strategy

### Phase 1: Frozen Backbone
Start with the image encoder backbone frozen to train only the projection heads:

```bash
python siglip_trainer.py \
    --train_data_dir /path/to/data \
    --freeze_image_backbone \
    --learning_rate 1e-3 \
    --max_steps 20000
```

### Phase 2: Fine-tuning
Unfreeze the backbone for end-to-end training:

```bash
python siglip_trainer.py \
    --train_data_dir /path/to/data \
    --learning_rate 1e-5 \
    --max_steps 50000 \
    --text_model_path /path/to/phase1/checkpoint.pt
```

## Model Architecture

```
ChestCTCLIP
├── ImageEncoder (Merlin-based)
│   ├── merlin_model (I3ResNet backbone)
│   └── projection (2048 -> 1024 -> 512)
├── TextEncoder (LLM2Vec-based)
│   ├── text_model (LLM2Vec)
│   └── projection (hidden_size -> 512)
└── SigLIPLoss
    ├── temperature
    └── bias
```

## Memory Requirements

- **Minimum GPU Memory**: 8GB (batch_size=2)
- **Recommended GPU Memory**: 16GB+ (batch_size=4+)
- **CPU Memory**: 16GB+ recommended for data loading

## Tips for Training

1. **Start Small**: Begin with a small batch size and increase gradually
2. **Monitor Loss**: SigLIP loss should decrease steadily during training
3. **Learning Rate**: Start with 1e-4, reduce to 1e-5 for fine-tuning
4. **Validation**: Use validation set to monitor overfitting
5. **Checkpointing**: Save checkpoints frequently in case of interruption

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Use gradient accumulation
- Enable gradient checkpointing

### Import Errors
Make sure all dependencies are installed:
```bash
pip install torch torchvision
pip install monai
pip install transformers
pip install tqdm
```

### Text Model Loading Issues
If the LLM2Vec model fails to load:
- Check internet connection for model download
- Verify the model path is correct
- Try running the tutorial.py first to test model loading

## Expected Output

During training, you should see output like:
```
Loading training data from: /path/to/data
Found 1000 NPZ files in training directory
Creating model...
Total parameters: 1,234,567,890
Trainable parameters: 123,456,789
Starting training...
Step 100: Loss = 2.3456, LR = 1.00e-04
Step 200: Loss = 2.1234, LR = 9.95e-05
...
```

## Citation

If you use this code, please cite the relevant papers:

```bibtex
@article{siglip2023,
    title={Sigmoid Loss for Language Image Pre-Training},
    author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
    journal={arXiv preprint arXiv:2303.15343},
    year={2023}
}
``` 