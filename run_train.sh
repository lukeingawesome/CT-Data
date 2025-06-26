#!/bin/bash

# SigLIP Chest CT CLIP Training Script
# This script trains the model using the CT-RATE dataset

# Set strict error handling
set -e

# Create necessary directories
mkdir -p ./checkpoints
mkdir -p ./logs

# Training parameters
CSV_FILE="/data2/data/CT-RATE/ct_rate.csv"
SAVE_DIR="./checkpoints/siglip_ct_clip_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="./logs/training_$(date +%Y%m%d_%H%M%S).log"

# Create save directory
mkdir -p "$SAVE_DIR"

echo "Starting SigLIP Chest CT CLIP training..."
echo "CSV file: $CSV_FILE"
echo "Save directory: $SAVE_DIR"
echo "Log file: $LOG_FILE"

# Run training with recommended parameters
python siglip_trainer.py \
    --csv_file "$CSV_FILE" \
    --train_split "train" \
    --val_split "val" \
    --findings_column "findings" \
    --img_path_column "img_path" \
    --split_column "split" \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --weight_decay 1e-4 \
    --warmup_steps 2000 \
    --max_steps 50000 \
    --num_workers 4 \
    --temperature 1.0 \
    --freeze_image_backbone \
    --save_dir "$SAVE_DIR" \
    --log_interval 50 \
    --save_interval 2500 \
    --use_wandb \
    2>&1 | tee "$LOG_FILE"

echo "Training completed!"
echo "Checkpoints saved to: $SAVE_DIR"
echo "Training log saved to: $LOG_FILE" 