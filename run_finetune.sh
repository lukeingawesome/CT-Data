#!/bin/bash

# SigLIP Finetuning Script for Chest CT Multilabel Classification
# This script provides different training configurations for the SigLIP model

set -e  # Exit on any error

# Configuration
CSV_FILE="all_ct_wlabels.csv"
PRETRAINED_PATH="/opt/project/logs/ckpt/pytorch_model.bin"
OUTPUT_BASE_DIR="./outputs"

# 18 pathology labels from all_ct_wlabels.csv
PATHOLOGY_LABELS=(
    "Medical material"
    "Arterial wall calcification"
    "Cardiomegaly"
    "Pericardial effusion"
    "Coronary artery wall calcification"
    "Hiatal hernia"
    "Lymphadenopathy"
    "Emphysema"
    "Atelectasis"
    "Lung nodule"
    "Lung opacity"
    "Pulmonary fibrotic sequela"
    "Pleural effusion"
    "Mosaic attenuation pattern"
    "Peribronchial thickening"
    "Consolidation"
    "Bronchiectasis"
    "Interlobular septal thickening"
)

# Common arguments
COMMON_ARGS=(
    --csv "$CSV_FILE"
    --pretrained "$PRETRAINED_PATH"
    --label-columns "${PATHOLOGY_LABELS[@]}"
    --use-3channel
    --seed 42
)

# Function to print usage
usage() {
    echo "Usage: $0 [basic|focal|balanced|inference|debug]"
    echo ""
    echo "Training modes:"
    echo "  basic     - Basic training with standard BCE loss"
    echo "  focal     - Training with focal loss for class imbalance"
    echo "  balanced  - Training with balanced loss weights"
    echo "  inference - Inference only with trained model"
    echo "  debug     - Quick debug run with minimal epochs"
    echo ""
    echo "Configuration:"
    echo "  CSV_FILE: $CSV_FILE"
    echo "  PRETRAINED_PATH: $PRETRAINED_PATH"
    echo "  OUTPUT_BASE_DIR: $OUTPUT_BASE_DIR"
    echo ""
    echo "Examples:"
    echo "  ./run_finetune.sh basic"
    echo "  ./run_finetune.sh focal"
    echo "  ./run_finetune.sh inference"
    exit 1
}

# Function to check if files exist
check_files() {
    if [ ! -f "$CSV_FILE" ]; then
        echo "Error: CSV file not found: $CSV_FILE"
        exit 1
    fi
    
    if [ ! -f "$PRETRAINED_PATH" ]; then
        echo "Error: Pretrained model not found: $PRETRAINED_PATH"
        exit 1
    fi
    
    if [ ! -f "finetune_siglip.py" ]; then
        echo "Error: finetune_siglip.py not found in current directory"
        exit 1
    fi
}

# Function for basic training
run_basic_training() {
    echo "=========================================="
    echo "Running Basic SigLIP Finetuning"
    echo "=========================================="
    
    OUTPUT_DIR="$OUTPUT_BASE_DIR/siglip_basic"
    mkdir -p "$OUTPUT_DIR"
    
    python finetune_siglip.py \
        "${COMMON_ARGS[@]}" \
        --epochs 50 \
        --batch-size 8 \
        --lr 1e-4 \
        --weight-decay 1e-4 \
        --optimizer adamw \
        --scheduler plateau \
        --loss bce \
        --dropout-rate 0.1 \
        --early-stopping-patience 10 \
        --output-dir "$OUTPUT_DIR" \
        --wandb-project "siglip-ct-basic" \
        --log-interval 25 \
        --num-workers 4
}

# Function for focal loss training
run_focal_training() {
    echo "=========================================="
    echo "Running SigLIP Finetuning with Focal Loss"
    echo "=========================================="
    
    OUTPUT_DIR="$OUTPUT_BASE_DIR/siglip_focal"
    mkdir -p "$OUTPUT_DIR"
    
    python finetune_siglip.py \
        "${COMMON_ARGS[@]}" \
        --epochs 100 \
        --batch-size 12 \
        --lr 1e-4 \
        --weight-decay 1e-4 \
        --optimizer adamw \
        --scheduler plateau \
        --loss focal \
        --focal-gamma 2.0 \
        --focal-alpha 1.0 \
        --freeze-up-to 30 \
        --dropout-rate 0.15 \
        --amp \
        --gradient-accumulation-steps 2 \
        --max-grad-norm 1.0 \
        --early-stopping-patience 15 \
        --output-dir "$OUTPUT_DIR" \
        --wandb-project "siglip-ct-focal" \
        --log-interval 25 \
        --num-workers 4
}

# Function for balanced loss training
run_balanced_training() {
    echo "=========================================="
    echo "Running SigLIP Finetuning with Balanced Loss"
    echo "=========================================="
    
    OUTPUT_DIR="$OUTPUT_BASE_DIR/siglip_balanced"
    mkdir -p "$OUTPUT_DIR"
    
    python finetune_siglip.py \
        "${COMMON_ARGS[@]}" \
        --epochs 75 \
        --batch-size 10 \
        --lr 1e-4 \
        --weight-decay 1e-4 \
        --optimizer adamw \
        --scheduler plateau \
        --loss balanced \
        --dropout-rate 0.1 \
        --amp \
        --gradient-accumulation-steps 1 \
        --max-grad-norm 1.0 \
        --early-stopping-patience 12 \
        --output-dir "$OUTPUT_DIR" \
        --wandb-project "siglip-ct-balanced" \
        --log-interval 25 \
        --num-workers 4
}

# Function for inference only
run_inference() {
    echo "=========================================="
    echo "Running Inference with Trained Model"
    echo "=========================================="
    
    # Check for available models
    MODELS=(
        "$OUTPUT_BASE_DIR/siglip_basic/best_model.pth"
        "$OUTPUT_BASE_DIR/siglip_focal/best_model.pth"
        "$OUTPUT_BASE_DIR/siglip_balanced/best_model.pth"
    )
    
    MODEL_PATH=""
    for model in "${MODELS[@]}"; do
        if [ -f "$model" ]; then
            MODEL_PATH="$model"
            break
        fi
    done
    
    if [ -z "$MODEL_PATH" ]; then
        echo "Error: No trained model found. Available paths:"
        for model in "${MODELS[@]}"; do
            echo "  $model"
        done
        echo "Please train a model first using: $0 basic|focal|balanced"
        exit 1
    fi
    
    echo "Using model: $MODEL_PATH"
    
    python finetune_siglip.py \
        "${COMMON_ARGS[@]}" \
        --test-only "$MODEL_PATH" \
        --test-output "./predictions.csv" \
        --threshold 0.4 \
        --batch-size 16 \
        --num-workers 4
    
    echo "Predictions saved to: ./predictions.csv"
}

# Function for debug/quick training
run_debug_training() {
    echo "=========================================="
    echo "Running Debug Training (Quick Test)"
    echo "=========================================="
    
    OUTPUT_DIR="$OUTPUT_BASE_DIR/siglip_debug"
    mkdir -p "$OUTPUT_DIR"
    
    python finetune_siglip.py \
        "${COMMON_ARGS[@]}" \
        --epochs 5 \
        --batch-size 4 \
        --lr 1e-4 \
        --weight-decay 1e-4 \
        --optimizer adamw \
        --scheduler plateau \
        --loss bce \
        --dropout-rate 0.1 \
        --early-stopping-patience 3 \
        --output-dir "$OUTPUT_DIR" \
        --wandb-project "siglip-ct-debug" \
        --log-interval 5 \
        --num-workers 2
}

# Function for advanced training with custom parameters
run_advanced_training() {
    echo "=========================================="
    echo "Running Advanced SigLIP Finetuning"
    echo "=========================================="
    
    OUTPUT_DIR="$OUTPUT_BASE_DIR/siglip_advanced"
    mkdir -p "$OUTPUT_DIR"
    
    python finetune_siglip.py \
        "${COMMON_ARGS[@]}" \
        --epochs 150 \
        --batch-size 16 \
        --lr 2e-4 \
        --weight-decay 1e-5 \
        --optimizer adamw \
        --scheduler warmup \
        --warmup 1000 \
        --loss focal \
        --focal-gamma 2.5 \
        --focal-alpha 0.8 \
        --freeze-up-to 50 \
        --dropout-rate 0.2 \
        --amp \
        --gradient-accumulation-steps 4 \
        --max-grad-norm 0.5 \
        --early-stopping-patience 20 \
        --use-ema \
        --output-dir "$OUTPUT_DIR" \
        --wandb-project "siglip-ct-advanced" \
        --log-interval 20 \
        --num-workers 6
}

# Function to show system information
show_system_info() {
    echo "System Information:"
    echo "===================="
    echo "Working directory: $(pwd)"
    echo "Python version: $(python --version 2>/dev/null || echo 'Python not found')"
    echo "GPU available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not available')"
    echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'CUDA not available')"
    echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $7}' || echo 'Unknown')"
    echo "Disk space: $(df -h . | tail -1 | awk '{print $4}' || echo 'Unknown')"
    echo ""
}

# Main script logic
main() {
    # Check if mode is provided
    if [ $# -eq 0 ]; then
        usage
    fi
    
    MODE=$1
    
    # Show system information
    show_system_info
    
    # Check required files
    check_files
    
    # Create output directory
    mkdir -p "$OUTPUT_BASE_DIR"
    
    # Run based on mode
    case $MODE in
        "basic")
            run_basic_training
            ;;
        "focal")
            run_focal_training
            ;;
        "balanced")
            run_balanced_training
            ;;
        "inference")
            run_inference
            ;;
        "debug")
            run_debug_training
            ;;
        "advanced")
            run_advanced_training
            ;;
        "help"|"-h"|"--help")
            usage
            ;;
        *)
            echo "Error: Unknown mode '$MODE'"
            echo ""
            usage
            ;;
    esac
    
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
}

# Run main function with all arguments
main "$@" 