#!/usr/bin/env python3
"""
Example script for training a SigLIP Chest CT CLIP model.

This script demonstrates how to use the siglip_trainer.py to train a model
that combines LLM2Vec text encoder with Merlin image encoder.

Usage:
    python train_example.py --train_data_dir /path/to/train/npz/files

Optional arguments for text reports:
    python train_example.py --train_data_dir /path/to/train/npz/files \
                           --text_reports_path /path/to/text_reports.json

The text_reports.json file should be in the format:
{
    "image_id_1": "Chest CT shows normal findings...",
    "image_id_2": "There is evidence of pleural effusion...",
    ...
}
"""

import argparse
import os
import json
from pathlib import Path

def create_example_text_reports(npz_dir: str, output_path: str):
    """
    Create an example text reports JSON file based on NPZ files in the directory.
    This is just for demonstration - in practice, you would have real radiology reports.
    """
    npz_files = list(Path(npz_dir).glob("*.npz"))
    
    # Example radiology report templates
    report_templates = [
        "Normal chest CT scan. No acute findings.",
        "Chest CT shows small bilateral pleural effusions.",
        "Evidence of consolidation in the right lower lobe.",
        "Pulmonary edema present. Recommend clinical correlation.",
        "No significant abnormalities detected on this chest CT.",
        "Mild emphysematous changes noted in bilateral upper lobes.",
        "Trace pleural effusion on the left side.",
        "Ground glass opacities in bilateral lung bases.",
        "Normal cardiac silhouette. Clear lung fields.",
        "Subpleural nodules present, recommend follow-up."
    ]
    
    reports = {}
    for i, npz_file in enumerate(npz_files):
        # Use modulo to cycle through templates
        template_idx = i % len(report_templates)
        reports[npz_file.stem] = report_templates[template_idx]
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(reports, f, indent=2)
    
    print(f"Created example text reports file: {output_path}")
    print(f"Generated reports for {len(reports)} NPZ files")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Example training script for SigLIP Chest CT CLIP')
    
    # Required arguments
    parser.add_argument('--train_data_dir', type=str, required=True,
                        help='Directory containing training NPZ files')
    
    # Optional arguments
    parser.add_argument('--val_data_dir', type=str, default=None,
                        help='Directory containing validation NPZ files')
    parser.add_argument('--text_reports_path', type=str, default=None,
                        help='Path to JSON file containing text reports')
    parser.add_argument('--create_example_reports', action='store_true',
                        help='Create example text reports file from NPZ directory')
    parser.add_argument('--text_model_path', type=str, default=None,
                        help='Path to pretrained text model checkpoint')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (reduce if running into memory issues)')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Maximum number of training steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Validate that training directory exists
    if not os.path.exists(args.train_data_dir):
        raise ValueError(f"Training data directory does not exist: {args.train_data_dir}")
    
    # Check if NPZ files exist
    npz_files = list(Path(args.train_data_dir).glob("*.npz"))
    if not npz_files:
        raise ValueError(f"No NPZ files found in {args.train_data_dir}")
    
    print(f"Found {len(npz_files)} NPZ files in training directory")
    
    # Create example text reports if requested
    text_reports_path = args.text_reports_path
    if args.create_example_reports:
        text_reports_path = os.path.join(args.train_data_dir, "example_reports.json")
        create_example_text_reports(args.train_data_dir, text_reports_path)
    
    # Import and run the trainer
    import subprocess
    import sys
    
    # Build command for the actual trainer
    cmd = [
        sys.executable, "siglip_trainer.py",
        "--train_data_dir", args.train_data_dir,
        "--batch_size", str(args.batch_size),
        "--max_steps", str(args.max_steps),
        "--learning_rate", str(args.learning_rate),
        "--save_dir", args.save_dir,
        "--freeze_image_backbone",  # Start with frozen backbone
        "--log_interval", "50",
        "--save_interval", "1000"
    ]
    
    # Add optional arguments
    if text_reports_path:
        cmd.extend(["--text_reports_path", text_reports_path])
    
    if args.val_data_dir:
        cmd.extend(["--val_data_dir", args.val_data_dir])
    
    if args.text_model_path:
        cmd.extend(["--text_model_path", args.text_model_path])
    
    print("Starting training with command:")
    print(" ".join(cmd))
    print()
    
    # Run the training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Training interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main() 