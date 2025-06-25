#!/usr/bin/env python3
"""
Example script for training a SigLIP Chest CT CLIP model with CSV input.

This script demonstrates how to use the siglip_trainer.py with CSV input format
that contains img_path, findings, and split columns.

Usage:
    python train_csv_example.py --csv_file /path/to/your/data.csv

Expected CSV format:
    img_path,findings,split
    /path/to/image1.npz,"Normal chest CT scan...",train
    /path/to/image2.npz,"Bilateral pleural effusions...",train
    /path/to/image3.npz,"No acute findings",val
    ...

The script supports:
- Automatic train/validation split handling
- Customizable column names
- Example CSV generation for testing
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path


def create_example_csv(output_path: str, npz_directory: str = None, num_samples: int = 100):
    """
    Create an example CSV file for testing.
    
    Args:
        output_path: Path where to save the example CSV
        npz_directory: Directory containing NPZ files (optional)
        num_samples: Number of example entries to create
    """
    
    # Example radiology findings
    findings_templates = [
        "Normal chest CT scan. No acute findings.",
        "Bilateral pleural effusions noted. Recommend clinical correlation.",
        "Evidence of consolidation in the right lower lobe consistent with pneumonia.",
        "Pulmonary edema present. Cardiomegaly observed.",
        "No significant abnormalities detected on this chest CT examination.",
        "Mild emphysematous changes noted in bilateral upper lobes.",
        "Small left-sided pleural effusion. No pneumothorax.",
        "Ground glass opacities in bilateral lung bases suggestive of atypical infection.",
        "Normal cardiac silhouette. Clear lung fields bilaterally.",
        "Multiple pulmonary nodules present. Recommend follow-up imaging.",
        "Mediastinal lymphadenopathy observed. Further evaluation recommended.",
        "Trace pleural effusion. Subsegmental atelectasis at lung bases.",
        "Post-surgical changes noted. No evidence of complication.",
        "Chronic changes consistent with previous inflammatory disease.",
        "Small pneumothorax on the right side. No tension component."
    ]
    
    data = []
    
    if npz_directory and os.path.exists(npz_directory):
        # Use actual NPZ files if directory is provided
        npz_files = list(Path(npz_directory).glob("*.npz"))
        if npz_files:
            print(f"Found {len(npz_files)} NPZ files in {npz_directory}")
            for i, npz_file in enumerate(npz_files[:num_samples]):
                # Assign split (80% train, 20% val)
                split = 'train' if i < len(npz_files) * 0.8 else 'val'
                
                # Cycle through findings templates
                findings = findings_templates[i % len(findings_templates)]
                
                data.append({
                    'img_path': str(npz_file),
                    'findings': findings,
                    'split': split
                })
        else:
            print(f"No NPZ files found in {npz_directory}")
    
    # If no real files or not enough samples, create dummy entries
    remaining_samples = num_samples - len(data)
    if remaining_samples > 0:
        print(f"Creating {remaining_samples} dummy entries")
        for i in range(remaining_samples):
            # Create dummy file paths
            img_path = f"/path/to/dummy_image_{i:04d}.npz"
            findings = findings_templates[i % len(findings_templates)]
            split = 'train' if i < remaining_samples * 0.8 else 'val'
            
            data.append({
                'img_path': img_path,
                'findings': findings,
                'split': split
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"\nCreated example CSV with {len(df)} entries:")
    print(f"  Train samples: {sum(df['split'] == 'train')}")
    print(f"  Validation samples: {sum(df['split'] == 'val')}")
    print(f"  Saved to: {output_path}")
    
    # Show preview
    print(f"\nPreview of CSV:")
    print(df.head())
    
    return output_path


def validate_csv(csv_path: str, required_columns: list = None):
    """Validate CSV format and show statistics."""
    if required_columns is None:
        required_columns = ['img_path', 'findings', 'split']
    
    df = pd.read_csv(csv_path)
    
    print(f"\nCSV Validation for: {csv_path}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return False
    else:
        print(f"✅ All required columns present: {required_columns}")
    
    # Check splits
    if 'split' in df.columns:
        split_counts = df['split'].value_counts()
        print(f"Split distribution:")
        for split, count in split_counts.items():
            print(f"  {split}: {count} samples")
    
    # Check for missing values
    missing_findings = df['findings'].isna().sum()
    missing_paths = df['img_path'].isna().sum()
    
    if missing_findings > 0:
        print(f"⚠️  Missing findings: {missing_findings}")
    if missing_paths > 0:
        print(f"⚠️  Missing image paths: {missing_paths}")
    
    # Check if image files exist (sample a few)
    sample_size = min(5, len(df))
    existing_files = 0
    for _, row in df.sample(sample_size).iterrows():
        if os.path.exists(row['img_path']):
            existing_files += 1
    
    print(f"Sample file check: {existing_files}/{sample_size} files exist")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Example CSV training for SigLIP Chest CT CLIP')
    
    # CSV file handling
    parser.add_argument('--csv_file', type=str, default=None,
                        help='Path to CSV file for training')
    parser.add_argument('--create_example_csv', action='store_true',
                        help='Create an example CSV file')
    parser.add_argument('--npz_directory', type=str, default=None,
                        help='Directory with NPZ files for creating example CSV')
    parser.add_argument('--output_csv', type=str, default='example_training_data.csv',
                        help='Output path for example CSV')
    parser.add_argument('--validate_csv', action='store_true',
                        help='Validate CSV format without training')
    
    # Column customization
    parser.add_argument('--findings_column', type=str, default='findings',
                        help='Column name for findings text')
    parser.add_argument('--img_path_column', type=str, default='img_path',
                        help='Column name for image paths')
    parser.add_argument('--split_column', type=str, default='split',
                        help='Column name for data splits')
    
    # Training configuration
    parser.add_argument('--batch_size', type=str, default='2',
                        help='Batch size for training')
    parser.add_argument('--max_steps', type=str, default='5000',
                        help='Maximum training steps')
    parser.add_argument('--learning_rate', type=str, default='1e-4',
                        help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_csv',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Create example CSV if requested
    if args.create_example_csv:
        csv_path = create_example_csv(
            output_path=args.output_csv,
            npz_directory=args.npz_directory,
            num_samples=100
        )
        args.csv_file = csv_path  # Use the created CSV
    
    # Validate CSV if file is provided
    if args.csv_file:
        if not os.path.exists(args.csv_file):
            raise FileNotFoundError(f"CSV file not found: {args.csv_file}")
        
        is_valid = validate_csv(
            args.csv_file,
            [args.img_path_column, args.findings_column, args.split_column]
        )
        
        if not is_valid:
            print("❌ CSV validation failed. Please fix the issues above.")
            return
        
        if args.validate_csv:
            print("✅ CSV validation completed successfully!")
            return
    
    # Check if we have a CSV file for training
    if not args.csv_file:
        print("No CSV file provided. Use --csv_file or --create_example_csv")
        return
    
    # Import and run the trainer
    import subprocess
    import sys
    
    print(f"\n🚀 Starting training with CSV: {args.csv_file}")
    
    # Build command for the actual trainer
    cmd = [
        sys.executable, "siglip_trainer.py",
        "--csv_file", args.csv_file,
        "--findings_column", args.findings_column,
        "--img_path_column", args.img_path_column,
        "--split_column", args.split_column,
        "--batch_size", args.batch_size,
        "--max_steps", args.max_steps,
        "--learning_rate", args.learning_rate,
        "--save_dir", args.save_dir,
        "--freeze_image_backbone",  # Start with frozen backbone
        "--log_interval", "50",
        "--save_interval", "1000"
    ]
    
    print("Training command:")
    print(" ".join(cmd))
    print()
    
    # Run the training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("🛑 Training interrupted by user")
        sys.exit(0)
    except FileNotFoundError:
        print("❌ siglip_trainer.py not found. Make sure you're in the correct directory.")
        sys.exit(1)


if __name__ == "__main__":
    main() 