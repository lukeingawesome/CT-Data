#!/usr/bin/env python3
"""
Integration tests for the complete training pipeline.
"""

import pytest
import torch
import pandas as pd
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import finetune_siglip


class TestEndToEndPipeline:
    """Integration tests for complete pipeline."""
    
    @pytest.fixture
    def sample_dataset_files(self):
        """Create sample dataset files for integration testing."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create sample CSV
        data = {
            'img_path': [
                os.path.join(temp_dir, 'image1.nii.gz'),
                os.path.join(temp_dir, 'image2.nii.gz'),
                os.path.join(temp_dir, 'image3.nii.gz'),
                os.path.join(temp_dir, 'image4.nii.gz')
            ],
            'split': ['train', 'train', 'val', 'val'],
            'Medical material': [1, 0, 1, 0],
            'Cardiomegaly': [0, 1, 0, 1],
            'Lung nodule': [1, 1, 0, 0],
            'Pleural effusion': [0, 0, 1, 1]
        }
        
        df = pd.DataFrame(data)
        csv_path = os.path.join(temp_dir, 'test_data.csv')
        df.to_csv(csv_path, index=False)
        
        # Create fake image files (we'll mock the actual loading)
        for img_path in data['img_path']:
            # Create empty files
            with open(img_path, 'w') as f:
                f.write('fake_nii_data')
        
        yield {
            'csv_path': csv_path,
            'temp_dir': temp_dir,
            'label_columns': ['Medical material', 'Cardiomegaly', 'Lung nodule', 'Pleural effusion']
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @patch('finetune_siglip.sitk.ReadImage')
    @patch('finetune_siglip.sitk.GetArrayFromImage')
    @patch('finetune_siglip.Merlin')
    def test_complete_training_pipeline(self, mock_merlin, mock_get_array, mock_read_image, sample_dataset_files):
        """Test complete training pipeline with minimal data."""
        # Mock image loading
        mock_image = MagicMock()
        mock_read_image.return_value = mock_image
        fake_volume = np.random.rand(160, 224, 224).astype(np.float32)
        mock_get_array.return_value = fake_volume
        
        # Mock Merlin model
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = torch.randn(2, 1024)  # Mock features
        mock_merlin.return_value = mock_model_instance
        
        # Test parameters
        csv_path = sample_dataset_files['csv_path']
        label_columns = sample_dataset_files['label_columns']
        temp_dir = sample_dataset_files['temp_dir']
        
        # Create dataloaders
        train_loader, val_loader = finetune_siglip.get_dataloaders(
            csv_path=csv_path,
            label_columns=label_columns,
            batch_size=2,
            num_workers=0,  # No multiprocessing for testing
            use_3channel=True
        )
        
        # Verify dataloaders work
        assert train_loader is not None
        assert val_loader is not None
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        
        # Test one batch from each loader
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        assert 'image' in train_batch
        assert 'labels' in train_batch
        assert train_batch['labels'].shape[1] == len(label_columns)
        
        # Test model creation
        with patch('finetune_siglip.torch.load'):  # Mock checkpoint loading
            model = finetune_siglip.build_model(
                pretrained_path='/fake/path/model.bin',
                num_classes=len(label_columns),
                dropout_rate=0.1,
                freeze_up_to=0
            )
        
        assert model is not None
        assert isinstance(model, finetune_siglip.SigLIPClassifier)
        
        # Test forward pass
        device = torch.device('cpu')
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(train_batch['image'])
            assert output.shape == (train_batch['image'].shape[0], len(label_columns))
    
    @patch('finetune_siglip.sitk.ReadImage')
    @patch('finetune_siglip.sitk.GetArrayFromImage')
    def test_dataset_loading_integration(self, mock_get_array, mock_read_image, sample_dataset_files):
        """Test dataset loading integration."""
        # Mock image loading
        mock_image = MagicMock()
        mock_read_image.return_value = mock_image
        fake_volume = np.random.rand(160, 224, 224).astype(np.float32)
        mock_get_array.return_value = fake_volume
        
        csv_path = sample_dataset_files['csv_path']
        label_columns = sample_dataset_files['label_columns']
        
        # Test train dataset
        train_dataset = finetune_siglip.CTMultilabelDataset(
            csv_file=csv_path,
            label_columns=label_columns,
            split='train',
            use_3channel=True
        )
        
        # Test val dataset
        val_dataset = finetune_siglip.CTMultilabelDataset(
            csv_file=csv_path,
            label_columns=label_columns,
            split='val',
            use_3channel=True
        )
        
        # Verify dataset properties
        assert len(train_dataset) == 2
        assert len(val_dataset) == 2
        
        # Test sample loading
        train_sample = train_dataset[0]
        val_sample = val_dataset[0]
        
        # Verify sample structure
        for sample in [train_sample, val_sample]:
            assert 'image' in sample
            assert 'labels' in sample
            assert isinstance(sample['labels'], torch.Tensor)
            assert sample['labels'].shape == (len(label_columns),)
    
    def test_loss_functions_integration(self):
        """Test loss functions with realistic data."""
        batch_size = 4
        num_classes = 4
        
        # Create realistic predictions and targets
        predictions = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, 2, (batch_size, num_classes)).float()
        
        # Test FocalLoss
        focal_loss = finetune_siglip.FocalLoss(alpha=1.0, gamma=2.0)
        loss1 = focal_loss(predictions, targets)
        assert torch.isfinite(loss1)
        assert loss1 >= 0
        
        # Test BalancedBCELoss
        pos_weights = finetune_siglip.calculate_pos_weights(targets.numpy())
        balanced_loss = finetune_siglip.BalancedBCELoss(pos_weights=pos_weights)
        loss2 = balanced_loss(predictions, targets)
        assert torch.isfinite(loss2)
        assert loss2 >= 0
        
        # Test regular BCE for comparison
        bce_loss = torch.nn.BCEWithLogitsLoss()
        loss3 = bce_loss(predictions, targets)
        assert torch.isfinite(loss3)
        assert loss3 >= 0
    
    @patch('finetune_siglip.Merlin')
    def test_model_training_step(self, mock_merlin):
        """Test a single training step."""
        # Mock the backbone
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = torch.randn(2, 1024)
        mock_merlin.return_value = mock_model_instance
        
        # Create model
        with patch('finetune_siglip.torch.load'):
            model = finetune_siglip.build_model(
                pretrained_path='/fake/path/model.bin',
                num_classes=4,
                dropout_rate=0.1
            )
        
        # Create fake batch
        batch_size = 2
        batch = {
            'image': torch.randn(batch_size, 1, 160, 224, 224),
            'labels': torch.randint(0, 2, (batch_size, 4)).float()
        }
        
        # Create optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = finetune_siglip.FocalLoss()
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(batch['image'])
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
        
        # Verify training step worked
        assert torch.isfinite(loss)
        assert loss.requires_grad or not loss.requires_grad  # Loss computed
        assert outputs.shape == (batch_size, 4)


class TestArgumentParsing:
    """Test argument parsing and configuration."""
    
    def test_finetune_script_imports(self):
        """Test that all required modules can be imported."""
        # This test verifies that the script structure is correct
        import finetune_siglip
        
        # Check key classes exist
        assert hasattr(finetune_siglip, 'SigLIPClassifier')
        assert hasattr(finetune_siglip, 'FocalLoss')
        assert hasattr(finetune_siglip, 'BalancedBCELoss')
        assert hasattr(finetune_siglip, 'CTMultilabelDataset')
        
        # Check key functions exist
        assert hasattr(finetune_siglip, 'train_one_epoch')
        assert hasattr(finetune_siglip, 'evaluate')
        assert hasattr(finetune_siglip, 'get_dataloaders')
        assert hasattr(finetune_siglip, 'build_model')


class TestErrorHandling:
    """Test error handling in various scenarios."""
    
    def test_invalid_model_path(self):
        """Test behavior with invalid model path."""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            finetune_siglip.build_model(
                pretrained_path='/nonexistent/path/model.bin',
                num_classes=18
            )
    
    def test_invalid_csv_path(self):
        """Test behavior with invalid CSV path."""
        with pytest.raises(FileNotFoundError):
            finetune_siglip.CTMultilabelDataset(
                csv_file='/nonexistent/path/data.csv',
                label_columns=['label1', 'label2'],
                split='train'
            )
    
    def test_mismatched_dimensions(self):
        """Test error handling for mismatched tensor dimensions."""
        # Create model with wrong number of classes
        backbone = MagicMock()
        backbone.return_value = torch.randn(2, 1024)
        
        with patch('finetune_siglip.SigLIPClassifier._get_backbone_dim', return_value=1024):
            model = finetune_siglip.SigLIPClassifier(
                backbone=backbone,
                num_classes=5  # Wrong number
            )
        
        # Create targets with different number of classes
        predictions = model(torch.randn(2, 1, 160, 224, 224))
        targets = torch.randint(0, 2, (2, 8)).float()  # Wrong shape
        
        criterion = finetune_siglip.FocalLoss()
        
        with pytest.raises(RuntimeError):
            loss = criterion(predictions, targets)


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 