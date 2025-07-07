#!/usr/bin/env python3
"""
Unit tests for data loading and preprocessing components.
"""

import pytest
import torch
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from finetune_siglip import CTMultilabelDataset, get_dataloaders, _hu_window_to_unit


class TestCTMultilabelDataset:
    """Test cases for CTMultilabelDataset."""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        data = {
            'img_path': [
                '/fake/path/image1.nii.gz',
                '/fake/path/image2.nii.gz',
                '/fake/path/image3.nii.gz',
                '/fake/path/image4.nii.gz'
            ],
            'split': ['train', 'train', 'val', 'val'],
            'Medical material': [1, 0, 1, 0],
            'Cardiomegaly': [0, 1, 0, 1],
            'Lung nodule': [1, 1, 0, 0],
            'Pleural effusion': [0, 0, 1, 1]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def label_columns(self):
        """Label columns for testing."""
        return ['Medical material', 'Cardiomegaly', 'Lung nodule', 'Pleural effusion']
    
    def test_dataset_initialization(self, temp_csv_file, label_columns):
        """Test dataset initialization."""
        dataset = CTMultilabelDataset(
            csv_file=temp_csv_file,
            label_columns=label_columns,
            split='train',
            use_3channel=True
        )
        
        assert len(dataset) == 2  # 2 train samples
        assert dataset.label_columns == label_columns
        assert dataset.split == 'train'
        assert dataset.use_3channel == True
    
    def test_dataset_length(self, temp_csv_file, label_columns):
        """Test dataset length for different splits."""
        train_dataset = CTMultilabelDataset(
            csv_file=temp_csv_file,
            label_columns=label_columns,
            split='train'
        )
        
        val_dataset = CTMultilabelDataset(
            csv_file=temp_csv_file,
            label_columns=label_columns,
            split='val'
        )
        
        assert len(train_dataset) == 2
        assert len(val_dataset) == 2
    
    @patch('finetune_siglip.sitk.ReadImage')
    @patch('finetune_siglip.sitk.GetArrayFromImage')
    def test_dataset_getitem(self, mock_get_array, mock_read_image, temp_csv_file, label_columns):
        """Test dataset __getitem__ method."""
        # Mock the medical image loading
        mock_image = MagicMock()
        mock_read_image.return_value = mock_image
        
        # Mock the array data - CT volume
        fake_volume = np.random.rand(160, 224, 224).astype(np.float32)
        mock_get_array.return_value = fake_volume
        
        dataset = CTMultilabelDataset(
            csv_file=temp_csv_file,
            label_columns=label_columns,
            split='train',
            use_3channel=True
        )
        
        # Test getting an item
        sample = dataset[0]
        
        assert 'image' in sample
        assert 'labels' in sample
        assert isinstance(sample['labels'], torch.Tensor)
        assert sample['labels'].shape == (len(label_columns),)
    
    def test_invalid_split(self, temp_csv_file, label_columns):
        """Test dataset with invalid split."""
        with pytest.raises(ValueError):
            CTMultilabelDataset(
                csv_file=temp_csv_file,
                label_columns=label_columns,
                split='invalid_split'
            )


class TestDataUtilities:
    """Test data utility functions."""
    
    def test_hu_window_to_unit(self):
        """Test HU windowing function."""
        # Create sample HU values
        volume = np.array([[-1000, -500, 0, 500, 1000]], dtype=np.float32)
        
        # Test lung window (center=-600, width=1500)
        windowed = _hu_window_to_unit(volume, center=-600, width=1500)
        
        assert windowed.shape == volume.shape
        assert np.all(windowed >= 0) and np.all(windowed <= 1)
        assert windowed.dtype == np.float32
    
    def test_hu_window_edge_cases(self):
        """Test HU windowing with edge cases."""
        # Test with extreme values
        volume = np.array([[-3000, -1000, 0, 1000, 3000]], dtype=np.float32)
        windowed = _hu_window_to_unit(volume, center=0, width=2000)
        
        # Values should be clipped to [0, 1]
        assert np.all(windowed >= 0) and np.all(windowed <= 1)
    
    @patch('finetune_siglip.CTMultilabelDataset')
    def test_get_dataloaders(self, mock_dataset_class):
        """Test dataloader creation."""
        # Mock dataset instances
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        mock_train_dataset.__len__.return_value = 100
        mock_val_dataset.__len__.return_value = 20
        
        # Configure the mock to return different datasets for train/val
        def dataset_side_effect(*args, **kwargs):
            if kwargs.get('split') == 'train':
                return mock_train_dataset
            else:
                return mock_val_dataset
        
        mock_dataset_class.side_effect = dataset_side_effect
        
        # Test dataloader creation
        train_loader, val_loader = get_dataloaders(
            csv_path='/fake/path.csv',
            label_columns=['label1', 'label2'],
            batch_size=32,
            num_workers=4,
            use_3channel=True
        )
        
        # Verify dataloaders were created
        assert train_loader is not None
        assert val_loader is not None
        
        # Verify dataset class was called correctly
        assert mock_dataset_class.call_count == 2


class TestDataValidation:
    """Test data validation and error handling."""
    
    def test_missing_csv_file(self, label_columns):
        """Test behavior with missing CSV file."""
        with pytest.raises(FileNotFoundError):
            CTMultilabelDataset(
                csv_file='/nonexistent/file.csv',
                label_columns=label_columns,
                split='train'
            )
    
    def test_missing_label_columns(self, temp_csv_file):
        """Test behavior with missing label columns."""
        # Use label columns that don't exist in the CSV
        missing_labels = ['NonexistentLabel1', 'NonexistentLabel2']
        
        with pytest.raises(KeyError):
            dataset = CTMultilabelDataset(
                csv_file=temp_csv_file,
                label_columns=missing_labels,
                split='train'
            )
            # Force loading of data
            len(dataset)
    
    def test_empty_split(self, sample_csv_data, label_columns):
        """Test behavior with empty split."""
        # Create CSV with no test samples
        data = sample_csv_data.copy()
        data['split'] = ['train'] * len(data)  # All samples are train
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            dataset = CTMultilabelDataset(
                csv_file=temp_file,
                label_columns=label_columns,
                split='test'  # No test samples exist
            )
            assert len(dataset) == 0
        finally:
            os.unlink(temp_file)


class TestTransforms:
    """Test CT transforms and preprocessing."""
    
    @patch('training.ct_transform.get_train_transform')
    @patch('training.ct_transform.get_val_transform')
    def test_transform_functions(self, mock_val_transform, mock_train_transform):
        """Test that transform functions are properly imported."""
        # Mock the transform functions
        mock_train_transform.return_value = MagicMock()
        mock_val_transform.return_value = MagicMock()
        
        # Import and test - this would normally be in get_dataloaders
        try:
            from training.ct_transform import get_train_transform, get_val_transform
            train_transform = get_train_transform()
            val_transform = get_val_transform()
            
            assert train_transform is not None
            assert val_transform is not None
        except ImportError:
            # If transforms not available, skip test
            pytest.skip("CT transforms not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 