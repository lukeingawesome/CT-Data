#!/usr/bin/env python3
"""
Unit tests for training pipeline components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from unittest.mock import MagicMock, patch
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from finetune_siglip import (
    train_one_epoch, evaluate, save_checkpoint, CheckpointManager,
    test_model, SigLIPClassifier
)


class TestTrainingLoop:
    """Test cases for training loop components."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock(spec=nn.Module)
        model.return_value = torch.randn(4, 18)  # batch_size=4, num_classes=18
        model.train = MagicMock()
        model.eval = MagicMock()
        return model
    
    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock dataloader for testing."""
        dataloader = MagicMock()
        # Mock batches
        batch1 = {
            'image': torch.randn(4, 1, 160, 224, 224),
            'labels': torch.randint(0, 2, (4, 18)).float()
        }
        batch2 = {
            'image': torch.randn(4, 1, 160, 224, 224),
            'labels': torch.randint(0, 2, (4, 18)).float()
        }
        dataloader.__iter__.return_value = iter([batch1, batch2])
        dataloader.__len__.return_value = 2
        return dataloader
    
    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer for testing."""
        optimizer = MagicMock()
        optimizer.zero_grad = MagicMock()
        optimizer.step = MagicMock()
        return optimizer
    
    @pytest.fixture
    def mock_criterion(self):
        """Create a mock loss function for testing."""
        criterion = MagicMock()
        criterion.return_value = torch.tensor(0.5)
        return criterion
    
    def test_train_one_epoch(self, mock_model, mock_dataloader, mock_optimizer, mock_criterion):
        """Test train_one_epoch function."""
        device = torch.device('cpu')
        scaler = None  # No mixed precision for testing
        
        metrics = train_one_epoch(
            model=mock_model,
            train_loader=mock_dataloader,
            criterion=mock_criterion,
            optimizer=mock_optimizer,
            scaler=scaler,
            device=device,
            epoch=1,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            log_interval=1
        )
        
        # Check that metrics are returned
        assert isinstance(metrics, dict)
        assert 'train_loss' in metrics
        assert 'train_lr' in metrics
        
        # Check that model was set to training mode
        mock_model.train.assert_called()
        
        # Check that optimizer was used
        assert mock_optimizer.zero_grad.call_count > 0
        assert mock_optimizer.step.call_count > 0
    
    def test_train_one_epoch_with_scaler(self, mock_model, mock_dataloader, mock_optimizer, mock_criterion):
        """Test train_one_epoch with gradient scaler."""
        device = torch.device('cpu')
        scaler = MagicMock()
        scaler.scale.return_value = MagicMock()
        scaler.step = MagicMock()
        scaler.update = MagicMock()
        
        metrics = train_one_epoch(
            model=mock_model,
            train_loader=mock_dataloader,
            criterion=mock_criterion,
            optimizer=mock_optimizer,
            scaler=scaler,
            device=device,
            epoch=1
        )
        
        assert isinstance(metrics, dict)
        # Scaler should be used
        assert scaler.scale.call_count > 0
        assert scaler.step.call_count > 0
        assert scaler.update.call_count > 0
    
    def test_evaluate(self, mock_model, mock_dataloader, mock_criterion):
        """Test evaluate function."""
        device = torch.device('cpu')
        label_names = [f'Label_{i}' for i in range(18)]
        
        # Mock model predictions
        mock_model.return_value = torch.randn(4, 18)
        
        metrics = evaluate(
            model=mock_model,
            val_loader=mock_dataloader,
            criterion=mock_criterion,
            device=device,
            label_names=label_names,
            threshold=0.5
        )
        
        # Check that metrics are returned
        assert isinstance(metrics, dict)
        assert 'val_loss' in metrics
        assert 'macro_f1' in metrics
        assert 'micro_f1' in metrics
        
        # Check that model was set to eval mode
        mock_model.eval.assert_called()
    
    def test_evaluate_with_metrics(self, mock_model, mock_dataloader, mock_criterion):
        """Test evaluate function with detailed metric checking."""
        device = torch.device('cpu')
        label_names = ['Label_1', 'Label_2']
        
        # Create predictable outputs for testing metrics
        # Set model to return fixed predictions
        predictions = torch.tensor([[0.8, 0.2], [0.3, 0.9], [0.6, 0.1], [0.4, 0.7]])
        mock_model.return_value = predictions
        
        # Create corresponding targets
        targets = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]]).float()
        
        # Mock the dataloader to return these specific targets
        batch = {
            'image': torch.randn(4, 1, 160, 224, 224),
            'labels': targets
        }
        mock_dataloader.__iter__.return_value = iter([batch])
        
        metrics = evaluate(
            model=mock_model,
            val_loader=mock_dataloader,
            criterion=mock_criterion,
            device=device,
            label_names=label_names,
            threshold=0.5
        )
        
        # Verify expected metrics exist
        expected_keys = ['val_loss', 'macro_f1', 'micro_f1', 'macro_precision', 'macro_recall']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (float, int))


class TestCheckpointManagement:
    """Test cases for checkpoint saving and management."""
    
    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        model = MagicMock()
        optimizer = MagicMock()
        
        # Mock state_dict returns
        model.state_dict.return_value = {'model_param': torch.tensor([1.0])}
        optimizer.state_dict.return_value = {'optim_param': 0.01}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=5,
                metrics={'val_loss': 0.5, 'macro_f1': 0.8},
                output_dir=temp_dir,
                prefix='test'
            )
            
            # Check that checkpoint file was created
            checkpoint_files = [f for f in os.listdir(temp_dir) if f.startswith('test')]
            assert len(checkpoint_files) > 0
    
    def test_checkpoint_manager(self):
        """Test CheckpointManager class."""
        manager = CheckpointManager(keep_top_k=2)
        
        # Add some checkpoints
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint1 = os.path.join(temp_dir, 'checkpoint_1.pth')
            checkpoint2 = os.path.join(temp_dir, 'checkpoint_2.pth')
            checkpoint3 = os.path.join(temp_dir, 'checkpoint_3.pth')
            
            # Create dummy checkpoint files
            for checkpoint in [checkpoint1, checkpoint2, checkpoint3]:
                torch.save({'dummy': 'data'}, checkpoint)
            
            # Add checkpoints with different F1 scores
            manager.add_checkpoint(0.7, checkpoint1)
            manager.add_checkpoint(0.9, checkpoint2)  # Best
            manager.add_checkpoint(0.8, checkpoint3)
            
            # Check that best checkpoint is returned
            best = manager.get_best_checkpoint()
            assert best == checkpoint2
            
            # Check that only top k are kept
            assert len(manager.checkpoints) == 2  # keep_top_k=2
    
    def test_checkpoint_manager_overflow(self):
        """Test CheckpointManager when exceeding keep_top_k."""
        manager = CheckpointManager(keep_top_k=2)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoints = []
            for i in range(4):
                checkpoint = os.path.join(temp_dir, f'checkpoint_{i}.pth')
                torch.save({'dummy': f'data_{i}'}, checkpoint)
                checkpoints.append(checkpoint)
            
            # Add checkpoints in random order
            f1_scores = [0.6, 0.9, 0.7, 0.8]
            for i, f1 in enumerate(f1_scores):
                manager.add_checkpoint(f1, checkpoints[i])
            
            # Should keep only top 2
            assert len(manager.checkpoints) == 2
            
            # Best should be the one with F1=0.9
            best = manager.get_best_checkpoint()
            assert best == checkpoints[1]  # checkpoint_1.pth had F1=0.9


class TestModelTesting:
    """Test cases for model testing functionality."""
    
    @pytest.fixture
    def mock_test_model_setup(self):
        """Set up mocks for test_model function."""
        model = MagicMock()
        model.return_value = torch.sigmoid(torch.randn(4, 18))  # Predictions
        
        dataloader = MagicMock()
        batch = {
            'image': torch.randn(4, 1, 160, 224, 224),
            'labels': torch.randint(0, 2, (4, 18)).float(),
            'img_path': [f'/fake/path/image_{i}.nii.gz' for i in range(4)]
        }
        dataloader.__iter__.return_value = iter([batch])
        
        return model, dataloader
    
    def test_test_model(self, mock_test_model_setup):
        """Test test_model function."""
        model, dataloader = mock_test_model_setup
        device = torch.device('cpu')
        label_names = [f'Label_{i}' for i in range(18)]
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            output_path = temp_file.name
        
        try:
            test_model(
                model=model,
                test_loader=dataloader,
                device=device,
                label_names=label_names,
                output_path=output_path,
                threshold=0.5
            )
            
            # Check that output file was created
            assert os.path.exists(output_path)
            
            # Read and verify output format
            import pandas as pd
            df = pd.read_csv(output_path)
            
            # Should have img_path column and label columns
            expected_columns = ['img_path'] + label_names
            for col in expected_columns:
                assert col in df.columns
            
            # Should have correct number of rows
            assert len(df) == 4  # batch size
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestTrainingUtilities:
    """Test utility functions for training."""
    
    def test_model_device_handling(self):
        """Test that models handle device placement correctly."""
        # This is more of an integration test
        device = torch.device('cpu')
        
        # Create a simple model
        model = nn.Linear(10, 5)
        model = model.to(device)
        
        # Test input
        x = torch.randn(2, 10).to(device)
        output = model(x)
        
        assert output.device == device
        assert output.shape == (2, 5)


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 