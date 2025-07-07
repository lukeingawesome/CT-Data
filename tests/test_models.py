#!/usr/bin/env python3
"""
Unit tests for model components and loss functions.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the modules to test
from finetune_siglip import SigLIPClassifier, FocalLoss, BalancedBCELoss, calculate_pos_weights


class TestSigLIPClassifier:
    """Test cases for SigLIPClassifier."""
    
    @pytest.fixture
    def mock_backbone(self):
        """Create a mock backbone for testing."""
        backbone = MagicMock()
        backbone.return_value = torch.randn(2, 1024)  # batch_size=2, feature_dim=1024
        return backbone
    
    @pytest.fixture
    def classifier(self, mock_backbone):
        """Create a SigLIPClassifier instance for testing."""
        with patch('finetune_siglip.SigLIPClassifier._get_backbone_dim', return_value=1024):
            return SigLIPClassifier(
                backbone=mock_backbone,
                num_classes=18,
                dropout_rate=0.1,
                freeze_up_to=0
            )
    
    def test_classifier_initialization(self, mock_backbone):
        """Test that classifier initializes correctly."""
        with patch('finetune_siglip.SigLIPClassifier._get_backbone_dim', return_value=1024):
            classifier = SigLIPClassifier(
                backbone=mock_backbone,
                num_classes=18,
                dropout_rate=0.1,
                freeze_up_to=0
            )
        
        assert classifier.num_classes == 18
        assert isinstance(classifier.classifier, nn.Sequential)
        # Check that classifier has correct output dimension
        assert classifier.classifier[-1].out_features == 18
    
    def test_forward_pass(self, classifier, mock_backbone):
        """Test forward pass through classifier."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, 160, 224, 224)
        
        # Mock the backbone to return features
        mock_backbone.return_value = torch.randn(batch_size, 1024)
        
        output = classifier(input_tensor)
        
        assert output.shape == (batch_size, 18)
        assert isinstance(output, torch.Tensor)
    
    def test_freeze_layers(self, mock_backbone):
        """Test layer freezing functionality."""
        # Create a mock backbone with named parameters
        backbone = MagicMock()
        param1 = torch.nn.Parameter(torch.randn(10, 10))
        param2 = torch.nn.Parameter(torch.randn(10, 10))
        param3 = torch.nn.Parameter(torch.randn(10, 10))
        
        backbone.named_parameters.return_value = [
            ('layer1.weight', param1),
            ('layer2.weight', param2),
            ('layer3.weight', param3)
        ]
        
        with patch('finetune_siglip.SigLIPClassifier._get_backbone_dim', return_value=1024):
            classifier = SigLIPClassifier(
                backbone=backbone,
                num_classes=18,
                freeze_up_to=2
            )
        
        # Check that freeze layers was called
        backbone.named_parameters.assert_called()


class TestLossFunctions:
    """Test cases for custom loss functions."""
    
    def test_focal_loss_initialization(self):
        """Test FocalLoss initialization."""
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
        
        assert focal_loss.alpha == 1.0
        assert focal_loss.gamma == 2.0
        assert focal_loss.reduction == 'mean'
    
    def test_focal_loss_forward(self):
        """Test FocalLoss forward pass."""
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
        
        batch_size = 4
        num_classes = 18
        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, 2, (batch_size, num_classes)).float()
        
        loss = focal_loss(inputs, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar loss
        assert loss >= 0  # loss should be non-negative
    
    def test_focal_loss_reduction_modes(self):
        """Test different reduction modes for FocalLoss."""
        batch_size = 4
        num_classes = 18
        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, 2, (batch_size, num_classes)).float()
        
        # Test mean reduction
        focal_loss_mean = FocalLoss(reduction='mean')
        loss_mean = focal_loss_mean(inputs, targets)
        assert loss_mean.dim() == 0
        
        # Test sum reduction
        focal_loss_sum = FocalLoss(reduction='sum')
        loss_sum = focal_loss_sum(inputs, targets)
        assert loss_sum.dim() == 0
        
        # Test no reduction
        focal_loss_none = FocalLoss(reduction='none')
        loss_none = focal_loss_none(inputs, targets)
        assert loss_none.shape == (batch_size, num_classes)
    
    def test_balanced_bce_loss(self):
        """Test BalancedBCELoss."""
        num_classes = 18
        pos_weights = torch.ones(num_classes) * 2.0
        balanced_loss = BalancedBCELoss(pos_weights=pos_weights)
        
        batch_size = 4
        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, 2, (batch_size, num_classes)).float()
        
        loss = balanced_loss(inputs, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss >= 0
    
    def test_calculate_pos_weights(self):
        """Test calculate_pos_weights function."""
        # Create mock label data
        labels = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ])
        
        pos_weights = calculate_pos_weights(labels)
        
        assert isinstance(pos_weights, torch.Tensor)
        assert pos_weights.shape == (4,)
        assert torch.all(pos_weights > 0)


class TestModelUtils:
    """Test utility functions for model operations."""
    
    def test_set_seed_reproducibility(self):
        """Test that set_seed produces reproducible results."""
        from finetune_siglip import set_seed
        
        # Set seed and generate random numbers
        set_seed(42)
        random1 = torch.rand(5)
        
        # Set same seed and generate again
        set_seed(42)
        random2 = torch.rand(5)
        
        # Should be identical
        assert torch.allclose(random1, random2)
    
    @patch('finetune_siglip.Merlin')
    def test_build_model(self, mock_merlin):
        """Test build_model function."""
        from finetune_siglip import build_model
        
        # Mock the Merlin model
        mock_model_instance = MagicMock()
        mock_model_instance.output_dim = 1024
        mock_merlin.return_value = mock_model_instance
        
        # Test model building
        model = build_model(
            pretrained_path='/fake/path/model.bin',
            num_classes=18,
            dropout_rate=0.1,
            freeze_up_to=0
        )
        
        assert isinstance(model, SigLIPClassifier)
        assert model.num_classes == 18


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 