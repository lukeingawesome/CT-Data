#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
import numpy as np
import os
import sys
import warnings

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set environment variables for testing
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.dirname(__file__))
    
    # Disable CUDA for testing (use CPU only)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    yield
    
    # Cleanup after all tests
    pass


@pytest.fixture
def device():
    """Provide CPU device for testing."""
    return torch.device('cpu')


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    batch_size = 4
    num_classes = 18
    
    return {
        'image': torch.randn(batch_size, 1, 160, 224, 224),
        'labels': torch.randint(0, 2, (batch_size, num_classes)).float(),
        'img_path': [f'/fake/path/image_{i}.nii.gz' for i in range(batch_size)]
    }


@pytest.fixture
def label_names():
    """Provide standard label names for testing."""
    return [
        "Medical material",
        "Arterial wall calcification", 
        "Cardiomegaly",
        "Pericardial effusion",
        "Coronary artery wall calcification",
        "Hiatal hernia",
        "Lymphadenopathy",
        "Emphysema",
        "Atelectasis",
        "Lung nodule",
        "Lung opacity",
        "Pulmonary fibrotic sequela",
        "Pleural effusion",
        "Mosaic attenuation pattern",
        "Peribronchial thickening",
        "Consolidation",
        "Bronchiectasis",
        "Interlobular septal thickening"
    ]


@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing."""
    return {
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 2,
        'num_classes': 18,
        'dropout_rate': 0.1,
        'weight_decay': 1e-4,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0
    }


# Skip tests that require GPU if not available
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle GPU tests."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu) 