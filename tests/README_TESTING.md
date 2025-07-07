# Testing and CI/CD Setup for SigLIP CT Training

This document describes the comprehensive testing framework and CI/CD pipeline implemented for the SigLIP CT training project.

## Overview

The testing framework includes:
- **Unit Tests**: Test individual components (models, data loading, loss functions)
- **Integration Tests**: Test complete workflows and component interactions
- **Performance Tests**: Benchmark model performance and training speed
- **Security Tests**: Check for vulnerabilities and security issues
- **Code Quality**: Linting, formatting, and type checking
- **CI/CD Pipeline**: Automated testing on GitHub Actions

## Quick Start

### Prerequisites

- Python 3.8+ 
- pip or conda
- Docker (optional, for containerized testing)

### Setup Test Environment

```bash
# Option 1: Using the test runner script
./run_tests.sh setup

# Option 2: Using Makefile
make setup

# Option 3: Manual setup
pip install -r requirements.txt
pip install -r test-requirements.txt
```

### Running Tests

#### Quick Commands

```bash
# Run all tests
./run_tests.sh all
# or
make test

# Run only unit tests
./run_tests.sh unit
# or 
make test-unit

# Run with coverage
./run_tests.sh coverage
# or
make test-coverage

# Run full CI pipeline locally
./run_tests.sh ci
# or
make ci
```

#### Individual Test Categories

```bash
# Unit tests (fast, isolated)
pytest tests/test_models.py -v
pytest tests/test_data.py -v 
pytest tests/test_training.py -v

# Integration tests (slower, end-to-end)
pytest tests/test_integration.py -v

# All tests with markers
pytest tests/ -v -m "not slow"  # Exclude slow tests
pytest tests/ -v -m "slow"      # Run only slow tests
```

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                # Shared fixtures and configuration
├── test_models.py             # Model component tests
├── test_data.py              # Data loading and preprocessing tests
├── test_training.py          # Training pipeline tests
└── test_integration.py       # End-to-end integration tests
```

### Test Coverage

The tests cover the following components:

#### Model Tests (`test_models.py`)
- ✅ SigLIP classifier initialization and forward pass
- ✅ Custom loss functions (FocalLoss, BalancedBCELoss)
- ✅ Model utilities and helper functions
- ✅ Checkpoint saving and loading
- ✅ Layer freezing functionality

#### Data Tests (`test_data.py`)
- ✅ Dataset loading and preprocessing
- ✅ Data transforms and augmentations
- ✅ HU windowing and normalization
- ✅ Dataloader creation and batching
- ✅ Error handling for invalid data

#### Training Tests (`test_training.py`)
- ✅ Training loop execution
- ✅ Evaluation metrics calculation
- ✅ Checkpoint management
- ✅ Mixed precision training
- ✅ Gradient accumulation and clipping

#### Integration Tests (`test_integration.py`)
- ✅ End-to-end training pipeline
- ✅ Model inference workflow
- ✅ Configuration validation
- ✅ Error handling scenarios

## CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline (`.github/workflows/ci.yml`) includes multiple jobs:

#### 1. Code Quality (`lint`)
- **Black**: Code formatting check
- **isort**: Import sorting
- **flake8**: Linting and style guide enforcement
- **mypy**: Type checking (optional)

#### 2. Security (`security`)
- **safety**: Dependency vulnerability scanning
- **bandit**: Security linting

#### 3. Unit Tests (`test`)
- **Matrix Strategy**: Tests across Python 3.8-3.11 on Ubuntu and macOS
- **CPU-only**: Uses CPU versions of PyTorch for CI efficiency
- **Caching**: pip dependencies cached for faster runs

#### 4. Integration Tests (`integration-test`)
- **Coverage**: Generates test coverage reports
- **Codecov**: Uploads coverage to Codecov.io
- **Artifacts**: Stores coverage reports

#### 5. Performance Tests (`performance`)
- **Benchmarks**: Performance testing (triggered on releases)
- **Metrics**: Stores benchmark results

#### 6. Docker Tests (`docker`)
- **Build**: Tests Docker image building
- **Smoke Test**: Verifies container functionality

#### 7. Deployment (`deploy`)
- **Release Only**: Runs only on GitHub releases
- **Artifacts**: Creates deployment packages
- **Production**: Placeholder for production deployment

### Workflow Triggers

- **Push**: `main` and `develop` branches
- **Pull Request**: All PRs to `main`
- **Release**: When GitHub releases are published
- **Schedule**: Weekly runs on Sundays
- **Manual**: Can be triggered manually

## Local Development

### Test Runner Script

The `run_tests.sh` script provides convenient commands:

```bash
./run_tests.sh help              # Show all available commands
./run_tests.sh setup             # Set up test environment
./run_tests.sh lint              # Run code quality checks
./run_tests.sh fix               # Fix code formatting
./run_tests.sh unit              # Run unit tests
./run_tests.sh integration       # Run integration tests
./run_tests.sh coverage          # Run tests with coverage
./run_tests.sh security          # Run security checks
./run_tests.sh ci                # Run full CI pipeline locally
./run_tests.sh cleanup           # Clean up test artifacts
```

### Makefile Commands

Alternative commands using `make`:

```bash
make help               # Show available targets
make setup              # Set up development environment
make test               # Run all tests
make test-coverage      # Run tests with coverage
make lint               # Run code quality checks
make format             # Format code
make security           # Run security checks
make clean              # Clean up artifacts
make docker-test        # Test Docker setup
make ci                 # Run full CI pipeline
```

## Configuration Files

### pytest Configuration (`pytest.ini`)
```ini
[tool:pytest]
testpaths = tests
addopts = --verbose --tb=short --cov=. --cov-report=term-missing
markers = 
    slow: marks tests as slow
    gpu: marks tests as requiring GPU
    integration: marks tests as integration tests
```

### Test Requirements (`test-requirements.txt`)
- `pytest`: Testing framework
- `pytest-cov`: Coverage reporting
- `pytest-mock`: Mocking utilities
- `black`: Code formatting
- `flake8`: Linting
- `isort`: Import sorting
- `safety`: Security scanning
- `bandit`: Security linting

## Best Practices

### Writing Tests

1. **Use descriptive test names**: `test_focal_loss_with_class_imbalance`
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Use fixtures**: Shared test data and setup
4. **Mock external dependencies**: File I/O, network calls, heavy computations
5. **Test edge cases**: Invalid inputs, boundary conditions
6. **Keep tests fast**: Use minimal data and mock expensive operations

### Test Organization

```python
class TestModelComponents:
    """Group related tests in classes."""
    
    @pytest.fixture
    def mock_model(self):
        """Shared fixture for test class."""
        return MagicMock()
    
    def test_specific_functionality(self, mock_model):
        """Test specific model functionality."""
        # Arrange
        input_data = torch.randn(4, 1, 160, 224, 224)
        
        # Act
        output = mock_model(input_data)
        
        # Assert
        assert output.shape == (4, 18)
```

### Continuous Integration

1. **Fast Feedback**: Unit tests run first, integration tests later
2. **Parallel Execution**: Multiple Python versions tested simultaneously
3. **Caching**: Dependencies cached to speed up builds
4. **Artifact Storage**: Test reports and coverage saved
5. **Security First**: Security checks run on every commit

## Monitoring and Reporting

### Coverage Reports

Coverage reports are generated in multiple formats:
- **Terminal**: Real-time coverage during test runs
- **HTML**: Detailed browser-viewable reports in `htmlcov/`
- **XML**: Machine-readable format for CI integration

### Performance Monitoring

Performance tests track:
- Model inference speed
- Training throughput
- Memory usage
- Data loading performance

### Security Monitoring

Security checks include:
- Dependency vulnerability scanning
- Code security analysis
- Docker image security (when applicable)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes project root
2. **CUDA Errors**: Tests run on CPU by default
3. **Missing Dependencies**: Run `./run_tests.sh setup`
4. **Permission Errors**: Ensure `run_tests.sh` is executable

### Debug Mode

```bash
# Run specific test with detailed output
pytest tests/test_models.py::TestSigLIPClassifier::test_forward_pass -v -s

# Run tests with pdb debugging
pytest tests/test_models.py --pdb

# Run tests with coverage and open HTML report
make test-coverage && open htmlcov/index.html
```

## Integration with IDEs

### VS Code

Add to `.vscode/settings.json`:
```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black"
}
```

### PyCharm

1. Go to Settings → Tools → Python Integrated Tools
2. Set Default test runner to `pytest`
3. Set Package requirements file to `test-requirements.txt`

## Contributing

When contributing:

1. **Write tests**: New features should include tests
2. **Run locally**: Use `./run_tests.sh ci` before pushing
3. **Check coverage**: Aim for >80% test coverage
4. **Follow standards**: Code should pass all linting checks
5. **Update docs**: Update this README if you change the test structure

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Code Coverage with pytest-cov](https://pytest-cov.readthedocs.io/)

## Support

For issues with the testing setup:

1. Check this README first
2. Run `./run_tests.sh help` or `make help`
3. Check GitHub Actions logs for CI failures
4. Create an issue in the repository with:
   - Python version
   - Operating system
   - Error message and full traceback
   - Steps to reproduce 