# Makefile for SigLIP CT Training Project

.PHONY: help setup test test-unit test-integration test-all test-coverage lint fix clean install install-dev docker-build docker-test ci format security

# Default target
help:
	@echo "Available targets:"
	@echo "  setup           - Set up development environment"
	@echo "  install         - Install project dependencies"
	@echo "  install-dev     - Install project + development dependencies"
	@echo "  test            - Run all tests"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration- Run integration tests only"
	@echo "  test-coverage   - Run tests with coverage report"
	@echo "  lint            - Run code quality checks"
	@echo "  format          - Format code with black and isort"
	@echo "  fix             - Alias for format"
	@echo "  security        - Run security checks"
	@echo "  clean           - Clean up build artifacts"
	@echo "  docker-build    - Build Docker image"
	@echo "  docker-test     - Test Docker image"
	@echo "  ci              - Run full CI pipeline locally"
	@echo "  help            - Show this help message"

# Environment setup
setup:
	@echo "Setting up development environment..."
	python3 -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r test-requirements.txt
	@echo "Development environment ready!"

install:
	@echo "Installing project dependencies..."
	python3 -m pip install --upgrade pip
	pip install -r requirements.txt

install-dev: install
	@echo "Installing development dependencies..."
	pip install -r test-requirements.txt

# Testing
test: test-unit test-integration

test-unit:
	@echo "Running unit tests..."
	pytest tests/test_models.py tests/test_data.py tests/test_training.py -v

test-integration:
	@echo "Running integration tests..."
	pytest tests/test_integration.py -v -m "not slow"

test-all:
	@echo "Running all tests..."
	pytest tests/ -v -m "not slow"

test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=. --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "Coverage report generated in htmlcov/"

test-fast:
	@echo "Running fast tests only..."
	pytest tests/ -v -m "not slow" -x

# Code quality
lint:
	@echo "Running code quality checks..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	black --check --diff .
	isort --check-only --diff .

format:
	@echo "Formatting code..."
	black .
	isort .
	@echo "Code formatting completed!"

fix: format

# Security
security:
	@echo "Running security checks..."
	safety check || true
	bandit -r . -ll || true

# Cleanup
clean:
	@echo "Cleaning up..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -f coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleanup completed!"

# Docker
docker-build:
	@echo "Building Docker image..."
	docker build -t siglip-ct:latest .

docker-test: docker-build
	@echo "Testing Docker image..."
	docker run --rm siglip-ct:latest python -c "import torch; print('PyTorch version:', torch.__version__)"
	docker run --rm siglip-ct:latest python -c "import finetune_siglip; print('SigLIP module imported successfully')"

# CI Pipeline
ci: setup lint test security
	@echo "✅ CI pipeline completed successfully!"

# Development helpers
dev-setup: setup
	@echo "Setting up development environment with pre-commit hooks..."
	pip install pre-commit || true
	pre-commit install || true

# Training shortcuts
train-debug:
	@echo "Running debug training..."
	./run_finetune.sh debug

train-basic:
	@echo "Running basic training..."
	./run_finetune.sh basic

train-focal:
	@echo "Running focal loss training..."
	./run_finetune.sh focal

# Quick test data generation
test-data:
	@echo "Creating test data..."
	mkdir -p csv
	python3 -c "import pandas as pd; import numpy as np; data = {'img_path': [f'/fake/image_{i}.nii.gz' for i in range(20)], 'split': ['train'] * 12 + ['val'] * 8, 'Medical material': np.random.randint(0, 2, 20), 'Cardiomegaly': np.random.randint(0, 2, 20), 'Lung nodule': np.random.randint(0, 2, 20), 'Pleural effusion': np.random.randint(0, 2, 20)}; pd.DataFrame(data).to_csv('csv/test_data.csv', index=False); print('Test data created!')"

# Check requirements
check-deps:
	@echo "Checking dependency compatibility..."
	pip check

# Update requirements
update-deps:
	@echo "Updating dependencies..."
	pip-compile requirements.in --upgrade || echo "pip-tools not installed, skipping..."
	pip-compile test-requirements.in --upgrade || echo "pip-tools not installed, skipping..." 