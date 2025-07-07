#!/bin/bash

# Test runner script for SigLIP CT training project
# This script provides easy commands to run different types of tests locally

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup test environment
setup_test_env() {
    print_status "Setting up test environment..."
    
    # Check if Python is installed
    if ! command_exists python3; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check if pip is installed
    if ! command_exists pip3; then
        print_error "pip3 is not installed"
        exit 1
    fi
    
    # Install test requirements if not already installed
    if [ ! -f "test-requirements-installed.flag" ]; then
        print_status "Installing test requirements..."
        pip3 install -r test-requirements.txt
        touch test-requirements-installed.flag
    fi
    
    # Create test data directory if it doesn't exist
    mkdir -p csv
    
    # Create minimal test data if it doesn't exist
    if [ ! -f "csv/test_data.csv" ]; then
        print_status "Creating minimal test data..."
        python3 -c "
import pandas as pd
import numpy as np

# Create minimal test dataset
data = {
    'img_path': [f'/fake/image_{i}.nii.gz' for i in range(20)],
    'split': ['train'] * 12 + ['val'] * 8,
    'Medical material': np.random.randint(0, 2, 20),
    'Cardiomegaly': np.random.randint(0, 2, 20),
    'Lung nodule': np.random.randint(0, 2, 20),
    'Pleural effusion': np.random.randint(0, 2, 20)
}

df = pd.DataFrame(data)
df.to_csv('csv/test_data.csv', index=False)
print('Created test CSV with', len(df), 'samples')
"
    fi
}

# Function to run linting
run_lint() {
    print_status "Running code quality checks..."
    
    # Check if linting tools are installed
    if ! command_exists black; then
        print_warning "black not found, installing..."
        pip3 install black
    fi
    
    if ! command_exists flake8; then
        print_warning "flake8 not found, installing..."
        pip3 install flake8
    fi
    
    if ! command_exists isort; then
        print_warning "isort not found, installing..."
        pip3 install isort
    fi
    
    print_status "Running black (code formatting)..."
    black --check --diff . || {
        print_warning "Code formatting issues found. Run 'black .' to fix them."
    }
    
    print_status "Running isort (import sorting)..."
    isort --check-only --diff . || {
        print_warning "Import sorting issues found. Run 'isort .' to fix them."
    }
    
    print_status "Running flake8 (linting)..."
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
}

# Function to fix code formatting
fix_formatting() {
    print_status "Fixing code formatting..."
    
    if command_exists black; then
        black .
        print_success "Applied black formatting"
    fi
    
    if command_exists isort; then
        isort .
        print_success "Applied isort import sorting"
    fi
}

# Function to run unit tests
run_unit_tests() {
    print_status "Running unit tests..."
    pytest tests/test_models.py tests/test_data.py tests/test_training.py -v --tb=short
}

# Function to run integration tests
run_integration_tests() {
    print_status "Running integration tests..."
    pytest tests/test_integration.py -v --tb=short -m "not slow"
}

# Function to run all tests
run_all_tests() {
    print_status "Running all tests..."
    pytest tests/ -v --tb=short -m "not slow"
}

# Function to run tests with coverage
run_tests_with_coverage() {
    print_status "Running tests with coverage..."
    pytest tests/ --cov=. --cov-report=term-missing --cov-report=html --cov-report=xml
    print_success "Coverage report generated in htmlcov/"
}

# Function to run performance tests
run_performance_tests() {
    print_status "Running performance tests..."
    pytest tests/ -m "slow" --benchmark-only || {
        print_warning "No performance tests found or benchmark plugin not installed"
    }
}

# Function to run security checks
run_security_checks() {
    print_status "Running security checks..."
    
    if command_exists safety; then
        print_status "Checking dependencies for security vulnerabilities..."
        safety check || print_warning "Security vulnerabilities found in dependencies"
    else
        print_warning "safety not installed, skipping dependency security check"
    fi
    
    if command_exists bandit; then
        print_status "Running bandit security linter..."
        bandit -r . || print_warning "Security issues found in code"
    else
        print_warning "bandit not installed, skipping security linting"
    fi
}

# Function to clean up test artifacts
cleanup() {
    print_status "Cleaning up test artifacts..."
    rm -rf .pytest_cache
    rm -rf htmlcov
    rm -rf .coverage
    rm -f coverage.xml
    rm -f test-requirements-installed.flag
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    print_success "Cleanup completed"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup           Set up test environment"
    echo "  lint            Run code quality checks"
    echo "  fix             Fix code formatting issues"
    echo "  unit            Run unit tests"
    echo "  integration     Run integration tests"
    echo "  all             Run all tests"
    echo "  coverage        Run tests with coverage report"
    echo "  performance     Run performance benchmarks"
    echo "  security        Run security checks"
    echo "  ci              Run full CI pipeline locally"
    echo "  cleanup         Clean up test artifacts"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup && $0 unit        # Setup and run unit tests"
    echo "  $0 ci                      # Run full CI pipeline"
    echo "  $0 fix && $0 lint          # Fix formatting and check code quality"
}

# Main script logic
main() {
    case "${1:-help}" in
        "setup")
            setup_test_env
            ;;
        "lint")
            setup_test_env
            run_lint
            ;;
        "fix")
            fix_formatting
            ;;
        "unit")
            setup_test_env
            run_unit_tests
            ;;
        "integration")
            setup_test_env
            run_integration_tests
            ;;
        "all")
            setup_test_env
            run_all_tests
            ;;
        "coverage")
            setup_test_env
            run_tests_with_coverage
            ;;
        "performance")
            setup_test_env
            run_performance_tests
            ;;
        "security")
            run_security_checks
            ;;
        "ci")
            print_status "Running full CI pipeline locally..."
            setup_test_env
            run_lint
            run_unit_tests
            run_integration_tests
            run_security_checks
            print_success "CI pipeline completed successfully!"
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 