"""Shared pytest fixtures and configuration for all tests."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator
import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files.
    
    Yields:
        Path: Path to the temporary directory
    """
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config():
    """Provide a mock configuration dictionary for testing.
    
    Returns:
        dict: Mock configuration with common settings
    """
    return {
        "model_name": "test_model",
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
        "device": "cpu",
        "seed": 42,
        "output_dir": "test_output",
    }


@pytest.fixture
def sample_text_data():
    """Provide sample text data for testing tokenizers and models.
    
    Returns:
        list: List of sample text strings
    """
    return [
        "Hello, world!",
        "This is a test sentence.",
        "Machine learning is fascinating.",
        "Python is a great programming language.",
        "Testing is important for code quality.",
    ]


@pytest.fixture
def mock_model_weights(temp_dir: Path):
    """Create mock model weight files for testing loading/saving.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path: Path to mock weights file
    """
    weights_file = temp_dir / "model_weights.pt"
    # Create a dummy file to simulate model weights
    weights_file.write_text("mock weights data")
    return weights_file


@pytest.fixture
def env_var_backup():
    """Backup and restore environment variables during tests.
    
    Yields:
        dict: Copy of original environment variables
    """
    original_env = os.environ.copy()
    yield original_env
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    # If using PyTorch, uncomment:
    # import torch
    # torch.manual_seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(42)


@pytest.fixture
def capture_logs():
    """Capture log messages during tests.
    
    Returns:
        list: List to collect log records
    """
    import logging
    
    logs = []
    handler = logging.Handler()
    handler.emit = lambda record: logs.append(record)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    
    yield logs
    
    root_logger.removeHandler(handler)


# Markers for test organization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")