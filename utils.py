import numpy as np
import os


def load_parameters(model_path):
    """
    Load model parameters from numpy file.
    
    Args:
        model_path: Path to .npz model file
    
    Returns:
        NumPy file object with model parameters
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model file is invalid
    """
    # Tour 3: Check file existence
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Lilith cannot be summoned without her weights.\n"
            f"Expected location: {os.path.abspath(model_path)}"
        )
    
    # Tour 3: Validate file extension
    if not model_path.endswith('.npz'):
        raise ValueError(
            f"Model file must be .npz format, got: {model_path}\n"
            f"Lilith requires NumPy parameter files."
        )
    
    try:
        params = np.load(model_path)
        return params
    except Exception as e:
        raise ValueError(
            f"Failed to load model from {model_path}: {e}\n"
            f"Model file may be corrupted."
        )
