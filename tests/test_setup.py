import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_project_structure():
    """Test that all required directories exist."""
    required_dirs = [
        'src', 'data', 'tests', 'results', 'notebooks',
        'src/data_processing', 'src/filtering', 'src/similarity',
        'src/visualization', 'src/utils',
        'data/raw', 'data/processed', 'data/reference'
    ]
    
    for dir_path in required_dirs:
        assert os.path.exists(dir_path), f"Directory {dir_path} does not exist"

def test_imports():
    """Test that core dependencies can be imported."""
    try:
        import pandas as pd
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        import matplotlib.pyplot as plt
        print("All core dependencies imported successfully!")
    except ImportError as e:
        pytest.fail(f"Failed to import required dependency: {e}")