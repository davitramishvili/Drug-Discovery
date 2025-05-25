#!/usr/bin/env python3
"""
Simple script to list available datasets for the antimalarial screening pipeline.
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.dataset_manager import DatasetManager

def main():
    """List available datasets."""
    dataset_manager = DatasetManager()
    dataset_manager.list_datasets()

if __name__ == "__main__":
    main() 