"""
Utility functions for Chapter 3 ML screening implementation.
Provides data persistence and molecular dataframe operations.
"""

import pickle
import os
from pathlib import Path
import pandas as pd
from typing import Optional, List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_molecular_dataframe(df: pd.DataFrame, filename: str, artifacts_dir: str = "artifacts/chapter3/") -> str:
    """
    Save a molecular dataframe with RDKit mol objects to disk.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing molecular data
        filename (str): Name of the file to save (without extension)
        artifacts_dir (str): Directory to save artifacts
        
    Returns:
        str: Full path to the saved file
    """
    # Ensure directory exists
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    
    # Create full filepath
    filepath = Path(artifacts_dir) / f"{filename}.pkl"
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(df, f)
        
        logger.info(f"Successfully saved molecular dataframe to {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.error(f"Error saving molecular dataframe: {e}")
        raise

def load_molecular_dataframe(filepath: str) -> pd.DataFrame:
    """
    Load a molecular dataframe with RDKit mol objects from disk.
    
    Parameters:
        filepath (str): Path to the saved dataframe file
        
    Returns:
        pandas.DataFrame: Loaded molecular dataframe
    """
    try:
        with open(filepath, 'rb') as f:
            df = pickle.load(f)
        
        logger.info(f"Successfully loaded molecular dataframe from {filepath}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading molecular dataframe: {e}")
        raise

def list_saved_dataframes(artifacts_dir: str = "artifacts/chapter3/") -> List[str]:
    """
    List all saved molecular dataframes in the artifacts directory.
    
    Parameters:
        artifacts_dir (str): Directory containing saved dataframes
        
    Returns:
        List[str]: List of saved dataframe filenames
    """
    artifacts_path = Path(artifacts_dir)
    
    if not artifacts_path.exists():
        logger.warning(f"Artifacts directory {artifacts_dir} does not exist")
        return []
    
    # Find all pickle files
    pkl_files = list(artifacts_path.glob("*.pkl"))
    filenames = [f.stem for f in pkl_files]
    
    logger.info(f"Found {len(filenames)} saved dataframes: {filenames}")
    return filenames

def setup_visualization_style():
    """Configure consistent visualization style for Chapter 3 plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    colors = ["#A20025", "#6C8EBF"]  # Define a color palette
    sns.set_palette(sns.color_palette(colors))
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16   
    plt.rcParams['xtick.labelsize'] = 16   
    plt.rcParams['ytick.labelsize'] = 16
    
    logger.info("Visualization style configured")

def setup_rdkit_drawing():
    """Configure RDKit drawing settings for consistent molecular visualizations."""
    try:
        from rdkit.Chem import Draw
        
        d2d = Draw.MolDraw2DSVG(-1, -1)
        dopts = d2d.drawOptions()
        dopts.useBWAtomPalette()
        dopts.setHighlightColour((.635, .0, .145, .4))
        dopts.baseFontSize = 1.0
        dopts.additionalAtomLabelPadding = 0.15
        
        logger.info("RDKit drawing options configured")
        return dopts
    
    except ImportError:
        logger.warning("RDKit not available - drawing options not configured")
        return None

def create_directory_structure(base_dir: str = ".") -> Dict[str, str]:
    """
    Create the complete directory structure for Chapter 3 implementation.
    
    Parameters:
        base_dir (str): Base directory for the project
        
    Returns:
        Dict[str, str]: Dictionary of created directory paths
    """
    directories = {
        'data': 'data/chapter3',
        'artifacts': 'artifacts/chapter3',
        'figures': 'figures/chapter3',
        'src': 'src/chapter3_ml_screening'
    }
    
    created_dirs = {}
    for name, path in directories.items():
        full_path = Path(base_dir) / path
        full_path.mkdir(parents=True, exist_ok=True)
        created_dirs[name] = str(full_path)
        logger.info(f"Created/verified directory: {full_path}")
    
    return created_dirs

def validate_smiles(smiles: str) -> bool:
    """
    Validate a SMILES string using RDKit.
    
    Parameters:
        smiles (str): SMILES string to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except ImportError:
        logger.warning("RDKit not available - cannot validate SMILES")
        return True  # Assume valid if RDKit not available
    except Exception:
        return False

def batch_process_molecules(smiles_list: List[str], process_func, batch_size: int = 1000) -> List[Any]:
    """
    Process molecules in batches to handle memory efficiently.
    
    Parameters:
        smiles_list (List[str]): List of SMILES strings
        process_func: Function to apply to each SMILES
        batch_size (int): Size of each batch
        
    Returns:
        List[Any]: List of processed results
    """
    results = []
    total_batches = len(smiles_list) // batch_size + (1 if len(smiles_list) % batch_size else 0)
    
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        batch_results = [process_func(smi) for smi in batch]
        results.extend(batch_results)
        
        current_batch = i // batch_size + 1
        logger.info(f"Processed batch {current_batch}/{total_batches}")
    
    return results 