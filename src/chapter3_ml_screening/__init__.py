"""
Chapter 3: Ligand-based Screening - Machine Learning for Drug Discovery

This module implements the complete pipeline from Flynn's ML4DD Chapter 3, including:
- Data acquisition and curation for hERG cardiotoxicity prediction
- SMILES standardization and molecular processing
- Morgan fingerprint generation for ML features
- Linear model training with regularization
- Hyperparameter tuning and model evaluation
- Model persistence and deployment utilities

Main Components:
- HERGDataProcessor: Data loading, cleaning, and standardization
- MolecularFeaturizer: SMILES to fingerprint conversion
- HERGClassifier: ML model training and evaluation
- ModelEvaluator: Comprehensive model assessment
- VisualizationTools: Data exploration and results plotting
"""

from .data_processing import HERGDataProcessor
from .molecular_features import MolecularFeaturizer
from .ml_models import HERGClassifier
from .evaluation import ModelEvaluator
from .visualization import VisualizationTools
from .utils import save_molecular_dataframe, load_molecular_dataframe, list_saved_dataframes

__version__ = "1.0.0"
__author__ = "Drug Discovery Pipeline - Chapter 3 Implementation"

__all__ = [
    "HERGDataProcessor",
    "MolecularFeaturizer", 
    "HERGClassifier",
    "ModelEvaluator",
    "VisualizationTools",
    "save_molecular_dataframe",
    "load_molecular_dataframe",
    "list_saved_dataframes"
]

# Set random seed for reproducibility
RANDOM_SEED = 42

# Configuration constants
DEFAULT_FINGERPRINT_PARAMS = {
    'radius': 2,
    'n_bits': 2048
}

DATA_PATHS = {
    'herg_data': 'data/chapter3/hERG_blockers.xlsx',
    'artifacts': 'artifacts/chapter3/',
    'figures': 'figures/chapter3/'
} 