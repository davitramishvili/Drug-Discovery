"""
Data processing module for Chapter 3 ML screening.
Handles hERG data loading, SMILES standardization, and molecular object creation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging
import urllib.request

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize.rdMolStandardize import (
        Cleanup, LargestFragmentChooser, TautomerEnumerator, Uncharger
    )
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available - some functionality will be limited")

logger = logging.getLogger(__name__)

class HERGDataProcessor:
    """
    Handles loading, processing, and standardization of hERG blocker data.
    
    This class implements the data processing pipeline from Flynn's Chapter 3,
    including SMILES standardization and molecular object creation.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the data processor.
        
        Parameters:
            random_seed (int): Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.data = None
        self.processed_data = None
        
        # Initialize standardization components
        if RDKIT_AVAILABLE:
            self.cleanup = Cleanup
            self.fragment_chooser = LargestFragmentChooser()
            self.uncharger = Uncharger()
            self.tautomer_enumerator = TautomerEnumerator()
        
        logger.info("HERGDataProcessor initialized")
    
    def load_herg_blockers_data(self) -> Optional[pd.DataFrame]:
        """
        Load hERG blockers dataset.
        Will try to download if not present locally.
        """
        # Try to find project root by looking for the 'data' directory
        current_path = Path(__file__).parent
        project_root = None
        
        # Search up the directory tree for project root
        for parent in [current_path] + list(current_path.parents):
            if (parent / 'data').exists():
                project_root = parent
                break
        
        if project_root is None:
            # Fallback to relative path
            data_path = Path("data/chapter3/hERG_blockers.xlsx")
        else:
            data_path = project_root / "data/chapter3/hERG_blockers.xlsx"
        
        # Download if not present
        if not data_path.exists():
            logger.info(f"Data file not found at {data_path}, attempting to download...")
            self._download_herg_data(data_path)
        
        # Load the data, skipping header rows and the footer
        try:
            df = pd.read_excel(
                data_path,
                usecols="A:F",
                header=None,
                skiprows=[0, 1],
                names=["SMILES", "Name", "pIC50", "Class", "Scaffold Split", "Random Split"],
            ).head(-68)  # Remove footer rows
            
            logger.info(f"Successfully loaded {len(df)} compounds.")
            
            # Store the data
            self.data = df
            
            # Display basic info
            self._display_data_info(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def _download_herg_data(self, data_path: Path) -> bool:
        """Download hERG data from the GitHub repository."""
        url = "https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/data/ch03/hERG_blockers.xlsx"
        
        try:
            # Ensure directory exists
            data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download the file
            urllib.request.urlretrieve(url, data_path)
            logger.info(f"Successfully downloaded hERG data to {data_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download hERG data: {e}")
            return False
    
    def _display_data_info(self, df: pd.DataFrame):
        """Display basic information about the loaded dataset."""
        logger.info("Dataset preview:")
        logger.info(f"\n{df.head()}")
        
        logger.info("\nDataset information:")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Non-null counts:\n{df.count()}")
        
        logger.info("\nBasic statistics:")
        logger.info(f"\n{df.describe()}")
    
    def process_smiles(self, smi: str) -> Optional[object]:
        """
        Process SMILES strings to create standardized molecule objects.
        
        This implements the standardization pipeline from the notebook:
        1. Convert SMILES to molecule objects
        2. Clean up the molecules
        3. Select the largest fragment (for salts or mixtures)
        4. Neutralize charges
        5. Canonicalize tautomers
        
        Parameters:
            smi (str): SMILES string representing a molecule
            
        Returns:
            rdkit.Chem.Mol: Standardized RDKit molecule object, or None if processing fails
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - returning SMILES string")
            return smi
        
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smi)
        
        if mol is None:
            return None
        
        try:
            # Apply standardization steps
            mol = self.cleanup(mol)  # Remove reactive groups, standardize bonds
            mol = self.fragment_chooser.choose(mol)  # Select largest fragment (removes salts)
            mol = self.uncharger.uncharge(mol)  # Neutralize charges where possible
            mol = self.tautomer_enumerator.Canonicalize(mol)  # Standardize tautomers
            
            return mol
            
        except Exception as e:
            logger.warning(f"Failed to standardize SMILES {smi}: {e}")
            return None
    
    def standardize_molecules(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply standardization to all molecules in the dataset.
        
        Parameters:
            df (pandas.DataFrame, optional): DataFrame to process. Uses self.data if None.
            
        Returns:
            pandas.DataFrame: DataFrame with added 'mol' column containing standardized molecules
        """
        if df is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_herg_blockers_data() first.")
            df = self.data.copy()
        else:
            df = df.copy()
        
        logger.info("Standardizing SMILES strings...")
        
        # Apply standardization to all molecules
        df["mol"] = df["SMILES"].apply(self.process_smiles)
        
        # Count non-standardizable molecules
        invalid_mols = df[df["mol"].isna()]
        if len(invalid_mols) > 0:
            logger.warning(f"{len(invalid_mols)} molecules could not be standardized.")
            # Remove invalid molecules
            df = df.dropna(subset=["mol"])
            logger.info(f"Dataset size after removing invalid molecules: {len(df)}")
        
        self.processed_data = df
        return df
    
    def split_data(self, df: Optional[pd.DataFrame] = None, 
                   split_col: str = "Random Split", 
                   train_pattern: str = "Train", 
                   test_pattern: str = "Test") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets based on a split column.
        
        Parameters:
            df (pandas.DataFrame, optional): DataFrame to split. Uses processed_data if None.
            split_col (str): Column name with split information
            train_pattern (str): Pattern to identify training examples
            test_pattern (str): Pattern to identify test examples
            
        Returns:
            tuple: (train_df, test_df) DataFrames with training and testing data
        """
        if df is None:
            if self.processed_data is None:
                raise ValueError("No processed data available. Call standardize_molecules() first.")
            df = self.processed_data
        
        # Extract indices for train and test sets
        train_mask = df[split_col].str.contains(train_pattern, na=False)
        test_mask = df[split_col].str.contains(test_pattern, na=False)
        
        # Create train and test dataframes
        train_df = df[train_mask].copy().reset_index(drop=True)
        test_df = df[test_mask].copy().reset_index(drop=True)
        
        # Shuffle the data
        np.random.seed(self.random_seed)
        train_df = train_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        logger.info(f"Split data into {len(train_df)} training and {len(test_df)} testing examples")
        
        return train_df, test_df
    
    def simulate_annotation_error(self, df: pd.DataFrame, 
                                  col: str = "pIC50", 
                                  error_value: float = 3.0) -> pd.Series:
        """
        Simulate annotation errors in a dataset by adding an error value to all entries.
        
        This demonstrates how data quality issues can affect distributions.
        
        Parameters:
            df (pandas.DataFrame): DataFrame containing the data
            col (str): Column name to modify
            error_value (float): Error value to add
            
        Returns:
            pandas.Series: Series with simulated errors
        """
        logger.info(f"Simulating annotation error by adding {error_value} to {col}")
        return df[col] + error_value
    
    def get_activity_distribution_stats(self, df: Optional[pd.DataFrame] = None, 
                                       activity_col: str = "pIC50") -> dict:
        """
        Get statistical summary of activity distribution.
        
        Parameters:
            df (pandas.DataFrame, optional): DataFrame to analyze
            activity_col (str): Column name with activity values
            
        Returns:
            dict: Dictionary with distribution statistics
        """
        if df is None:
            df = self.processed_data or self.data
        
        if df is None or activity_col not in df.columns:
            return {}
        
        stats = {
            'count': df[activity_col].count(),
            'mean': df[activity_col].mean(),
            'std': df[activity_col].std(),
            'min': df[activity_col].min(),
            'max': df[activity_col].max(),
            'median': df[activity_col].median(),
            'q25': df[activity_col].quantile(0.25),
            'q75': df[activity_col].quantile(0.75)
        }
        
        logger.info(f"Activity distribution stats for {activity_col}:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.3f}")
        
        return stats
    
    def get_extreme_molecules(self, df: Optional[pd.DataFrame] = None,
                             activity_col: str = "pIC50",
                             n_top: int = 4,
                             n_bottom: int = 4) -> pd.DataFrame:
        """
        Get molecules with extreme activity values for visualization.
        
        Parameters:
            df (pandas.DataFrame, optional): DataFrame to analyze
            activity_col (str): Column name with activity values
            n_top (int): Number of top molecules to return
            n_bottom (int): Number of bottom molecules to return
            
        Returns:
            pandas.DataFrame: DataFrame with extreme molecules
        """
        if df is None:
            df = self.processed_data or self.data
        
        if df is None:
            return pd.DataFrame()
        
        # Sort by activity
        df_sorted = df.sort_values(activity_col, ascending=False)
        
        # Get extremes
        top_molecules = df_sorted.head(n_top)
        bottom_molecules = df_sorted.dropna(subset=[activity_col]).tail(n_bottom)
        
        extremes = pd.concat([top_molecules, bottom_molecules])
        
        logger.info(f"Retrieved {len(extremes)} extreme molecules")
        return extremes 