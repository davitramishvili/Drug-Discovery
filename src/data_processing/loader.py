"""
Module for loading and processing molecular data from SDF files.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging
from rdkit import Chem
from rdkit.Chem import PandasTools
from typing import List, Optional, Dict, Any

# Add project root to path for imports to work from any directory
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import the centralized descriptor calculator
try:
    from src.utils.molecular_descriptors import descriptor_calculator
except ImportError:
    # Fallback for when running from examples subdirectories
    sys.path.insert(0, str(project_root / "src"))
    from utils.molecular_descriptors import descriptor_calculator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MoleculeLoader:
    """Class for loading and processing molecular data from SDF files."""
    
    def __init__(self):
        """Initialize the MoleculeLoader."""
        self.molecules_df = None
        
    def load_sdf(self, file_path: str, mol_col_name: str = 'ROMol') -> pd.DataFrame:
        """
        Load molecules from an SDF file.
        
        Args:
            file_path: Path to the SDF file
            mol_col_name: Name for the molecule column in the DataFrame
            
        Returns:
            DataFrame containing molecules and their properties
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"SDF file not found: {file_path}")
                
            logger.info(f"Loading molecules from {file_path}")
            
            # Load SDF file using PandasTools
            df = PandasTools.LoadSDF(str(file_path), molColName=mol_col_name)
            
            if df.empty:
                logger.warning(f"No molecules found in {file_path}")
                return df
                
            # Remove molecules that failed to load
            df = df[df[mol_col_name].notna()]
            
            logger.info(f"Successfully loaded {len(df)} molecules")
            self.molecules_df = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading SDF file {file_path}: {str(e)}")
            raise
            
    def add_descriptors(self, df: pd.DataFrame, mol_col_name: str = 'ROMol') -> pd.DataFrame:
        """
        Add molecular descriptors to the DataFrame.
        
        Args:
            df: DataFrame containing molecules
            mol_col_name: Name of the molecule column
            
        Returns:
            DataFrame with added descriptor columns
        """
        try:
            logger.info("Calculating molecular descriptors")
            
            # Add molecular descriptors for enhanced analysis
            print("   📊 Computing molecular descriptors...")
            
            # Use centralized descriptor calculator
            for idx, row in df.iterrows():
                mol = row.get(mol_col_name)
                if mol is not None:
                    try:
                        descriptors = descriptor_calculator.calculate_all_descriptors(mol)
                        for desc_name, desc_value in descriptors.items():
                            df.at[idx, desc_name] = desc_value
                    except Exception as e:
                        # Set default values for failed calculations
                        for desc_name in ['MW', 'LogP', 'HBA', 'HBD', 'TPSA', 'RotBonds']:
                            df.at[idx, desc_name] = 0
            
            logger.info("Molecular descriptors calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating descriptors: {str(e)}")
            raise
            
    def load_and_process(self, file_path: str, mol_col_name: str = 'ROMol') -> pd.DataFrame:
        """
        Load SDF file and add molecular descriptors in one step.
        
        Args:
            file_path: Path to the SDF file
            mol_col_name: Name for the molecule column
            
        Returns:
            DataFrame with molecules and descriptors
        """
        df = self.load_sdf(file_path, mol_col_name)
        if not df.empty:
            df = self.add_descriptors(df, mol_col_name)
        return df
        
    def get_molecule_count(self) -> int:
        """Get the number of loaded molecules."""
        if self.molecules_df is not None:
            return len(self.molecules_df)
        return 0
        
    def get_valid_molecules(self, mol_col_name: str = 'ROMol') -> pd.DataFrame:
        """
        Get only molecules that loaded successfully.
        
        Args:
            mol_col_name: Name of the molecule column
            
        Returns:
            DataFrame containing only valid molecules
        """
        if self.molecules_df is not None:
            return self.molecules_df[self.molecules_df[mol_col_name].notna()]
        return pd.DataFrame() 