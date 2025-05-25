"""
Module for loading and processing molecular data from SDF files.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

from .descriptors import calculate_lipinski_descriptors

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
            
            descriptor_data = []
            for idx, row in df.iterrows():
                mol = row[mol_col_name]
                if mol is not None:
                    descriptors = calculate_lipinski_descriptors(mol)
                    if descriptors:
                        descriptor_data.append(descriptors)
                    else:
                        # Add NaN values for failed calculations
                        descriptor_data.append({
                            'MW': None, 'LogP': None, 'HBA': None,
                            'HBD': None, 'TPSA': None, 'RotBonds': None
                        })
                else:
                    descriptor_data.append({
                        'MW': None, 'LogP': None, 'HBA': None,
                        'HBD': None, 'TPSA': None, 'RotBonds': None
                    })
            
            # Add descriptors to DataFrame
            descriptors_df = pd.DataFrame(descriptor_data)
            result_df = pd.concat([df.reset_index(drop=True), descriptors_df], axis=1)
            
            logger.info("Molecular descriptors calculated successfully")
            return result_df
            
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