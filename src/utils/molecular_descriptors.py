"""
Centralized molecular descriptor calculations.

This module provides a unified interface for calculating molecular descriptors
to eliminate duplication across the project.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available - descriptor calculations will be limited")

logger = logging.getLogger(__name__)


class MolecularDescriptorCalculator:
    """
    Centralized calculator for molecular descriptors.
    
    This class provides a unified interface for calculating common molecular
    descriptors and eliminates code duplication across the project.
    """
    
    def __init__(self):
        """Initialize the descriptor calculator."""
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - limited functionality")
        
        # Standard descriptor definitions
        self.standard_descriptors = {
            'MW': 'Molecular Weight',
            'LogP': 'Partition Coefficient (LogP)',
            'HBA': 'Hydrogen Bond Acceptors',
            'HBD': 'Hydrogen Bond Donors', 
            'TPSA': 'Topological Polar Surface Area',
            'RotBonds': 'Rotatable Bonds'
        }
        
        logger.info(f"MolecularDescriptorCalculator initialized")
    
    def calculate_single_descriptor(self, mol: Chem.Mol, descriptor: str) -> Optional[float]:
        """
        Calculate a single descriptor for a molecule.
        
        Args:
            mol: RDKit molecule object
            descriptor: Descriptor name (MW, LogP, HBA, HBD, TPSA, RotBonds)
            
        Returns:
            Calculated descriptor value or None if calculation fails
        """
        if not RDKIT_AVAILABLE or mol is None:
            return None
        
        try:
            if descriptor == 'MW':
                return Descriptors.MolWt(mol)
            elif descriptor == 'LogP':
                return Crippen.MolLogP(mol)
            elif descriptor == 'HBA':
                return Descriptors.NumHAcceptors(mol)
            elif descriptor == 'HBD':
                return Descriptors.NumHDonors(mol)
            elif descriptor == 'TPSA':
                return Descriptors.TPSA(mol)
            elif descriptor == 'RotBonds':
                return Descriptors.NumRotatableBonds(mol)
            else:
                logger.warning(f"Unknown descriptor: {descriptor}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to calculate {descriptor}: {e}")
            return None
    
    def calculate_all_descriptors(self, mol: Chem.Mol) -> Dict[str, Optional[float]]:
        """
        Calculate all standard descriptors for a molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with descriptor names as keys and values as values
        """
        results = {}
        for descriptor in self.standard_descriptors.keys():
            results[descriptor] = self.calculate_single_descriptor(mol, descriptor)
        return results
    
    def calculate_descriptors_batch(self, molecules: List[Chem.Mol], 
                                  descriptors: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate descriptors for a list of molecules.
        
        Args:
            molecules: List of RDKit molecule objects
            descriptors: List of descriptor names to calculate (default: all)
            
        Returns:
            DataFrame with molecules as rows and descriptors as columns
        """
        if descriptors is None:
            descriptors = list(self.standard_descriptors.keys())
        
        logger.info(f"Calculating {len(descriptors)} descriptors for {len(molecules)} molecules")
        
        results = []
        for i, mol in enumerate(molecules):
            mol_descriptors = {}
            for desc in descriptors:
                mol_descriptors[desc] = self.calculate_single_descriptor(mol, desc)
            results.append(mol_descriptors)
            
            # Progress reporting
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(molecules)} molecules")
        
        return pd.DataFrame(results)
    
    def add_descriptors_to_dataframe(self, df: pd.DataFrame, 
                                   mol_col: str = 'ROMol',
                                   descriptors: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Add molecular descriptors to an existing DataFrame.
        
        Args:
            df: DataFrame containing molecules
            mol_col: Name of the column containing RDKit molecule objects
            descriptors: List of descriptor names to calculate (default: all)
            
        Returns:
            DataFrame with added descriptor columns
        """
        if mol_col not in df.columns:
            logger.error(f"Molecule column '{mol_col}' not found in DataFrame")
            return df.copy()
        
        if descriptors is None:
            descriptors = list(self.standard_descriptors.keys())
        
        logger.info(f"Adding {len(descriptors)} descriptors to DataFrame with {len(df)} rows")
        
        df_copy = df.copy()
        
        # Calculate descriptors for all molecules
        for desc in descriptors:
            df_copy[desc] = df_copy[mol_col].apply(
                lambda mol: self.calculate_single_descriptor(mol, desc)
            )
        
        return df_copy
    
    def check_lipinski_rule_of_five(self, mol: Chem.Mol) -> Dict[str, bool]:
        """
        Check Lipinski's Rule of Five compliance.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with rule compliance for each criterion
        """
        descriptors = self.calculate_all_descriptors(mol)
        
        if any(v is None for v in [descriptors['MW'], descriptors['LogP'], 
                                 descriptors['HBA'], descriptors['HBD']]):
            return {
                'mw_ok': False,
                'logp_ok': False, 
                'hba_ok': False,
                'hbd_ok': False,
                'passes_lipinski': False
            }
        
        mw_ok = descriptors['MW'] <= 500
        logp_ok = descriptors['LogP'] <= 5
        hba_ok = descriptors['HBA'] <= 10
        hbd_ok = descriptors['HBD'] <= 5
        
        passes_lipinski = sum([mw_ok, logp_ok, hba_ok, hbd_ok]) >= 3  # Allow 1 violation
        
        return {
            'mw_ok': mw_ok,
            'logp_ok': logp_ok,
            'hba_ok': hba_ok, 
            'hbd_ok': hbd_ok,
            'passes_lipinski': passes_lipinski
        }
    
    def get_descriptor_summary_stats(self, df: pd.DataFrame, 
                                   descriptors: Optional[List[str]] = None) -> Dict:
        """
        Get summary statistics for molecular descriptors.
        
        Args:
            df: DataFrame with descriptor columns
            descriptors: List of descriptor names to analyze (default: all)
            
        Returns:
            Dictionary with summary statistics
        """
        if descriptors is None:
            descriptors = [col for col in df.columns if col in self.standard_descriptors]
        
        stats = {}
        for desc in descriptors:
            if desc in df.columns:
                values = df[desc].dropna()
                if len(values) > 0:
                    stats[desc] = {
                        'count': len(values),
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'median': values.median()
                    }
        
        return stats
    
    def validate_drug_likeness(self, mol: Chem.Mol, 
                             custom_ranges: Optional[Dict[str, tuple]] = None) -> Dict:
        """
        Validate drug-likeness based on descriptor ranges.
        
        Args:
            mol: RDKit molecule object
            custom_ranges: Custom descriptor ranges (default: standard drug-like ranges)
            
        Returns:
            Dictionary with validation results
        """
        # Standard drug-like ranges
        default_ranges = {
            'MW': (150, 500),
            'LogP': (-3, 5),
            'HBA': (0, 10),
            'HBD': (0, 5),
            'TPSA': (20, 140),
            'RotBonds': (0, 10)
        }
        
        ranges = custom_ranges or default_ranges
        descriptors = self.calculate_all_descriptors(mol)
        
        results = {
            'descriptors': descriptors,
            'violations': [],
            'passes_filters': True
        }
        
        for desc, (min_val, max_val) in ranges.items():
            if desc in descriptors and descriptors[desc] is not None:
                value = descriptors[desc]
                if not (min_val <= value <= max_val):
                    results['violations'].append(f"{desc}: {value} not in range [{min_val}, {max_val}]")
                    results['passes_filters'] = False
        
        return results


# Global instance for easy access
descriptor_calculator = MolecularDescriptorCalculator()


def calculate_descriptor(mol: Chem.Mol, descriptor: str) -> Optional[float]:
    """
    Convenience function to calculate a single descriptor.
    
    Args:
        mol: RDKit molecule object
        descriptor: Descriptor name
        
    Returns:
        Calculated descriptor value
    """
    return descriptor_calculator.calculate_single_descriptor(mol, descriptor)


def add_standard_descriptors(df: pd.DataFrame, mol_col: str = 'ROMol') -> pd.DataFrame:
    """
    Convenience function to add all standard descriptors to a DataFrame.
    
    Args:
        df: DataFrame with molecule column
        mol_col: Name of the molecule column
        
    Returns:
        DataFrame with added descriptor columns
    """
    return descriptor_calculator.add_descriptors_to_dataframe(df, mol_col) 