"""
Utility functions for applying different types of structural filters to compounds.
These functions make it easier to apply specific filters independently.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from typing import Tuple, List, Dict
import logging
from .structural_alerts import StructuralAlertFilter

logger = logging.getLogger(__name__)

def apply_pains_filter(df: pd.DataFrame, mol_col: str = 'mol') -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Apply only PAINS filter to the compound library.
    
    Args:
        df: DataFrame containing molecules
        mol_col: Name of the molecule column
        
    Returns:
        Tuple of (filtered DataFrame, statistics dictionary)
    """
    filter_obj = StructuralAlertFilter()
    
    # Apply only PAINS filter
    filtered_df = filter_obj.filter_dataframe(df, apply_pains=True, apply_brenk=False, apply_nih=False)
    
    # Get statistics
    stats = {
        'total_molecules': len(df),
        'remaining_molecules': len(filtered_df),
        'filtered_molecules': len(df) - len(filtered_df),
        'pass_rate': len(filtered_df) / len(df) * 100 if len(df) > 0 else 0
    }
    
    return filtered_df, stats

def apply_brenk_filter(df: pd.DataFrame, mol_col: str = 'mol') -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Apply only BRENK filter to the compound library.
    
    Args:
        df: DataFrame containing molecules
        mol_col: Name of the molecule column
        
    Returns:
        Tuple of (filtered DataFrame, statistics dictionary)
    """
    filter_obj = StructuralAlertFilter()
    
    # Apply only BRENK filter
    filtered_df = filter_obj.filter_dataframe(df, apply_pains=False, apply_brenk=True, apply_nih=False)
    
    # Get statistics
    stats = {
        'total_molecules': len(df),
        'remaining_molecules': len(filtered_df),
        'filtered_molecules': len(df) - len(filtered_df),
        'pass_rate': len(filtered_df) / len(df) * 100 if len(df) > 0 else 0
    }
    
    return filtered_df, stats

def apply_nih_filter(df: pd.DataFrame, mol_col: str = 'mol') -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Apply only NIH filter to the compound library.
    
    Args:
        df: DataFrame containing molecules
        mol_col: Name of the molecule column
        
    Returns:
        Tuple of (filtered DataFrame, statistics dictionary)
    """
    filter_obj = StructuralAlertFilter()
    
    # Apply only NIH filter
    filtered_df = filter_obj.filter_dataframe(df, apply_pains=False, apply_brenk=False, apply_nih=True)
    
    # Get statistics
    stats = {
        'total_molecules': len(df),
        'remaining_molecules': len(filtered_df),
        'filtered_molecules': len(df) - len(filtered_df),
        'pass_rate': len(filtered_df) / len(df) * 100 if len(df) > 0 else 0
    }
    
    return filtered_df, stats

def apply_custom_filter_combination(df: pd.DataFrame, 
                                 use_pains: bool = True,
                                 use_brenk: bool = True,
                                 use_nih: bool = False,
                                 mol_col: str = 'mol') -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Apply a custom combination of structural filters.
    
    Args:
        df: DataFrame containing molecules
        use_pains: Whether to apply PAINS filter
        use_brenk: Whether to apply BRENK filter
        use_nih: Whether to apply NIH filter
        mol_col: Name of the molecule column
        
    Returns:
        Tuple of (filtered DataFrame, statistics dictionary)
    """
    filter_obj = StructuralAlertFilter()
    
    # Apply selected filters
    filtered_df = filter_obj.filter_dataframe(df, 
                                           apply_pains=use_pains,
                                           apply_brenk=use_brenk,
                                           apply_nih=use_nih)
    
    # Get detailed statistics
    stats = {
        'total_molecules': len(df),
        'remaining_molecules': len(filtered_df),
        'filtered_molecules': len(df) - len(filtered_df),
        'pass_rate': len(filtered_df) / len(df) * 100 if len(df) > 0 else 0,
        'filters_applied': {
            'pains': use_pains,
            'brenk': use_brenk,
            'nih': use_nih
        }
    }
    
    # Add per-filter statistics if available
    if use_pains and 'passes_pains' in filtered_df.columns:
        stats['pains_pass_rate'] = len(filtered_df[filtered_df['passes_pains']]) / len(df) * 100
        
    if use_brenk and 'passes_brenk' in filtered_df.columns:
        stats['brenk_pass_rate'] = len(filtered_df[filtered_df['passes_brenk']]) / len(df) * 100
        
    if use_nih and 'passes_nih' in filtered_df.columns:
        stats['nih_pass_rate'] = len(filtered_df[filtered_df['passes_nih']]) / len(df) * 100
    
    return filtered_df, stats

def analyze_filter_results(df: pd.DataFrame) -> Dict[str, any]:
    """
    Analyze the results of structural filtering in detail.
    
    Args:
        df: DataFrame with filter results
        
    Returns:
        Dictionary containing detailed analysis
    """
    filter_obj = StructuralAlertFilter()
    
    # Get basic statistics
    stats = filter_obj.get_alert_statistics(df)
    
    # Get pattern analysis
    pattern_analysis = filter_obj.analyze_alert_patterns(df)
    
    # Combine results
    analysis = {
        'statistics': stats,
        'pattern_analysis': pattern_analysis,
        'filter_correlations': {}
    }
    
    # Calculate correlations between filters if multiple were applied
    filter_cols = [col for col in ['passes_pains', 'passes_brenk', 'passes_nih'] 
                  if col in df.columns]
    
    if len(filter_cols) > 1:
        correlations = df[filter_cols].corr()
        analysis['filter_correlations'] = correlations.to_dict()
    
    return analysis 