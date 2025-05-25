"""
Module for drug-likeness filtering based on molecular descriptors.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DrugLikeFilter:
    """Class for applying drug-likeness filters to molecular datasets."""
    
    def __init__(self, violations_allowed: int = 1):
        """
        Initialize the DrugLikeFilter.
        
        Args:
            violations_allowed: Number of Lipinski violations allowed (default: 1)
        """
        self.violations_allowed = violations_allowed
        
        # Lipinski's Rule of Five criteria
        self.lipinski_criteria = {
            'MW': (0, 500),           # Molecular weight <= 500 Da
            'LogP': (-5, 5),          # LogP <= 5
            'HBA': (0, 10),           # Hydrogen bond acceptors <= 10
            'HBD': (0, 5),            # Hydrogen bond donors <= 5
        }
        
        # Additional drug-like criteria
        self.additional_criteria = {
            'TPSA': (0, 140),         # Topological polar surface area <= 140 Ų
            'RotBonds': (0, 10),      # Rotatable bonds <= 10
        }
        
    def check_lipinski_rule(self, row: pd.Series) -> Tuple[bool, int, Dict[str, bool]]:
        """
        Check if a molecule passes Lipinski's Rule of Five.
        
        Args:
            row: DataFrame row containing molecular descriptors
            
        Returns:
            Tuple of (passes_rule, num_violations, violation_details)
        """
        violations = {}
        violation_count = 0
        
        for descriptor, (min_val, max_val) in self.lipinski_criteria.items():
            if descriptor in row and pd.notna(row[descriptor]):
                value = row[descriptor]
                if value < min_val or value > max_val:
                    violations[descriptor] = True
                    violation_count += 1
                else:
                    violations[descriptor] = False
            else:
                # Missing descriptor counts as violation
                violations[descriptor] = True
                violation_count += 1
                
        passes_rule = violation_count <= self.violations_allowed
        
        return passes_rule, violation_count, violations
        
    def check_additional_criteria(self, row: pd.Series) -> Tuple[bool, Dict[str, bool]]:
        """
        Check additional drug-likeness criteria.
        
        Args:
            row: DataFrame row containing molecular descriptors
            
        Returns:
            Tuple of (passes_criteria, violation_details)
        """
        violations = {}
        
        for descriptor, (min_val, max_val) in self.additional_criteria.items():
            if descriptor in row and pd.notna(row[descriptor]):
                value = row[descriptor]
                violations[descriptor] = value < min_val or value > max_val
            else:
                violations[descriptor] = True
                
        passes_criteria = not any(violations.values())
        
        return passes_criteria, violations
        
    def filter_dataframe(self, df: pd.DataFrame, 
                        apply_lipinski: bool = True,
                        apply_additional: bool = True) -> pd.DataFrame:
        """
        Filter a DataFrame based on drug-likeness criteria.
        
        Args:
            df: DataFrame containing molecules and descriptors
            apply_lipinski: Whether to apply Lipinski's Rule of Five
            apply_additional: Whether to apply additional criteria
            
        Returns:
            Filtered DataFrame with drug-like molecules
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for filtering")
            return df
            
        logger.info(f"Filtering {len(df)} molecules for drug-likeness")
        
        # Initialize filter columns
        df = df.copy()
        df['passes_lipinski'] = True
        df['lipinski_violations'] = 0
        df['passes_additional'] = True
        
        # Apply Lipinski filtering
        if apply_lipinski:
            lipinski_results = df.apply(
                lambda row: self.check_lipinski_rule(row), axis=1
            )
            df['passes_lipinski'] = [result[0] for result in lipinski_results]
            df['lipinski_violations'] = [result[1] for result in lipinski_results]
            
        # Apply additional criteria
        if apply_additional:
            additional_results = df.apply(
                lambda row: self.check_additional_criteria(row), axis=1
            )
            df['passes_additional'] = [result[0] for result in additional_results]
            
        # Create overall filter
        if apply_lipinski and apply_additional:
            df['drug_like'] = df['passes_lipinski'] & df['passes_additional']
        elif apply_lipinski:
            df['drug_like'] = df['passes_lipinski']
        elif apply_additional:
            df['drug_like'] = df['passes_additional']
        else:
            df['drug_like'] = True
            
        # Filter the DataFrame
        filtered_df = df[df['drug_like']].copy()
        
        logger.info(f"Filtered to {len(filtered_df)} drug-like molecules "
                   f"({len(filtered_df)/len(df)*100:.1f}% pass rate)")
        
        return filtered_df
        
    def get_filter_statistics(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get statistics about filtering results.
        
        Args:
            df: DataFrame with filtering results
            
        Returns:
            Dictionary containing filter statistics
        """
        if df.empty:
            return {}
            
        # Check if drug_like column exists
        if 'drug_like' in df.columns:
            drug_like_count = len(df[df['drug_like']])
        else:
            drug_like_count = len(df)  # If no filter applied, all molecules pass
            
        stats = {
            'total_molecules': len(df),
            'drug_like_molecules': drug_like_count,
            'pass_rate': drug_like_count / len(df) * 100,
        }
        
        if 'passes_lipinski' in df.columns:
            stats['lipinski_pass_rate'] = len(df[df['passes_lipinski']]) / len(df) * 100
            stats['avg_lipinski_violations'] = df['lipinski_violations'].mean()
            
        if 'passes_additional' in df.columns:
            stats['additional_pass_rate'] = len(df[df['passes_additional']]) / len(df) * 100
            
        return stats
        
    def analyze_violations(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze which criteria are most commonly violated.
        
        Args:
            df: DataFrame containing molecules and descriptors
            
        Returns:
            Dictionary with violation analysis
        """
        violation_analysis = {}
        
        # Analyze Lipinski violations
        lipinski_violations = {}
        for descriptor in self.lipinski_criteria.keys():
            if descriptor in df.columns:
                min_val, max_val = self.lipinski_criteria[descriptor]
                violations = ((df[descriptor] < min_val) | 
                            (df[descriptor] > max_val) | 
                            df[descriptor].isna())
                lipinski_violations[descriptor] = {
                    'violation_count': violations.sum(),
                    'violation_rate': violations.mean() * 100,
                    'mean_value': df[descriptor].mean(),
                    'criteria': f"{min_val} <= {descriptor} <= {max_val}"
                }
                
        violation_analysis['lipinski'] = lipinski_violations
        
        # Analyze additional criteria violations
        additional_violations = {}
        for descriptor in self.additional_criteria.keys():
            if descriptor in df.columns:
                min_val, max_val = self.additional_criteria[descriptor]
                violations = ((df[descriptor] < min_val) | 
                            (df[descriptor] > max_val) | 
                            df[descriptor].isna())
                additional_violations[descriptor] = {
                    'violation_count': violations.sum(),
                    'violation_rate': violations.mean() * 100,
                    'mean_value': df[descriptor].mean(),
                    'criteria': f"{min_val} <= {descriptor} <= {max_val}"
                }
                
        violation_analysis['additional'] = additional_violations
        
        return violation_analysis 