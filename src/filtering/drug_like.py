"""
Module for drug-likeness filtering based on molecular descriptors.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial

from .structural_alerts import StructuralAlertFilter

logger = logging.getLogger(__name__)

class DrugLikeFilter:
    """Class for applying drug-likeness filters to molecular datasets with multi-threading support."""
    
    def __init__(self, violations_allowed: int = 1, apply_pains: bool = True, 
                 apply_brenk: bool = True, apply_nih: bool = False,
                 n_threads: Optional[int] = None):
        """
        Initialize the DrugLikeFilter.
        
        Args:
            violations_allowed: Number of Lipinski violations allowed (default: 1)
            apply_pains: Whether to apply PAINS filters (default: True)
            apply_brenk: Whether to apply BRENK filters (default: True)
            apply_nih: Whether to apply NIH filters (default: False)
            n_threads: Number of threads to use (default: None, auto-detect)
        """
        self.violations_allowed = violations_allowed
        self.apply_pains = apply_pains
        self.apply_brenk = apply_brenk
        self.apply_nih = apply_nih
        
        # Threading configuration
        import os
        self.n_threads = n_threads or min(8, (os.cpu_count() or 1) + 4)
        self._lock = threading.Lock()
        
        # Initialize structural alert filter if needed
        if self.apply_pains or self.apply_brenk or self.apply_nih:
            self.structural_filter = StructuralAlertFilter()
        else:
            self.structural_filter = None
        
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
        
    def _process_chunk_lipinski(self, chunk: pd.DataFrame) -> List[Tuple[bool, int, Dict[str, bool]]]:
        """
        Process a chunk of molecules for Lipinski filtering in a thread-safe manner.
        
        Args:
            chunk: DataFrame chunk to process
            
        Returns:
            List of tuples containing (passes_rule, num_violations, violation_details)
        """
        results = []
        for _, row in chunk.iterrows():
            result = self.check_lipinski_rule(row)
            results.append(result)
        return results
    
    def _process_chunk_additional(self, chunk: pd.DataFrame) -> List[Tuple[bool, Dict[str, bool]]]:
        """
        Process a chunk of molecules for additional criteria filtering in a thread-safe manner.
        
        Args:
            chunk: DataFrame chunk to process
            
        Returns:
            List of tuples containing (passes_criteria, violation_details)
        """
        results = []
        for _, row in chunk.iterrows():
            result = self.check_additional_criteria(row)
            results.append(result)
        return results
    
    def filter_dataframe_threaded(self, df: pd.DataFrame, 
                                apply_lipinski: bool = True,
                                apply_additional: bool = True,
                                apply_structural_alerts: bool = True,
                                chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Filter a DataFrame based on drug-likeness criteria using multi-threading.
        
        Args:
            df: DataFrame containing molecules and descriptors
            apply_lipinski: Whether to apply Lipinski's Rule of Five
            apply_additional: Whether to apply additional criteria
            apply_structural_alerts: Whether to apply structural alert filters
            chunk_size: Size of chunks for threading (default: auto-calculate)
            
        Returns:
            Filtered DataFrame with drug-like molecules
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for filtering")
            return df
            
        logger.info(f"Filtering {len(df)} molecules for drug-likeness using {self.n_threads} threads")
        
        # Initialize filter columns
        df = df.copy()
        df['passes_lipinski'] = True
        df['lipinski_violations'] = 0
        df['passes_additional'] = True
        df['passes_structural_alerts'] = True
        
        # Calculate chunk size
        if chunk_size is None:
            chunk_size = max(1, len(df) // (self.n_threads * 2))
        
        # Apply Lipinski filtering with threading
        if apply_lipinski:
            logger.info("Applying Lipinski Rule of Five filtering...")
            chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
            
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                future_to_chunk = {
                    executor.submit(self._process_chunk_lipinski, chunk): i 
                    for i, chunk in enumerate(chunks)
                }
                
                lipinski_results = [None] * len(chunks)
                for future in as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        lipinski_results[chunk_idx] = future.result()
                    except Exception as e:
                        logger.error(f"Error processing Lipinski chunk {chunk_idx}: {e}")
                        lipinski_results[chunk_idx] = []
            
            # Flatten results and assign to DataFrame
            all_results = []
            for chunk_results in lipinski_results:
                all_results.extend(chunk_results)
            
            if len(all_results) == len(df):
                df['passes_lipinski'] = [result[0] for result in all_results]
                df['lipinski_violations'] = [result[1] for result in all_results]
            
        # Apply additional criteria with threading
        if apply_additional:
            logger.info("Applying additional drug-likeness criteria...")
            chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
            
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                future_to_chunk = {
                    executor.submit(self._process_chunk_additional, chunk): i 
                    for i, chunk in enumerate(chunks)
                }
                
                additional_results = [None] * len(chunks)
                for future in as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        additional_results[chunk_idx] = future.result()
                    except Exception as e:
                        logger.error(f"Error processing additional criteria chunk {chunk_idx}: {e}")
                        additional_results[chunk_idx] = []
            
            # Flatten results and assign to DataFrame
            all_results = []
            for chunk_results in additional_results:
                all_results.extend(chunk_results)
            
            if len(all_results) == len(df):
                df['passes_additional'] = [result[0] for result in all_results]
        
        # Apply structural alert filtering (structural alerts module needs to handle its own threading)
        if apply_structural_alerts and self.structural_filter is not None:
            if 'mol' in df.columns or 'ROMol' in df.columns:
                logger.info("Applying structural alert filtering...")
                df = self.structural_filter.filter_dataframe(
                    df, 
                    apply_pains=self.apply_pains,
                    apply_brenk=self.apply_brenk,
                    apply_nih=self.apply_nih
                )
            else:
                logger.warning("No 'mol' or 'ROMol' column found - skipping structural alert filtering")
                df['passes_structural_alerts'] = True
            
        # Create overall filter
        filters_to_apply = []
        if apply_lipinski:
            filters_to_apply.append('passes_lipinski')
        if apply_additional:
            filters_to_apply.append('passes_additional')
        if apply_structural_alerts:
            filters_to_apply.append('passes_structural_alerts')
        
        if filters_to_apply:
            df['drug_like'] = df[filters_to_apply].all(axis=1)
        else:
            df['drug_like'] = True
            
        # Filter the DataFrame
        filtered_df = df[df['drug_like']].copy()
        
        logger.info(f"Filtered to {len(filtered_df)} drug-like molecules "
                   f"({len(filtered_df)/len(df)*100:.1f}% pass rate)")
        
        return filtered_df
        
    def filter_dataframe(self, df: pd.DataFrame, 
                        apply_lipinski: bool = True,
                        apply_additional: bool = True,
                        apply_structural_alerts: bool = True,
                        use_threading: bool = True,
                        chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Filter a DataFrame based on drug-likeness criteria.
        
        Args:
            df: DataFrame containing molecules and descriptors
            apply_lipinski: Whether to apply Lipinski's Rule of Five
            apply_additional: Whether to apply additional criteria
            apply_structural_alerts: Whether to apply structural alert filters
            use_threading: Whether to use multi-threading (default: True)
            chunk_size: Size of chunks for threading (default: auto-calculate)
            
        Returns:
            Filtered DataFrame with drug-like molecules
        """
        if use_threading and len(df) > 100:  # Use threading for larger datasets
            return self.filter_dataframe_threaded(
                df, apply_lipinski, apply_additional, apply_structural_alerts, chunk_size
            )
        else:
            return self._filter_dataframe_single_threaded(
                df, apply_lipinski, apply_additional, apply_structural_alerts
            )

    def _filter_dataframe_single_threaded(self, df: pd.DataFrame, 
                                        apply_lipinski: bool = True,
                                        apply_additional: bool = True,
                                        apply_structural_alerts: bool = True) -> pd.DataFrame:
        """
        Filter a DataFrame based on drug-likeness criteria using single thread (original implementation).
        
        Args:
            df: DataFrame containing molecules and descriptors
            apply_lipinski: Whether to apply Lipinski's Rule of Five
            apply_additional: Whether to apply additional criteria
            apply_structural_alerts: Whether to apply structural alert filters
            
        Returns:
            Filtered DataFrame with drug-like molecules
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for filtering")
            return df
            
        logger.info(f"Filtering {len(df)} molecules for drug-likeness (single-threaded)")
        
        # Initialize filter columns
        df = df.copy()
        df['passes_lipinski'] = True
        df['lipinski_violations'] = 0
        df['passes_additional'] = True
        df['passes_structural_alerts'] = True
        
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
        
        # Apply structural alert filtering
        if apply_structural_alerts and self.structural_filter is not None:
            if 'mol' in df.columns or 'ROMol' in df.columns:
                df = self.structural_filter.filter_dataframe(
                    df, 
                    apply_pains=self.apply_pains,
                    apply_brenk=self.apply_brenk,
                    apply_nih=self.apply_nih
                )
            else:
                logger.warning("No 'mol' or 'ROMol' column found - skipping structural alert filtering")
                df['passes_structural_alerts'] = True
            
        # Create overall filter
        filters_to_apply = []
        if apply_lipinski:
            filters_to_apply.append('passes_lipinski')
        if apply_additional:
            filters_to_apply.append('passes_additional')
        if apply_structural_alerts:
            filters_to_apply.append('passes_structural_alerts')
        
        if filters_to_apply:
            df['drug_like'] = df[filters_to_apply].all(axis=1)
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
    
    def calculate_violations_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate violations for all molecules without filtering them out.
        This is used for visualization purposes to show the full distribution.
        
        Args:
            df: DataFrame containing molecules and descriptors
            
        Returns:
            DataFrame with violation calculations but no filtering applied
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for violation calculation")
            return df
            
        # Initialize columns
        df = df.copy()
        df['passes_lipinski'] = True
        df['lipinski_violations'] = 0
        df['passes_additional'] = True
        
        # Calculate Lipinski violations
        lipinski_results = df.apply(
            lambda row: self.check_lipinski_rule(row), axis=1
        )
        df['passes_lipinski'] = [result[0] for result in lipinski_results]
        df['lipinski_violations'] = [result[1] for result in lipinski_results]
        
        # Calculate additional criteria
        additional_results = df.apply(
            lambda row: self.check_additional_criteria(row), axis=1
        )
        df['passes_additional'] = [result[0] for result in additional_results]
        
        # Create overall drug-likeness assessment
        df['drug_like'] = df['passes_lipinski'] & df['passes_additional']
        
        return df
        
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

    def _apply_filters_batch(self, molecules_batch: List[Tuple]) -> List[Dict]:
        """Apply filters to a batch of molecules using centralized infrastructure."""
        # Use centralized descriptor calculator
        from src.utils.molecular_descriptors import descriptor_calculator
        
        results = []
        for idx, (mol, original_data) in enumerate(molecules_batch):
            result = {
                'original_index': original_data.get('original_index', idx),
                'mol': mol,
                'passes_filter': False,
                'violations': [],
                'descriptors': {}
            }
            
            if mol is None:
                result['violations'].append('Invalid molecule')
                results.append(result)
                continue
            
            # Calculate descriptors using centralized calculator
            descriptors = descriptor_calculator.calculate_all_descriptors(mol)
            result['descriptors'] = descriptors
            
            # Check Lipinski filters
            violations = []
            if self.apply_lipinski:
                for desc, (min_val, max_val) in self.lipinski_ranges.items():
                    if desc in descriptors and descriptors[desc] is not None:
                        value = descriptors[desc]
                        if not (min_val <= value <= max_val):
                            violations.append(f"{desc}: {value:.2f} not in [{min_val}, {max_val}]")
            
            # Check additional filters if enabled
            if self.apply_additional:
                if descriptors.get('RotBonds', 0) > 10:
                    violations.append(f"RotBonds: {descriptors['RotBonds']} > 10")
            
            # Determine if molecule passes
            violation_count = len(violations)
            result['violations'] = violations
            result['passes_filter'] = violation_count <= self.violations_allowed
            
            results.append(result) 