"""
Structural alert filters for identifying potentially problematic compounds.
Includes PAINS (Pan-Assay Interference Compounds) and BRENK filters.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from typing import Dict, List, Optional, Tuple, Set
import logging

logger = logging.getLogger(__name__)

class StructuralAlertFilter:
    """Class for applying structural alert filters (PAINS, BRENK, etc.)."""
    
    def __init__(self):
        """Initialize the StructuralAlertFilter with common alert patterns."""
        
        # Simplified PAINS patterns (most critical ones only for performance)
        self.pains_patterns = [
            # Quinones (most important PAINS)
            "c1ccc(=O)c(=O)cc1",                             # Simple quinone
            
            # Reactive carbonyls
            "C(=O)C(=O)",                                    # 1,2-dicarbonyl
            
            # Aldehydes (reactive)
            "[CH]=O",                                        # Aldehyde
            
            # Catechols
            "c1ccc(O)c(O)c1",                               # Catechol
            
            # Rhodanines
            "S1C(=O)NC(=O)S1",                              # Rhodanine
        ]
        
        # Simplified BRENK patterns (most critical unwanted functionalities)
        self.brenk_patterns = [
            # Heavy halogens
            "[Br]",                                          # Bromine
            "[I]",                                           # Iodine
            
            # Reactive groups
            "C(=O)Cl",                                       # Acyl chloride
            "S(=O)(=O)Cl",                                   # Sulfonyl chloride
            
            # Epoxides
            "C1OC1",                                         # Epoxide
            
            # Azides
            "N=[N+]=[N-]",                                   # Azide
        ]
        
        # Compile patterns with error handling
        self.compiled_pains = []
        self.compiled_brenk = []
        
        # Compile PAINS patterns
        for i, pattern in enumerate(self.pains_patterns):
            try:
                mol = Chem.MolFromSmarts(pattern)
                if mol is not None:
                    self.compiled_pains.append(mol)
                else:
                    logger.warning(f"Invalid PAINS pattern {i+1}: {pattern}")
            except Exception as e:
                logger.warning(f"Error compiling PAINS pattern {i+1}: {e}")
        
        # Compile BRENK patterns
        for i, pattern in enumerate(self.brenk_patterns):
            try:
                mol = Chem.MolFromSmarts(pattern)
                if mol is not None:
                    self.compiled_brenk.append(mol)
                else:
                    logger.warning(f"Invalid BRENK pattern {i+1}: {pattern}")
            except Exception as e:
                logger.warning(f"Error compiling BRENK pattern {i+1}: {e}")
        
        logger.info(f"Loaded {len(self.compiled_pains)} PAINS patterns")
        logger.info(f"Loaded {len(self.compiled_brenk)} BRENK patterns")
    
    def check_pains(self, mol: Chem.Mol) -> Tuple[bool, List[str]]:
        """
        Check if molecule contains PAINS patterns.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Tuple of (has_pains, list_of_matched_patterns)
        """
        if mol is None:
            return True, ["Invalid molecule"]
        
        matched_patterns = []
        
        for i, pattern in enumerate(self.compiled_pains):
            if mol.HasSubstructMatch(pattern):
                matched_patterns.append(f"PAINS_{i+1}")
        
        has_pains = len(matched_patterns) > 0
        return has_pains, matched_patterns
    
    def check_brenk(self, mol: Chem.Mol) -> Tuple[bool, List[str]]:
        """
        Check if molecule contains BRENK patterns.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Tuple of (has_brenk, list_of_matched_patterns)
        """
        if mol is None:
            return True, ["Invalid molecule"]
        
        matched_patterns = []
        
        for i, pattern in enumerate(self.compiled_brenk):
            if mol.HasSubstructMatch(pattern):
                matched_patterns.append(f"BRENK_{i+1}")
        
        has_brenk = len(matched_patterns) > 0
        return has_brenk, matched_patterns
    
    def filter_dataframe(self, df: pd.DataFrame, 
                        apply_pains: bool = True,
                        apply_brenk: bool = True) -> pd.DataFrame:
        """
        Filter DataFrame based on structural alerts.
        
        Args:
            df: DataFrame containing molecules with 'mol' column
            apply_pains: Whether to apply PAINS filters
            apply_brenk: Whether to apply BRENK filters
            
        Returns:
            Filtered DataFrame without problematic structures
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for structural alert filtering")
            return df
        
        # Check for molecule column (can be 'mol' or 'ROMol')
        mol_col = None
        if 'mol' in df.columns:
            mol_col = 'mol'
        elif 'ROMol' in df.columns:
            mol_col = 'ROMol'
        else:
            logger.error("DataFrame must contain 'mol' or 'ROMol' column with RDKit molecule objects")
            return df
        
        logger.info(f"Applying structural alert filters to {len(df)} molecules")
        
        df = df.copy()
        df['passes_pains'] = True
        df['passes_brenk'] = True
        df['pains_alerts'] = ""
        df['brenk_alerts'] = ""
        
        # Apply PAINS filtering
        if apply_pains:
            pains_results = df[mol_col].apply(self.check_pains)
            df['passes_pains'] = ~pd.Series([result[0] for result in pains_results])
            df['pains_alerts'] = pd.Series([';'.join(result[1]) for result in pains_results])
        
        # Apply BRENK filtering
        if apply_brenk:
            brenk_results = df[mol_col].apply(self.check_brenk)
            df['passes_brenk'] = ~pd.Series([result[0] for result in brenk_results])
            df['brenk_alerts'] = pd.Series([';'.join(result[1]) for result in brenk_results])
        
        # Create overall structural alert filter
        if apply_pains and apply_brenk:
            df['passes_structural_alerts'] = df['passes_pains'] & df['passes_brenk']
        elif apply_pains:
            df['passes_structural_alerts'] = df['passes_pains']
        elif apply_brenk:
            df['passes_structural_alerts'] = df['passes_brenk']
        else:
            df['passes_structural_alerts'] = True
        
        # Filter the DataFrame
        filtered_df = df[df['passes_structural_alerts']].copy()
        
        pains_filtered = len(df) - len(df[df['passes_pains']])
        brenk_filtered = len(df) - len(df[df['passes_brenk']])
        total_filtered = len(df) - len(filtered_df)
        
        logger.info(f"Structural alert filtering results:")
        logger.info(f"  PAINS alerts: {pains_filtered} molecules")
        logger.info(f"  BRENK alerts: {brenk_filtered} molecules")
        logger.info(f"  Total filtered: {total_filtered} molecules")
        logger.info(f"  Remaining: {len(filtered_df)} molecules ({len(filtered_df)/len(df)*100:.1f}% pass rate)")
        
        return filtered_df
    
    def get_alert_statistics(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get statistics about structural alert filtering.
        
        Args:
            df: DataFrame with structural alert results
            
        Returns:
            Dictionary containing alert statistics
        """
        if df.empty or 'passes_structural_alerts' not in df.columns:
            return {}
        
        stats = {
            'total_molecules': len(df),
            'passes_structural_alerts': len(df[df['passes_structural_alerts']]),
            'structural_alert_pass_rate': len(df[df['passes_structural_alerts']]) / len(df) * 100,
        }
        
        if 'passes_pains' in df.columns:
            stats['pains_violations'] = len(df[~df['passes_pains']])
            stats['pains_pass_rate'] = len(df[df['passes_pains']]) / len(df) * 100
        
        if 'passes_brenk' in df.columns:
            stats['brenk_violations'] = len(df[~df['passes_brenk']])
            stats['brenk_pass_rate'] = len(df[df['passes_brenk']]) / len(df) * 100
        
        return stats
    
    def analyze_alert_patterns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze which alert patterns are most commonly found.
        
        Args:
            df: DataFrame with alert results
            
        Returns:
            Dictionary with pattern analysis
        """
        analysis = {}
        
        if 'pains_alerts' in df.columns:
            # Analyze PAINS patterns
            pains_counts = {}
            for alerts in df['pains_alerts']:
                if alerts:
                    for alert in alerts.split(';'):
                        if alert:
                            pains_counts[alert] = pains_counts.get(alert, 0) + 1
            
            analysis['pains'] = {
                'pattern_counts': pains_counts,
                'most_common': sorted(pains_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        
        if 'brenk_alerts' in df.columns:
            # Analyze BRENK patterns
            brenk_counts = {}
            for alerts in df['brenk_alerts']:
                if alerts:
                    for alert in alerts.split(';'):
                        if alert:
                            brenk_counts[alert] = brenk_counts.get(alert, 0) + 1
            
            analysis['brenk'] = {
                'pattern_counts': brenk_counts,
                'most_common': sorted(brenk_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        
        return analysis 