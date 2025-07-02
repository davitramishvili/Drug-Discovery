"""
Molecular features module for Chapter 3 ML screening.
Handles fingerprint generation and molecular feature transformations.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple
import logging
from sklearn.base import BaseEstimator, TransformerMixin

# RDKit imports
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdFingerprintGenerator import AdditionalOutput, GetMorganGenerator
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available - fingerprint functionality will be limited")

logger = logging.getLogger(__name__)

class MolecularFeaturizer:
    """
    Handles molecular fingerprint generation and feature extraction.
    
    This class implements the fingerprint generation pipeline from Flynn's Chapter 3,
    focusing on Morgan (ECFP) fingerprints for machine learning applications.
    """
    
    def __init__(self, radius: int = 2, n_bits: int = 2048, use_features: bool = False):
        """
        Initialize the molecular featurizer.
        
        Parameters:
            radius (int): Radius for Morgan fingerprints (default: 2)
            n_bits (int): Number of bits in the fingerprint (default: 2048)
            use_features (bool): Whether to use feature-based fingerprints
        """
        self.radius = radius
        self.n_bits = n_bits
        self.use_features = use_features
        
        if RDKIT_AVAILABLE:
            # Initialize Morgan fingerprint generator
            self.morgan_generator = GetMorganGenerator(
                radius=self.radius, 
                fpSize=self.n_bits,
                includeChirality=True
            )
        
        logger.info(f"MolecularFeaturizer initialized with radius={radius}, n_bits={n_bits}")
    
    def compute_fingerprint(self, mol, radius: Optional[int] = None, n_bits: Optional[int] = None) -> np.ndarray:
        """
        Compute Morgan fingerprint for a molecule.
        
        This implements the fingerprint computation from the notebook.
        
        Parameters:
            mol: RDKit molecule object or SMILES string
            radius (int, optional): Fingerprint radius (uses instance default if None)
            n_bits (int, optional): Number of bits (uses instance default if None)
            
        Returns:
            numpy.ndarray: Binary fingerprint array
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - returning dummy fingerprint")
            return np.zeros(self.n_bits)
        
        # Use instance defaults if not provided
        if radius is None:
            radius = self.radius
        if n_bits is None:
            n_bits = self.n_bits
        
        # Convert SMILES to molecule if needed
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        if mol is None:
            logger.warning("Invalid molecule - returning zero fingerprint")
            return np.zeros(n_bits)
        
        try:
            # Generate fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            
            # Convert to numpy array
            arr = np.zeros((n_bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            
            return arr
            
        except Exception as e:
            logger.warning(f"Error computing fingerprint: {e}")
            return np.zeros(n_bits)
    
    def compute_fingerprints_batch(self, molecules: List, 
                                   radius: Optional[int] = None, 
                                   n_bits: Optional[int] = None) -> np.ndarray:
        """
        Compute fingerprints for a batch of molecules.
        
        Parameters:
            molecules (List): List of RDKit molecule objects or SMILES strings
            radius (int, optional): Fingerprint radius
            n_bits (int, optional): Number of bits
            
        Returns:
            numpy.ndarray: Array of fingerprints (n_molecules, n_bits)
        """
        fingerprints = []
        
        for i, mol in enumerate(molecules):
            fp = self.compute_fingerprint(mol, radius, n_bits)
            fingerprints.append(fp)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(molecules)} molecules")
        
        return np.array(fingerprints)
    
    def explore_fingerprint_features(self, fingerprints: np.ndarray) -> dict:
        """
        Explore fingerprint features to understand the data distribution.
        
        Parameters:
            fingerprints (numpy.ndarray): Array of fingerprints
            
        Returns:
            dict: Dictionary with feature statistics
        """
        if fingerprints.size == 0:
            return {}
        
        stats = {
            'n_molecules': fingerprints.shape[0],
            'n_bits': fingerprints.shape[1],
            'mean_bits_per_molecule': np.mean(np.sum(fingerprints, axis=1)),
            'std_bits_per_molecule': np.std(np.sum(fingerprints, axis=1)),
            'min_bits_per_molecule': np.min(np.sum(fingerprints, axis=1)),
            'max_bits_per_molecule': np.max(np.sum(fingerprints, axis=1)),
            'bit_frequency': np.sum(fingerprints, axis=0),
            'sparsity': 1 - np.mean(fingerprints),
            'most_common_bits': np.argsort(np.sum(fingerprints, axis=0))[-10:][::-1],
            'least_common_bits': np.argsort(np.sum(fingerprints, axis=0))[:10]
        }
        
        logger.info(f"Fingerprint feature exploration:")
        logger.info(f"  Dataset: {stats['n_molecules']} molecules, {stats['n_bits']} bits")
        logger.info(f"  Average bits per molecule: {stats['mean_bits_per_molecule']:.1f} ± {stats['std_bits_per_molecule']:.1f}")
        logger.info(f"  Sparsity: {stats['sparsity']:.3f}")
        
        return stats
    
    def get_bit_info(self, mol, bit_number: int) -> dict:
        """
        Get information about what substructure a specific bit represents.
        
        Parameters:
            mol: RDKit molecule object
            bit_number (int): Bit number to analyze
            
        Returns:
            dict: Information about the bit
        """
        if not RDKIT_AVAILABLE:
            return {}
        
        try:
            # Generate fingerprint with additional output
            ao = AdditionalOutput()
            ao.AllocateBitInfoMap()
            fp = self.morgan_generator.GetFingerprint(mol, additionalOutput=ao)
            
            bit_info_map = ao.GetBitInfoMap()
            
            if bit_number in bit_info_map:
                return {
                    'bit_number': bit_number,
                    'environments': bit_info_map[bit_number],
                    'molecule': mol
                }
            else:
                return {'bit_number': bit_number, 'found': False}
                
        except Exception as e:
            logger.warning(f"Error getting bit info: {e}")
            return {}
    
    def draw_fragment_from_bit(self, mol, bit_number: int):
        """
        Draw the molecular fragment corresponding to a specific fingerprint bit.
        
        Parameters:
            mol: RDKit molecule object
            bit_number (int): Bit number to visualize
            
        Returns:
            SVG string or None
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - cannot draw fragments")
            return None
        
        try:
            from rdkit.Chem import Draw
            
            ao = AdditionalOutput()
            ao.AllocateBitInfoMap()
            fp = self.morgan_generator.GetFingerprint(mol, additionalOutput=ao)
            
            svg = Draw.DrawMorganBit(mol, bit_number, ao.GetBitInfoMap(), useSVG=True)
            return svg
            
        except Exception as e:
            logger.warning(f"Error drawing fragment for bit {bit_number}: {e}")
            return None
    
    def get_examples_for_bit(self, bit_number: int, molecules: List, fingerprints: np.ndarray) -> List:
        """
        Get visual examples of molecules that have a specific bit set.
        
        Parameters:
            bit_number (int): Bit number to find examples for
            molecules (List): List of RDKit molecule objects
            fingerprints (numpy.ndarray): Array of fingerprints
            
        Returns:
            List: List of SVG visualizations
        """
        if not RDKIT_AVAILABLE:
            return []
        
        # Find molecules with this bit set
        molecules_with_bit = np.where(fingerprints[:, bit_number] == 1)[0]
        
        if len(molecules_with_bit) == 0:
            logger.warning(f"No molecules found with bit {bit_number} set")
            return []
        
        examples = []
        for idx in molecules_with_bit[:5]:  # Limit to first 5 examples
            mol = molecules[idx]
            svg = self.draw_fragment_from_bit(mol, bit_number)
            if svg:
                examples.append(svg)
        
        logger.info(f"Found {len(examples)} examples for bit {bit_number}")
        return examples


class SmilesToMols(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer to convert SMILES strings to RDKit molecule objects.
    
    This transformer is used in ML pipelines to standardize the molecular representation step.
    """
    
    def __init__(self, standardize: bool = True):
        """
        Initialize the transformer.
        
        Parameters:
            standardize (bool): Whether to apply standardization to molecules
        """
        self.standardize = standardize
        
        if RDKIT_AVAILABLE and standardize:
            from rdkit.Chem.MolStandardize.rdMolStandardize import (
                Cleanup, LargestFragmentChooser, TautomerEnumerator, Uncharger
            )
            self.cleanup = Cleanup
            self.fragment_chooser = LargestFragmentChooser()
            self.uncharger = Uncharger()
            self.tautomer_enumerator = TautomerEnumerator()
    
    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """
        Transform SMILES strings to RDKit molecule objects.
        
        Parameters:
            X: Array-like of SMILES strings
            
        Returns:
            List of RDKit molecule objects
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - returning SMILES strings")
            return X
        
        molecules = []
        for smiles in X:
            mol = self._process_smiles(smiles)
            molecules.append(mol)
        
        return molecules
    
    def _process_smiles(self, smiles: str):
        """Process a single SMILES string using centralized infrastructure."""
        if not isinstance(smiles, str):
            return None
        
        if not self.standardize:
            # Simple conversion without standardization
            if not RDKIT_AVAILABLE:
                return smiles
            return Chem.MolFromSmiles(smiles)
        
        # Use centralized standardization
        from .data_processing import HERGDataProcessor
        processor = HERGDataProcessor()
        return processor.process_smiles(smiles)


class FingerprintFeaturizer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer to convert molecules to fingerprint features.
    
    This transformer is used in ML pipelines for the feature extraction step.
    """
    
    def __init__(self, radius: int = 2, n_bits: int = 2048):
        """
        Initialize the featurizer.
        
        Parameters:
            radius (int): Morgan fingerprint radius
            n_bits (int): Number of bits in fingerprint
        """
        self.radius = radius
        self.n_bits = n_bits
        self.featurizer = MolecularFeaturizer(radius=radius, n_bits=n_bits)
    
    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """
        Transform molecules to fingerprint features.
        
        Parameters:
            X: List of RDKit molecule objects or SMILES strings
            
        Returns:
            numpy.ndarray: Array of fingerprints
        """
        if isinstance(X, (list, np.ndarray)):
            return self.featurizer.compute_fingerprints_batch(X, self.radius, self.n_bits)
        else:
            # Single molecule
            return self.featurizer.compute_fingerprint(X, self.radius, self.n_bits).reshape(1, -1) 