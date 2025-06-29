"""
Module for generating molecular fingerprints for similarity searching.

Morgan fingerprints are circular fingerprints that capture molecular features at different radii:
- radius=0: Only considers the atom itself
- radius=1: Considers atoms 1 bond away
- radius=2 (default): Considers atoms up to 2 bonds away (recommended for most cases)
- radius=3: Considers atoms up to 3 bonds away (captures more extended features)

The number of bits affects the fingerprint resolution:
- n_bits=256: Very compact but higher collision risk
- n_bits=512: Good balance for small molecules
- n_bits=1024: Standard for drug-like molecules
- n_bits=2048 (default): Recommended for most cases
- n_bits=4096: Higher resolution, better for large diverse libraries
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem
from rdkit import DataStructs
from typing import List, Optional, Union, Dict, Any, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

class FingerprintGenerator:
    """Class for generating various types of molecular fingerprints with multi-threading support."""
    
    def __init__(self, fingerprint_type: str = "morgan", 
                 radius: int = 2, n_bits: int = 2048,
                 use_features: bool = False,
                 include_chirality: bool = False,
                 n_threads: Optional[int] = None):
        """
        Initialize the FingerprintGenerator.
        
        Args:
            fingerprint_type: Type of fingerprint ("morgan", "rdkit", "maccs")
            radius: Radius for Morgan fingerprints (default: 2)
                   - 0: Only atom types
                   - 1: Atom types + immediate neighbors
                   - 2: Extended environment (recommended)
                   - 3: Larger environment
            n_bits: Number of bits for fingerprints (default: 2048)
                   - 256: Very compact, higher collision risk
                   - 512: Good for small molecules
                   - 1024: Standard for drug-like molecules
                   - 2048: Recommended default
                   - 4096: Higher resolution
            use_features: Whether to use chemical features for Morgan FPs
                        instead of connectivity (default: False)
            include_chirality: Whether to include chirality in Morgan FPs
                             (default: False)
            n_threads: Number of threads to use (default: None, auto-detect)
        """
        self.fingerprint_type = fingerprint_type.lower()
        self.radius = radius
        self.n_bits = n_bits
        self.use_features = use_features
        self.include_chirality = include_chirality
        
        # Threading configuration
        import os
        self.n_threads = n_threads or min(8, (os.cpu_count() or 1) + 4)
        self._lock = threading.Lock()
        
        # Validate fingerprint type
        valid_types = ["morgan", "rdkit", "maccs"]
        if self.fingerprint_type not in valid_types:
            raise ValueError(f"Fingerprint type must be one of {valid_types}")
            
        # Validate radius
        if self.radius < 0:
            raise ValueError("Radius must be non-negative")
            
        # Validate n_bits
        valid_n_bits = [256, 512, 1024, 2048, 4096]
        if self.n_bits not in valid_n_bits:
            logger.warning(f"Unusual number of bits: {n_bits}. Common values are {valid_n_bits}")
            
    def generate_fingerprint(self, mol: Chem.Mol) -> Optional[np.ndarray]:
        """
        Generate a fingerprint for a single molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Numpy array representing the fingerprint, or None if generation fails
        """
        if mol is None:
            return None
            
        try:
            if self.fingerprint_type == "morgan":
                # Use Morgan features if specified
                if self.use_features:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                        mol, self.radius, nBits=self.n_bits,
                        useFeatures=True, useChirality=self.include_chirality
                    )
                else:
                    # Use the newer MorganGenerator to avoid deprecation warnings
                    try:
                        from rdkit.Chem.rdMolDescriptors import GetMorganGenerator
                        generator = GetMorganGenerator(
                            radius=self.radius, 
                            fpSize=self.n_bits,
                            includeChirality=self.include_chirality
                        )
                        fp = generator.GetFingerprint(mol)
                    except ImportError:
                        # Fallback to older method if newer API not available
                        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                            mol, self.radius, nBits=self.n_bits,
                            useChirality=self.include_chirality
                        )
            elif self.fingerprint_type == "rdkit":
                fp = Chem.RDKFingerprint(mol, fpSize=self.n_bits)
            elif self.fingerprint_type == "maccs":
                fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            else:
                raise ValueError(f"Unknown fingerprint type: {self.fingerprint_type}")
                
            # Convert to numpy array
            fp_array = np.zeros((len(fp),), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, fp_array)
            
            return fp_array
            
        except Exception as e:
            logger.warning(f"Failed to generate fingerprint: {str(e)}")
            return None
            
    def get_bit_info(self, mol: Chem.Mol) -> Dict[int, List[Tuple[int, ...]]]:
        """
        Get information about which molecular features set specific bits.
        Only available for Morgan fingerprints.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary mapping bit positions to lists of atom environments
        """
        if self.fingerprint_type != "morgan":
            raise ValueError("Bit info is only available for Morgan fingerprints")
            
        if mol is None:
            return {}
            
        # Initialize bit information dictionary
        bit_info = {}
        
        # Generate Morgan fingerprint with bit information
        _ = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, self.radius, nBits=self.n_bits,
            useFeatures=self.use_features,
            useChirality=self.include_chirality,
            bitInfo=bit_info
        )
        
        return bit_info
            
    def _process_molecule_chunk(self, molecules: List[Chem.Mol]) -> List[Optional[np.ndarray]]:
        """
        Process a chunk of molecules for fingerprint generation in a thread-safe manner.
        
        Args:
            molecules: List of RDKit molecule objects
            
        Returns:
            List of fingerprints (or None for failed molecules)
        """
        fingerprints = []
        for mol in molecules:
            fp = self.generate_fingerprint(mol)
            fingerprints.append(fp)
        return fingerprints
    
    def generate_fingerprints_batch_threaded(self, molecules: List[Chem.Mol],
                                           chunk_size: Optional[int] = None) -> np.ndarray:
        """
        Generate fingerprints for a batch of molecules using multi-threading.
        
        Args:
            molecules: List of RDKit molecule objects
            chunk_size: Size of chunks for threading (default: auto-calculate)
            
        Returns:
            2D numpy array where each row is a fingerprint
        """
        if not molecules:
            return np.array([])
            
        logger.info(f"Generating fingerprints for {len(molecules)} molecules using {self.n_threads} threads")
        
        # Calculate chunk size
        if chunk_size is None:
            chunk_size = max(1, len(molecules) // (self.n_threads * 2))
        
        # Split molecules into chunks
        chunks = [molecules[i:i+chunk_size] for i in range(0, len(molecules), chunk_size)]
        
        # Process chunks in parallel
        all_fingerprints = []
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            future_to_chunk = {
                executor.submit(self._process_molecule_chunk, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            chunk_results = [None] * len(chunks)
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results[chunk_idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing fingerprint chunk {chunk_idx}: {e}")
                    chunk_results[chunk_idx] = []
        
        # Flatten results
        for chunk_fps in chunk_results:
            if chunk_fps:
                all_fingerprints.extend(chunk_fps)
        
        # Convert to proper format
        final_fingerprints = []
        for fp in all_fingerprints:
            if fp is not None:
                final_fingerprints.append(fp)
            else:
                # Add zero fingerprint for failed molecules
                if self.fingerprint_type == "maccs":
                    final_fingerprints.append(np.zeros(167, dtype=np.int8))
                else:
                    final_fingerprints.append(np.zeros(self.n_bits, dtype=np.int8))
        
        if not final_fingerprints:
            return np.array([])
            
        return np.array(final_fingerprints)
        
    def add_fingerprints_to_dataframe(self, df: pd.DataFrame, 
                                    mol_col: str = 'ROMol',
                                    use_threading: bool = True) -> pd.DataFrame:
        """
        Add fingerprints to a DataFrame containing molecules.
        
        Args:
            df: DataFrame containing molecules
            mol_col: Name of the molecule column
            use_threading: Whether to use multi-threading for fingerprint generation
            
        Returns:
            DataFrame with added fingerprint column
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for fingerprint generation")
            return df
            
        logger.info(f"Generating {self.fingerprint_type} fingerprints for {len(df)} molecules")
        
        if use_threading and len(df) > 50:  # Use threading for larger datasets
            fingerprints = self.generate_fingerprints_batch_threaded(df[mol_col].tolist())
        else:
            fingerprints = self.generate_fingerprints_batch(df[mol_col].tolist())
        
        # Add fingerprints to DataFrame
        df = df.copy()
        df['fingerprint'] = list(fingerprints) if len(fingerprints) > 0 else [None] * len(df)
        
        return df
        
    def generate_fingerprints_batch(self, molecules: List[Chem.Mol]) -> np.ndarray:
        """
        Generate fingerprints for a batch of molecules (single-threaded version).
        
        Args:
            molecules: List of RDKit molecule objects
            
        Returns:
            2D numpy array where each row is a fingerprint
        """
        fingerprints = []
        
        for mol in molecules:
            fp = self.generate_fingerprint(mol)
            if fp is not None:
                fingerprints.append(fp)
            else:
                # Add zero fingerprint for failed molecules
                if self.fingerprint_type == "maccs":
                    fingerprints.append(np.zeros(167, dtype=np.int8))
                else:
                    fingerprints.append(np.zeros(self.n_bits, dtype=np.int8))
                    
        if not fingerprints:
            return np.array([])
            
        return np.array(fingerprints)
        
    def get_fingerprint_info(self) -> Dict[str, Any]:
        """
        Get information about the fingerprint configuration.
        
        Returns:
            Dictionary containing fingerprint information
        """
        info = {
            'type': self.fingerprint_type,
            'radius': self.radius if self.fingerprint_type == 'morgan' else None,
            'n_bits': self.n_bits if self.fingerprint_type != 'maccs' else 167,
            'description': self._get_fingerprint_description()
        }
        
        return info
        
    def _get_fingerprint_description(self) -> str:
        """Get a description of the fingerprint type."""
        descriptions = {
            'morgan': f"Morgan (circular) fingerprints with radius {self.radius} and {self.n_bits} bits",
            'rdkit': f"RDKit path-based fingerprints with {self.n_bits} bits",
            'maccs': "MACCS keys (166 structural keys)"
        }
        
        return descriptions.get(self.fingerprint_type, "Unknown fingerprint type")


class FingerprintSimilarity:
    """Class for calculating similarity between molecular fingerprints."""
    
    def __init__(self, metric: str = "tanimoto"):
        """
        Initialize the FingerprintSimilarity calculator.
        
        Args:
            metric: Similarity metric ("tanimoto", "dice", "cosine", "jaccard")
        """
        self.metric = metric.lower()
        
        valid_metrics = ["tanimoto", "dice", "cosine", "jaccard"]
        if self.metric not in valid_metrics:
            raise ValueError(f"Similarity metric must be one of {valid_metrics}")
            
    def calculate_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """
        Calculate similarity between two fingerprints.
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
            
        Returns:
            Similarity score between 0 and 1
        """
        if fp1 is None or fp2 is None:
            return 0.0
            
        if len(fp1) != len(fp2):
            raise ValueError("Fingerprints must have the same length")
            
        # Convert to boolean arrays for bit-based metrics
        fp1_bool = fp1.astype(bool)
        fp2_bool = fp2.astype(bool)
        
        if self.metric == "tanimoto":
            return self._tanimoto_similarity(fp1_bool, fp2_bool)
        elif self.metric == "dice":
            return self._dice_similarity(fp1_bool, fp2_bool)
        elif self.metric == "cosine":
            return self._cosine_similarity(fp1, fp2)
        elif self.metric == "jaccard":
            return self._jaccard_similarity(fp1_bool, fp2_bool)
        else:
            raise ValueError(f"Unknown similarity metric: {self.metric}")
            
    def _tanimoto_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Tanimoto similarity."""
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
            
        return intersection / union
        
    def _dice_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Dice similarity."""
        intersection = np.sum(fp1 & fp2)
        total_bits = np.sum(fp1) + np.sum(fp2)
        
        if total_bits == 0:
            return 1.0 if intersection == 0 else 0.0
            
        return 2 * intersection / total_bits
        
    def _cosine_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot_product = np.dot(fp1, fp2)
        norm1 = np.linalg.norm(fp1)
        norm2 = np.linalg.norm(fp2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0 if dot_product == 0 else 0.0
            
        return dot_product / (norm1 * norm2)
        
    def _jaccard_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Jaccard similarity (same as Tanimoto for binary data)."""
        return self._tanimoto_similarity(fp1, fp2)
        
    def calculate_similarity_matrix(self, fingerprints: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise similarity matrix for a set of fingerprints.
        
        Args:
            fingerprints: 2D array where each row is a fingerprint
            
        Returns:
            Symmetric similarity matrix
        """
        n_molecules = fingerprints.shape[0]
        similarity_matrix = np.zeros((n_molecules, n_molecules))
        
        for i in range(n_molecules):
            for j in range(i, n_molecules):
                sim = self.calculate_similarity(fingerprints[i], fingerprints[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
                
        return similarity_matrix 