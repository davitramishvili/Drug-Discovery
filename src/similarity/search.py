"""
Module for performing similarity-based virtual screening.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .fingerprints import FingerprintGenerator, FingerprintSimilarity

logger = logging.getLogger(__name__)

class SimilaritySearcher:
    """Class for performing similarity-based virtual screening with multi-threading support."""
    
    def __init__(self, fingerprint_type: str = "morgan", 
                 similarity_metric: str = "tanimoto",
                 radius: int = 2, n_bits: int = 2048,
                 n_threads: Optional[int] = None):
        """
        Initialize the SimilaritySearcher.
        
        Args:
            fingerprint_type: Type of fingerprint to use
            similarity_metric: Similarity metric to use
            radius: Radius for Morgan fingerprints
            n_bits: Number of bits for fingerprints
            n_threads: Number of threads to use (default: None, auto-detect)
        """
        self.fp_generator = FingerprintGenerator(
            fingerprint_type=fingerprint_type,
            radius=radius,
            n_bits=n_bits,
            n_threads=n_threads
        )
        self.similarity_calculator = FingerprintSimilarity(metric=similarity_metric)
        
        # Threading configuration
        import os
        self.n_threads = n_threads or min(8, (os.cpu_count() or 1) + 4)
        self._lock = threading.Lock()
        
        self.reference_fingerprints = None
        self.reference_data = None
        
    def load_reference_compounds(self, reference_df: pd.DataFrame, 
                               mol_col: str = 'ROMol') -> None:
        """
        Load reference compounds for similarity searching.
        
        Args:
            reference_df: DataFrame containing reference molecules
            mol_col: Name of the molecule column
        """
        if reference_df.empty:
            raise ValueError("Reference DataFrame is empty")
            
        logger.info(f"Loading {len(reference_df)} reference compounds")
        
        # Generate fingerprints for reference compounds
        self.reference_data = reference_df.copy()
        self.reference_data = self.fp_generator.add_fingerprints_to_dataframe(
            self.reference_data, mol_col
        )
        
        # Extract fingerprints as numpy array
        fingerprints = []
        for fp in self.reference_data['fingerprint']:
            if fp is not None:
                fingerprints.append(fp)
            else:
                # Add zero fingerprint for failed molecules
                if self.fp_generator.fingerprint_type == "maccs":
                    fingerprints.append(np.zeros(167, dtype=np.int8))
                else:
                    fingerprints.append(np.zeros(self.fp_generator.n_bits, dtype=np.int8))
                    
        self.reference_fingerprints = np.array(fingerprints)
        
        logger.info(f"Reference fingerprints generated: {self.reference_fingerprints.shape}")
        
    def _process_library_chunk(self, chunk_data: Tuple[pd.DataFrame, float, int]) -> List[Dict]:
        """
        Process a chunk of library molecules for similarity searching in a thread-safe manner.
        
        Args:
            chunk_data: Tuple containing (chunk_df, threshold, max_results_per_chunk)
            
        Returns:
            List of result dictionaries
        """
        chunk_df, threshold, max_results_per_chunk = chunk_data
        results = []
        
        for lib_idx, lib_row in chunk_df.iterrows():
            lib_fp = lib_row['fingerprint']
            
            if lib_fp is None:
                continue
                
            # Calculate similarity to all reference compounds
            max_similarity = 0.0
            best_ref_idx = -1
            
            for ref_idx, ref_fp in enumerate(self.reference_fingerprints):
                similarity = self.similarity_calculator.calculate_similarity(lib_fp, ref_fp)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_ref_idx = ref_idx
                    
            # Check if similarity meets threshold
            if max_similarity >= threshold:
                result_row = lib_row.copy().to_dict()
                result_row['similarity'] = max_similarity
                result_row['best_reference_idx'] = best_ref_idx
                
                if self.reference_data is not None and 'ID' in self.reference_data.columns:
                    result_row['best_reference_id'] = self.reference_data.iloc[best_ref_idx]['ID']
                
                results.append(result_row)
                
                # Limit results per chunk to avoid memory issues
                if len(results) >= max_results_per_chunk:
                    break
                
        return results

    def search_library_threaded(self, library_df: pd.DataFrame,
                              threshold: float = 0.7,
                              max_results: int = 1000,
                              mol_col: str = 'ROMol',
                              chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Search a compound library for molecules similar to reference compounds using multi-threading.
        
        Args:
            library_df: DataFrame containing library molecules
            threshold: Minimum similarity threshold
            max_results: Maximum number of results to return
            mol_col: Name of the molecule column
            chunk_size: Size of chunks for threading (default: auto-calculate)
            
        Returns:
            DataFrame containing similar molecules with similarity scores
        """
        if self.reference_fingerprints is None:
            raise ValueError("No reference compounds loaded. Call load_reference_compounds first.")
            
        if library_df.empty:
            logger.warning("Empty library DataFrame provided")
            return pd.DataFrame()
            
        logger.info(f"Searching library of {len(library_df)} molecules using {self.n_threads} threads")
        
        # Generate fingerprints for library molecules
        library_with_fps = self.fp_generator.add_fingerprints_to_dataframe(
            library_df.copy(), mol_col, use_threading=True
        )
        
        # Calculate chunk size
        if chunk_size is None:
            chunk_size = max(1, len(library_with_fps) // (self.n_threads * 2))
        
        # Split library into chunks
        chunks = []
        max_results_per_chunk = max_results // self.n_threads + 100  # Some buffer
        
        for i in range(0, len(library_with_fps), chunk_size):
            chunk_df = library_with_fps.iloc[i:i+chunk_size]
            chunks.append((chunk_df, threshold, max_results_per_chunk))
        
        # Process chunks in parallel
        all_results = []
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            future_to_chunk = {
                executor.submit(self._process_library_chunk, chunk_data): i 
                for i, chunk_data in enumerate(chunks)
            }
            
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Error processing similarity search chunk {chunk_idx}: {e}")
        
        if not all_results:
            logger.info("No molecules found above similarity threshold")
            return pd.DataFrame()
            
        # Convert to DataFrame and sort by similarity
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('similarity', ascending=False)
        
        # Limit results
        if len(results_df) > max_results:
            results_df = results_df.head(max_results)
            
        logger.info(f"Found {len(results_df)} similar molecules above threshold {threshold}")
        
        return results_df
        
    def search_library(self, library_df: pd.DataFrame,
                      threshold: float = 0.7,
                      max_results: int = 1000,
                      mol_col: str = 'ROMol',
                      use_threading: bool = True) -> pd.DataFrame:
        """
        Search a compound library for molecules similar to reference compounds.
        
        Args:
            library_df: DataFrame containing library molecules
            threshold: Minimum similarity threshold
            max_results: Maximum number of results to return
            mol_col: Name of the molecule column
            use_threading: Whether to use multi-threading
            
        Returns:
            DataFrame containing similar molecules with similarity scores
        """
        if use_threading and len(library_df) > 100:  # Use threading for larger datasets
            return self.search_library_threaded(
                library_df, threshold, max_results, mol_col
            )
        else:
            return self._search_library_single_threaded(
                library_df, threshold, max_results, mol_col
            )

    def _search_library_single_threaded(self, library_df: pd.DataFrame,
                                      threshold: float = 0.7,
                                      max_results: int = 1000,
                                      mol_col: str = 'ROMol') -> pd.DataFrame:
        """
        Search a compound library for molecules similar to reference compounds (original single-threaded implementation).
        
        Args:
            library_df: DataFrame containing library molecules
            threshold: Minimum similarity threshold
            max_results: Maximum number of results to return
            mol_col: Name of the molecule column
            
        Returns:
            DataFrame containing similar molecules with similarity scores
        """
        if self.reference_fingerprints is None:
            raise ValueError("No reference compounds loaded. Call load_reference_compounds first.")
            
        if library_df.empty:
            logger.warning("Empty library DataFrame provided")
            return pd.DataFrame()
            
        logger.info(f"Searching library of {len(library_df)} molecules (single-threaded)")
        
        # Generate fingerprints for library molecules
        library_with_fps = self.fp_generator.add_fingerprints_to_dataframe(
            library_df.copy(), mol_col, use_threading=False
        )
        
        # Calculate similarities
        results = []
        
        for lib_idx, lib_row in library_with_fps.iterrows():
            lib_fp = lib_row['fingerprint']
            
            if lib_fp is None:
                continue
                
            # Calculate similarity to all reference compounds
            max_similarity = 0.0
            best_ref_idx = -1
            
            for ref_idx, ref_fp in enumerate(self.reference_fingerprints):
                similarity = self.similarity_calculator.calculate_similarity(lib_fp, ref_fp)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_ref_idx = ref_idx
                    
            # Check if similarity meets threshold
            if max_similarity >= threshold:
                result_row = lib_row.copy()
                result_row['similarity'] = max_similarity
                result_row['best_reference_idx'] = best_ref_idx
                
                if self.reference_data is not None and 'ID' in self.reference_data.columns:
                    result_row['best_reference_id'] = self.reference_data.iloc[best_ref_idx]['ID']
                
                results.append(result_row)
                
        if not results:
            logger.info("No molecules found above similarity threshold")
            return pd.DataFrame()
            
        # Convert to DataFrame and sort by similarity
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('similarity', ascending=False)
        
        # Limit results
        if len(results_df) > max_results:
            results_df = results_df.head(max_results)
            
        logger.info(f"Found {len(results_df)} similar molecules above threshold {threshold}")
        
        return results_df
        
    def get_top_similar_molecules(self, query_mol: Chem.Mol, 
                                library_df: pd.DataFrame,
                                n_results: int = 10,
                                mol_col: str = 'ROMol') -> pd.DataFrame:
        """
        Find the most similar molecules to a single query molecule.
        
        Args:
            query_mol: Query molecule
            library_df: DataFrame containing library molecules
            n_results: Number of top results to return
            mol_col: Name of the molecule column
            
        Returns:
            DataFrame containing top similar molecules
        """
        if library_df.empty:
            return pd.DataFrame()
            
        # Generate fingerprint for query molecule
        query_fp = self.fp_generator.generate_fingerprint(query_mol)
        if query_fp is None:
            logger.warning("Failed to generate fingerprint for query molecule")
            return pd.DataFrame()
            
        # Generate fingerprints for library
        library_with_fps = self.fp_generator.add_fingerprints_to_dataframe(
            library_df.copy(), mol_col
        )
        
        # Calculate similarities
        similarities = []
        for idx, row in library_with_fps.iterrows():
            lib_fp = row['fingerprint']
            if lib_fp is not None:
                sim = self.similarity_calculator.calculate_similarity(query_fp, lib_fp)
                similarities.append((idx, sim))
            else:
                similarities.append((idx, 0.0))
                
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, sim in similarities[:n_results]]
        
        # Create results DataFrame
        results_df = library_with_fps.loc[top_indices].copy()
        results_df['similarity'] = [sim for idx, sim in similarities[:n_results]]
        
        return results_df
        
    def _process_similarity_matrix_chunk(self, chunk_data: Tuple[np.ndarray, int, int]) -> np.ndarray:
        """
        Process a chunk of the similarity matrix calculation in a thread-safe manner.
        
        Args:
            chunk_data: Tuple containing (fingerprints, start_idx, end_idx)
            
        Returns:
            Partial similarity matrix chunk
        """
        fingerprints, start_idx, end_idx = chunk_data
        n_molecules = len(fingerprints)
        chunk_matrix = np.zeros((end_idx - start_idx, n_molecules))
        
        for i in range(start_idx, end_idx):
            for j in range(n_molecules):
                if i <= j:  # Only calculate upper triangle
                    similarity = self.similarity_calculator.calculate_similarity(
                        fingerprints[i], fingerprints[j]
                    )
                    chunk_matrix[i - start_idx, j] = similarity
        
        return chunk_matrix, start_idx, end_idx
        
    def calculate_diversity_matrix(self, molecules_df: pd.DataFrame,
                                 mol_col: str = 'ROMol',
                                 use_threading: bool = True) -> np.ndarray:
        """
        Calculate diversity matrix for a set of molecules.
        
        Args:
            molecules_df: DataFrame containing molecules
            mol_col: Name of the molecule column
            use_threading: Whether to use multi-threading
            
        Returns:
            Similarity matrix
        """
        if molecules_df.empty:
            return np.array([])
            
        # Generate fingerprints
        molecules_with_fps = self.fp_generator.add_fingerprints_to_dataframe(
            molecules_df.copy(), mol_col
        )
        
        # Extract fingerprints
        fingerprints = []
        for fp in molecules_with_fps['fingerprint']:
            if fp is not None:
                fingerprints.append(fp)
            else:
                if self.fp_generator.fingerprint_type == "maccs":
                    fingerprints.append(np.zeros(167, dtype=np.int8))
                else:
                    fingerprints.append(np.zeros(self.fp_generator.n_bits, dtype=np.int8))
                    
        fingerprints_array = np.array(fingerprints)
        
        # Calculate similarity matrix
        if use_threading:
            # Split the matrix calculation into chunks
            chunk_size = max(1, len(fingerprints_array) // (self.n_threads * 2))
            chunks = [(fingerprints_array[i:i+chunk_size], i, i+chunk_size) for i in range(0, len(fingerprints_array), chunk_size)]
            
            # Process chunks in parallel
            results = []
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                future_to_chunk = {
                    executor.submit(self._process_similarity_matrix_chunk, chunk_data): i 
                    for i, chunk_data in enumerate(chunks)
                }
                
                for future in as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        chunk_results = future.result()
                        results.append(chunk_results)
                    except Exception as e:
                        logger.error(f"Error processing similarity matrix chunk {chunk_idx}: {e}")
            
            # Combine results
            similarity_matrix = np.zeros((len(fingerprints_array), len(fingerprints_array)))
            for chunk_matrix, start_idx, end_idx in results:
                similarity_matrix[start_idx:end_idx, start_idx:end_idx] = chunk_matrix
        else:
        similarity_matrix = self.similarity_calculator.calculate_similarity_matrix(
            fingerprints_array
        )
        
        return similarity_matrix
        
    def get_search_statistics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about similarity search results.
        
        Args:
            results_df: DataFrame containing search results
            
        Returns:
            Dictionary containing search statistics
        """
        if results_df.empty:
            return {'total_hits': 0}
            
        stats = {
            'total_hits': len(results_df),
            'mean_similarity': results_df['similarity'].mean(),
            'max_similarity': results_df['similarity'].max(),
            'min_similarity': results_df['similarity'].min(),
            'std_similarity': results_df['similarity'].std(),
        }
        
        # Similarity distribution
        similarity_ranges = [
            (0.9, 1.0, 'very_high'),
            (0.8, 0.9, 'high'),
            (0.7, 0.8, 'medium'),
            (0.6, 0.7, 'low'),
            (0.0, 0.6, 'very_low')
        ]
        
        for min_sim, max_sim, label in similarity_ranges:
            count = len(results_df[
                (results_df['similarity'] >= min_sim) & 
                (results_df['similarity'] < max_sim)
            ])
            stats[f'{label}_similarity_count'] = count
            stats[f'{label}_similarity_percentage'] = count / len(results_df) * 100
            
        return stats 