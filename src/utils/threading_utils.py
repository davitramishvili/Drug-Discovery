"""
Threading utilities for molecular data processing.

This module consolidates common threading patterns used throughout the project
to eliminate code duplication and provide consistent threading behavior.
"""

import time
import threading
from typing import List, Callable, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import numpy as np
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Thread-safe progress tracking for batch operations."""
    
    def __init__(self, total_items: int = 0, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
            description: Description of the operation
        """
        self.total_items = total_items
        self.processed_items = 0
        self.description = description
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def update(self, increment: int = 1) -> None:
        """
        Update progress (thread-safe).
        
        Args:
            increment: Number of items to add to progress
        """
        with self.lock:
            self.processed_items += increment
            if self.total_items > 0:
                percentage = (self.processed_items / self.total_items) * 100
                elapsed_time = time.time() - self.start_time
                items_per_sec = self.processed_items / elapsed_time if elapsed_time > 0 else 0
                
                print(f"\r   ⚡ {self.description}: {self.processed_items}/{self.total_items} "
                      f"({percentage:.1f}%) - {items_per_sec:.1f} items/sec", 
                      end="", flush=True)
    
    def complete(self) -> None:
        """Mark processing as complete."""
        print()  # New line after progress
        elapsed_time = time.time() - self.start_time
        items_per_sec = self.processed_items / elapsed_time if elapsed_time > 0 else 0
        logger.info(f"✅ {self.description} completed: {self.processed_items} items "
                   f"in {elapsed_time:.2f}s ({items_per_sec:.1f} items/sec)")


class ThreadedBatchProcessor:
    """Generic threaded batch processor for molecular data operations."""
    
    def __init__(self, n_threads: Optional[int] = None):
        """
        Initialize batch processor.
        
        Args:
            n_threads: Number of threads to use (None for CPU count)
        """
        import os
        self.n_threads = n_threads or os.cpu_count()
        self.progress_tracker: Optional[ProgressTracker] = None
    
    def process_batches(self, 
                       data: List[Any], 
                       batch_processor: Callable[[List[Any], int], Tuple[int, List[Any]]], 
                       description: str = "Processing") -> List[Any]:
        """
        Process data in parallel batches.
        
        Args:
            data: List of items to process
            batch_processor: Function to process a batch (batch_data, batch_id) -> (batch_id, results)
            description: Description for progress tracking
            
        Returns:
            List of processed results in original order
        """
        if not data:
            return []
        
        # Initialize progress tracking
        self.progress_tracker = ProgressTracker(len(data), description)
        
        # Split data into batches
        batch_size = max(1, len(data) // self.n_threads)
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        
        logger.info(f"🔄 Processing {len(data)} items in {len(batches)} batches using {self.n_threads} threads")
        
        # Initialize results array
        results = [None] * len(data)
        
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(batch_processor, batch, i): i 
                for i, batch in enumerate(batches)
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                try:
                    batch_id, batch_results = future.result()
                    
                    # Insert batch results in correct position
                    start_idx = batch_id * batch_size
                    for i, result in enumerate(batch_results):
                        if start_idx + i < len(results):
                            results[start_idx + i] = result
                            
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    # Fill with None for failed batch
                    batch_id = future_to_batch[future]
                    start_idx = batch_id * batch_size
                    batch_size_actual = min(batch_size, len(data) - start_idx)
                    for i in range(batch_size_actual):
                        if start_idx + i < len(results):
                            results[start_idx + i] = None
        
        self.progress_tracker.complete()
        return results
    
    def process_batches_with_callback(self, 
                                    data: List[Any], 
                                    batch_processor: Callable[[List[Any], int, ProgressTracker], Tuple[int, List[Any]]], 
                                    description: str = "Processing") -> List[Any]:
        """
        Process data in parallel batches with progress callback.
        
        Args:
            data: List of items to process
            batch_processor: Function to process a batch with progress callback
            description: Description for progress tracking
            
        Returns:
            List of processed results in original order
        """
        if not data:
            return []
        
        # Initialize progress tracking
        self.progress_tracker = ProgressTracker(len(data), description)
        
        # Split data into batches
        batch_size = max(1, len(data) // self.n_threads)
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        
        logger.info(f"🔄 Processing {len(data)} items in {len(batches)} batches using {self.n_threads} threads")
        
        # Initialize results array
        results = [None] * len(data)
        
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(batch_processor, batch, i, self.progress_tracker): i 
                for i, batch in enumerate(batches)
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                try:
                    batch_id, batch_results = future.result()
                    
                    # Insert batch results in correct position
                    start_idx = batch_id * batch_size
                    for i, result in enumerate(batch_results):
                        if start_idx + i < len(results):
                            results[start_idx + i] = result
                            
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    # Fill with None for failed batch
                    batch_id = future_to_batch[future]
                    start_idx = batch_id * batch_size
                    batch_size_actual = min(batch_size, len(data) - start_idx)
                    for i in range(batch_size_actual):
                        if start_idx + i < len(results):
                            results[start_idx + i] = None
        
        self.progress_tracker.complete()
        return results


def create_batches(data: List[Any], n_batches: int) -> List[List[Any]]:
    """
    Split data into approximately equal batches.
    
    Args:
        data: List of items to split
        n_batches: Number of batches to create
        
    Returns:
        List of batches
    """
    if not data or n_batches <= 0:
        return []
    
    batch_size = max(1, len(data) // n_batches)
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def run_threaded_operation(operation: Callable, 
                          data_batches: List[List[Any]], 
                          n_threads: Optional[int] = None,
                          description: str = "Operation") -> List[Any]:
    """
    Run a threaded operation on data batches.
    
    Args:
        operation: Function to apply to each batch
        data_batches: List of data batches
        n_threads: Number of threads (None for CPU count)
        description: Description for logging
        
    Returns:
        List of results from each batch
    """
    import os
    n_threads = n_threads or os.cpu_count()
    
    logger.info(f"🔄 Running {description} with {n_threads} threads on {len(data_batches)} batches")
    
    results = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(operation, batch) for batch in data_batches]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error in threaded operation: {e}")
                results.append(None)
    
    logger.info(f"✅ {description} completed")
    return results


def compute_similarities_batch(batch_lib_fps: List, ref_fps: List) -> List[float]:
    """
    Compute maximum Tanimoto similarities for a batch of library fingerprints.
    
    This function is optimized for Chapter 2 similarity workflows and handles
    numpy-based fingerprint arrays efficiently.
    
    Args:
        batch_lib_fps: List of library fingerprints (numpy arrays)
        ref_fps: List of reference fingerprints (numpy arrays)
    
    Returns:
        List of maximum similarity scores for each library fingerprint
    """
    max_similarities = []
    
    for lib_fp in batch_lib_fps:
        if lib_fp is not None:
            max_sim = 0.0
            for ref_fp in ref_fps:
                if ref_fp is not None:
                    try:
                        # Use numpy-based Tanimoto calculation for numpy arrays
                        similarity = compute_tanimoto_similarity(lib_fp, ref_fp)
                        max_sim = max(max_sim, similarity)
                    except Exception:
                        continue
            max_similarities.append(max_sim)
        else:
            max_similarities.append(0.0)
    
    return max_similarities


def compute_tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Compute Tanimoto similarity between two binary fingerprints.
    
    Optimized numpy implementation that's used across all similarity calculations.
    
    Args:
        fp1: First fingerprint as numpy array
        fp2: Second fingerprint as numpy array
    
    Returns:
        Tanimoto similarity coefficient (0.0 to 1.0)
    """
    try:
        intersection = np.logical_and(fp1, fp2).sum()
        union = np.logical_or(fp1, fp2).sum()
        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0


def threaded_similarity_search(lib_fingerprints: List, ref_fingerprints: List, 
                             n_threads: int = None, description: str = "similarities") -> List[float]:
    """
    Perform threaded similarity search between library and reference fingerprints.
    
    Centralized function that eliminates duplicate similarity search patterns
    across Chapter 2 workflows.
    
    Args:
        lib_fingerprints: List of library fingerprints
        ref_fingerprints: List of reference fingerprints  
        n_threads: Number of threads (auto-detected if None)
        description: Progress description
    
    Returns:
        List of maximum similarity scores for each library fingerprint
    """
    import time
    
    if n_threads is None:
        n_threads = min(8, cpu_count())
    
    print(f"\n🔍 Computing {description} with {n_threads} threads...")
    
    # Filter out None fingerprints from references
    valid_ref_fps = [fp for fp in ref_fingerprints if fp is not None]
    print(f"   Using {len(valid_ref_fps)} valid reference fingerprints")
    
    if not valid_ref_fps:
        print("   ❌ No valid reference fingerprints!")
        return [0.0] * len(lib_fingerprints)
    
    processor = ThreadedBatchProcessor(n_threads)
    
    def similarity_batch_processor(batch_lib_fps, batch_id):
        """Process a batch of library fingerprints for similarity computation."""
        batch_similarities = compute_similarities_batch(batch_lib_fps, valid_ref_fps)
        return batch_id, batch_similarities
    
    similarities = processor.process_batches(lib_fingerprints, similarity_batch_processor, description)
    print(f"   ✅ Computed similarities for {len(similarities)} compounds")
    return similarities 