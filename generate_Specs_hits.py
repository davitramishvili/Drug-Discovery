#!/usr/bin/env python3
"""
Generate Specs Hits - Production Version
========================================

This script generates the top 1000 hits from Specs library similarity screening
using optimized multi-threading for parallel processing and saves them for reuse.

PRODUCTION FEATURES:
- Multi-threaded fingerprint computation
- Parallel similarity calculations  
- Progress tracking with threading
- Optimized memory usage
- Enhanced drug-likeness filtering

Usage:
    python generate_Specs_hits.py
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import multiprocessing as mp
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors

# Import pipeline components
from similarity.fingerprints import FingerprintGenerator
from data_processing.loader import MoleculeLoader
from filtering.drug_like import DrugLikeFilter

# Suppress warnings
warnings.filterwarnings('ignore')

class SpecsHitsGenerator:
    """Production-grade multi-threaded generator for Specs library hits."""
    
    def __init__(self, random_seed: int = 42, n_threads: int = None):
        """Initialize the hits generator with threading support."""
        self.random_seed = random_seed
        self.n_threads = n_threads or min(8, mp.cpu_count())
        
        # Output configuration
        self.output_file = "results/threaded_Specs_hits.csv"
        
        # Create output directory
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking - use centralized infrastructure
        self.progress_tracker = None
        
        print(f"🧵 Initialized with {self.n_threads} threads")
    
    def compute_fingerprints_batch(self, molecules: List[Chem.Mol], batch_id: int) -> Tuple[int, List]:
        """Compute fingerprints for a batch of molecules (thread worker)."""
        fingerprints = []
        fp_generator = FingerprintGenerator(fingerprint_type="morgan", radius=2, n_bits=2048)
        
        for mol in molecules:
            if mol is not None:
                try:
                    fp = fp_generator.generate_fingerprint(mol)
                    fingerprints.append(fp)
                except Exception:
                    fingerprints.append(None)
            else:
                fingerprints.append(None)
            
            # Update progress using centralized tracker
            if self.progress_tracker:
                self.progress_tracker.update(1)
        
        return batch_id, fingerprints
    
    def compute_similarities_batch(self, lib_fps: List, ref_fps: List, batch_id: int) -> Tuple[int, List[float]]:
        """Compute similarities for a batch of library fingerprints against all references."""
        max_similarities = []
        
        for lib_fp in lib_fps:
            if lib_fp is not None:
                max_sim = 0.0
                for ref_fp in ref_fps:
                    if ref_fp is not None:
                        try:
                            # Use numpy-based Tanimoto calculation for numpy arrays
                            intersection = np.logical_and(lib_fp, ref_fp).sum()
                            union = np.logical_or(lib_fp, ref_fp).sum()
                            sim = intersection / union if union > 0 else 0.0
                            max_sim = max(max_sim, sim)
                        except Exception:
                            continue
                max_similarities.append(max_sim)
            else:
                max_similarities.append(0.0)
            
            self.progress_tracker.update(1)
        
        return batch_id, max_similarities
    
    def load_and_filter_data(self, max_library_size: int = 15000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and filter molecular data."""
        print("\n📚 Loading molecular data...")
        
        # Load data files
        loader = MoleculeLoader()
        
        specs_path = Path("data/raw/Specs.sdf")
        malaria_path = Path("data/reference/malaria_box_400.sdf")
        
        if not specs_path.exists() or not malaria_path.exists():
            print("❌ Required data files not found")
            return None, None
        
        # Load library (with size limit for performance)
        library_df = loader.load_sdf(str(specs_path))
        if len(library_df) > max_library_size:
            print(f"⚡ Limiting library to first {max_library_size} compounds (from {len(library_df)})")
            library_df = library_df.head(max_library_size)
        
        # Load reference
        reference_df = loader.load_sdf(str(malaria_path))
        
        print(f"✅ Loaded {len(library_df)} library compounds")
        print(f"✅ Loaded {len(reference_df)} reference compounds")
        
        # Apply drug-likeness filters with relaxed criteria
        print("\n🧪 Applying drug-likeness filters...")
        filter_engine = DrugLikeFilter(violations_allowed=2)  # Allow up to 2 Lipinski violations
        
        # The data already has ROMol column from the loader
        # No need to convert column names
        
        # Apply enhanced filtering that allows some violations
        library_filtered = filter_engine.filter_dataframe(
            library_df, 
            apply_lipinski=True,
            apply_additional=False,  # Skip additional filters to be more permissive
            apply_structural_alerts=False,  # Skip structural alerts for now
            use_threading=True
        )
        
        print(f"   Library after filtering: {len(library_filtered)} compounds")
        
        return library_filtered, reference_df
    
    def threaded_fingerprint_computation(self, molecules: List[Chem.Mol], description: str) -> List:
        """Compute fingerprints using multiple threads - using centralized infrastructure."""
        # Use centralized threading infrastructure
        from src.utils.threading_utils import ThreadedBatchProcessor, ProgressTracker
        
        print(f"\n🔢 Computing {description} fingerprints with {self.n_threads} threads...")
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(len(molecules), f"{description} fingerprints")
        
        processor = ThreadedBatchProcessor(self.n_threads)
        
        def fingerprint_batch_processor(batch_molecules, batch_id):
            """Process a batch of molecules for fingerprint computation."""
            return self.compute_fingerprints_batch(batch_molecules, batch_id)
        
        results = processor.process_batches(molecules, fingerprint_batch_processor, f"{description} fingerprints")
        
        # Complete progress tracking
        self.progress_tracker.complete()
        
        return results
    
    def threaded_similarity_search(self, lib_fingerprints: List, ref_fingerprints: List) -> List[float]:
        """Perform similarity search using multiple threads - using centralized infrastructure."""
        # Use centralized threading infrastructure
        from src.utils.threading_utils import ThreadedBatchProcessor, compute_similarities_batch
        
        print(f"\n🔍 Computing similarities with {self.n_threads} threads...")
        
        # Filter out None fingerprints from references
        valid_ref_fps = [fp for fp in ref_fingerprints if fp is not None]
        print(f"   Using {len(valid_ref_fps)} valid reference fingerprints")
        
        processor = ThreadedBatchProcessor(self.n_threads)
        
        def similarity_batch_processor(batch_lib_fps, batch_id):
            """Process a batch of library fingerprints for similarity computation."""
            batch_similarities = compute_similarities_batch(batch_lib_fps, valid_ref_fps)
            return batch_id, batch_similarities
        
        similarities = processor.process_batches(lib_fingerprints, similarity_batch_processor, "similarities")
        print(f"\n   ✅ Computed similarities for {len(similarities)} compounds")
        return similarities
    
    def generate_hits_threaded(self, max_hits: int = 1000, max_library_size: int = 15000, similarity_threshold: float = 0.58) -> pd.DataFrame:
        """Generate hits using threaded similarity search."""
        print(f"\n🚀 THREADED HITS GENERATION")
        print("="*60)
        
        start_time = time.time()
        
        # Load and filter data
        library_df, reference_df = self.load_and_filter_data(max_library_size)
        if library_df is None or reference_df is None:
            return None
        
        # Extract molecules
        lib_molecules = [row.get('ROMol') for _, row in library_df.iterrows()]
        ref_molecules = [row.get('ROMol') for _, row in reference_df.iterrows()]
        
        # Compute fingerprints in parallel
        lib_fingerprints = self.threaded_fingerprint_computation(lib_molecules, "library")
        ref_fingerprints = self.threaded_fingerprint_computation(ref_molecules, "reference")
        
        # Compute similarities in parallel
        similarities = self.threaded_similarity_search(lib_fingerprints, ref_fingerprints)
        
        # Add similarities to dataframe
        library_df = library_df.copy()
        library_df['similarity'] = similarities
        
        # Filter by similarity threshold and get top hits
        hits_df = library_df[library_df['similarity'] >= similarity_threshold].copy()
        hits_df = hits_df.nlargest(max_hits, 'similarity')
        
        # Add molecular properties
        print(f"\n📊 Computing molecular properties for {len(hits_df)} hits...")
        for idx, row in hits_df.iterrows():
            mol = row.get('mol')
            if mol:
                try:
                    hits_df.at[idx, 'MW'] = Descriptors.MolWt(mol)
                    hits_df.at[idx, 'LogP'] = Descriptors.MolLogP(mol)
                    hits_df.at[idx, 'HBA'] = Descriptors.NumHAcceptors(mol)
                    hits_df.at[idx, 'HBD'] = Descriptors.NumHDonors(mol)
                    hits_df.at[idx, 'TPSA'] = Descriptors.TPSA(mol)
                except Exception:
                    continue
            
            elapsed_time = time.time() - start_time
            
        print(f"\n🎉 THREADED GENERATION COMPLETE!")
        print(f"   ⏱️  Total time: {elapsed_time:.1f} seconds")
        print(f"   📊 Generated {len(hits_df)} hits")
                
        if len(hits_df) > 0:
            avg_sim = hits_df['similarity'].mean()
            min_sim = hits_df['similarity'].min()
            max_sim = hits_df['similarity'].max()
            print(f"   📈 Similarity range: {min_sim:.3f} - {max_sim:.3f} (avg: {avg_sim:.3f})")
            
            # Performance metrics
            compounds_per_second = len(library_df) / elapsed_time
            print(f"   ⚡ Processing speed: {compounds_per_second:.1f} compounds/second")
            print(f"   🧵 Threading efficiency: {self.n_threads}x parallel processing")
                
        return hits_df
    
    def save_hits(self, hits_df: pd.DataFrame) -> bool:
        """Save hits to CSV file."""
        if hits_df is None or hits_df.empty:
            print("❌ No hits to save")
            return False
        
        try:
            # Remove mol column for CSV saving
            save_df = hits_df.drop(columns=['mol'], errors='ignore')
            
            output_path = Path(self.output_file)
            save_df.to_csv(output_path, index=False)
            
            print(f"\n💾 THREADED HITS SAVED!")
            print(f"   📁 File: {output_path.absolute()}")
            print(f"   📊 Compounds: {len(save_df)}")
            print(f"   💽 File size: {output_path.stat().st_size / 1024:.1f} KB")
            print(f"   🧵 Generated using {self.n_threads} threads")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving hits: {e}")
            return False

def main():
    """Main execution function."""
    print("="*70)
    print("🚀 THREADED CHAPTER 2 HITS GENERATOR")
    print("Multi-threaded parallel processing for maximum performance!")
    print("="*70)
    
    # Auto-detect optimal thread count
    n_threads = min(mp.cpu_count(), 8)
    print(f"🧵 Detected {mp.cpu_count()} CPU cores, using {n_threads} threads")
    
    generator = SpecsHitsGenerator(random_seed=42, n_threads=n_threads)
    
    # Check if hits already exist
    hits_path = Path(generator.output_file)
    if hits_path.exists():
        print(f"\n✅ Found existing threaded hits: {hits_path}")
        try:
            existing_df = pd.read_csv(hits_path)
            print(f"   📊 Contains {len(existing_df)} compounds")
            
            user_input = input("\nRegenerate with threading? (y/N): ").lower().strip()
            if user_input not in ['y', 'yes']:
                print("✅ Using existing hits file")
                return existing_df
        except Exception as e:
            print(f"⚠️  Error reading existing file: {e}")
    
    # Generate hits with threading
    print("\n🚀 Starting THREADED hits generation...")
    print("   ⚡ This will be much faster than the single-threaded version!")
    
    hits_df = generator.generate_hits_threaded(
        max_hits=1000,
        max_library_size=15000,  # Adjust based on your RAM
        similarity_threshold=0.58
    )
    
    if hits_df is not None and len(hits_df) > 0:
        # Save hits
        if generator.save_hits(hits_df):
            print("\n🎉 THREADED SETUP COMPLETE!")
            print("   Ready for lightning-fast safety analysis!")
            return hits_df
        else:
            print("❌ Failed to save hits")
            return None
    else:
        print("❌ Failed to generate hits")
        return None

if __name__ == "__main__":
    hits = main() 