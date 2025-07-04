#!/usr/bin/env python3
"""
Generate Chapter 2 Hits - SIMPLIFIED THREADED VERSION
====================================================

This script demonstrates multi-threaded similarity search using a simplified approach.
It uses the first 5000 compounds from the library and basic Lipinski filtering only.

Usage:
    python generate_chapter2_hits_threaded_simple.py
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
import multiprocessing as mp

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors

# Import components
from similarity.fingerprints import FingerprintGenerator
from data_processing.loader import MoleculeLoader

# Suppress warnings
warnings.filterwarnings('ignore')

class SimpleThreadedGenerator:
    """Simplified multi-threaded hits generator for demonstration."""
    
    def __init__(self, random_seed: int = 42, n_threads: int = None):
        """Initialize the threaded generator."""
        self.random_seed = random_seed
        self.output_file = "chapter2_hits_threaded_simple.csv"
        
        # Auto-detect optimal thread count
        if n_threads is None:
            self.n_threads = min(mp.cpu_count(), 6)  # Conservative limit
        else:
            self.n_threads = n_threads
            
        # Thread-safe progress tracking
        self.progress_lock = threading.Lock()
        self.processed_compounds = 0
        self.total_compounds = 0
        
        print("🚀 SIMPLE THREADED Generator Initialized")
        print(f"   🧵 Using {self.n_threads} threads for parallel processing")
        print(f"   📁 Will save results to: {self.output_file}")
    
    def update_progress(self, increment: int = 1):
        """Thread-safe progress update."""
        with self.progress_lock:
            self.processed_compounds += increment
            if self.total_compounds > 0:
                percentage = (self.processed_compounds / self.total_compounds) * 100
                print(f"\r   ⚡ Processing: {self.processed_compounds}/{self.total_compounds} ({percentage:.1f}%)", end="", flush=True)
    
    def check_drug_likeness(self, mol: Chem.Mol) -> bool:
        """Enhanced drug-likeness check using centralized infrastructure."""
        if mol is None:
            return False
            
        # Use centralized descriptor calculator
        from src.utils.molecular_descriptors import descriptor_calculator
        
        descriptors = descriptor_calculator.calculate_all_descriptors(mol)
        
        # Extract values with defaults
        mw = descriptors.get('MW', 0)
        logp = descriptors.get('LogP', 0)
        hbd = descriptors.get('HBD', 0)
        hba = descriptors.get('HBA', 0)
        
        # Count Lipinski violations (more lenient)
        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1
        
        # Allow up to 1 Lipinski violation, plus basic sanity checks
        return (violations <= 1 and 
                mw >= 150 and mw <= 1000 and  # Basic MW range
                logp >= -3 and logp <= 8)      # Reasonable LogP range
    
    def compute_fingerprints_batch(self, molecules: List[Chem.Mol], batch_id: int) -> Tuple[int, List]:
        """Compute fingerprints for a batch of molecules (thread worker) - using centralized infrastructure."""
        # Use existing infrastructure instead of duplicating fingerprint computation
        from similarity.fingerprints import FingerprintGenerator
        
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
            
            self.update_progress(1)
        
        return batch_id, fingerprints
    
    def compute_similarities_batch(self, lib_fps: List, ref_fps: List, batch_id: int) -> Tuple[int, List[float]]:
        """Compute similarities for a batch of library fingerprints - using centralized infrastructure."""
        # Use centralized similarity computation
        from src.utils.threading_utils import compute_similarities_batch
        
        max_similarities = compute_similarities_batch(lib_fps, ref_fps)
        
        # Update progress tracking
        for _ in lib_fps:
            self.update_progress(1)
        
        return batch_id, max_similarities
    
    def load_and_filter_data(self, max_library_size: int = 5000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and filter molecular data with simple filtering."""
        print("\n📚 Loading molecular data...")
        
        # Load data files using centralized path management
        from src.utils.config import data_paths
        loader = MoleculeLoader()
        
        specs_path = Path(data_paths.resolve_specs_path(from_examples=True))
        malaria_path = Path(data_paths.resolve_malaria_path(from_examples=True))
        
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
        
        # Apply SIMPLE Lipinski filtering
        print("\n🧪 Applying simple Lipinski filters...")
        
        # Filter library
        library_filtered = []
        for _, mol_data in library_df.iterrows():
            mol = mol_data.get('ROMol')  # Use correct column name
            if mol and self.check_drug_likeness(mol):
                library_filtered.append(mol_data)
        
        library_df = pd.DataFrame(library_filtered)
        print(f"   Library after filtering: {len(library_df)} compounds")
        
        # Filter reference (keep more permissive for antimalarial compounds)
        ref_filtered = []
        for _, mol_data in reference_df.iterrows():
            mol = mol_data.get('ROMol')  # Use correct column name
            if mol:  # Keep all valid antimalarial compounds
                ref_filtered.append(mol_data)
        
        reference_df = pd.DataFrame(ref_filtered)
        print(f"   Reference after filtering: {len(reference_df)} compounds")
        
        return library_df, reference_df
    
    def threaded_fingerprint_computation(self, molecules: List[Chem.Mol], description: str) -> List:
        """Compute fingerprints using multiple threads."""
        print(f"\n🔢 Computing {description} fingerprints with {self.n_threads} threads...")
        
        # Split molecules into batches
        batch_size = max(1, len(molecules) // self.n_threads)
        batches = [molecules[i:i + batch_size] for i in range(0, len(molecules), batch_size)]
        
        # Reset progress tracking
        self.processed_compounds = 0
        self.total_compounds = len(molecules)
        
        fingerprints = [None] * len(molecules)
        
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.compute_fingerprints_batch, batch, i): i 
                for i, batch in enumerate(batches)
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch_id, batch_fps = future.result()
                
                # Insert batch results in correct position
                start_idx = batch_id * batch_size
                for i, fp in enumerate(batch_fps):
                    if start_idx + i < len(fingerprints):
                        fingerprints[start_idx + i] = fp
        
        valid_fps = [fp for fp in fingerprints if fp is not None]
        print(f"\n   ✅ Computed {len(valid_fps)} valid fingerprints")
        return fingerprints
    
    def threaded_similarity_search(self, lib_fingerprints: List, ref_fingerprints: List) -> List[float]:
        """Perform similarity search using multiple threads - using centralized infrastructure."""
        # Use centralized similarity search
        from src.utils.threading_utils import threaded_similarity_search
        
        return threaded_similarity_search(
            lib_fingerprints, 
            ref_fingerprints, 
            n_threads=self.n_threads, 
            description="similarities"
        )
    
    def generate_hits_threaded(self, max_hits: int = 1000, max_library_size: int = 5000, similarity_threshold: float = 0.3) -> pd.DataFrame:
        """Generate hits using threaded similarity search."""
        print(f"\n🚀 SIMPLE THREADED HITS GENERATION")
        print("="*60)
        
        start_time = time.time()
        
        # Load and filter data
        library_df, reference_df = self.load_and_filter_data(max_library_size)
        if library_df is None or reference_df is None or len(library_df) == 0:
            print("❌ No compounds available for processing")
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
        hits_df = hits_df.nlargest(min(max_hits, len(hits_df)), 'similarity')
        
        # Add molecular properties
        print(f"\n📊 Computing molecular properties for {len(hits_df)} hits...")
        
        # Use centralized descriptor calculator
        from src.utils.molecular_descriptors import descriptor_calculator
        
        # Add descriptors efficiently using the centralized calculator
        hits_df_with_descriptors = descriptor_calculator.add_descriptors_to_dataframe(
            hits_df, mol_col='ROMol', descriptors=['MW', 'LogP', 'HBA', 'HBD']
        )
        
        # Update the original dataframe
        for desc in ['MW', 'LogP', 'HBA', 'HBD']:
            if desc in hits_df_with_descriptors.columns:
                hits_df[desc] = hits_df_with_descriptors[desc]
        
        elapsed_time = time.time() - start_time
        
        print(f"\n🎉 SIMPLE THREADED GENERATION COMPLETE!")
        print(f"   ⏱️  Total time: {elapsed_time:.1f} seconds")
        print(f"   📊 Generated {len(hits_df)} hits")
        
        if len(hits_df) > 0:
            avg_sim = hits_df['similarity'].mean()
            min_sim = hits_df['similarity'].min()
            max_sim = hits_df['similarity'].max()
            print(f"   📈 Similarity range: {min_sim:.3f} - {max_sim:.3f} (avg: {avg_sim:.3f})")
            
            # Performance metrics
            total_compounds = len(library_df)
            compounds_per_second = total_compounds / elapsed_time
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
            
            print(f"\n💾 SIMPLE THREADED HITS SAVED!")
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
    print("🚀 SIMPLE THREADED HITS GENERATOR")
    print("Demonstrating multi-threaded parallel processing!")
    print("="*70)
    
    # Auto-detect optimal thread count
    n_threads = min(mp.cpu_count(), 6)
    print(f"🧵 Detected {mp.cpu_count()} CPU cores, using {n_threads} threads")
    
    generator = SimpleThreadedGenerator(random_seed=42, n_threads=n_threads)
    
    # Generate hits with threading
    print("\n🚀 Starting SIMPLE THREADED hits generation...")
    print("   ⚡ Using simple Lipinski filtering for better compatibility!")
    
    hits_df = generator.generate_hits_threaded(
        max_hits=1000,
        max_library_size=5000,  # Smaller for demo
        similarity_threshold=0.3  # Lower threshold to get more hits
    )
    
    if hits_df is not None and len(hits_df) > 0:
        # Save hits
        if generator.save_hits(hits_df):
            print("\n🎉 SIMPLE THREADED DEMO COMPLETE!")
            print("   🧵 Threading demonstration successful!")
            return hits_df
        else:
            print("❌ Failed to save hits")
            return None
    else:
        print("❌ Failed to generate hits")
        return None

if __name__ == "__main__":
    hits = main() 