#!/usr/bin/env python3
"""
Demonstration of multi-threading capabilities added to filtering and similarity modules.

This script shows how the enhanced modules can be used everywhere in the pipeline
for improved performance with minimal code changes.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import time
import logging
import sys
import os
from pathlib import Path

# Add the parent directory to Python path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the enhanced modules
from filtering.drug_like import DrugLikeFilter
from similarity.fingerprints import FingerprintGenerator
from similarity.search import SimilaritySearcher

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_demo_data():
    """Create demo dataset for testing."""
    logger.info("Creating demo dataset...")
    
    # Use centralized descriptor calculator
    from src.utils.molecular_descriptors import descriptor_calculator
    
    smiles_list = [
        "CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC", "C1=CC=C(C=C1)C(=O)O",
        "CC(C)(C)C1=CC=C(C=C1)O", "CC1=CC=CC=C1", "CCC1=CC=CC=C1",
        "C1=CC=C2C(=C1)C=CC=C2", "CC(=O)NC1=CC=CC=C1",
        "CC1=CC=C(C=C1)C", "COC1=CC=CC=C1", "CC(C)C1=CC=CC=C1",
        "C1=CC=C(C=C1)N", "CC1=CC=C(C=C1)N", "C1=CC=C(C=C1)Cl",
        "CC1=CC=C(C=C1)Cl", "C1=CC=C(C=C1)Br", "CC1=CC=C(C=C1)Br",
        "C1=CC=C(C=C1)F", "CC1=CC=C(C=C1)F", "C1=CC=C(C=C1)I"
    ]
    
    # Add more diverse compounds for better testing
    additional_smiles = [
        "CCCCCCCCCCCCCCCC(=O)O",  # Palmitic acid
        "CCCCCCCCC=CCCCCCCCC(=O)O",  # Oleic acid
        "NC(=O)C1=CC=CC=C1",  # Benzamide
        "CC(C)(C)OC(=O)NC1=CC=CC=C1",  # Boc-aniline
        "C1=CC=C2C(=C1)NC=N2",  # Benzimidazole
        "C1=CC=C2C(=C1)NC3=CC=CC=C32",  # Carbazole
        "CC1=CC2=C(C=C1)C=CC=C2",  # 2-Methylnaphthalene
        "C1=CC=C(C=C1)C2=CC=CC=C2",  # Biphenyl
        "CC(C)(C)C1=CC=C(C=C1)C(C)(C)C",  # Di-tert-butylbenzene
        "C1=CC=C(C=C1)C(=O)C2=CC=CC=C2",  # Benzophenone
    ] * 15  # Multiply to get more test data
    
    smiles_list.extend(additional_smiles)
    
    molecules = []
    data = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            molecules.append(mol)
            
            # Use centralized descriptor calculator
            descriptors = descriptor_calculator.calculate_all_descriptors(mol)
            
            row = {
                'ID': f'MOL_{i:04d}',
                'SMILES': smiles,
                'ROMol': mol,
                'MW': descriptors.get('MW', 0),
                'LogP': descriptors.get('LogP', 0),
                'HBA': descriptors.get('HBA', 0),
                'HBD': descriptors.get('HBD', 0),
                'TPSA': descriptors.get('TPSA', 0),
                'RotBonds': descriptors.get('RotBonds', 0),
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    logger.info(f"Created demo dataset with {len(df)} molecules")
    return df

def demo_enhanced_filtering():
    """Demonstrate enhanced drug-like filtering with threading."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Enhanced Drug-like Filtering")
    logger.info("="*60)
    
    # Create demo data
    df = create_demo_data()
    
    # Original way (single-threaded)
    logger.info("\nUsing original single-threaded approach:")
    filter_original = DrugLikeFilter(violations_allowed=1)
    
    start_time = time.time()
    filtered_original = filter_original.filter_dataframe(df, use_threading=False)
    original_time = time.time() - start_time
    
    logger.info(f"Original: {len(filtered_original)}/{len(df)} molecules passed in {original_time:.3f}s")
    
    # Enhanced way (multi-threaded) - automatically chooses best method
    logger.info("\nUsing enhanced multi-threaded approach:")
    filter_enhanced = DrugLikeFilter(violations_allowed=1, n_threads=4)
    
    start_time = time.time()
    filtered_enhanced = filter_enhanced.filter_dataframe(df, use_threading=True)
    enhanced_time = time.time() - start_time
    
    logger.info(f"Enhanced: {len(filtered_enhanced)}/{len(df)} molecules passed in {enhanced_time:.3f}s")
    
    if enhanced_time > 0 and original_time > 0:
        speedup = original_time / enhanced_time
        logger.info(f"Speedup: {speedup:.2f}x (for larger datasets, speedup will be more significant)")
    
    return df

def demo_enhanced_fingerprints():
    """Demonstrate enhanced fingerprint generation with threading."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Enhanced Fingerprint Generation")
    logger.info("="*60)
    
    # Create demo data
    df = create_demo_data()
    molecules = df['ROMol'].tolist()
    
    # Original way (single-threaded)
    logger.info("\nUsing original single-threaded approach:")
    fp_gen_original = FingerprintGenerator(fingerprint_type="morgan", n_bits=1024)
    
    start_time = time.time()
    fps_original = fp_gen_original.generate_fingerprints_batch(molecules)
    original_time = time.time() - start_time
    
    logger.info(f"Original: Generated {fps_original.shape} fingerprints in {original_time:.3f}s")
    
    # Enhanced way (multi-threaded)
    logger.info("\nUsing enhanced multi-threaded approach:")
    fp_gen_enhanced = FingerprintGenerator(fingerprint_type="morgan", n_bits=1024, n_threads=4)
    
    start_time = time.time()
    fps_enhanced = fp_gen_enhanced.generate_fingerprints_batch_threaded(molecules)
    enhanced_time = time.time() - start_time
    
    logger.info(f"Enhanced: Generated {fps_enhanced.shape} fingerprints in {enhanced_time:.3f}s")
    
    if enhanced_time > 0 and original_time > 0:
        speedup = original_time / enhanced_time
        logger.info(f"Speedup: {speedup:.2f}x (for larger datasets, speedup will be more significant)")
    
    return df

def demo_enhanced_similarity_search():
    """Demonstrate enhanced similarity search with threading."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Enhanced Similarity Search")
    logger.info("="*60)
    
    # Create demo data
    df = create_demo_data()
    
    # Split into reference and library
    reference_df = df.head(10).copy()
    library_df = df.tail(400).copy()
    
    # Original way (single-threaded)
    logger.info("\nUsing original single-threaded approach:")
    searcher_original = SimilaritySearcher(fingerprint_type="morgan", n_threads=1)
    searcher_original.load_reference_compounds(reference_df)
    
    start_time = time.time()
    results_original = searcher_original.search_library(
        library_df, threshold=0.5, max_results=50, use_threading=False
    )
    original_time = time.time() - start_time
    
    logger.info(f"Original: Found {len(results_original)} hits in {original_time:.3f}s")
    
    # Enhanced way (multi-threaded)
    logger.info("\nUsing enhanced multi-threaded approach:")
    searcher_enhanced = SimilaritySearcher(fingerprint_type="morgan", n_threads=4)
    searcher_enhanced.load_reference_compounds(reference_df)
    
    start_time = time.time()
    results_enhanced = searcher_enhanced.search_library(
        library_df, threshold=0.5, max_results=50, use_threading=True
    )
    enhanced_time = time.time() - start_time
    
    logger.info(f"Enhanced: Found {len(results_enhanced)} hits in {enhanced_time:.3f}s")
    
    if enhanced_time > 0 and original_time > 0:
        speedup = original_time / enhanced_time
        logger.info(f"Speedup: {speedup:.2f}x (for larger datasets, speedup will be more significant)")

def demo_backward_compatibility():
    """Show that old code still works without changes."""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Backward Compatibility")
    logger.info("="*60)
    
    # Create demo data
    df = create_demo_data()
    
    logger.info("\nOld code still works exactly the same:")
    logger.info("# This is how you used to call the filtering")
    logger.info("filter = DrugLikeFilter(violations_allowed=1)")
    logger.info("filtered_df = filter.filter_dataframe(df)")
    
    # This works exactly as before
    old_filter = DrugLikeFilter(violations_allowed=1)
    filtered_df = old_filter.filter_dataframe(df)
    
    logger.info(f"✓ Old code works: {len(filtered_df)}/{len(df)} molecules passed")
    logger.info("\nNow you can optionally enable threading for better performance:")
    logger.info("# Just add threading parameters for better performance")
    logger.info("filter = DrugLikeFilter(violations_allowed=1, n_threads=4)")
    logger.info("filtered_df = filter.filter_dataframe(df, use_threading=True)")

def test_molecular_descriptor_calculation():
    """Test molecular descriptor calculations with threading capabilities."""
    print("🧪 Testing molecular descriptor calculations...")
    
    # Use centralized descriptor calculator
    from src.utils.molecular_descriptors import descriptor_calculator
    
    test_smiles = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid  
        "c1ccccc1",  # Benzene
        "CCN(CC)CC",  # Triethylamine
        "C1=CC=C(C=C1)C(=O)O"  # Benzoic acid
    ]
    
    print(f"   Testing {len(test_smiles)} molecules...")
    
    for i, smiles in enumerate(test_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Use centralized descriptor calculator
            descriptors = descriptor_calculator.calculate_all_descriptors(mol)
            
            print(f"   Molecule {i+1}: {smiles}")
            print(f"      MW: {descriptors.get('MW', 0):.2f}")
            print(f"      LogP: {descriptors.get('LogP', 0):.2f}")
            print(f"      HBA: {descriptors.get('HBA', 0)}")
            print(f"      HBD: {descriptors.get('HBD', 0)}")
            print(f"      TPSA: {descriptors.get('TPSA', 0):.2f}")
            print(f"      RotBonds: {descriptors.get('RotBonds', 0)}")

def main():
    """Run all demonstrations."""
    logger.info("Multi-Threading Enhancement Demonstration")
    logger.info("========================================")
    logger.info("\nThis demo shows how the core modules now support multi-threading")
    logger.info("for improved performance while maintaining backward compatibility.")
    
    try:
        # Demo enhanced filtering
        demo_enhanced_filtering()
        
        # Demo enhanced fingerprints
        demo_enhanced_fingerprints()
        
        # Demo enhanced similarity search
        demo_enhanced_similarity_search()
        
        # Demo backward compatibility
        demo_backward_compatibility()
        
        # Test molecular descriptor calculations
        test_molecular_descriptor_calculation()
        
        logger.info("\n" + "="*60)
        logger.info("SUMMARY: Multi-Threading Enhancements")
        logger.info("="*60)
        logger.info("\n✓ Drug-like filtering now supports multi-threading")
        logger.info("✓ Fingerprint generation now supports multi-threading")
        logger.info("✓ Similarity search now supports multi-threading")
        logger.info("✓ All enhancements are backward compatible")
        logger.info("✓ Threading is automatically enabled for larger datasets")
        logger.info("✓ Thread count auto-detects CPU cores")
        logger.info("\nTo use in your existing scripts:")
        logger.info("1. Add n_threads parameter when creating objects")
        logger.info("2. Add use_threading=True to method calls")
        logger.info("3. Enjoy faster processing on multi-core systems!")
        
        logger.info("\nThe modules can now be used everywhere in the pipeline")
        logger.info("for improved performance with minimal code changes.")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise

if __name__ == "__main__":
    main() 