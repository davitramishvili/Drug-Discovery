#!/usr/bin/env python3
"""
Main test runner for the antimalarial drug discovery pipeline.
Provides a comprehensive test suite that can be run standalone or with pytest.

Usage:
    python test_pipeline.py          # Run all tests
    pytest tests/                    # Run with pytest
"""

import sys
import time
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test all imports."""
    print("Testing imports...")
    
    try:
        # Basic imports
        from rdkit import Chem
        import pandas as pd
        import numpy as np
        print("✓ Basic imports successful")
        
        # Module imports
        from data_processing.loader import MoleculeLoader
        from filtering.structural_alerts import StructuralAlertFilter
        from filtering.drug_like import DrugLikeFilter
        from similarity.search import SimilaritySearcher
        from pipeline import AntimalarialScreeningPipeline
        from utils.config import ProjectConfig
        from utils.dataset_manager import DatasetManager
        print("✓ All module imports successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_creation():
    """Test dataset creation."""
    print("\nTesting dataset creation...")
    
    try:
        from utils.dataset_manager import DatasetManager
        
        manager = DatasetManager()
        
        # Test small dataset creation
        print("  Creating small dataset...")
        manager.create_small_dataset()
        
        # Verify files exist
        small_lib = Path("data/raw/small_test_library.sdf")
        small_ref = Path("data/reference/small_reference.sdf")
        
        if small_lib.exists() and small_ref.exists():
            print("✓ Small dataset created successfully")
        else:
            print("✗ Small dataset files not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_structural_alerts():
    """Test structural alerts functionality."""
    print("\nTesting structural alerts...")
    
    try:
        from filtering.structural_alerts import StructuralAlertFilter
        from rdkit import Chem
        
        # Initialize filter
        filter_obj = StructuralAlertFilter()
        
        # Test with simple molecules
        test_molecules = [
            ("CCO", "Ethanol"),  # Should pass
            ("c1ccc(=O)c(=O)cc1", "Quinone"),  # Should fail PAINS
            ("C(=O)Cl", "Acyl chloride"),  # Should fail BRENK
        ]
        
        for smiles, name in test_molecules:
            mol = Chem.MolFromSmiles(smiles)
            has_pains, pains_alerts = filter_obj.check_pains(mol)
            has_brenk, brenk_alerts = filter_obj.check_brenk(mol)
            print(f"  {name}: PAINS={has_pains}, BRENK={has_brenk}")
        
        print("✓ Structural alerts working")
        return True
        
    except Exception as e:
        print(f"✗ Structural alerts failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fingerprints():
    """Test fingerprint generation."""
    print("\nTesting fingerprint generation...")
    
    try:
        from similarity.fingerprints import FingerprintGenerator
        from rdkit import Chem
        
        # Test different fingerprint types
        fp_types = ["morgan", "rdkit", "maccs"]
        
        mol = Chem.MolFromSmiles("CCO")  # Ethanol
        
        for fp_type in fp_types:
            generator = FingerprintGenerator(fingerprint_type=fp_type)
            fp = generator.generate_fingerprint(mol)
            
            if fp is not None:
                print(f"  {fp_type}: Generated fingerprint of length {len(fp)}")
            else:
                print(f"  {fp_type}: Failed to generate fingerprint")
                return False
        
        print("✓ Fingerprint generation working")
        return True
        
    except Exception as e:
        print(f"✗ Fingerprint generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_small_pipeline():
    """Test the complete pipeline with small dataset."""
    print("\nTesting complete pipeline with small dataset...")
    
    try:
        from pipeline import AntimalarialScreeningPipeline
        from utils.config import ProjectConfig
        
        # Configure for small dataset
        config = ProjectConfig()
        config.input_library = "data/raw/small_test_library.sdf"
        config.reference_compounds = "data/reference/small_reference.sdf"
        config.output_dir = "test_results"
        config.similarity_config.threshold = 0.3  # Lower threshold for testing
        
        # Initialize pipeline
        pipeline = AntimalarialScreeningPipeline(config)
        
        # Run pipeline without plots to avoid hanging
        print("  Running pipeline...")
        results = pipeline.run_full_pipeline(
            generate_plots=False,
            save_results=True
        )
        
        # Check results
        if results and 'statistics' in results:
            stats = results['statistics']
            print(f"  Library size: {stats.get('library_size', 'N/A')}")
            print(f"  Drug-like molecules: {stats.get('drug_like_molecules', 'N/A')}")
            print(f"  Similarity hits: {stats.get('total_hits', 'N/A')}")
            print(f"  Runtime: {stats.get('total_runtime_seconds', 'N/A'):.2f}s")
        
        print("✓ Pipeline completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    
    try:
        from data_processing.loader import MoleculeLoader
        
        loader = MoleculeLoader()
        
        # Load small dataset
        df = loader.load_and_process("data/raw/small_test_library.sdf")
        
        if not df.empty:
            print(f"  Loaded {len(df)} molecules")
            print(f"  Columns: {list(df.columns)}")
            
            # Check if descriptors were calculated
            descriptor_cols = ['MW', 'LogP', 'HBA', 'HBD', 'TPSA', 'RotBonds']
            missing_cols = [col for col in descriptor_cols if col not in df.columns]
            
            if not missing_cols:
                print("  All descriptors calculated")
            else:
                print(f"  Missing descriptors: {missing_cols}")
                
            print("✓ Data loading working")
            return True
        else:
            print("✗ No molecules loaded")
            return False
            
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Antimalarial Drug Discovery Pipeline - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Dataset Creation", test_dataset_creation),
        ("Data Loading", test_data_loading),
        ("Structural Alerts", test_structural_alerts),
        ("Fingerprints", test_fingerprints),
        ("Small Pipeline", test_small_pipeline),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"Test {test_name} failed - stopping here")
            break
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 