#!/usr/bin/env python3
"""
Pytest test suite for the antimalarial drug discovery pipeline.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestPipeline:
    """Test suite for the complete pipeline."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment before each test."""
        # Create temporary directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp())
        yield
        # Cleanup after test
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_imports(self):
        """Test that all modules can be imported successfully."""
        # Basic imports
        from rdkit import Chem
        import pandas as pd
        import numpy as np
        
        # Module imports
        from data_processing.loader import MoleculeLoader
        from filtering.structural_alerts import StructuralAlertFilter
        from filtering.drug_like import DrugLikeFilter
        from similarity.search import SimilaritySearcher
        from pipeline import AntimalarialScreeningPipeline
        from utils.config import ProjectConfig
        from utils.dataset_manager import DatasetManager
        
        assert True  # If we get here, all imports succeeded
    
    def test_dataset_manager(self):
        """Test dataset manager functionality."""
        from utils.dataset_manager import DatasetManager
        
        manager = DatasetManager(data_dir=str(self.test_dir))
        
        # Test dataset info retrieval
        datasets = manager.get_available_datasets()
        assert "small" in datasets
        assert "diverse" in datasets
        assert "large" in datasets
        
        # Test small dataset creation
        manager.create_small_dataset()
        
        library_path = self.test_dir / "raw" / "small_test_library.sdf"
        reference_path = self.test_dir / "reference" / "small_reference.sdf"
        
        assert library_path.exists()
        assert reference_path.exists()
    
    def test_structural_alerts(self):
        """Test structural alerts filtering."""
        from filtering.structural_alerts import StructuralAlertFilter
        from rdkit import Chem
        import pandas as pd
        
        filter_obj = StructuralAlertFilter()
        
        # Test with safe molecule
        safe_mol = Chem.MolFromSmiles("CCO")  # Ethanol
        has_pains, pains_alerts = filter_obj.check_pains(safe_mol)
        has_brenk, brenk_alerts = filter_obj.check_brenk(safe_mol)
        
        # Ethanol should be safe
        assert not has_pains
        assert not has_brenk
        
        # Test DataFrame filtering
        test_df = pd.DataFrame({
            'ID': ['mol1'],
            'SMILES': ['CCO'],
            'ROMol': [safe_mol]
        })
        
        filtered_df = filter_obj.filter_dataframe(test_df)
        assert len(filtered_df) == 1  # Should pass all filters
    
    def test_fingerprint_generation(self):
        """Test fingerprint generation."""
        from similarity.fingerprints import FingerprintGenerator
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles("CCO")  # Ethanol
        
        # Test different fingerprint types
        for fp_type in ["morgan", "rdkit", "maccs"]:
            generator = FingerprintGenerator(fingerprint_type=fp_type)
            fp = generator.generate_fingerprint(mol)
            
            assert fp is not None
            assert len(fp) > 0
    
    def test_data_loading(self):
        """Test data loading functionality."""
        from data_processing.loader import MoleculeLoader
        from utils.dataset_manager import DatasetManager
        
        # Create test dataset
        manager = DatasetManager(data_dir=str(self.test_dir))
        manager.create_small_dataset()
        
        # Load data
        loader = MoleculeLoader()
        library_path = self.test_dir / "raw" / "small_test_library.sdf"
        df = loader.load_and_process(str(library_path))
        
        assert not df.empty
        assert len(df) == 5  # Small dataset has 5 molecules
        
        # Check required columns
        required_cols = ['MW', 'LogP', 'HBA', 'HBD', 'TPSA', 'RotBonds']
        for col in required_cols:
            assert col in df.columns
    
    def test_pipeline_execution(self):
        """Test complete pipeline execution."""
        from pipeline import AntimalarialScreeningPipeline
        from utils.config import ProjectConfig
        from utils.dataset_manager import DatasetManager
        
        # Setup test dataset
        manager = DatasetManager(data_dir=str(self.test_dir))
        manager.create_small_dataset()
        
        # Configure pipeline
        config = ProjectConfig()
        config.input_library = str(self.test_dir / "raw" / "small_test_library.sdf")
        config.reference_compounds = str(self.test_dir / "reference" / "small_reference.sdf")
        config.output_dir = str(self.test_dir / "results")
        config.similarity_config.threshold = 0.3
        
        # Run pipeline
        pipeline = AntimalarialScreeningPipeline(config)
        results = pipeline.run_full_pipeline(
            generate_plots=False,  # Skip plots for testing
            save_results=True
        )
        
        # Verify results
        assert results is not None
        assert 'statistics' in results
        
        stats = results['statistics']
        assert stats['library_size'] == 5
        assert 'drug_like_molecules' in stats
        assert 'total_hits' in stats
        assert 'total_runtime_seconds' in stats

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 