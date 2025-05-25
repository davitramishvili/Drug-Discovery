#!/usr/bin/env python3
"""
Dataset manager for antimalarial drug discovery pipeline.
Provides different dataset configurations for testing and analysis.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdDepictor
import logging

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages different datasets for the antimalarial screening pipeline."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the dataset manager.
        
        Args:
            data_dir: Base directory for data files
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.reference_dir = self.data_dir / "reference"
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        
    def get_available_datasets(self) -> Dict[str, Dict[str, str]]:
        """
        Get information about available datasets.
        
        Returns:
            Dictionary with dataset information
        """
        datasets = {
            "small": {
                "name": "Small Test Dataset",
                "description": "5 well-known drug molecules for quick testing",
                "library": "small_test_library.sdf",
                "reference": "small_reference.sdf",
                "molecules": 5,
                "violations_expected": "Minimal (0-1 violations per molecule)"
            },
            "diverse": {
                "name": "Diverse Test Dataset", 
                "description": "20 molecules with varying Lipinski violations for comprehensive analysis",
                "library": "test_library.sdf",
                "reference": "malaria_box.sdf",
                "molecules": 20,
                "violations_expected": "Wide range (0-3+ violations per molecule)"
            },
            "large": {
                "name": "Large Test Dataset",
                "description": "50 molecules including natural products and complex structures",
                "library": "large_test_library.sdf",
                "reference": "extended_malaria_box.sdf", 
                "molecules": 50,
                "violations_expected": "Full spectrum of drug-likeness"
            },
            "custom": {
                "name": "Custom Dataset",
                "description": "Use your own SDF files",
                "library": "custom_library.sdf",
                "reference": "custom_reference.sdf",
                "molecules": "Variable",
                "violations_expected": "Depends on your data"
            }
        }
        return datasets
        
    def create_small_dataset(self) -> None:
        """Create a small dataset with well-known drug molecules."""
        
        # Small library molecules (mostly drug-like)
        library_molecules = [
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin", "Anti-inflammatory"),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine", "Stimulant"),
            ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen", "Anti-inflammatory"),
            ("CC(=O)NC1=CC=C(C=C1)O", "Paracetamol", "Analgesic"),
            ("C1=CC=CC=C1", "Benzene", "Solvent"),
        ]
        
        # Small reference compounds
        reference_molecules = [
            ("CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl", "Chloroquine", "Antimalarial"),
            ("COC1=CC=C(C=C1)C(C2=CC=CC=N2)C3=CC=CC=N3", "Mefloquine_analog", "Antimalarial"),
        ]
        
        self._create_sdf_file(library_molecules, self.raw_dir / "small_test_library.sdf")
        self._create_sdf_file(reference_molecules, self.reference_dir / "small_reference.sdf")
        
        logger.info("Created small test dataset")
        
    def create_diverse_dataset(self) -> None:
        """Create the diverse dataset (current test_library.sdf)."""
        
        # This is the dataset we already created
        diverse_molecules = [
            # Drug-like molecules (0 violations)
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin", "Anti-inflammatory"),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine", "Stimulant"),
            ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen", "Anti-inflammatory"),
            ("CC(=O)NC1=CC=C(C=C1)O", "Paracetamol", "Analgesic"),
            
            # Molecules with 1 Lipinski violation
            ("CC(C)(C)C1=CC=C(C=C1)C(C)(C)C2=CC=C(C=C2)C(C)(C)C3=CC=C(C=C3)C(C)(C)C", "Large_hydrophobic_1", "Test_compound"),
            ("CCCCCCCCCCCCCCCCCC(=O)O", "Stearic_acid", "Fatty_acid"),
            
            # Molecules with 2 Lipinski violations
            ("CC(C)(C)C1=CC=C(C=C1)C(C)(C)C2=CC=C(C=C2)C(C)(C)C3=CC=C(C=C3)C(C)(C)C4=CC=C(C=C4)C(C)(C)C", "Large_hydrophobic_2", "Test_compound"),
            ("O=C(O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCC", "Very_long_fatty_acid", "Fatty_acid"),
            
            # Molecules with 3+ Lipinski violations
            ("CC1=C(C(=C(C(=C1C)C(=O)O)C)C(=O)O)C(=O)O", "Multi_carboxylic_acid", "Test_compound"),
            ("O=C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)CO", "Sugar_chain", "Carbohydrate"),
            
            # Very large molecules (4+ violations)
            ("CC(C)(C)C1=CC=C(C=C1)C(C)(C)C2=CC=C(C=C2)C(C)(C)C3=CC=C(C=C3)C(C)(C)C4=CC=C(C=C4)C(C)(C)C5=CC=C(C=C5)C(C)(C)C6=CC=C(C=C6)C(C)(C)C", "Very_large_hydrophobic", "Test_compound"),
            ("O=C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)CO", "Very_long_sugar", "Carbohydrate"),
            
            # Complex molecules
            ("NCCNCCNCCNCCNCCNCCNCCNCCNCCNCCN", "Polyamine", "Test_compound"),
            ("O=C(O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", "Ultra_long_fatty_acid", "Fatty_acid"),
            
            # Antimalarial-like structures
            ("CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl", "Chloroquine_analog", "Antimalarial"),
            ("COC1=CC=C(C=C1)C(C2=CC=CC=N2)C3=CC=CC=N3", "Quinoline_derivative", "Antimalarial"),
            
            # Natural product-like
            ("CC1=C(C2=CC=CC=C2N1)CC(=O)N3CCC(CC3)N4C=NC5=C4C=CC(=C5)C", "Complex_heterocycle", "Natural_product"),
            ("CC(C)CCCC(C)CCCC(C)CCCC(C)CCCC(C)CCCC(C)CCCC(C)CCCC(C)CCCC(C)C", "Squalene_analog", "Natural_product"),
            
            # Peptide-like molecules
            ("CC(C)CC(NC(=O)C(CC1=CC=CC=C1)NC(=O)C(CC(=O)O)NC(=O)C(CC(=O)N)NC(=O)C(CC2=CC=C(C=C2)O)N)C(=O)O", "Pentapeptide", "Peptide"),
            ("NCCCC(NC(=O)CCC(NC(=O)CC(N)C(=O)O)C(=O)O)C(=O)NCCCC(NC(=O)CCC(NC(=O)CC(N)C(=O)O)C(=O)O)C(=O)O", "Oligopeptide", "Peptide"),
        ]
        
        # Reference compounds
        reference_molecules = [
            ("CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl", "Chloroquine", "Antimalarial"),
            ("COC1=CC=C(C=C1)C(C2=CC=CC=N2)C3=CC=CC=N3", "Mefloquine_analog", "Antimalarial"),
        ]
        
        self._create_sdf_file(diverse_molecules, self.raw_dir / "test_library.sdf")
        self._create_sdf_file(reference_molecules, self.reference_dir / "malaria_box.sdf")
        
        logger.info("Created diverse test dataset")
        
    def create_large_dataset(self) -> None:
        """Create a large dataset with 50 molecules including natural products."""
        
        # Start with diverse dataset molecules
        large_molecules = [
            # Drug-like molecules (0 violations)
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin", "Anti-inflammatory"),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine", "Stimulant"),
            ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen", "Anti-inflammatory"),
            ("CC(=O)NC1=CC=C(C=C1)O", "Paracetamol", "Analgesic"),
            ("CC1=CC=C(C=C1)C(=O)O", "p-Toluic_acid", "Aromatic_acid"),
            ("CC(C)(C)C1=CC=C(C=C1)O", "BHT", "Antioxidant"),
            ("CC1=CC(=CC(=C1)C)C(=O)O", "Mesitylenic_acid", "Aromatic_acid"),
            ("CC1=CC=CC=C1N", "p-Toluidine", "Aromatic_amine"),
            
            # Molecules with 1-2 violations
            ("CC(C)(C)C1=CC=C(C=C1)C(C)(C)C2=CC=C(C=C2)C(C)(C)C3=CC=C(C=C3)C(C)(C)C", "Large_hydrophobic_1", "Test_compound"),
            ("CCCCCCCCCCCCCCCCCC(=O)O", "Stearic_acid", "Fatty_acid"),
            ("CC(C)(C)C1=CC=C(C=C1)C(C)(C)C2=CC=C(C=C2)C(C)(C)C3=CC=C(C=C3)C(C)(C)C4=CC=C(C=C4)C(C)(C)C", "Large_hydrophobic_2", "Test_compound"),
            ("CCCCCCCCCCCCCCCCCCCCCCCCC(=O)O", "Very_long_fatty_acid", "Fatty_acid"),
            ("CC1=C(C(=C(C(=C1C)C(=O)O)C)C(=O)O)C(=O)O", "Multi_carboxylic_acid", "Test_compound"),
            
            # Natural products and complex molecules
            ("CC1=C2C=C(C=CC2=C(C=C1)O)C(C)C", "Thymol", "Natural_product"),
            ("CC(C)C1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen_analog", "NSAID"),
            ("COC1=CC=C(C=C1)C=CC(=O)O", "Ferulic_acid", "Natural_product"),
            ("CC1=CC=C(C=C1)S(=O)(=O)N", "Toluenesulfonamide", "Sulfonamide"),
            ("CC1=CC=C(C=C1)S(=O)(=O)NC2=CC=CC=C2", "N-Phenyl_tosylamide", "Sulfonamide"),
            
            # Antimalarial and related compounds
            ("CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl", "Chloroquine_analog", "Antimalarial"),
            ("COC1=CC=C(C=C1)C(C2=CC=CC=N2)C3=CC=CC=N3", "Quinoline_derivative", "Antimalarial"),
            ("CC1=NC2=CC=CC=C2C(=C1)C(=O)O", "Quinaldic_acid", "Quinoline"),
            ("C1=CC=C2C(=C1)C=CC=N2", "Quinoline", "Heterocycle"),
            ("CC1=CC=NC2=CC=CC=C12", "Methylquinoline", "Quinoline"),
            
            # Peptides and amino acids
            ("CC(C)CC(NC(=O)C(CC1=CC=CC=C1)NC(=O)C(CC(=O)O)NC(=O)C(CC(=O)N)NC(=O)C(CC2=CC=C(C=C2)O)N)C(=O)O", "Pentapeptide", "Peptide"),
            ("NCCCC(NC(=O)CCC(NC(=O)CC(N)C(=O)O)C(=O)O)C(=O)NCCCC(NC(=O)CCC(NC(=O)CC(N)C(=O)O)C(=O)O)C(=O)O", "Oligopeptide", "Peptide"),
            ("CC(C)CC(N)C(=O)O", "Leucine", "Amino_acid"),
            ("CC(C(N)C(=O)O)O", "Threonine", "Amino_acid"),
            ("CC1=CC=C(C=C1)CC(N)C(=O)O", "Tyrosine", "Amino_acid"),
            
            # Carbohydrates and polyols
            ("O=C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)CO", "Sugar_chain", "Carbohydrate"),
            ("OCC(O)C(O)C(O)C(O)CO", "Sorbitol", "Polyol"),
            ("OCC(O)C(O)C(O)C(O)C=O", "Glucose", "Sugar"),
            ("OCC1OC(O)C(O)C(O)C1O", "Glucose_cyclic", "Sugar"),
            
            # Steroids and terpenes
            ("CC(C)CCCC(C)CCCC(C)CCCC(C)CCCC(C)CCCC(C)CCCC(C)CCCC(C)CCCC(C)C", "Squalene_analog", "Natural_product"),
            ("CC1CCC2C(C1)CCC3C2CCC4C3CCC(C4)O", "Cholesterol_analog", "Steroid"),
            ("CC(C)C1CCC(C)CC1", "Menthol_analog", "Terpene"),
            
            # Nucleotides and bases
            ("NC1=NC=NC2=C1N=CN2", "Adenine", "Nucleobase"),
            ("NC1=NC(=O)NC=C1", "Cytosine", "Nucleobase"),
            ("O=C1NC(=O)NC=C1", "Uracil", "Nucleobase"),
            ("NC1=NC(=O)C2=C(N1)N=CN2", "Guanine", "Nucleobase"),
            
            # Complex heterocycles
            ("CC1=C(C2=CC=CC=C2N1)CC(=O)N3CCC(CC3)N4C=NC5=C4C=CC(=C5)C", "Complex_heterocycle", "Natural_product"),
            ("NCCNCCNCCNCCNCCNCCNCCNCCNCCNCCN", "Polyamine", "Test_compound"),
            ("C1=CC=C2C(=C1)C=CC=N2", "Quinoline_base", "Heterocycle"),
            ("C1=CC=C2C(=C1)C=CN=C2", "Isoquinoline", "Heterocycle"),
            
            # Very large molecules
            ("CC(C)(C)C1=CC=C(C=C1)C(C)(C)C2=CC=C(C=C2)C(C)(C)C3=CC=C(C=C3)C(C)(C)C4=CC=C(C=C4)C(C)(C)C5=CC=C(C=C5)C(C)(C)C6=CC=C(C=C6)C(C)(C)C", "Very_large_hydrophobic", "Test_compound"),
            ("O=C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)CO", "Very_long_sugar", "Carbohydrate"),
            ("O=C(O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", "Ultra_long_fatty_acid", "Fatty_acid"),
            
            # Additional drug-like molecules
            ("CC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2", "N-Phenyl_toluamide", "Amide"),
            ("CC1=CC=C(C=C1)C(=O)O", "p-Toluic_acid_2", "Aromatic_acid"),
            ("CC(C)NC(=O)C1=CC=CC=C1", "N-Isopropyl_benzamide", "Amide"),
        ]
        
        # Extended reference compounds
        extended_reference = [
            ("CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl", "Chloroquine", "Antimalarial"),
            ("COC1=CC=C(C=C1)C(C2=CC=CC=N2)C3=CC=CC=N3", "Mefloquine_analog", "Antimalarial"),
            ("CC1=NC2=CC=CC=C2C(=C1)C(=O)O", "Quinaldic_acid", "Antimalarial"),
            ("C1=CC=C2C(=C1)C=CC=N2", "Quinoline", "Antimalarial"),
            ("CC1=CC=NC2=CC=CC=C12", "Methylquinoline", "Antimalarial"),
        ]
        
        self._create_sdf_file(large_molecules, self.raw_dir / "large_test_library.sdf")
        self._create_sdf_file(extended_reference, self.reference_dir / "extended_malaria_box.sdf")
        
        logger.info("Created large test dataset")
        
    def _create_sdf_file(self, molecules: List[Tuple[str, str, str]], filename: Path) -> None:
        """
        Create an SDF file from a list of (SMILES, name, activity) tuples.
        
        Args:
            molecules: List of (SMILES, name, activity) tuples
            filename: Output SDF file path
        """
        writer = Chem.SDWriter(str(filename))
        
        for i, (smiles, name, activity) in enumerate(molecules, 1):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Add hydrogens and generate 2D coordinates
                mol = Chem.AddHs(mol)
                rdDepictor.Compute2DCoords(mol)
                
                # Set properties
                mol.SetProp("ID", f"COMPOUND_{i:03d}")
                mol.SetProp("NAME", name)
                mol.SetProp("ACTIVITY", activity)
                mol.SetProp("SMILES", smiles)
                
                writer.write(mol)
            else:
                logger.warning(f"Failed to create molecule from SMILES: {smiles}")
        
        writer.close()
        logger.info(f"Created SDF file: {filename}")
        
    def setup_dataset(self, dataset_name: str) -> Tuple[str, str]:
        """
        Setup a specific dataset and return the file paths.
        
        Args:
            dataset_name: Name of the dataset to setup
            
        Returns:
            Tuple of (library_path, reference_path)
        """
        datasets = self.get_available_datasets()
        
        if dataset_name not in datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}")
            
        dataset_info = datasets[dataset_name]
        
        # Create the dataset if it doesn't exist
        library_path = self.raw_dir / dataset_info["library"]
        reference_path = self.reference_dir / dataset_info["reference"]
        
        if dataset_name == "small" and not library_path.exists():
            self.create_small_dataset()
        elif dataset_name == "diverse" and not library_path.exists():
            self.create_diverse_dataset()
        elif dataset_name == "large" and not library_path.exists():
            self.create_large_dataset()
        elif dataset_name == "custom":
            if not library_path.exists() or not reference_path.exists():
                logger.warning(f"Custom dataset files not found. Please place your files at:")
                logger.warning(f"Library: {library_path}")
                logger.warning(f"Reference: {reference_path}")
                
        return str(library_path), str(reference_path)
        
    def list_datasets(self) -> None:
        """Print information about available datasets."""
        datasets = self.get_available_datasets()
        
        print("\nAvailable Datasets:")
        print("=" * 50)
        
        for key, info in datasets.items():
            print(f"\n{key.upper()}: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Molecules: {info['molecules']}")
            print(f"  Expected violations: {info['violations_expected']}")
            print(f"  Library file: {info['library']}")
            print(f"  Reference file: {info['reference']}") 