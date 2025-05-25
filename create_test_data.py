#!/usr/bin/env python3
"""
Script to create test SDF files with valid molecular structures.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import os

def create_test_library():
    """Create a test library SDF file with diverse drug-like molecules."""
    
    # Test molecules with SMILES and properties
    test_molecules = [
        {
            'ID': 'COMPOUND_001',
            'NAME': 'Aspirin',
            'SMILES': 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'ACTIVITY': 'Anti-inflammatory'
        },
        {
            'ID': 'COMPOUND_002', 
            'NAME': 'Caffeine',
            'SMILES': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'ACTIVITY': 'Stimulant'
        },
        {
            'ID': 'COMPOUND_003',
            'NAME': 'Ibuprofen', 
            'SMILES': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
            'ACTIVITY': 'Anti-inflammatory'
        },
        {
            'ID': 'COMPOUND_004',
            'NAME': 'Paracetamol',
            'SMILES': 'CC(=O)NC1=CC=C(C=C1)O',
            'ACTIVITY': 'Analgesic'
        },
        {
            'ID': 'COMPOUND_005',
            'NAME': 'Benzene',
            'SMILES': 'C1=CC=CC=C1',
            'ACTIVITY': 'Solvent'
        }
    ]
    
    # Create molecules and write to SDF
    writer = Chem.SDWriter('data/raw/test_library.sdf')
    
    for mol_data in test_molecules:
        mol = Chem.MolFromSmiles(mol_data['SMILES'])
        if mol is not None:
            # Add 2D coordinates
            AllChem.Compute2DCoords(mol)
            
            # Set properties
            mol.SetProp('ID', mol_data['ID'])
            mol.SetProp('NAME', mol_data['NAME'])
            mol.SetProp('ACTIVITY', mol_data['ACTIVITY'])
            
            writer.write(mol)
    
    writer.close()
    print(f"Created test library with {len(test_molecules)} molecules")

def create_reference_compounds():
    """Create reference antimalarial compounds SDF file."""
    
    # Known antimalarial compounds
    antimalarial_compounds = [
        {
            'ID': 'CHLOROQUINE',
            'NAME': 'Chloroquine',
            'SMILES': 'CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl',
            'ACTIVITY': 'Antimalarial'
        },
        {
            'ID': 'QUININE',
            'NAME': 'Quinine',
            'SMILES': 'C=CC1CC2CCN1CC2C(C3=CC=NC4=CC=C(C=C34)OC)O',
            'ACTIVITY': 'Antimalarial'
        },
        {
            'ID': 'ARTEMISININ',
            'NAME': 'Artemisinin', 
            'SMILES': 'C[C@H]1CC[C@H]2[C@@H](C)C(=O)O[C@H]3O[C@]4(C)CC[C@@H]1[C@]2(C)OO[C@@]34C',
            'ACTIVITY': 'Antimalarial'
        }
    ]
    
    # Create molecules and write to SDF
    writer = Chem.SDWriter('data/reference/malaria_box.sdf')
    
    for mol_data in antimalarial_compounds:
        mol = Chem.MolFromSmiles(mol_data['SMILES'])
        if mol is not None:
            # Add 2D coordinates
            AllChem.Compute2DCoords(mol)
            
            # Set properties
            mol.SetProp('ID', mol_data['ID'])
            mol.SetProp('NAME', mol_data['NAME'])
            mol.SetProp('ACTIVITY', mol_data['ACTIVITY'])
            
            writer.write(mol)
    
    writer.close()
    print(f"Created reference compounds with {len(antimalarial_compounds)} molecules")

def main():
    """Create test data files."""
    print("Creating test data files...")
    
    # Ensure directories exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/reference', exist_ok=True)
    
    # Create test files
    create_test_library()
    create_reference_compounds()
    
    print("Test data files created successfully!")

if __name__ == "__main__":
    main() 