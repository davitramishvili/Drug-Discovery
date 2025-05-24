"""
Module for calculating molecular descriptors.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from typing import Dict, Optional

def calculate_lipinski_descriptors(mol: Chem.Mol) -> Optional[Dict[str, float]]:
    """
    Calculate Lipinski's Rule of Five descriptors for a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary of descriptor values or None if calculation fails
    """
    try:
        descriptors = {
            'MW': Descriptors.ExactMolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol)
        }
        return descriptors
    except Exception as e:
        print(f"Descriptor calculation failed: {str(e)}")
        return None 