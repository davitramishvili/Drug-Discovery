#!/usr/bin/env python3
"""
Script to convert malaria box compounds from CSV to SDF format.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path

def convert_csv_to_sdf(csv_path, sdf_path):
    """Convert CSV file containing SMILES to SDF format."""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Create writer for SDF file
    writer = Chem.SDWriter(str(sdf_path))
    
    # Convert each SMILES to molecule and write to SDF
    for idx, row in df.iterrows():
        try:
            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(row['Smiles'])
            if mol is None:
                print(f"Failed to create molecule for compound {row['HEOS_COMPOUND_ID']}")
                continue
                
            # Add hydrogens and generate 3D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            
            # Set properties
            mol.SetProp("_Name", row['HEOS_COMPOUND_ID'])
            mol.SetProp("Batch_No", row['Batch_No_March2012'])
            mol.SetProp("EC50_nM", str(row['EC50_nM']))
            mol.SetProp("Set", row['Set'])
            mol.SetProp("Molecular_Weight", str(row['Molecular_Weight']))
            mol.SetProp("ALogP", str(row['ALogP']))
            
            # Write molecule to SDF
            writer.write(mol)
            
        except Exception as e:
            print(f"Error processing compound {row['HEOS_COMPOUND_ID']}: {str(e)}")
    
    writer.close()
    print(f"Conversion completed. Output saved to {sdf_path}")

if __name__ == "__main__":
    # Set paths
    data_dir = Path("data")
    csv_path = data_dir / "reference" / "malaria_box_400_compounds.csv"
    sdf_path = data_dir / "reference" / "malaria_box_400.sdf"
    
    # Convert file
    convert_csv_to_sdf(csv_path, sdf_path) 