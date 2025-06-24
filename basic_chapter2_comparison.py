#!/usr/bin/env python3
"""
Basic Chapter 2 Drug Discovery Demo
Demonstrating simple filtering concepts vs advanced pipeline
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def basic_lipinski_filter(smiles_list):
    """
    BASIC Chapter 2 Implementation - Simple Lipinski filtering
    This is what you might start with before implementing advanced exercises
    """
    drug_like = []
    non_drug_like = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
            
        # Basic Lipinski Rule of 5 implementation
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        # Simple boolean check - no flexibility or customization
        if (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10):
            drug_like.append({
                'SMILES': smiles,
                'MW': mw,
                'LogP': logp,
                'HBD': hbd,
                'HBA': hba,
                'Status': 'Drug-like'
            })
        else:
            non_drug_like.append({
                'SMILES': smiles,
                'MW': mw,
                'LogP': logp,
                'HBD': hbd,
                'HBA': hba,
                'Status': 'Non-drug-like'
            })
    
    return drug_like, non_drug_like

def basic_similarity_search(target_smiles, library_smiles):
    """
    BASIC Chapter 2 Implementation - Simple similarity without advanced fingerprints
    Just basic Tanimoto similarity with default fingerprints
    """
    from rdkit import DataStructs
    from rdkit.Chem import rdMolDescriptors
    
    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is None:
        return []
    
    target_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(target_mol, 2)
    similarities = []
    
    for smiles in library_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
            
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
        similarity = DataStructs.TanimotoSimilarity(target_fp, fp)
        
        similarities.append({
            'SMILES': smiles,
            'Similarity': similarity
        })
    
    # Simple sorting - no threshold or advanced filtering
    similarities.sort(key=lambda x: x['Similarity'], reverse=True)
    return similarities

def create_basic_plots(drug_like, non_drug_like, output_dir):
    """
    BASIC Chapter 2 Visualization - Simple plots without advanced styling
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Simple pie chart
    plt.figure(figsize=(8, 6))
    labels = ['Drug-like', 'Non-drug-like']
    sizes = [len(drug_like), len(non_drug_like)]
    colors = ['lightblue', 'lightcoral']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Basic Drug-likeness Filter Results')
    plt.axis('equal')
    plt.savefig(f'{output_dir}/basic_druglikeness_pie.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Simple scatter plot
    all_data = drug_like + non_drug_like
    if all_data:
        df = pd.DataFrame(all_data)
        
        plt.figure(figsize=(8, 6))
        drug_like_df = df[df['Status'] == 'Drug-like']
        non_drug_like_df = df[df['Status'] == 'Non-drug-like']
        
        plt.scatter(drug_like_df['MW'], drug_like_df['LogP'], 
                   c='blue', label='Drug-like', alpha=0.7)
        plt.scatter(non_drug_like_df['MW'], non_drug_like_df['LogP'], 
                   c='red', label='Non-drug-like', alpha=0.7)
        
        plt.xlabel('Molecular Weight')
        plt.ylabel('LogP')
        plt.title('Basic MW vs LogP Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/basic_mw_logp_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()

def run_basic_demo():
    """
    Run the basic Chapter 2 demonstration
    """
    print("="*60)
    print("BASIC CHAPTER 2 DRUG DISCOVERY DEMO")
    print("Simple implementation without advanced exercises")
    print("="*60)
    
    # Create simple synthetic dataset for basic demo
    # These represent typical molecules you might encounter in Chapter 2
    smiles_list = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin - drug-like
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine - drug-like
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen - drug-like
        "CC(=O)NC1=CC=C(C=C1)O",  # Paracetamol - drug-like
        "CCCCCCCCCCCCCCCCCC(=O)O",  # Stearic acid - too lipophilic
        "C1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3C4=CC=CC=C4",  # Too large/complex
        "O=C(O)CC(O)C(O)C(O)CO",  # Sugar-like - too hydrophilic
        "NCCCCCCCCCCCCCCCCN",  # Long chain diamine - poor drug-like
        "CC(C)(C)C1=CC=C(C=C1)O",  # Simple phenol - borderline
        "CCCCCCCCCCCCCCCCCCCC(=O)O"  # Very long fatty acid - non-drug-like
    ]
    
    print(f"📊 Processing {len(smiles_list)} molecules...")
    
    # Basic filtering
    drug_like, non_drug_like = basic_lipinski_filter(smiles_list)
    
    print(f"✅ Drug-like molecules: {len(drug_like)}")
    print(f"❌ Non-drug-like molecules: {len(non_drug_like)}")
    print(f"📈 Drug-likeness rate: {len(drug_like)/(len(drug_like)+len(non_drug_like))*100:.1f}%")
    
    # Basic similarity search (if we have drug-like molecules)
    if drug_like:
        target_smiles = drug_like[0]['SMILES']  # Use first drug-like as target
        library_smiles = [mol['SMILES'] for mol in drug_like[1:]]  # Search in rest
        
        if library_smiles:
            similarities = basic_similarity_search(target_smiles, library_smiles)
            print(f"🔍 Found {len(similarities)} similarity matches")
            if similarities:
                print(f"📊 Best similarity: {similarities[0]['Similarity']:.3f}")
    
    # Create basic plots
    output_dir = "comparison_basic"
    create_basic_plots(drug_like, non_drug_like, output_dir)
    print(f"📁 Basic plots saved to: {output_dir}")
    
    # Save basic results
    all_results = drug_like + non_drug_like
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{output_dir}/basic_results.csv", index=False)
        print(f"💾 Basic results saved to: {output_dir}/basic_results.csv")
    
    print("\n" + "="*60)
    print("BASIC DEMO COMPLETED")
    print("Compare this with your advanced implementation!")
    print("="*60)

if __name__ == "__main__":
    run_basic_demo() 