#!/usr/bin/env python3
"""
Demo for Exercise 2: Apply hERG Model to Malaria Box Compounds
==============================================================

This script demonstrates the implementation of Exercise 2 from Chapter 3,
applying the best hERG prediction model to top 1000 Malaria Box hits.

Usage:
    python demo_exercise2_malaria_box.py
"""

import sys
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Suppress warnings
warnings.filterwarnings('ignore')

# Import our modules
from chapter3_ml_screening import (
    HERGDataProcessor, MolecularFeaturizer, HERGClassifier,
    ModelEvaluator, VisualizationTools
)
from chapter3_ml_screening.utils import save_molecular_dataframe, load_molecular_dataframe


def load_or_simulate_malaria_box_hits(n_compounds: int = 1000):
    """Load Malaria Box hits from Chapter 2 or simulate them."""
    
    # Check for existing results from Chapter 2
    results_files = [
        'results/top_50_hits.csv',
        'results/similarity_results.csv',
        'examples/diverse_results/top_50_hits.csv'
    ]
    
    for file_path in results_files:
        if Path(file_path).exists():
            print(f"Loading hits from {file_path}...")
            df = pd.read_csv(file_path)
            if 'SMILES' in df.columns or 'smiles' in df.columns:
                smiles_col = 'SMILES' if 'SMILES' in df.columns else 'smiles'
                # Use available data, even if less than n_compounds
                available_compounds = min(len(df), n_compounds)
                return df[smiles_col].head(available_compounds).tolist(), df.head(available_compounds)
    
    # Check for Malaria Box reference files
    malaria_files = list(Path('data/reference').glob('*malaria*.sdf'))
    if malaria_files:
        print(f"Loading compounds from {malaria_files[0]}...")
        suppl = Chem.SDMolSupplier(str(malaria_files[0]))
        compounds = []
        compound_info = []
        
        for i, mol in enumerate(suppl):
            if mol is not None and i < n_compounds:
                smiles = Chem.MolToSmiles(mol)
                compounds.append(smiles)
                
                # Extract properties
                info = {
                    'compound_id': mol.GetProp('_Name') if mol.HasProp('_Name') else f'MB_{i:04d}',
                    'SMILES': smiles,
                    'source': 'Malaria Box'
                }
                
                # Add any other properties
                for prop in mol.GetPropNames():
                    if prop != '_Name':
                        info[prop] = mol.GetProp(prop)
                
                compound_info.append(info)
        
        return compounds, pd.DataFrame(compound_info)
    
    # Simulate compounds if no real data available
    print(f"No real Malaria Box data found. Simulating {n_compounds} compounds for demonstration...")
    
    # Use a diverse set of drug-like SMILES for simulation
    example_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin-like
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine-like
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen-like
        "CC1=CC=C(C=C1)C(C)C(=O)O",  # Similar to ibuprofen
        "COC1=CC=CC=C1OCCN",  # Ether compound
        "CC(C)NCC(COC1=CC=CC2=CC=CC=C12)O",  # Beta blocker-like
        "CN1CCN(CC1)C2=CC=CC=N2",  # Pyridine derivative
        "CC1=CC(=NO1)C2=CC=CC=C2",  # Isoxazole derivative
        "C1CC1CN2C=NC3=C2C=CC=C3",  # Indole derivative
        "CC(C)(C)NCC(COC1=CC=CC2=CC=CC=C12)O"  # Propranolol-like
    ]
    
    # Repeat and modify to get n_compounds
    compounds = []
    compound_info = []
    
    for i in range(n_compounds):
        base_smiles = example_smiles[i % len(example_smiles)]
        # Add some variation
        if i > len(example_smiles):
            # Simple modifications to create diversity
            if i % 3 == 0:
                base_smiles = base_smiles.replace('C', 'CC', 1)  # Add methyl
            elif i % 3 == 1:
                base_smiles = base_smiles.replace('O', 'S', 1)  # O to S
            elif i % 3 == 2:
                base_smiles = base_smiles.replace('N', 'NC', 1)  # Add to N
        
        compounds.append(base_smiles)
        compound_info.append({
            'compound_id': f'MB_{i:04d}',
            'SMILES': base_smiles,
            'similarity_score': np.random.uniform(0.7, 0.95),  # Simulated similarity
            'source': 'Simulated Malaria Box'
        })
    
    return compounds, pd.DataFrame(compound_info)


def run_exercise2_malaria_box_screening():
    """Run Exercise 2: Apply hERG model to Malaria Box compounds."""
    
    print("="*70)
    print("EXERCISE 2: Apply hERG Model to Malaria Box Compounds")
    print("="*70)
    print()
    
    # Initialize components
    processor = HERGDataProcessor(random_seed=42)
    featurizer = MolecularFeaturizer(radius=2, n_bits=2048)
    evaluator = ModelEvaluator(random_seed=42)
    visualizer = VisualizationTools()
    
    # Step 1: Load the best model
    print("Step 1: Loading the best hERG prediction model...")
    
    model_path = 'artifacts/chapter3/exercise1_best_model.pkl'
    if not Path(model_path).exists():
        # If Exercise 1 model doesn't exist, train a quick one
        print("Exercise 1 model not found. Training a model...")
        classifier = HERGClassifier(random_seed=42)
        
        # Load hERG data
        herg_data = processor.load_herg_blockers_data()
        standardized_data = processor.standardize_molecules(herg_data)
        train_data, test_data = processor.split_data(standardized_data)
        
        # Generate fingerprints
        train_fingerprints = featurizer.compute_fingerprints_batch(train_data['mol'].tolist())
        train_labels = train_data['Class'].values
        
        # Train model
        model = classifier.train_sgd_classifier(train_fingerprints, train_labels, 
                                              penalty='l2', alpha=0.01)
        Path('artifacts/chapter3').mkdir(parents=True, exist_ok=True)
        classifier.save_model(model_path, model)
    else:
        model = joblib.load(model_path)
        print(f"✓ Loaded model from {model_path}")
    
    # Step 2: Load Malaria Box compounds
    print("\nStep 2: Loading Malaria Box compounds...")
    smiles_list, compound_info = load_or_simulate_malaria_box_hits(1000)
    print(f"✓ Loaded {len(smiles_list)} compounds")
    
    # Step 3: Process and standardize SMILES
    print("\nStep 3: Processing and standardizing SMILES...")
    valid_mols = []
    valid_indices = []
    
    for i, smiles in enumerate(smiles_list):
        mol = processor.process_smiles(smiles)
        if mol is not None:
            valid_mols.append(mol)
            valid_indices.append(i)
    
    # Filter compound info
    compound_info_clean = compound_info.iloc[valid_indices].reset_index(drop=True)
    print(f"✓ Successfully processed {len(valid_mols)} compounds")
    
    # Step 4: Generate fingerprints
    print("\nStep 4: Generating molecular fingerprints...")
    fingerprints = featurizer.compute_fingerprints_batch(valid_mols)
    print(f"✓ Generated fingerprints: shape {fingerprints.shape}")
    
    # Step 5: Apply model
    print("\nStep 5: Applying hERG model to compounds...")
    results = evaluator.apply_to_compound_library(
        model, fingerprints, compound_info_clean,
        model_name="hERG Classifier"
    )
    
    # Step 6: Analyze results
    print("\nStep 6: Analyzing results...")
    
    results_df = results['results_df']
    safe_compounds = results['safe_compounds']
    blockers = results['blockers']
    
    print(f"\nRESULTS SUMMARY:")
    print(f"Total compounds screened: {results['n_total']}")
    print(f"Predicted hERG blockers: {results['n_blockers']} ({results['percent_blockers']:.1f}%)")
    print(f"Predicted safe compounds: {results['n_safe']} ({results['percent_safe']:.1f}%)")
    
    if 'blocker_probability' in results_df.columns:
        print(f"\nHigh confidence predictions:")
        print(f"  High confidence safe: {results['n_high_conf_safe']}")
        print(f"  High confidence blockers: {results['n_high_conf_blockers']}")
    
    # Step 7: Create visualizations
    print("\nStep 7: Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('hERG Screening Results for Malaria Box Compounds', fontsize=16)
    
    # Plot 1: Pie chart of predictions
    ax1 = axes[0, 0]
    labels = ['Safe (Non-blockers)', 'hERG Blockers']
    sizes = [results['n_safe'], results['n_blockers']]
    colors = ['#2ecc71', '#e74c3c']
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, 
                                       autopct='%1.1f%%', startangle=90)
    ax1.set_title('Classification Distribution')
    
    # Plot 2: Probability distribution (if available)
    ax2 = axes[0, 1]
    if 'blocker_probability' in results_df.columns:
        ax2.hist(results_df['blocker_probability'], bins=50, 
                edgecolor='black', alpha=0.7, color='#3498db')
        ax2.axvline(x=0.5, color='red', linestyle='--', label='Decision threshold')
        ax2.set_xlabel('hERG Blocker Probability')
        ax2.set_ylabel('Number of Compounds')
        ax2.set_title('Probability Distribution')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Probability scores not available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Probability Distribution')
    
    # Plot 3: Confidence distribution
    ax3 = axes[1, 0]
    if 'confidence' in results_df.columns:
        safe_conf = results_df[results_df['prediction'] == 0]['confidence']
        blocker_conf = results_df[results_df['prediction'] == 1]['confidence']
        
        ax3.hist([safe_conf, blocker_conf], bins=30, 
                label=['Safe compounds', 'Blockers'],
                color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax3.set_xlabel('Prediction Confidence')
        ax3.set_ylabel('Number of Compounds')
        ax3.set_title('Confidence Score Distribution')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Confidence scores not available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Confidence Score Distribution')
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    Screening Summary:
    
    • Total compounds: {results['n_total']}
    • Safe compounds: {results['n_safe']} ({results['percent_safe']:.1f}%)
    • hERG blockers: {results['n_blockers']} ({results['percent_blockers']:.1f}%)
    
    Retention after hERG filter: {results['percent_safe']:.1f}%
    
    This means that out of the initial 1000 
    Malaria Box hits, approximately {results['n_safe']} 
    compounds are predicted to be safe from 
    hERG-related cardiotoxicity.
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    Path('figures/chapter3').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/chapter3/exercise2_malaria_box_screening.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to figures/chapter3/exercise2_malaria_box_screening.png")
    
    # Step 8: Save results
    print("\nStep 8: Saving results...")
    
    # Save full results
    Path('artifacts/chapter3').mkdir(parents=True, exist_ok=True)
    results_df.to_csv('artifacts/chapter3/malaria_box_herg_predictions.csv', index=False)
    print("✓ Full results saved to artifacts/chapter3/malaria_box_herg_predictions.csv")
    
    # Save safe compounds separately
    safe_compounds.to_csv('artifacts/chapter3/malaria_box_safe_compounds.csv', index=False)
    print("✓ Safe compounds saved to artifacts/chapter3/malaria_box_safe_compounds.csv")
    
    # Display sample of results
    print("\nSample of screening results:")
    display_cols = ['predicted_label']
    
    # Add available columns
    if 'ID' in results_df.columns:
        display_cols.insert(0, 'ID')
    elif 'compound_id' in results_df.columns:
        display_cols.insert(0, 'compound_id')
    
    if 'NAME' in results_df.columns:
        display_cols.insert(1, 'NAME')
        
    if 'blocker_probability' in results_df.columns:
        display_cols.append('blocker_probability')
    if 'similarity_score' in results_df.columns:
        display_cols.append('similarity_score')
    
    # Only display columns that exist
    display_cols = [col for col in display_cols if col in results_df.columns]
    
    print(results_df[display_cols].head(10).to_string(index=False))
    
    return results


if __name__ == "__main__":
    results = run_exercise2_malaria_box_screening()
    print("\n✓ Exercise 2 completed successfully!")
    if results['n_total'] > 0:
        print(f"\nFinal answer: {results['n_safe']} compounds out of {results['n_total']} are predicted to be safe from hERG blockage.")
    else:
        print("\nNo compounds passed the hERG safety filter.")