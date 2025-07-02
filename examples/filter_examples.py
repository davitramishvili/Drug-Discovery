"""
Example script demonstrating how to use the structural filter utility functions.
"""

import pandas as pd
from rdkit import Chem
from src.filtering.filter_utils import (
    apply_pains_filter,
    apply_brenk_filter,
    apply_nih_filter,
    apply_custom_filter_combination,
    analyze_filter_results
)

def load_sdf_as_dataframe(sdf_path: str) -> pd.DataFrame:
    """Load SDF file into a pandas DataFrame with RDKit molecules."""
    molecules = []
    supplier = Chem.SDMolSupplier(sdf_path)
    
    for mol in supplier:
        if mol is not None:
            molecules.append({
                'mol': mol,
                'smiles': Chem.MolToSmiles(mol)
            })
    
    return pd.DataFrame(molecules)

def main():
    # Load your compound library
    df = load_sdf_as_dataframe('../data/raw/Specs.sdf.gz')
    print(f"Loaded {len(df)} molecules")
    
    # Example 1: Apply only PAINS filter
    print("\nApplying PAINS filter...")
    pains_filtered_df, pains_stats = apply_pains_filter(df)
    print(f"PAINS filter results:")
    print(f"- Total molecules: {pains_stats['total_molecules']}")
    print(f"- Remaining molecules: {pains_stats['remaining_molecules']}")
    print(f"- Pass rate: {pains_stats['pass_rate']:.1f}%")
    
    # Example 2: Apply only BRENK filter
    print("\nApplying BRENK filter...")
    brenk_filtered_df, brenk_stats = apply_brenk_filter(df)
    print(f"BRENK filter results:")
    print(f"- Total molecules: {brenk_stats['total_molecules']}")
    print(f"- Remaining molecules: {brenk_stats['remaining_molecules']}")
    print(f"- Pass rate: {brenk_stats['pass_rate']:.1f}%")
    
    # Example 3: Apply only NIH filter
    print("\nApplying NIH filter...")
    nih_filtered_df, nih_stats = apply_nih_filter(df)
    print(f"NIH filter results:")
    print(f"- Total molecules: {nih_stats['total_molecules']}")
    print(f"- Remaining molecules: {nih_stats['remaining_molecules']}")
    print(f"- Pass rate: {nih_stats['pass_rate']:.1f}%")
    
    # Example 4: Apply custom combination of filters
    print("\nApplying custom filter combination (PAINS + NIH)...")
    custom_filtered_df, custom_stats = apply_custom_filter_combination(
        df, use_pains=True, use_brenk=False, use_nih=True
    )
    print(f"Custom filter combination results:")
    print(f"- Total molecules: {custom_stats['total_molecules']}")
    print(f"- Remaining molecules: {custom_stats['remaining_molecules']}")
    print(f"- Pass rate: {custom_stats['pass_rate']:.1f}%")
    print(f"- PAINS pass rate: {custom_stats.get('pains_pass_rate', 0):.1f}%")
    print(f"- NIH pass rate: {custom_stats.get('nih_pass_rate', 0):.1f}%")
    
    # Example 5: Analyze filter results in detail
    print("\nAnalyzing filter results...")
    analysis = analyze_filter_results(custom_filtered_df)
    
    # Print most common alerts
    if 'pattern_analysis' in analysis and 'pains' in analysis['pattern_analysis']:
        print("\nMost common PAINS alerts:")
        for pattern, count in analysis['pattern_analysis']['pains']['most_common']:
            print(f"- {pattern}: {count} occurrences")
            
    if 'filter_correlations' in analysis and analysis['filter_correlations']:
        print("\nFilter correlations:")
        for filter1, corrs in analysis['filter_correlations'].items():
            for filter2, corr in corrs.items():
                if filter1 < filter2:  # Avoid printing both A->B and B->A
                    print(f"- {filter1} vs {filter2}: {corr:.2f}")

if __name__ == '__main__':
    main() 