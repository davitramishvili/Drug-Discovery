#!/usr/bin/env python3
"""
Debug script to test data loading and filtering.
"""

import sys
sys.path.insert(0, 'src')

from src.data_processing.loader import MoleculeLoader
from src.filtering.drug_like import DrugLikeFilter

def test_data_loading():
    """Test data loading."""
    print("Testing data loading...")
    
    # Test library loading
    loader = MoleculeLoader()
    df = loader.load_and_process('data/raw/test_library.sdf')
    
    print(f"Loaded {len(df)} molecules")
    print(f"Columns: {df.columns.tolist()}")
    print("\nSample data:")
    print(df.head())
    
    return df

def test_filtering(df):
    """Test filtering."""
    print("\nTesting filtering...")
    
    filter_obj = DrugLikeFilter(violations_allowed=1)
    
    try:
        filtered_df = filter_obj.filter_dataframe(df)
        print(f"Filtered to {len(filtered_df)} molecules")
        print(f"Filter columns: {[col for col in filtered_df.columns if 'drug' in col or 'lipinski' in col or 'passes' in col]}")
        
        # Test statistics
        stats = filter_obj.get_filter_statistics(df)
        print(f"Filter statistics: {stats}")
        
        return filtered_df
        
    except Exception as e:
        print(f"Error during filtering: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run debug tests."""
    try:
        # Test data loading
        df = test_data_loading()
        
        # Test filtering
        filtered_df = test_filtering(df)
        
        print("\nDebug test completed successfully!")
        
    except Exception as e:
        print(f"Error in debug test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 