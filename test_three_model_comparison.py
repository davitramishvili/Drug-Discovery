#!/usr/bin/env python3
"""
Test script for Three-Model Comparison and Exercise 1 Interpretation
===================================================================

This script runs the comprehensive three-model comparison with molecular 
interpretation that completes Exercise 1 from Chapter 3.6.

Usage:
    python test_three_model_comparison.py
"""

from demo_chapter3 import run_three_model_comparison

def main():
    """Run the three-model comparison test."""
    print("Starting Three-Model Comparison with Exercise 1 Interpretation...")
    print("This will compare SGD vs Random Forest models with molecular insights.")
    print()
    
    # Run the comprehensive comparison
    results = run_three_model_comparison()
    
    if results:
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best Model: {results['best_model_name']}")
        print(f"Best Matthews CC: {results['best_mcc_score']:.4f}")
        print()
        print("Check the following directories for results:")
        print("- artifacts/chapter3/ for model comparison reports")
        print("- figures/chapter3/ for molecular visualizations")
    else:
        print("❌ Test failed - check error messages above")

if __name__ == "__main__":
    main() 