#!/usr/bin/env python3
"""
Random Forest hERG Classifier Example
=====================================

This example demonstrates how to use the integrated RandomForestHERGClassifier
that leverages the existing project infrastructure for data processing and
molecular feature extraction.

The new approach:
- Uses existing HERGDataProcessor for data loading and SMILES standardization
- Uses existing MolecularFeaturizer for fingerprint computation
- Integrates seamlessly with the chapter3_ml_screening module
- Maintains consistent project structure and coding patterns

Usage:
    python examples/random_forest_example.py
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chapter3_ml_screening.random_forest_model import RandomForestHERGClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Demonstrate the integrated RandomForest hERG classifier."""
    print("="*70)
    print("🧬 INTEGRATED RANDOM FOREST hERG CLASSIFIER EXAMPLE")
    print("="*70)
    print("Using existing project infrastructure for optimal integration")
    print("="*70)
    
    try:
        # Initialize the classifier
        print("\n🔧 Initializing RandomForest hERG Classifier...")
        rf_classifier = RandomForestHERGClassifier(random_seed=42)
        
        # Load and prepare data using existing infrastructure
        print("\n📊 Loading and preparing hERG data...")
        fingerprints, labels = rf_classifier.load_and_prepare_data()
        
        # Quick hyperparameter optimization with smaller grid for demonstration
        print("\n🔍 Running hyperparameter optimization...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt'],
            'bootstrap': [True],
            'class_weight': [None, 'balanced']
        }
        
        optimization_results = rf_classifier.optimize_hyperparameters(
            cv_folds=3, 
            param_grid=param_grid
        )
        
        # Evaluate the model
        print("\n📊 Evaluating model performance...")
        test_metrics = rf_classifier.evaluate_model()
        
        # Compare with baselines
        print("\n⚖️ Comparing with baseline models...")
        comparison_results = rf_classifier.compare_with_baselines()
        
        # Analyze feature importance
        print("\n🔍 Analyzing feature importance...")
        importance_df = rf_classifier.analyze_feature_importance(top_n=10)
        
        # Test predictions on example molecules
        print("\n🧪 Testing predictions on example molecules...")
        example_smiles = [
            "CCO",  # Ethanol
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CC1=CC=CC=C1C(=O)O"  # Toluic acid
        ]
        
        prediction_results = rf_classifier.predict_molecules(example_smiles)
        
        print("\n🔬 Example Predictions:")
        for smiles, pred, prob in zip(
            prediction_results['smiles'],
            prediction_results['predictions'],
            prediction_results['probabilities']
        ):
            status = "🚨 hERG Blocker" if pred == 1 else "✅ Non-blocker"
            print(f"   {smiles}: {status} (P = {prob:.3f})")
        
        # Save the model
        model_path = "artifacts/chapter3/integrated_random_forest_herg_model.pkl"
        rf_classifier.save_model(model_path)
        
        # Get model summary
        summary = rf_classifier.get_model_summary()
        
        # Final results summary
        print("\n" + "="*70)
        print("🏆 INTEGRATED RANDOM FOREST CLASSIFIER - RESULTS SUMMARY")
        print("="*70)
        print(f"✅ Model Training: COMPLETED")
        print(f"🎯 Best CV MCC: {optimization_results['best_score']:.4f}")
        print(f"🎯 Test Set MCC: {test_metrics['matthews_corrcoef']:.4f}")
        print(f"🎯 Test Set Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"🎯 Test Set ROC AUC: {test_metrics['roc_auc']:.4f}")
        print(f"💾 Model saved to: {model_path}")
        print("\n🔧 Model Configuration:")
        print(f"   n_estimators: {summary['n_estimators']}")
        print(f"   max_depth: {summary['max_depth']}")
        print(f"   min_samples_split: {summary['min_samples_split']}")
        print(f"   max_features: {summary['max_features']}")
        print(f"   class_weight: {summary['class_weight']}")
        
        print("\n✨ Key Advantages of Integrated Approach:")
        print("   🏗️  Uses existing HERGDataProcessor for data loading")
        print("   🧬 Uses existing MolecularFeaturizer for fingerprints")
        print("   📁 Maintains consistent project structure")
        print("   🔄 Avoids code duplication")
        print("   🎯 Leverages established standardization pipelines")
        print("   📊 Integrates with existing logging and error handling")
        
        print("\n✅ Example completed successfully!")
        
        return rf_classifier
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    classifier = main() 