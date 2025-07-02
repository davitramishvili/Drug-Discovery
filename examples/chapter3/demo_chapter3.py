#!/usr/bin/env python3
"""
Demo script for Chapter 3: Ligand-based ML Screening
=====================================================

This script demonstrates the complete workflow from Flynn's ML for Drug Discovery Chapter 3:
1. Data loading and preprocessing
2. SMILES standardization
3. Molecular fingerprint generation
4. Linear model training with regularization
5. Hyperparameter optimization
6. Model evaluation and feature analysis
7. Visualization and reporting

Usage:
    python demo_chapter3.py
"""

import sys
import os
import warnings
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    """
    Run the complete Chapter 3 ML screening pipeline demonstration.
    """
    print("=" * 70)
    print("Chapter 3: Ligand-based ML Screening - DEMONSTRATION")
    print("Based on Flynn's ML for Drug Discovery Book")
    print("=" * 70)
    print()
    
    try:
        # Import our Chapter 3 modules
        from chapter3_ml_screening import (
            HERGDataProcessor, MolecularFeaturizer, HERGClassifier,
            ModelEvaluator, VisualizationTools
        )
        
        print("✓ Successfully imported Chapter 3 modules")
        
        # Step 1: Data Processing
        print("\n" + "="*50)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("="*50)
        
        processor = HERGDataProcessor(random_seed=42)
        
        # Load hERG data (will download if not present)
        print("Loading hERG blockers dataset...")
        herg_data = processor.load_herg_blockers_data()
        
        if herg_data is not None:
            print(f"✓ Loaded {len(herg_data)} compounds")
            print(f"  Columns: {list(herg_data.columns)}")
            print(f"  Sample data:\n{herg_data.head(3)}")
        else:
            print("✗ Failed to load data")
            return
        
        # Step 2: SMILES Standardization
        print("\n" + "="*50)
        print("STEP 2: SMILES STANDARDIZATION")
        print("="*50)
        
        print("Standardizing SMILES strings...")
        standardized_data = processor.standardize_molecules(herg_data)
        print(f"✓ Standardized {len(standardized_data)} molecules")
        
        # Split data
        train_data, test_data = processor.split_data(standardized_data)
        print(f"✓ Split into {len(train_data)} training and {len(test_data)} test samples")
        
        # Step 3: Molecular Fingerprints
        print("\n" + "="*50)
        print("STEP 3: MOLECULAR FINGERPRINT GENERATION")
        print("="*50)
        
        featurizer = MolecularFeaturizer(radius=2, n_bits=2048)
        
        print("Computing Morgan fingerprints for training set...")
        train_fingerprints = featurizer.compute_fingerprints_batch(train_data['mol'].tolist())
        train_labels = train_data['Class'].values
        
        print("Computing Morgan fingerprints for test set...")
        test_fingerprints = featurizer.compute_fingerprints_batch(test_data['mol'].tolist())
        test_labels = test_data['Class'].values
        
        print(f"✓ Generated fingerprints: {train_fingerprints.shape}")
        
        # Explore fingerprint features
        fp_stats = featurizer.explore_fingerprint_features(train_fingerprints)
        print(f"✓ Fingerprint sparsity: {fp_stats.get('sparsity', 'N/A'):.3f}")
        
        # Step 4: Machine Learning Models
        print("\n" + "="*50)
        print("STEP 4: MACHINE LEARNING MODEL TRAINING")
        print("="*50)
        
        classifier = HERGClassifier(random_seed=42)
        
        # Train basic SGD classifier
        print("Training SGD classifier...")
        sgd_model = classifier.train_sgd_classifier(train_fingerprints, train_labels)
        print("✓ SGD classifier trained")
        
        # Train baseline dummy classifier
        print("Training baseline dummy classifier...")
        dummy_model = classifier.train_baseline_dummy(train_fingerprints, train_labels)
        dummy_accuracy = dummy_model.score(train_fingerprints, train_labels)
        print(f"✓ Dummy classifier accuracy: {dummy_accuracy:.4f}")
        
        # Cross-validation evaluation
        print("Performing cross-validation...")
        cv_results = classifier.cross_validate_model(sgd_model, train_fingerprints, train_labels, cv=5)
        print(f"✓ Cross-validation completed")
        print(f"  Mean CV accuracy: {cv_results['test_acc'].mean():.4f} ± {cv_results['test_acc'].std():.4f}")
        
        # Step 5: Hyperparameter Optimization
        print("\n" + "="*50)
        print("STEP 5: HYPERPARAMETER OPTIMIZATION")
        print("="*50)
        
        print("Running grid search for hyperparameter optimization...")
        print("This may take a few minutes...")
        
        # Simplified parameter grid for demo
        param_grid = {
            'alpha': [0.001, 0.01, 0.1],
            'penalty': ['l1', 'l2', 'elasticnet']
        }
        
        grid_search = classifier.grid_search_hyperparameters(
            train_fingerprints, train_labels, 
            param_grid=param_grid, cv=3  # Reduced CV for demo speed
        )
        
        print(f"✓ Grid search completed")
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        
        # Step 6: Model Evaluation
        print("\n" + "="*50)
        print("STEP 6: MODEL EVALUATION")
        print("="*50)
        
        evaluator = ModelEvaluator(random_seed=42)
        
        # Evaluate on test set
        print("Evaluating final model on test set...")
        final_metrics = evaluator.evaluate_final_model(
            best_model, test_fingerprints, test_labels,
            model_name="optimized_sgd",
            output_file="artifacts/chapter3/final_evaluation.txt"
        )
        
        print(f"✓ Final model evaluation completed")
        print(f"  Test accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  Test F1 (macro): {final_metrics['f1_macro']:.4f}")
        print(f"  Matthews CC: {final_metrics['matthews_corrcoef']:.4f}")
        
        # Feature importance analysis
        print("\nAnalyzing feature importance...")
        feature_analysis = classifier.analyze_feature_importance(best_model)
        
        if feature_analysis:
            print(f"✓ Feature analysis completed")
            print(f"  Total features: {feature_analysis['n_features']}")
            print(f"  Non-zero weights: {feature_analysis['n_nonzero_weights']}")
            print(f"  Weight range: [{feature_analysis['min_weight']:.4f}, {feature_analysis['max_weight']:.4f}]")
        
        # Step 7: Visualization
        print("\n" + "="*50)
        print("STEP 7: VISUALIZATION AND REPORTING")
        print("="*50)
        
        visualizer = VisualizationTools()
        
        print("Creating visualizations...")
        
        # Plot activity distribution
        activity_fig = visualizer.plot_activity_distribution(herg_data)
        print("✓ Activity distribution plot created")
        
        # Plot fingerprint exploration
        fp_fig = visualizer.explore_fingerprint_features(train_fingerprints)
        print("✓ Fingerprint features plot created")
        
        # Plot cross-validation results
        cv_fig = visualizer.visualize_cv_results(cv_results)
        print("✓ Cross-validation results plot created")
        
        # Plot confusion matrix
        from sklearn.metrics import confusion_matrix
        y_pred_test = best_model.predict(test_fingerprints)
        cm = confusion_matrix(test_labels, y_pred_test)
        cm_fig = visualizer.plot_confusion_matrix(cm, class_names=['Non-blocker', 'Blocker'])
        print("✓ Confusion matrix plot created")
        
        # Save model
        print("\nSaving trained model...")
        classifier.save_model("artifacts/chapter3/herg_classifier_final.pkl", best_model)
        print("✓ Model saved")
        
        # Model summary
        model_summary = classifier.get_model_summary(best_model)
        print(f"\n✓ Model Summary:")
        print(f"  Type: {model_summary['model_type']}")
        print(f"  Classes: {model_summary.get('classes_', 'N/A')}")
        print(f"  Coefficient shape: {model_summary.get('coef_shape', 'N/A')}")
        
        # Step 8: Demo Predictions
        print("\n" + "="*50)
        print("STEP 8: DEMONSTRATION PREDICTIONS")
        print("="*50)
        
        # Example molecules for prediction
        example_smiles = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
        ]
        
        example_names = ["Aspirin", "Caffeine", "Ibuprofen"]
        
        print("Making predictions on example molecules...")
        
        for name, smiles in zip(example_names, example_smiles):
            try:
                # Process SMILES
                mol = processor.process_smiles(smiles)
                if mol is not None:
                    # Generate fingerprint
                    fp = featurizer.compute_fingerprint(mol).reshape(1, -1)
                    
                    # Make prediction
                    pred = best_model.predict(fp)[0]
                    pred_proba = best_model.predict_proba(fp)[0] if hasattr(best_model, 'predict_proba') else None
                    
                    result = "hERG Blocker" if pred == 1 else "Non-blocker"
                    prob_text = f" (confidence: {pred_proba[pred]:.3f})" if pred_proba is not None else ""
                    
                    print(f"  {name}: {result}{prob_text}")
                else:
                    print(f"  {name}: Failed to process SMILES")
            except Exception as e:
                print(f"  {name}: Error - {e}")
        
        print("\n" + "="*70)
        print("CHAPTER 3 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print()
        print("Generated files:")
        print("  - artifacts/chapter3/final_evaluation.txt")
        print("  - artifacts/chapter3/herg_classifier_final.pkl")
        print("  - figures/chapter3/*.svg and *.png")
        print()
        print("Next steps:")
        print("  1. Explore the generated visualizations in figures/chapter3/")
        print("  2. Review the evaluation report in artifacts/chapter3/")
        print("  3. Use the saved model for new predictions")
        print("  4. Integrate with your existing drug discovery pipeline")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install scikit-learn pandas numpy matplotlib seaborn")
        print("  pip install rdkit-pypi  # For molecular functionality")
        
    except Exception as e:
        print(f"✗ Error during execution: {e}")
        logger.exception("Full error details:")
        
    finally:
        print(f"\nDemo completed at: {Path.cwd()}")

if __name__ == "__main__":
    main() 