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
    
    # For three-model comparison with Exercise 1 interpretation:
    from demo_chapter3 import run_three_model_comparison
    run_three_model_comparison()
"""

import sys
import os
import warnings
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_three_model_comparison():
    """
    Run comprehensive three-model comparison with Exercise 1 molecular interpretation.
    
    This method implements:
    1. Three-model training: SGD basic, SGD optimized, Random Forest optimized
    2. Comprehensive performance comparison
    3. Feature importance analysis for both linear and tree-based models
    4. Molecular interpretation (Exercise 1 from Chapter 3.6)
    5. Substructure visualization and chemical insights
    """
    print("=" * 80)
    print("CHAPTER 3: THREE-MODEL COMPARISON + EXERCISE 1 INTERPRETATION")
    print("Based on Flynn's ML for Drug Discovery Book")
    print("=" * 80)
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
        
        # Step 4: Three-Model Training
        print("\n" + "="*50)
        print("STEP 4: THREE-MODEL COMPARISON TRAINING")
        print("="*50)
        
        classifier = HERGClassifier(random_seed=42)
        
        # Model 1: Basic SGD classifier
        print("1. Training basic SGD classifier...")
        sgd_basic = classifier.train_sgd_classifier(train_fingerprints, train_labels)
        print("✓ Basic SGD classifier trained")
        
        # Store basic SGD in comparison models
        classifier.models_comparison['sgd_basic'] = sgd_basic
        
        # Train baseline dummy classifier for reference
        print("\nTraining baseline dummy classifier...")
        dummy_model = classifier.train_baseline_dummy(train_fingerprints, train_labels)
        dummy_accuracy = dummy_model.score(train_fingerprints, train_labels)
        print(f"✓ Dummy classifier accuracy: {dummy_accuracy:.4f}")
        
        # Model 2: Optimized SGD with Grid Search
        print("\n2. Training optimized SGD with grid search...")
        print("This may take a few minutes...")
        
        # Simplified parameter grid for demo
        param_grid_sgd = {
            'alpha': [0.001, 0.01, 0.1],
            'penalty': ['l1', 'l2', 'elasticnet']
        }
        
        grid_search_sgd = classifier.grid_search_hyperparameters(
            train_fingerprints, train_labels, 
            param_grid=param_grid_sgd, cv=3
        )
        
        print(f"✓ SGD grid search completed")
        print(f"  Best parameters: {grid_search_sgd.best_params_}")
        print(f"  Best CV score: {grid_search_sgd.best_score_:.4f}")
        
        sgd_optimized = grid_search_sgd.best_estimator_
        classifier.models_comparison['sgd_optimized'] = sgd_optimized
        
        # Model 3: Random Forest
        print("\n3. Training Random Forest classifier...")
        rf_basic = classifier.train_random_forest(train_fingerprints, train_labels)
        print("✓ Basic Random Forest trained")
        
        # Random Forest with Grid Search
        print("\n4. Training optimized Random Forest with grid search...")
        print("This may take several minutes...")
        
        # Simplified RF parameter grid for demo
        param_grid_rf = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']
        }
        
        grid_search_rf = classifier.grid_search_random_forest(
            train_fingerprints, train_labels,
            param_grid=param_grid_rf, cv=3
        )
        
        print(f"✓ Random Forest grid search completed")
        print(f"  Best parameters: {grid_search_rf.best_params_}")
        print(f"  Best CV score: {grid_search_rf.best_score_:.4f}")
        
        # Step 5: Comprehensive Model Evaluation & Comparison
        print("\n" + "="*50)
        print("STEP 5: COMPREHENSIVE MODEL EVALUATION & COMPARISON")
        print("="*50)
        
        evaluator = ModelEvaluator(random_seed=42)
        
        # Compare all models on test set
        print("Comparing all models on test set...")
        comparison_results = classifier.compare_models(test_fingerprints, test_labels)
        
        print("\n" + "="*80)
        print("THREE-MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(comparison_results.to_string(index=False))
        print("="*80)
        
        # Find best model by Matthews CC
        best_mcc_idx = comparison_results['Matthews_CC'].idxmax()
        best_model_name = comparison_results.iloc[best_mcc_idx]['Model']
        best_mcc_score = comparison_results.iloc[best_mcc_idx]['Matthews_CC']
        
        print(f"\n🏆 BEST MODEL BY MATTHEWS CORRELATION COEFFICIENT:")
        print(f"   Model: {best_model_name}")
        print(f"   Matthews CC: {best_mcc_score:.4f}")
        
        # Save comprehensive evaluation report
        print(f"\nSaving comprehensive evaluation report...")
        with open("artifacts/chapter3/models_comparison.txt", "w") as f:
            f.write("CHAPTER 3: THREE-MODEL COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write("Models Compared:\n")
            f.write("1. SGD Basic - Original Stochastic Gradient Descent\n")
            f.write("2. SGD Optimized - Grid Search Hyperparameter Tuned SGD\n")
            f.write("3. Random Forest - Tree-based ensemble method\n")
            f.write("4. Random Forest Optimized - Grid Search Tuned RF\n\n")
            f.write("PERFORMANCE COMPARISON:\n")
            f.write("-" * 30 + "\n")
            f.write(comparison_results.to_string(index=False))
            f.write(f"\n\nBEST MODEL: {best_model_name} (Matthews CC: {best_mcc_score:.4f})")
        
        print("✓ Comprehensive comparison report saved")
        
        # Step 6: Feature Importance Analysis
        print(f"\n" + "="*50)
        print("STEP 6: FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Analyze SGD model (linear)
        print("\n1. SGD Model Feature Importance (Linear Coefficients):")
        sgd_analysis = classifier.analyze_feature_importance(classifier.models_comparison['sgd_optimized'])
        if sgd_analysis:
            print(f"   Model type: {sgd_analysis.get('model_type', 'unknown')}")
            print(f"   Total features: {sgd_analysis['n_features']}")
            print(f"   Non-zero weights: {sgd_analysis['n_nonzero_weights']}")
            print(f"   Weight range: [{sgd_analysis['min_weight']:.4f}, {sgd_analysis['max_weight']:.4f}]")
        
        # Analyze Random Forest model (tree-based)
        print("\n2. Random Forest Feature Importance (Gini-based):")
        rf_analysis = None
        if 'random_forest_optimized' in classifier.models_comparison:
            rf_analysis = classifier.analyze_feature_importance(classifier.models_comparison['random_forest_optimized'])
            if rf_analysis:
                print(f"   Model type: {rf_analysis.get('model_type', 'unknown')}")
                print(f"   Total features: {rf_analysis['n_features']}")
                print(f"   Non-zero importances: {rf_analysis['n_nonzero_importances']}")
                print(f"   Importance range: [{rf_analysis['min_importance']:.4f}, {rf_analysis['max_importance']:.4f}]")
        
        print("✓ Feature importance analysis completed for both model types")
        
        # Step 7: Molecular Interpretation (Exercise 1 Component)
        print(f"\n" + "="*50)
        print("STEP 7: MOLECULAR INTERPRETATION - EXERCISE 1")
        print("="*50)
        
        # Interpret SGD model features
        print("\n1. Molecular Interpretation of SGD Features:")
        sgd_interpretation = None
        if sgd_analysis:
            sgd_interpretation = featurizer.interpret_feature_importance(
                sgd_analysis, train_data['mol'].tolist(), train_fingerprints, top_n=10
            )
            
            print(f"   ✓ Interpreted {len(sgd_interpretation['top_features'])} key molecular features")
            
            # Show molecular insights
            for insight in sgd_interpretation['molecular_insights']:
                print(f"   • {insight}")
            
            # Show top 3 most important features
            print(f"\n   Top 3 Most Important Molecular Features (SGD):")
            for feature in sgd_interpretation['top_features'][:3]:
                effect = feature['effect']
                frequency = feature['frequency_percent']
                weight = feature.get('weight', 0)
                print(f"   Rank {feature['rank']}: Bit {feature['bit_index']} - {effect}")
                print(f"     Weight: {weight:+.4f}, Found in {frequency:.1f}% of molecules")
        
        # Interpret Random Forest model features
        print("\n2. Molecular Interpretation of Random Forest Features:")
        rf_interpretation = None
        if 'random_forest_optimized' in classifier.models_comparison and rf_analysis:
            rf_interpretation = featurizer.interpret_feature_importance(
                rf_analysis, train_data['mol'].tolist(), train_fingerprints, top_n=10
            )
            
            print(f"   ✓ Interpreted {len(rf_interpretation['top_features'])} key molecular features")
            
            # Show molecular insights
            for insight in rf_interpretation['molecular_insights']:
                print(f"   • {insight}")
            
            # Show top 3 most important features
            print(f"\n   Top 3 Most Important Molecular Features (Random Forest):")
            for feature in rf_interpretation['top_features'][:3]:
                effect = feature['effect']
                frequency = feature['frequency_percent']
                importance = feature.get('importance', 0)
                print(f"   Rank {feature['rank']}: Bit {feature['bit_index']} - {effect}")
                print(f"     Importance: {importance:.4f}, Found in {frequency:.1f}% of molecules")
        
        # Create molecular substructure visualizations
        print("\n3. Creating Molecular Substructure Visualizations:")
        
        if sgd_interpretation:
            print("   Creating SGD molecular visualizations...")
            sgd_viz_files = featurizer.visualize_important_substructures(
                sgd_interpretation, train_data['mol'].tolist(), 
                save_dir="figures/chapter3/sgd_substructures"
            )
            print(f"   ✓ Created {len(sgd_viz_files)} SGD substructure visualizations")
        
        if rf_interpretation:
            print("   Creating Random Forest molecular visualizations...")
            rf_viz_files = featurizer.visualize_important_substructures(
                rf_interpretation, train_data['mol'].tolist(),
                save_dir="figures/chapter3/rf_substructures"
            )
            print(f"   ✓ Created {len(rf_viz_files)} Random Forest substructure visualizations")
        
        # Save molecular interpretation reports
        print("\n4. Saving Molecular Interpretation Reports:")
        
        # Save SGD interpretation
        if sgd_interpretation:
            import json
            with open("artifacts/chapter3/sgd_molecular_interpretation.json", "w") as f:
                # Convert numpy types to native Python types for JSON serialization
                sgd_clean = {}
                for key, value in sgd_interpretation.items():
                    if key == 'top_features':
                        sgd_clean[key] = value  # Already converted to native types
                    elif key == 'substructure_examples':
                        sgd_clean[key] = {str(k): v for k, v in value.items()}
                    else:
                        sgd_clean[key] = value
                json.dump(sgd_clean, f, indent=2)
            print("   ✓ SGD molecular interpretation saved")
        
        # Save RF interpretation  
        if rf_interpretation:
            with open("artifacts/chapter3/rf_molecular_interpretation.json", "w") as f:
                # Convert numpy types to native Python types for JSON serialization
                rf_clean = {}
                for key, value in rf_interpretation.items():
                    if key == 'top_features':
                        rf_clean[key] = value  # Already converted to native types
                    elif key == 'substructure_examples':
                        rf_clean[key] = {str(k): v for k, v in value.items()}
                    else:
                        rf_clean[key] = value
                json.dump(rf_clean, f, indent=2)
            print("   ✓ Random Forest molecular interpretation saved")
        
        print("\n✓ Exercise 1 molecular interpretation components completed!")
        print("   This completes the missing parts from Chapter 3.6 Exercise 1")
        
        # Final Summary
        print("\n" + "="*80)
        print("THREE-MODEL COMPARISON + EXERCISE 1 COMPLETED!")
        print("="*80)
        print()
        print("🎯 MODELS COMPARED:")
        print("   1. SGD Basic - Original Stochastic Gradient Descent")
        print("   2. SGD Optimized - Grid Search Hyperparameter Tuned")
        print("   3. Random Forest Basic - Tree-based ensemble")
        print("   4. Random Forest Optimized - Grid Search Tuned RF")
        print()
        print("📊 PERFORMANCE ANALYSIS:")
        print(f"   Best Model: {best_model_name}")
        print(f"   Best Matthews CC: {best_mcc_score:.4f}")
        print("   (See detailed comparison in artifacts/chapter3/models_comparison.txt)")
        print()
        print("🧬 MOLECULAR INTERPRETATION (Exercise 1):")
        print("   ✓ Feature importance → molecular substructures")
        print("   ✓ Chemical insights for both linear and tree-based models")
        print("   ✓ Molecular visualizations of important features")
        print("   ✓ JSON reports with detailed interpretations")
        print()
        print("📁 Generated Files:")
        print("   Model Comparison:")
        print("     - artifacts/chapter3/models_comparison.txt")
        print()
        print("   Molecular Interpretation (Exercise 1):")
        print("     - artifacts/chapter3/sgd_molecular_interpretation.json")
        print("     - artifacts/chapter3/rf_molecular_interpretation.json")
        print("     - figures/chapter3/sgd_substructures/*.svg")
        print("     - figures/chapter3/rf_substructures/*.svg")
        print()
        print("🚀 Key Achievements:")
        print("   ✅ Comprehensive three-model comparison")
        print("   ✅ Both linear (SGD) and tree-based (RF) algorithms")
        print("   ✅ Complete Exercise 1 molecular interpretation")
        print("   ✅ Molecular substructure analysis and visualization")
        print("   ✅ Feature importance → chemical insights mapping")
        print()
        print("📈 Next Steps:")
        print("   1. Compare model performance using Matthews CC scores")
        print("   2. Explore molecular substructure visualizations")
        print("   3. Review JSON interpretation reports for chemical insights")
        print("   4. Use best model for hERG screening in drug discovery")
        print("   5. Integrate molecular interpretation for compound optimization")
        
        return {
            'comparison_results': comparison_results,
            'best_model_name': best_model_name,
            'best_mcc_score': best_mcc_score,
            'sgd_interpretation': sgd_interpretation,
            'rf_interpretation': rf_interpretation
        }
        
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

def run_exercise1_interpretation(featurizer, feature_analysis, train_molecules, train_fingerprints, model_name="model"):
    """
    Run Exercise 1 molecular interpretation for a trained model.
    
    This method implements the missing Exercise 1 component from Chapter 3.6:
    - Feature importance → molecular substructure interpretation
    - Molecular fragment visualization
    - Chemical insights generation
    
    Parameters:
        featurizer: MolecularFeaturizer instance
        feature_analysis: Output from analyze_feature_importance()
        train_molecules: List of RDKit molecule objects
        train_fingerprints: Training fingerprint array
        model_name: Name of the model for file naming
    """
    print(f"\n" + "="*50)
    print(f"EXERCISE 1: MOLECULAR INTERPRETATION - {model_name.upper()}")
    print("="*50)
    
    if not feature_analysis:
        print(f"   ✗ No feature analysis available for {model_name}")
        return None
    
    # Interpret molecular features
    print(f"1. Interpreting {model_name} molecular features...")
    interpretation = featurizer.interpret_feature_importance(
        feature_analysis, train_molecules, train_fingerprints, top_n=10
    )
    
    if not interpretation or len(interpretation['top_features']) == 0:
        print(f"   ✗ Failed to interpret {model_name} features")
        return None
    
    print(f"   ✓ Interpreted {len(interpretation['top_features'])} key molecular features")
    
    # Show molecular insights
    print(f"\n2. Chemical Insights for {model_name}:")
    for insight in interpretation['molecular_insights']:
        print(f"   • {insight}")
    
    # Show top 3 most important features
    print(f"\n3. Top 3 Most Important Molecular Features:")
    for feature in interpretation['top_features'][:3]:
        effect = feature['effect']
        frequency = feature['frequency_percent']
        
        if feature_analysis.get('model_type') == 'linear':
            weight = feature.get('weight', 0)
            print(f"   Rank {feature['rank']}: Bit {feature['bit_index']} - {effect}")
            print(f"     Weight: {weight:+.4f}, Found in {frequency:.1f}% of molecules")
        else:
            importance = feature.get('importance', 0)
            print(f"   Rank {feature['rank']}: Bit {feature['bit_index']} - {effect}")
            print(f"     Importance: {importance:.4f}, Found in {frequency:.1f}% of molecules")
    
    # Create molecular visualizations
    print(f"\n4. Creating molecular substructure visualizations...")
    viz_files = featurizer.visualize_important_substructures(
        interpretation, train_molecules,
        save_dir=f"figures/chapter3/{model_name}_substructures"
    )
    print(f"   ✓ Created {len(viz_files)} molecular visualizations")
    
    # Save interpretation report
    print(f"5. Saving molecular interpretation report...")
    import json
    import numpy as np
    
    def convert_numpy_types(obj):
        """Recursively convert numpy types and RDKit objects to native Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__class__') and 'rdkit' in str(obj.__class__):
            # Skip RDKit objects (Mol, etc.)
            return f"<RDKit {obj.__class__.__name__}>"
        elif isinstance(obj, dict):
            return {str(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    with open(f"artifacts/chapter3/{model_name}_molecular_interpretation.json", "w") as f:
        # Convert all numpy types to native Python types for JSON serialization
        clean_interpretation = convert_numpy_types(interpretation)
        json.dump(clean_interpretation, f, indent=2)
    print(f"   ✓ {model_name} molecular interpretation saved")
    
    print(f"\n✓ Exercise 1 molecular interpretation completed for {model_name}!")
    return interpretation

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
            if feature_analysis.get('model_type') == 'linear':
                print(f"  Non-zero weights: {feature_analysis['n_nonzero_weights']}")
                print(f"  Weight range: [{feature_analysis['min_weight']:.4f}, {feature_analysis['max_weight']:.4f}]")
            else:
                print(f"  Non-zero importances: {feature_analysis.get('n_nonzero_importances', 'N/A')}")
                print(f"  Importance range: [{feature_analysis.get('min_importance', 0):.4f}, {feature_analysis.get('max_importance', 0):.4f}]")
        
        # EXERCISE 1: Molecular Interpretation (NEW)
        run_exercise1_interpretation(
            featurizer, feature_analysis, train_data['mol'].tolist(), 
            train_fingerprints, model_name="optimized_sgd"
        )
        
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
        
        print("DEMO COMPLETED!")
        print("="*70)
        print(f"Final model performance: {final_metrics['matthews_corrcoef']:.4f} Matthews CC")
        print(f"Model saved to: artifacts/chapter3/herg_classifier_final.pkl")
        print("Visualizations and reports saved to figures/chapter3/ and artifacts/chapter3/")
        
        print("\n📊 EXERCISE 1 COMPLETED!")
        print("   ✓ Molecular interpretation for feature importance")
        print("   ✓ Chemical insights from statistical models")
        print("   ✓ Molecular substructure visualizations")
        print("   ✓ JSON reports with detailed interpretations")
        
        print("\n🚀 NEXT STEPS:")
        print("   • Review molecular interpretation files in artifacts/chapter3/")
        print("   • Explore substructure visualizations in figures/chapter3/")
        print("   • Run three-model comparison: run_three_model_comparison()")
        print("   • Use model for hERG screening in drug discovery")
        
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