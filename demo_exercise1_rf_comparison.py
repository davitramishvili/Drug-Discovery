#!/usr/bin/env python3
"""
Demo for Exercise 1: RandomForestClassifier vs SGDClassifier Comparison
======================================================================

This script demonstrates the implementation of Exercise 1 from Chapter 3,
comparing RandomForestClassifier with SGDClassifier on hERG prediction task.

Usage:
    python demo_exercise1_rf_comparison.py
"""

import sys
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Suppress warnings
warnings.filterwarnings('ignore')

# Import our modules
from chapter3_ml_screening import (
    HERGDataProcessor, MolecularFeaturizer, HERGClassifier,
    ModelEvaluator, VisualizationTools
)


def run_exercise1_comparison():
    """Run Exercise 1: Compare RandomForest with SGDClassifier."""
    
    print("="*70)
    print("EXERCISE 1: RandomForestClassifier vs SGDClassifier")
    print("="*70)
    print()
    
    # Initialize components
    processor = HERGDataProcessor(random_seed=42)
    featurizer = MolecularFeaturizer(radius=2, n_bits=2048)
    classifier = HERGClassifier(random_seed=42)
    evaluator = ModelEvaluator(random_seed=42)
    
    # Step 1: Load and prepare data
    print("Step 1: Loading and preparing hERG dataset...")
    herg_data = processor.load_herg_blockers_data()
    standardized_data = processor.standardize_molecules(herg_data)
    train_data, test_data = processor.split_data(standardized_data)
    
    print(f"✓ Data loaded: {len(train_data)} training, {len(test_data)} test samples")
    
    # Step 2: Generate fingerprints
    print("\nStep 2: Generating molecular fingerprints...")
    train_fingerprints = featurizer.compute_fingerprints_batch(train_data['mol'].tolist())
    train_labels = train_data['Class'].values
    test_fingerprints = featurizer.compute_fingerprints_batch(test_data['mol'].tolist())
    test_labels = test_data['Class'].values
    
    print(f"✓ Fingerprints generated: shape {train_fingerprints.shape}")
    
    # Step 3: Train and optimize RandomForest
    print("\nStep 3: Training RandomForest with Grid Search...")
    print("This may take a few minutes...")
    
    # Define parameter grid for RandomForest
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Perform grid search
    rf_grid_search = classifier.grid_search_random_forest(
        train_fingerprints, train_labels,
        param_grid=rf_param_grid,
        cv=5,
        scoring='matthews_corrcoef'
    )
    
    print(f"\n✓ RandomForest Grid Search completed")
    print(f"  Best parameters: {rf_grid_search.best_params_}")
    print(f"  Best CV MCC: {rf_grid_search.best_score_:.4f}")
    
    # Step 4: Train and optimize SGDClassifier
    print("\nStep 4: Training SGDClassifier with Grid Search...")
    
    # Define parameter grid for SGD
    sgd_param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'l1_ratio': [0.15, 0.5, 0.85]  # For elastic net
    }
    
    # Perform grid search for SGD
    sgd_grid_search = classifier.grid_search_hyperparameters(
        train_fingerprints, train_labels,
        param_grid=sgd_param_grid,
        cv=5,
        scoring='matthews_corrcoef'
    )
    
    print(f"\n✓ SGDClassifier Grid Search completed")
    print(f"  Best parameters: {sgd_grid_search.best_params_}")
    print(f"  Best CV MCC: {sgd_grid_search.best_score_:.4f}")
    
    # Step 5: Compare models on test set
    print("\nStep 5: Evaluating models on test set...")
    
    # Get best models
    best_rf = rf_grid_search.best_estimator_
    best_sgd = sgd_grid_search.best_estimator_
    
    # Evaluate RandomForest
    rf_metrics = evaluator.evaluate_final_model(
        best_rf, test_fingerprints, test_labels,
        model_name="RandomForest_Best"
    )
    
    # Evaluate SGDClassifier
    sgd_metrics = evaluator.evaluate_final_model(
        best_sgd, test_fingerprints, test_labels,
        model_name="SGDClassifier_Best"
    )
    
    # Step 6: Create comparison visualization
    print("\nStep 6: Creating comparison visualizations...")
    
    # Create comparison dataframe
    comparison_data = pd.DataFrame({
        'Model': ['RandomForest', 'SGDClassifier'],
        'Accuracy': [rf_metrics['accuracy'], sgd_metrics['accuracy']],
        'F1_macro': [rf_metrics['f1_macro'], sgd_metrics['f1_macro']],
        'Precision_macro': [rf_metrics['precision_macro'], sgd_metrics['precision_macro']],
        'Recall_macro': [rf_metrics['recall_macro'], sgd_metrics['recall_macro']],
        'Matthews_CC': [rf_metrics['matthews_corrcoef'], sgd_metrics['matthews_corrcoef']]
    })
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('RandomForest vs SGDClassifier Comparison', fontsize=16)
    
    # Plot 1: Bar chart of all metrics
    ax1 = axes[0, 0]
    metrics_df = comparison_data.set_index('Model').T
    metrics_df.plot(kind='bar', ax=ax1)
    ax1.set_title('Model Performance Metrics')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Matthews CC comparison (main metric)
    ax2 = axes[0, 1]
    mcc_data = comparison_data[['Model', 'Matthews_CC']]
    bars = ax2.bar(mcc_data['Model'], mcc_data['Matthews_CC'], 
                    color=['#2ecc71' if x > y else '#e74c3c' 
                           for x, y in zip(mcc_data['Matthews_CC'], 
                                         [mcc_data['Matthews_CC'].min()]*2)])
    ax2.set_title('Matthews Correlation Coefficient (Primary Metric)')
    ax2.set_ylabel('MCC Score')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mcc_data['Matthews_CC']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    # Plot 3: Confusion matrices side by side
    from sklearn.metrics import confusion_matrix
    
    # RF confusion matrix
    ax3 = axes[1, 0]
    rf_cm = rf_metrics['confusion_matrix']
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title('RandomForest Confusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # SGD confusion matrix
    ax4 = axes[1, 1]
    sgd_cm = sgd_metrics['confusion_matrix']
    sns.heatmap(sgd_cm, annot=True, fmt='d', cmap='Oranges', ax=ax4)
    ax4.set_title('SGDClassifier Confusion Matrix')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    
    plt.tight_layout()
    
    # Save figure
    Path('figures/chapter3').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/chapter3/exercise1_model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Comparison plot saved to figures/chapter3/exercise1_model_comparison.png")
    
    # Step 7: Summary and conclusion
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print("\nTest Set Performance:")
    print(comparison_data.to_string(index=False))
    
    print(f"\nConclusion:")
    if rf_metrics['matthews_corrcoef'] > sgd_metrics['matthews_corrcoef']:
        improvement = (rf_metrics['matthews_corrcoef'] - sgd_metrics['matthews_corrcoef']) * 100
        print(f"✓ RandomForest OUTPERFORMS SGDClassifier by {improvement:.1f}% in MCC!")
        print(f"  RandomForest MCC: {rf_metrics['matthews_corrcoef']:.4f}")
        print(f"  SGDClassifier MCC: {sgd_metrics['matthews_corrcoef']:.4f}")
        winner = best_rf
    else:
        improvement = (sgd_metrics['matthews_corrcoef'] - rf_metrics['matthews_corrcoef']) * 100
        print(f"✗ SGDClassifier performs better than RandomForest by {improvement:.1f}% in MCC")
        print(f"  SGDClassifier MCC: {sgd_metrics['matthews_corrcoef']:.4f}")
        print(f"  RandomForest MCC: {rf_metrics['matthews_corrcoef']:.4f}")
        winner = best_sgd
    
    # Save the best model
    Path('artifacts/chapter3').mkdir(parents=True, exist_ok=True)
    classifier.save_model('artifacts/chapter3/exercise1_best_model.pkl', winner)
    print(f"\n✓ Best model saved to artifacts/chapter3/exercise1_best_model.pkl")
    
    return {
        'rf_model': best_rf,
        'sgd_model': best_sgd,
        'rf_metrics': rf_metrics,
        'sgd_metrics': sgd_metrics,
        'comparison': comparison_data,
        'best_model': winner
    }


if __name__ == "__main__":
    results = run_exercise1_comparison()
    print("\n✓ Exercise 1 completed successfully!") 