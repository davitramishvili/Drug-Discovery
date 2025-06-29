# Chapter 3: Ligand-based Screening - Machine Learning

This directory contains a complete implementation of **Chapter 3** from Flynn's "Machine Learning for Drug Discovery" book, focusing on ligand-based virtual screening using machine learning techniques.

## Overview

Chapter 3 implements an end-to-end machine learning pipeline for predicting **hERG channel cardiotoxicity** using molecular fingerprints and linear models. This is a critical safety assessment in drug discovery, as hERG channel blocking can cause dangerous cardiac arrhythmias.

### Key Features
- Data acquisition, curation, and quality assessment
- SMILES standardization and molecular preprocessing  
- Morgan (ECFP) fingerprint generation for ML features
- Linear model training with regularization techniques
- Hyperparameter optimization (Grid Search & Random Search)
- Model evaluation and performance analysis
- Feature importance and interpretability
- Model deployment and persistence

## Project Structure

```
src/chapter3_ml_screening/
├── __init__.py                 # Module interface and exports
├── data_processing.py          # Data loading and SMILES standardization
├── molecular_features.py       # Fingerprint generation and transformers
├── ml_models.py               # SGD classifiers and hyperparameter tuning
├── evaluation.py              # Model assessment and metrics
├── visualization.py           # Plotting and data exploration
└── utils.py                   # Utility functions and helpers

data/chapter3/                 # Dataset storage
├── hERG_blockers.xlsx         # Main dataset (auto-downloaded)

artifacts/chapter3/            # Model outputs and reports
├── final_evaluation.txt       # Model performance report
├── herg_classifier_final.pkl  # Saved trained model

figures/chapter3/              # Generated visualizations
├── distribution_pic50.svg     # Activity distribution plots
├── fingerprint_eda.svg        # Fingerprint analysis
├── cv_metrics_comparison.svg  # Cross-validation results
└── confusion_matrix.svg       # Model performance visualization
```

## Installation

### Prerequisites
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
pip install rdkit-pypi  # For molecular functionality
```

## Usage

### Running the Demo
```bash
python demo_chapter3.py
```

The demo executes the complete Chapter 3 pipeline:
1. **Data Loading**: Downloads hERG dataset (587 compounds)
2. **Preprocessing**: SMILES standardization and validation
3. **Feature Generation**: Morgan fingerprints (radius=2, 2048 bits)
4. **Model Training**: SGD classifier with cross-validation
5. **Optimization**: Grid search for best hyperparameters
6. **Evaluation**: Comprehensive performance assessment
7. **Visualization**: Generate plots and analysis charts
8. **Deployment**: Save trained model for production use

## Technical Implementation

### Core Components

#### 1. HERGDataProcessor
- **Purpose**: Data loading and molecular standardization
- **Key Methods**:
  - `load_herg_blockers_data()`: Download/load hERG dataset
  - `process_smiles()`: Standardize SMILES with RDKit
  - `split_data()`: Train/test splitting with predefined splits

#### 2. MolecularFeaturizer  
- **Purpose**: Convert molecules to ML-ready features
- **Key Methods**:
  - `compute_fingerprint()`: Generate Morgan fingerprints
  - `compute_fingerprints_batch()`: Batch processing for efficiency
  - `explore_fingerprint_features()`: Feature distribution analysis

#### 3. HERGClassifier
- **Purpose**: Machine learning model training and optimization
- **Key Methods**:
  - `train_sgd_classifier()`: Basic SGD training
  - `grid_search_hyperparameters()`: Systematic parameter search
  - `randomized_search_hyperparameters()`: Efficient random search
  - `analyze_feature_importance()`: Interpretability analysis

#### 4. ModelEvaluator
- **Purpose**: Comprehensive model assessment
- **Key Methods**:
  - `evaluate_predictions()`: Calculate multiple metrics
  - `evaluate_final_model()`: Test set evaluation with reporting
  - `benchmark_against_baseline()`: Compare vs dummy classifier

#### 5. VisualizationTools
- **Purpose**: Data exploration and results visualization
- **Key Methods**:
  - `plot_activity_distribution()`: pIC50 value distribution
  - `explore_fingerprint_features()`: Feature analysis plots
  - `visualize_cv_results()`: Cross-validation performance
  - `plot_confusion_matrix()`: Classification results

## Performance Metrics

### Model Performance
- **Test Accuracy**: ~85-90% (depending on train/test split)
- **F1 Score (macro)**: ~0.85-0.90
- **Matthews Correlation**: ~0.70-0.80
- **ROC AUC**: ~0.90-0.95

### Dataset Statistics
- **Compounds**: 587 compounds with hERG activity labels
- **Feature Sparsity**: ~95% (typical for molecular fingerprints)
- **Important Features**: Specific molecular substructures related to hERG binding
- **Regularization**: L1/L2 penalties help prevent overfitting

## Integration

This Chapter 3 implementation integrates with existing drug discovery pipelines:

```python
# Example integration
from src.chapter3_ml_screening import HERGClassifier, MolecularFeaturizer
from src.similarity import SimilaritySearcher  # Existing tool

# Load trained model
classifier = HERGClassifier()
model = classifier.load_model("artifacts/chapter3/herg_classifier_final.pkl")
featurizer = MolecularFeaturizer()

# Use with existing similarity search
searcher = SimilaritySearcher()
similar_compounds = searcher.find_similar(query_mol, database)

# Add hERG safety assessment
for compound in similar_compounds:
    fingerprint = featurizer.compute_fingerprint(compound.mol)
    herg_risk = model.predict_proba(fingerprint.reshape(1, -1))[0][1]
    compound.herg_risk_score = herg_risk
    compound.is_safe = herg_risk < 0.5  # Safety threshold
```

## Advanced Usage

### Custom Hyperparameter Optimization
```python
# Define custom parameter space
param_distributions = {
    'alpha': uniform(1e-6, 1e-1),
    'l1_ratio': uniform(0, 1),
    'penalty': ['l1', 'l2', 'elasticnet'],
    'max_iter': [500, 1000, 2000]
}

# Run randomized search
classifier = HERGClassifier()
search = classifier.randomized_search_hyperparameters(
    X_train, y_train,
    param_distributions=param_distributions,
    n_iter=200,  # More thorough search
    cv=10,       # More robust validation
    scoring='f1_macro'
)
```

### Feature Engineering Pipeline
```python
from sklearn.pipeline import Pipeline
from chapter3_ml_screening.molecular_features import SmilesToMols, FingerprintFeaturizer

# Create complete pipeline from SMILES to predictions
pipeline = Pipeline([
    ('smiles_to_mols', SmilesToMols(standardize=True)),
    ('fingerprints', FingerprintFeaturizer(radius=3, n_bits=4096)),  # Larger FPs
    ('classifier', SGDClassifier(alpha=0.01, penalty='l2'))
])

# Train on raw SMILES
pipeline.fit(smiles_train, y_train)

# Predict directly from SMILES
predictions = pipeline.predict(new_smiles)