#!/usr/bin/env python3
"""
Random Forest hERG Classifier
============================

Dedicated RandomForestClassifier implementation for hERG cardiotoxicity prediction.
This script focuses on thorough hyperparameter optimization and evaluation of 
Random Forest models for the hERG blocking prediction task.

Features:
- Comprehensive hyperparameter grid search
- Cross-validation evaluation
- Feature importance analysis
- Model interpretation and visualization
- Performance comparison with baseline models

Usage:
    python random_forest_herg_model.py
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, List

# Machine Learning
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, 
    StratifiedKFold, learning_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, f1_score, 
    precision_score, recall_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)

# RDKit for molecular processing
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import ExplicitBitVect

# Suppress warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

class RandomForestHERGClassifier:
    """
    Random Forest classifier specifically designed for hERG cardiotoxicity prediction.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the Random Forest hERG classifier.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.model = None
        self.best_params = None
        self.cv_results = None
        self.feature_importance = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        # Set random seeds
        np.random.seed(random_seed)
        
    def load_herg_data(self) -> pd.DataFrame:
        """Load hERG blocker dataset."""
        data_path = Path("../../../data/chapter3/hERG_blockers.xlsx")
        
        if not data_path.exists():
            raise FileNotFoundError(f"hERG dataset not found at {data_path}")
        
        try:
            df = pd.read_excel(data_path)
            print(f"✅ Loaded {len(df)} hERG compounds")
            return df
        except Exception as e:
            raise Exception(f"Error loading hERG data: {e}")
    
    def prepare_molecular_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare molecular fingerprints and labels from DataFrame.
        
        Args:
            df: DataFrame with SMILES and Classa columns
            
        Returns:
            Tuple of (fingerprints, labels)
        """
        print("🧬 Preparing molecular data...")
        
        # Standardize SMILES
        df['smiles_std'] = df['SMILES'].apply(self._standardize_smiles)
        df = df[df['smiles_std'].notna()].copy()
        
        # Convert to RDKit molecules
        df['mol'] = df['smiles_std'].apply(Chem.MolFromSmiles)
        df = df[df['mol'].notna()].copy()
        
        # Remove rows with missing class labels and convert to integers
        df = df[df['Classa'].notna()].copy()
        df['Classa'] = df['Classa'].astype(int)
        
        print(f"📊 Using {len(df)} valid molecules")
        
        # Compute fingerprints
        print("🔢 Computing Morgan fingerprints...")
        fingerprints = self._compute_morgan_fingerprints(df['mol'].tolist())
        labels = df['Classa'].values
        
        print(f"📈 Fingerprint matrix shape: {fingerprints.shape}")
        print(f"📊 Class distribution: {np.bincount(labels)} (0: non-blocker, 1: blocker)")
        
        return fingerprints, labels
    
    def _standardize_smiles(self, smiles: str) -> Optional[str]:
        """Standardize a SMILES string using existing infrastructure."""
        # Use existing HERGDataProcessor instead of duplicate code
        from src.chapter3_ml_screening.data_processing import HERGDataProcessor
        
        processor = HERGDataProcessor()
        mol = processor.process_smiles(smiles)
        
        if mol is not None:
            try:
                return Chem.MolToSmiles(mol)
            except:
                return None
        return None
    
    def _compute_morgan_fingerprints(self, molecules: List, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
        """Compute Morgan fingerprints for molecules using existing infrastructure."""
        # Use existing MolecularFeaturizer instead of duplicate code
        from src.chapter3_ml_screening.molecular_features import MolecularFeaturizer
        
        featurizer = MolecularFeaturizer(radius=radius, n_bits=n_bits)
        return featurizer.compute_fingerprints_batch(molecules)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.33) -> None:
        """Split data into training and test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed, stratify=y
        )
        
        print(f"📊 Data split:")
        print(f"   Training set: {self.X_train.shape[0]} samples")
        print(f"   Test set: {self.X_test.shape[0]} samples")
        print(f"   Training class distribution: {np.bincount(self.y_train)}")
        print(f"   Test class distribution: {np.bincount(self.y_test)}")
    
    def optimize_hyperparameters(self, cv_folds: int = 5, n_jobs: int = -1) -> Dict:
        """
        Perform comprehensive hyperparameter optimization for Random Forest.
        
        Args:
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with optimization results
        """
        print("\n🔍 Starting Random Forest hyperparameter optimization...")
        
        # Comprehensive parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3],
            'bootstrap': [True],
            'class_weight': [None, 'balanced']
        }
        
        print(f"🎯 Grid search parameters:")
        for param, values in param_grid.items():
            print(f"   {param}: {values}")
        
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"🔢 Total parameter combinations: {total_combinations}")
        
        # Create stratified k-fold
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_seed)
        
        # Grid search with MCC as primary metric
        rf = RandomForestClassifier(random_state=self.random_seed)
        
        grid_search = GridSearchCV(
            rf, param_grid,
            scoring='matthews_corrcoef',  # Primary metric: MCC
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        print("🚀 Starting grid search...")
        grid_search.fit(self.X_train, self.y_train)
        
        # Store results
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results = pd.DataFrame(grid_search.cv_results_)
        
        print(f"\n✅ Hyperparameter optimization completed!")
        print(f"🏆 Best parameters:")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        print(f"🎯 Best CV MCC score: {grid_search.best_score_:.4f}")
        
        return {
            'best_estimator': self.model,
            'best_params': self.best_params,
            'best_score': grid_search.best_score_,
            'cv_results': self.cv_results
        }
    
    def evaluate_model(self) -> Dict:
        """
        Comprehensive model evaluation on test set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print("\n📊 Evaluating Random Forest model...")
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'matthews_corrcoef': matthews_corrcoef(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        print(f"🎯 Random Forest Test Set Performance:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Matthews Correlation Coefficient: {metrics['matthews_corrcoef']:.4f}")
        print(f"   F1 Score: {metrics['f1_score']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Classification report
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Non-blocker', 'Blocker']))
        
        return metrics
    
    def compare_with_baselines(self) -> pd.DataFrame:
        """
        Compare Random Forest with baseline models.
        
        Returns:
            DataFrame with comparison results
        """
        print("\n⚖️ Comparing with baseline models...")
        
        models = {
            'Random Forest': self.model,
            'SGD Classifier': SGDClassifier(random_state=self.random_seed, loss='log_loss', max_iter=1000),
            'Dummy (Most Frequent)': DummyClassifier(strategy='most_frequent', random_state=self.random_seed)
        }
        
        results = []
        
        for name, model in models.items():
            if name != 'Random Forest':
                model.fit(self.X_train, self.y_train)
            
            y_pred = model.predict(self.X_test)
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            else:
                roc_auc = np.nan
            
            results.append({
                'Model': name,
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'MCC': matthews_corrcoef(self.y_test, y_pred),
                'F1': f1_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'ROC_AUC': roc_auc
            })
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('MCC', ascending=False)
        
        print(f"🏆 Model Comparison Results:")
        print(comparison_df.round(4))
        
        return comparison_df
    
    def analyze_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Analyze and report feature importance.
        
        Args:
            top_n: Number of top features to display
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print(f"\n🔍 Analyzing feature importance (top {top_n})...")
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_names = [f'Bit_{i}' for i in range(len(importances))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Store for later use
        self.feature_importance = importance_df
        
        print(f"📊 Top {top_n} most important features:")
        print(importance_df.head(top_n))
        
        return importance_df
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        import joblib
        
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance
        }, filepath)
        
        print(f"💾 Model saved to {filepath}")
    
    def predict_molecules(self, smiles_list: List[str]) -> Dict:
        """
        Predict hERG blockage for a list of SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Convert SMILES to molecules
        molecules = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        
        # Compute fingerprints
        fingerprints = self._compute_morgan_fingerprints(molecules)
        
        # Make predictions
        predictions = self.model.predict(fingerprints)
        probabilities = self.model.predict_proba(fingerprints)[:, 1]
        
        return {
            'smiles': smiles_list,
            'predictions': predictions,
            'probabilities': probabilities,
            'herg_blockers': [smi for smi, pred in zip(smiles_list, predictions) if pred == 1],
            'non_blockers': [smi for smi, pred in zip(smiles_list, predictions) if pred == 0]
        }


def main():
    """Main execution function."""
    print("="*70)
    print("🧬 RANDOM FOREST hERG CARDIOTOXICITY CLASSIFIER")
    print("="*70)
    print("Comprehensive Random Forest implementation for hERG blocking prediction")
    print("="*70)
    
    try:
        # Initialize classifier
        rf_classifier = RandomForestHERGClassifier(random_seed=42)
        
        # Load and prepare data
        print("\n📊 Loading hERG dataset...")
        herg_df = rf_classifier.load_herg_data()
        
        print("\n🧬 Preparing molecular fingerprints...")
        X, y = rf_classifier.prepare_molecular_data(herg_df)
        
        print("\n📊 Splitting data...")
        rf_classifier.split_data(X, y)
        
        # Hyperparameter optimization
        print("\n🔍 Optimizing hyperparameters...")
        optimization_results = rf_classifier.optimize_hyperparameters(cv_folds=5)
        
        # Model evaluation
        print("\n📊 Evaluating model performance...")
        test_metrics = rf_classifier.evaluate_model()
        
        # Baseline comparison
        print("\n⚖️ Comparing with baselines...")
        comparison_results = rf_classifier.compare_with_baselines()
        
        # Feature importance analysis
        print("\n🔍 Analyzing feature importance...")
        importance_df = rf_classifier.analyze_feature_importance(top_n=20)
        
        # Save model
        model_path = "artifacts/chapter3/random_forest_herg_model.pkl"
        rf_classifier.save_model(model_path)
        
        # Final summary
        print("\n" + "="*70)
        print("🏆 RANDOM FOREST hERG CLASSIFIER - SUMMARY")
        print("="*70)
        print(f"✅ Model Training: COMPLETED")
        print(f"🎯 Best CV MCC: {optimization_results['best_score']:.4f}")
        print(f"🎯 Test Set MCC: {test_metrics['matthews_corrcoef']:.4f}")
        print(f"🎯 Test Set Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"🎯 Test Set ROC AUC: {test_metrics['roc_auc']:.4f}")
        print(f"💾 Model saved to: {model_path}")
        
        # Example prediction
        print(f"\n🧪 Example predictions:")
        example_smiles = [
            "CCO",  # Ethanol (likely non-blocker)
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
        ]
        
        example_results = rf_classifier.predict_molecules(example_smiles)
        for smi, pred, prob in zip(example_results['smiles'], 
                                   example_results['predictions'], 
                                   example_results['probabilities']):
            status = "🚨 hERG Blocker" if pred == 1 else "✅ Non-blocker"
            print(f"   {smi}: {status} (P = {prob:.3f})")
        
        print("\n✅ Analysis completed successfully!")
        
        return rf_classifier
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise


if __name__ == "__main__":
    classifier = main() 