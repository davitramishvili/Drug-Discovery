"""
Random Forest model module for Chapter 3 ML screening.
Implements a dedicated RandomForestClassifier for hERG cardiotoxicity prediction,
integrating with existing data processing and molecular feature infrastructure.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union

# Machine Learning
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, f1_score, 
    precision_score, recall_score, classification_report,
    roc_auc_score
)

# Local imports
from .data_processing import HERGDataProcessor
from .molecular_features import MolecularFeaturizer

logger = logging.getLogger(__name__)

class RandomForestHERGClassifier:
    """
    Random Forest classifier for hERG cardiotoxicity prediction.
    
    This class leverages the existing infrastructure from the chapter3_ml_screening
    module for data processing and molecular feature extraction.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the Random Forest hERG classifier.
        
        Parameters:
            random_seed (int): Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.model = None
        self.best_params = None
        self.cv_results = None
        self.feature_importance = None
        
        # Initialize data processor and featurizer
        self.data_processor = HERGDataProcessor(random_seed=random_seed)
        self.featurizer = MolecularFeaturizer(radius=2, n_bits=2048)
        
        # Data storage
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.processed_data = None
        
        # Set random seeds
        np.random.seed(random_seed)
        
        logger.info(f"RandomForestHERGClassifier initialized with random_seed={random_seed}")
    
    def load_and_prepare_data(self, data_path: Optional[str] = None, 
                             test_size: float = 0.33) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare hERG data using existing infrastructure.
        
        Parameters:
            data_path (str, optional): Path to hERG data file
            test_size (float): Fraction of data to use for testing
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Fingerprints and labels
        """
        logger.info("Loading and preparing hERG data...")
        
        # Load data using existing data processor
        if data_path:
            herg_df = self.data_processor.load_herg_blockers_data(data_path)
        else:
            herg_df = self.data_processor.load_herg_blockers_data()
        
        if herg_df is None or herg_df.empty:
            raise ValueError("Failed to load hERG data")
        
        # Standardize molecules
        processed_df = self.data_processor.standardize_molecules(herg_df)
        
        # Remove rows with missing class labels
        processed_df = processed_df[processed_df['Class'].notna()].copy()
        processed_df['Class'] = processed_df['Class'].astype(int)
        
        logger.info(f"Using {len(processed_df)} valid molecules after processing")
        
        # Compute fingerprints using existing featurizer
        molecules = processed_df['mol'].tolist()
        fingerprints = self.featurizer.compute_fingerprints_batch(molecules)
        labels = processed_df['Class'].values
        
        logger.info(f"Fingerprint matrix shape: {fingerprints.shape}")
        logger.info(f"Class distribution: {np.bincount(labels)} (0: non-blocker, 1: blocker)")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            fingerprints, labels, 
            test_size=test_size, 
            random_state=self.random_seed, 
            stratify=labels
        )
        
        logger.info(f"Data split - Training: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}")
        logger.info(f"Training class distribution: {np.bincount(self.y_train)}")
        logger.info(f"Test class distribution: {np.bincount(self.y_test)}")
        
        # Store processed data for later use
        self.processed_data = processed_df
        
        return fingerprints, labels
    
    def optimize_hyperparameters(self, cv_folds: int = 5, n_jobs: int = -1, 
                                 param_grid: Optional[Dict] = None) -> Dict:
        """
        Perform comprehensive hyperparameter optimization for Random Forest.
        
        Parameters:
            cv_folds (int): Number of cross-validation folds
            n_jobs (int): Number of parallel jobs
            param_grid (dict, optional): Custom parameter grid
            
        Returns:
            Dict: Optimization results
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Call load_and_prepare_data() first.")
        
        logger.info("Starting Random Forest hyperparameter optimization...")
        
        # Default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3],
                'bootstrap': [True],
                'class_weight': [None, 'balanced']
            }
        
        logger.info(f"Parameter grid: {param_grid}")
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        logger.info(f"Total parameter combinations: {total_combinations}")
        
        # Create stratified k-fold
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_seed)
        
        # Grid search with MCC as primary metric
        rf = RandomForestClassifier(random_state=self.random_seed)
        
        grid_search = GridSearchCV(
            rf, param_grid,
            scoring='matthews_corrcoef',
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        logger.info("Starting grid search...")
        grid_search.fit(self.X_train, self.y_train)
        
        # Store results
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results = pd.DataFrame(grid_search.cv_results_)
        
        logger.info("Hyperparameter optimization completed!")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV MCC score: {grid_search.best_score_:.4f}")
        
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
            Dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call optimize_hyperparameters() first.")
        
        logger.info("Evaluating Random Forest model...")
        
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
        
        logger.info("Random Forest Test Set Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Classification report
        logger.info("\nDetailed Classification Report:")
        report = classification_report(
            self.y_test, y_pred, 
            target_names=['Non-blocker', 'Blocker']
        )
        logger.info(f"\n{report}")
        
        return metrics
    
    def compare_with_baselines(self) -> pd.DataFrame:
        """
        Compare Random Forest with baseline models.
        
        Returns:
            pd.DataFrame: Comparison results
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Call load_and_prepare_data() first.")
        
        logger.info("Comparing with baseline models...")
        
        models = {
            'Random Forest': self.model,
            'SGD Classifier': SGDClassifier(
                random_state=self.random_seed, 
                loss='log_loss', 
                max_iter=1000
            ),
            'Dummy (Most Frequent)': DummyClassifier(
                strategy='most_frequent', 
                random_state=self.random_seed
            )
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
        
        logger.info("Model Comparison Results:")
        logger.info(f"\n{comparison_df.round(4)}")
        
        return comparison_df
    
    def analyze_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Analyze and report feature importance.
        
        Parameters:
            top_n (int): Number of top features to display
            
        Returns:
            pd.DataFrame: Feature importance results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call optimize_hyperparameters() first.")
        
        logger.info(f"Analyzing feature importance (top {top_n})...")
        
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
        
        logger.info(f"Top {top_n} most important features:")
        logger.info(f"\n{importance_df.head(top_n)}")
        
        return importance_df
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        import joblib
        
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'featurizer_params': {
                'radius': self.featurizer.radius,
                'n_bits': self.featurizer.n_bits
            }
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model."""
        import joblib
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.best_params = model_data['best_params']
        self.feature_importance = model_data.get('feature_importance')
        
        # Restore featurizer parameters if available
        if 'featurizer_params' in model_data:
            params = model_data['featurizer_params']
            self.featurizer = MolecularFeaturizer(
                radius=params['radius'],
                n_bits=params['n_bits']
            )
        
        logger.info(f"Model loaded from {filepath}")
    
    def predict_molecules(self, smiles_list: List[str]) -> Dict:
        """
        Predict hERG blockage for a list of SMILES.
        
        Parameters:
            smiles_list (List[str]): List of SMILES strings
            
        Returns:
            Dict: Predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call optimize_hyperparameters() first.")
        
        logger.info(f"Predicting hERG blockage for {len(smiles_list)} molecules...")
        
        # Standardize SMILES and convert to molecules
        molecules = []
        valid_smiles = []
        
        for smiles in smiles_list:
            mol = self.data_processor.process_smiles(smiles)
            if mol is not None:
                molecules.append(mol)
                valid_smiles.append(smiles)
        
        if not molecules:
            logger.warning("No valid molecules found")
            return {
                'smiles': smiles_list,
                'predictions': [],
                'probabilities': [],
                'herg_blockers': [],
                'non_blockers': []
            }
        
        # Compute fingerprints
        fingerprints = self.featurizer.compute_fingerprints_batch(molecules)
        
        # Make predictions
        predictions = self.model.predict(fingerprints)
        probabilities = self.model.predict_proba(fingerprints)[:, 1]
        
        # Create results mapping
        results = {
            'smiles': valid_smiles,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'herg_blockers': [smiles for smiles, pred in zip(valid_smiles, predictions) if pred == 1],
            'non_blockers': [smiles for smiles, pred in zip(valid_smiles, predictions) if pred == 0]
        }
        
        logger.info(f"Prediction completed: {len(results['herg_blockers'])} blockers, {len(results['non_blockers'])} non-blockers")
        
        return results
    
    def get_model_summary(self) -> Dict:
        """Get a summary of the trained model."""
        if self.model is None:
            return {"status": "Model not trained"}
        
        summary = {
            "model_type": "RandomForestClassifier",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "min_samples_split": self.model.min_samples_split,
            "min_samples_leaf": self.model.min_samples_leaf,
            "max_features": self.model.max_features,
            "bootstrap": self.model.bootstrap,
            "class_weight": self.model.class_weight,
            "random_state": self.model.random_state,
            "feature_importances_available": self.feature_importance is not None,
            "training_data_shape": (self.X_train.shape if self.X_train is not None else None),
            "test_data_shape": (self.X_test.shape if self.X_test is not None else None)
        }
        
        return summary 