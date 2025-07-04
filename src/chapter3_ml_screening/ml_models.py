"""
Machine learning models module for Chapter 3 ML screening.
Handles hERG classification with linear models, regularization, and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Any, List
import logging
import joblib
from pathlib import Path

# Scikit-learn imports
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, 
    matthews_corrcoef, precision_score, recall_score
)
from scipy.stats import uniform as sp_rand

from .molecular_features import SmilesToMols, FingerprintFeaturizer

logger = logging.getLogger(__name__)

class HERGClassifier:
    """
    Machine learning classifier for hERG channel blocking prediction.
    
    This class implements the complete ML pipeline from Flynn's Chapter 3,
    including linear models, regularization, and hyperparameter optimization.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the hERG classifier.
        
        Parameters:
            random_seed (int): Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.model = None
        self.pipeline = None
        self.cv_results = None
        self.best_params = None
        
        logger.info("HERGClassifier initialized")
    
    def train_sgd_classifier(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> SGDClassifier:
        """
        Train a Stochastic Gradient Descent classifier.
        
        Parameters:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            **kwargs: Additional parameters for SGDClassifier
            
        Returns:
            sklearn.linear_model.SGDClassifier: Trained classifier
        """
        # Default parameters - using log_loss to enable probability predictions
        params = {
            'loss': 'log_loss',  # Changed from default 'hinge' to enable predict_proba
            'random_state': self.random_seed,
            'max_iter': 1000,
            'tol': 1e-3
        }
        
        # Update with any provided parameters
        params.update(kwargs)
        
        # Create and train the classifier
        clf = SGDClassifier(**params)
        clf.fit(X_train, y_train)
        
        self.model = clf
        logger.info("SGD classifier training completed")
        
        return clf
    
    def train_random_forest_classifier(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestClassifier:
        """
        Train a Random Forest classifier.
        
        Parameters:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            **kwargs: Additional parameters for RandomForestClassifier
            
        Returns:
            sklearn.ensemble.RandomForestClassifier: Trained classifier
        """
        # Default parameters
        params = {
            'random_state': self.random_seed,
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'n_jobs': -1
        }
        
        # Update with any provided parameters
        params.update(kwargs)
        
        # Create and train the classifier
        rf_clf = RandomForestClassifier(**params)
        rf_clf.fit(X_train, y_train)
        
        self.model = rf_clf
        logger.info("Random Forest classifier training completed")
        
        return rf_clf
    
    def grid_search_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                                 param_grid: Optional[Dict] = None,
                                 cv: int = 5,
                                 scoring: str = 'matthews_corrcoef') -> GridSearchCV:
        """
        Perform grid search for Random Forest hyperparameter optimization.
        
        Parameters:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            param_grid (dict, optional): Parameter grid to search
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric for optimization (default: MCC)
            
        Returns:
            sklearn.model_selection.GridSearchCV: Fitted grid search object
        """
        if param_grid is None:
            # Default parameter grid for Random Forest
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        # Create base classifier
        rf_clf = RandomForestClassifier(
            random_state=self.random_seed,
            n_jobs=-1
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            rf_clf,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        logger.info("Random Forest grid search completed")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search
    
    def compare_models(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      models: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Compare multiple models on the same dataset.
        
        Parameters:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            models (dict, optional): Dictionary of models to compare
            
        Returns:
            pandas.DataFrame: Comparison results
        """
        if models is None:
            # Default models to compare
            models = {
                'SGDClassifier': SGDClassifier(loss='log_loss', random_state=self.random_seed),
                'RandomForest': RandomForestClassifier(random_state=self.random_seed, n_jobs=-1),
                'Dummy': DummyClassifier(strategy='most_frequent', random_state=self.random_seed)
            }
        
        results = []
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            result = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'F1_macro': f1_score(y_test, y_pred, average='macro'),
                'Precision_macro': precision_score(y_test, y_pred, average='macro'),
                'Recall_macro': recall_score(y_test, y_pred, average='macro'),
                'Matthews_CC': matthews_corrcoef(y_test, y_pred)
            }
            
            results.append(result)
            
            logger.info(f"{name} - MCC: {result['Matthews_CC']:.4f}")
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('Matthews_CC', ascending=False)
        
        return comparison_df
    
    def create_pipeline(self, radius: int = 2, n_bits: int = 2048, 
                       sgd_params: Optional[Dict] = None) -> Pipeline:
        """
        Create a complete ML pipeline from SMILES to predictions.
        
        Parameters:
            radius (int): Morgan fingerprint radius
            n_bits (int): Number of fingerprint bits
            sgd_params (dict, optional): SGD classifier parameters
            
        Returns:
            sklearn.pipeline.Pipeline: Complete ML pipeline
        """
        if sgd_params is None:
            sgd_params = {
                'loss': 'log_loss',  # Enable probability predictions
                'random_state': self.random_seed,
                'max_iter': 1000,
                'tol': 1e-3
            }
        
        # Create pipeline
        pipeline = Pipeline([
            ('smiles_to_mols', SmilesToMols(standardize=True)),
            ('fingerprints', FingerprintFeaturizer(radius=radius, n_bits=n_bits)),
            ('classifier', SGDClassifier(**sgd_params))
        ])
        
        self.pipeline = pipeline
        logger.info(f"Created pipeline with radius={radius}, n_bits={n_bits}")
        
        return pipeline
    
    def train_baseline_dummy(self, X_train: np.ndarray, y_train: np.ndarray, 
                           strategy: str = "most_frequent") -> DummyClassifier:
        """
        Train a dummy classifier baseline.
        
        Parameters:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            strategy (str): Strategy for dummy classifier
            
        Returns:
            sklearn.dummy.DummyClassifier: Trained dummy classifier
        """
        dummy_clf = DummyClassifier(strategy=strategy, random_state=self.random_seed)
        dummy_clf.fit(X_train, y_train)
        
        logger.info(f"Dummy classifier trained with strategy '{strategy}'")
        return dummy_clf
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                           cv: int = 5, scoring: Optional[Dict] = None) -> pd.DataFrame:
        """
        Perform cross-validation evaluation of a model.
        
        Parameters:
            model: Trained model to evaluate
            X (numpy.ndarray): Features
            y (numpy.ndarray): Labels
            cv (int): Number of cross-validation folds
            scoring (dict, optional): Scoring metrics to compute
            
        Returns:
            pandas.DataFrame: Cross-validation results
        """
        if scoring is None:
            scoring = {
                'acc': 'accuracy',
                'prec_macro': 'precision_macro',
                'rec_macro': 'recall_macro',
                'f1_macro': 'f1_macro',
            }
        
        cv_scores = cross_validate(
            model, X, y, 
            scoring=scoring, 
            cv=cv, 
            return_train_score=True
        )
        
        cv_df = pd.DataFrame.from_dict(cv_scores)
        self.cv_results = cv_df
        
        logger.info("Cross-validation completed")
        logger.info(f"Mean test accuracy: {cv_df['test_acc'].mean():.4f} ± {cv_df['test_acc'].std():.4f}")
        
        return cv_df
    
    def grid_search_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                                  param_grid: Optional[Dict] = None, 
                                  cv: int = 5, 
                                  scoring: str = 'f1_macro') -> GridSearchCV:
        """
        Perform grid search for hyperparameter optimization.
        
        Parameters:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            param_grid (dict, optional): Parameter grid to search
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric for optimization
            
        Returns:
            sklearn.model_selection.GridSearchCV: Fitted grid search object
        """
        if param_grid is None:
            # Default parameter grid focusing on regularization
            param_grid = {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.0, 0.15, 0.5, 0.85, 1.0],  # For elastic net
                'penalty': ['l1', 'l2', 'elasticnet']
            }
        
        # Create base classifier with log_loss for probability predictions
        sgd_clf = SGDClassifier(
            loss='log_loss',
            random_state=self.random_seed,
            max_iter=1000,
            tol=1e-3
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            sgd_clf,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        logger.info("Grid search completed")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search
    
    def randomized_search_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                        param_distributions: Optional[Dict] = None,
                                        n_iter: int = 100,
                                        cv: int = 5,
                                        scoring: str = 'f1_macro') -> RandomizedSearchCV:
        """
        Perform randomized search for hyperparameter optimization.
        
        Parameters:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            param_distributions (dict, optional): Parameter distributions to sample
            n_iter (int): Number of parameter combinations to try
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric for optimization
            
        Returns:
            sklearn.model_selection.RandomizedSearchCV: Fitted randomized search object
        """
        if param_distributions is None:
            # Default parameter distributions
            param_distributions = {
                'alpha': sp_rand(loc=1e-6, scale=1e-1),
                'l1_ratio': sp_rand(),
                'penalty': ['l1', 'l2', 'elasticnet']
            }
        
        # Create base classifier with log_loss for probability predictions
        sgd_clf = SGDClassifier(
            loss='log_loss',
            random_state=self.random_seed,
            max_iter=1000,
            tol=1e-3
        )
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            sgd_clf,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=self.random_seed
        )
        
        random_search.fit(X_train, y_train)
        
        self.best_params = random_search.best_params_
        self.model = random_search.best_estimator_
        
        logger.info("Randomized search completed")
        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        return random_search
    
    def create_polynomial_features_pipeline(self, degree: int = 2, 
                                          sgd_params: Optional[Dict] = None) -> Pipeline:
        """
        Create a pipeline with polynomial feature transformation.
        
        Parameters:
            degree (int): Degree of polynomial features
            sgd_params (dict, optional): SGD classifier parameters
            
        Returns:
            sklearn.pipeline.Pipeline: Pipeline with polynomial features
        """
        if sgd_params is None:
            sgd_params = {
                'random_state': self.random_seed,
                'max_iter': 1000,
                'tol': 1e-3
            }
        
        pipeline = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            SGDClassifier(**sgd_params)
        )
        
        logger.info(f"Created polynomial features pipeline with degree={degree}")
        return pipeline
    
    def analyze_feature_importance(self, model=None) -> Dict[str, np.ndarray]:
        """
        Analyze feature importance from a trained linear model.
        
        Parameters:
            model: Trained model (uses self.model if None)
            
        Returns:
            dict: Dictionary with feature importance information
        """
        if model is None:
            model = self.model
        
        if model is None:
            logger.warning("No trained model available for feature importance analysis")
            return {}
        
        # Get model weights/coefficients
        if hasattr(model, 'coef_'):
            weights = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        else:
            logger.warning("Model does not have feature coefficients")
            return {}
        
        # Analyze weights
        analysis = {
            'weights': weights,
            'abs_weights': np.abs(weights),
            'n_features': len(weights),
            'n_zero_weights': np.sum(weights == 0),
            'n_nonzero_weights': np.sum(weights != 0),
            'mean_weight': np.mean(weights),
            'std_weight': np.std(weights),
            'min_weight': np.min(weights),
            'max_weight': np.max(weights),
            'top_positive_indices': np.argsort(weights)[-10:],
            'top_negative_indices': np.argsort(weights)[:10],
            'most_important_indices': np.argsort(np.abs(weights))[-10:]
        }
        
        logger.info("Feature importance analysis completed")
        logger.info(f"  Total features: {analysis['n_features']}")
        logger.info(f"  Non-zero weights: {analysis['n_nonzero_weights']}")
        logger.info(f"  Weight range: [{analysis['min_weight']:.4f}, {analysis['max_weight']:.4f}]")
        
        return analysis
    
    def save_model(self, filepath: str, model=None):
        """
        Save a trained model to disk.
        
        Parameters:
            filepath (str): Path to save the model
            model: Model to save (uses self.model if None)
        """
        if model is None:
            model = self.model or self.pipeline
        
        if model is None:
            logger.error("No model available to save")
            return
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            joblib.dump(model, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Parameters:
            filepath (str): Path to the saved model
            
        Returns:
            Loaded model object
        """
        try:
            model = joblib.load(filepath)
            self.model = model
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def predict(self, X: np.ndarray, model=None) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Parameters:
            X (numpy.ndarray): Features to predict
            model: Model to use (uses self.model if None)
            
        Returns:
            numpy.ndarray: Predictions
        """
        if model is None:
            model = self.model
        
        if model is None:
            logger.error("No trained model available for predictions")
            return np.array([])
        
        return model.predict(X)
    
    def predict_proba(self, X: np.ndarray, model=None) -> np.ndarray:
        """
        Predict class probabilities with the trained model.
        
        Parameters:
            X (numpy.ndarray): Features to predict
            model: Model to use (uses self.model if None)
            
        Returns:
            numpy.ndarray: Class probabilities
        """
        if model is None:
            model = self.model
        
        if model is None:
            logger.error("No trained model available for probability predictions")
            return np.array([])
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            logger.warning("Model does not support probability predictions")
            return np.array([])
    
    def get_model_summary(self, model=None) -> Dict[str, Any]:
        """
        Get a summary of the trained model.
        
        Parameters:
            model: Model to summarize (uses self.model if None)
            
        Returns:
            dict: Model summary information
        """
        if model is None:
            model = self.model
        
        if model is None:
            return {"error": "No trained model available"}
        
        summary = {
            "model_type": type(model).__name__,
            "model_params": model.get_params() if hasattr(model, 'get_params') else {},
        }
        
        # Add SGD-specific information
        if isinstance(model, SGDClassifier):
            summary.update({
                "n_iter_": getattr(model, 'n_iter_', None),
                "classes_": getattr(model, 'classes_', None),
                "coef_shape": model.coef_.shape if hasattr(model, 'coef_') else None,
                "intercept_": getattr(model, 'intercept_', None)
            })
        
        return summary 