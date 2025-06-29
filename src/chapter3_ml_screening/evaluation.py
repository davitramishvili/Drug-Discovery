"""
Evaluation module for Chapter 3 ML screening.
Handles comprehensive model assessment, metrics calculation, and performance analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
import logging
from pathlib import Path

# Scikit-learn metrics
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, 
    matthews_corrcoef, precision_score, recall_score,
    roc_auc_score, precision_recall_curve, roc_curve,
    classification_report
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation for hERG classification.
    
    This class implements the evaluation pipeline from Flynn's Chapter 3,
    including cross-validation analysis, feature importance, and test set evaluation.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the model evaluator.
        
        Parameters:
            random_seed (int): Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.evaluation_results = {}
        
        logger.info("ModelEvaluator initialized")
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_pred_proba: Optional[np.ndarray] = None,
                           model_name: str = "model") -> Dict[str, Any]:
        """
        Evaluate model predictions with comprehensive metrics.
        
        Parameters:
            y_true (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted labels
            y_pred_proba (numpy.ndarray, optional): Predicted probabilities
            model_name (str): Name of the model being evaluated
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        # Basic classification metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Add probabilistic metrics if probabilities are provided
        if y_pred_proba is not None:
            try:
                # For binary classification, use probabilities for positive class
                if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                    pos_proba = y_pred_proba[:, 1]
                else:
                    pos_proba = y_pred_proba
                
                metrics.update({
                    'roc_auc': roc_auc_score(y_true, pos_proba),
                    'roc_curve': roc_curve(y_true, pos_proba),
                    'precision_recall_curve': precision_recall_curve(y_true, pos_proba)
                })
            except Exception as e:
                logger.warning(f"Could not calculate probabilistic metrics: {e}")
        
        # Store results
        self.evaluation_results[model_name] = metrics
        
        # Log key metrics
        logger.info(f"Evaluation results for {model_name}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  Matthews CC: {metrics['matthews_corrcoef']:.4f}")
        
        return metrics
    
    def compare_models(self, models_results: Dict[str, Dict[str, Any]], 
                      primary_metric: str = 'f1_macro') -> pd.DataFrame:
        """
        Compare multiple models based on evaluation metrics.
        
        Parameters:
            models_results (dict): Dictionary of model results
            primary_metric (str): Primary metric for ranking models
            
        Returns:
            pandas.DataFrame: Comparison table of models
        """
        comparison_data = []
        
        for model_name, results in models_results.items():
            row = {
                'Model': model_name,
                'Accuracy': results.get('accuracy', np.nan),
                'F1_Macro': results.get('f1_macro', np.nan),
                'F1_Weighted': results.get('f1_weighted', np.nan),
                'Precision_Macro': results.get('precision_macro', np.nan),
                'Recall_Macro': results.get('recall_macro', np.nan),
                'Matthews_CC': results.get('matthews_corrcoef', np.nan),
                'ROC_AUC': results.get('roc_auc', np.nan)
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric
        if primary_metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(primary_metric, ascending=False)
        
        logger.info(f"Model comparison completed (sorted by {primary_metric})")
        return comparison_df
    
    def evaluate_cross_validation_results(self, cv_results: pd.DataFrame,
                                        model_name: str = "model") -> Dict[str, Any]:
        """
        Analyze cross-validation results to assess model stability and performance.
        
        Parameters:
            cv_results (pandas.DataFrame): Cross-validation results
            model_name (str): Name of the model
            
        Returns:
            dict: Analysis of CV results
        """
        analysis = {}
        
        # Get metric columns
        test_metrics = [col for col in cv_results.columns if col.startswith('test_')]
        train_metrics = [col for col in cv_results.columns if col.startswith('train_')]
        
        # Analyze each metric
        for test_col in test_metrics:
            metric_name = test_col.replace('test_', '')
            train_col = f'train_{metric_name}'
            
            test_scores = cv_results[test_col]
            train_scores = cv_results[train_col] if train_col in cv_results.columns else None
            
            metric_analysis = {
                'test_mean': test_scores.mean(),
                'test_std': test_scores.std(),
                'test_min': test_scores.min(),
                'test_max': test_scores.max(),
                'test_scores': test_scores.values
            }
            
            if train_scores is not None:
                metric_analysis.update({
                    'train_mean': train_scores.mean(),
                    'train_std': train_scores.std(),
                    'overfitting_gap': train_scores.mean() - test_scores.mean(),
                    'train_scores': train_scores.values
                })
            
            analysis[metric_name] = metric_analysis
        
        # Overall assessment
        if 'acc' in analysis:
            analysis['overall'] = {
                'stable': analysis['acc']['test_std'] < 0.05,
                'overfitting': analysis['acc'].get('overfitting_gap', 0) > 0.1,
                'performance_level': 'high' if analysis['acc']['test_mean'] > 0.8 else 'medium' if analysis['acc']['test_mean'] > 0.7 else 'low'
            }
        
        self.evaluation_results[f"{model_name}_cv"] = analysis
        
        logger.info(f"Cross-validation analysis for {model_name}:")
        for metric, stats in analysis.items():
            if isinstance(stats, dict) and 'test_mean' in stats:
                logger.info(f"  {metric}: {stats['test_mean']:.4f} ± {stats['test_std']:.4f}")
        
        return analysis
    
    def evaluate_final_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                           model_name: str = "final_model",
                           output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the final model on the test set.
        
        Parameters:
            model: Trained model
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            model_name (str): Name of the model
            output_file (str, optional): Path to save evaluation report
            
        Returns:
            dict: Final evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")
        
        # Evaluate predictions
        metrics = self.evaluate_predictions(y_test, y_pred, y_pred_proba, model_name)
        
        # Create detailed report
        report = self._create_evaluation_report(y_test, y_pred, y_pred_proba, metrics, model_name)
        
        # Save report if requested
        if output_file:
            self._save_report(report, output_file)
        
        logger.info("Final model evaluation completed")
        return metrics
    
    def _create_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_pred_proba: Optional[np.ndarray], 
                                 metrics: Dict[str, Any],
                                 model_name: str) -> str:
        """Create a detailed evaluation report."""
        report_lines = [
            f"Final Model Evaluation Report: {model_name}",
            "=" * 50,
            "",
            "Performance Metrics:",
            f"  Accuracy: {metrics['accuracy']:.4f}",
            f"  F1 Score (macro): {metrics['f1_macro']:.4f}",
            f"  F1 Score (weighted): {metrics['f1_weighted']:.4f}",
            f"  Precision (macro): {metrics['precision_macro']:.4f}",
            f"  Recall (macro): {metrics['recall_macro']:.4f}",
            f"  Matthews Correlation Coefficient: {metrics['matthews_corrcoef']:.4f}",
            ""
        ]
        
        # Add ROC AUC if available
        if 'roc_auc' in metrics:
            report_lines.append(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            report_lines.append("")
        
        # Add confusion matrix
        cm = metrics['confusion_matrix']
        report_lines.extend([
            "Confusion Matrix:",
            f"  True Negative: {cm[0,0]}, False Positive: {cm[0,1]}",
            f"  False Negative: {cm[1,0]}, True Positive: {cm[1,1]}",
            ""
        ])
        
        return "\n".join(report_lines)
    
    def _save_report(self, report: str, output_file: str):
        """Save evaluation report to file."""
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving evaluation report: {e}")
    
    def calculate_feature_stability(self, feature_weights_list: List[np.ndarray]) -> Dict[str, Any]:
        """
        Calculate stability of feature weights across different models/folds.
        
        Parameters:
            feature_weights_list (list): List of feature weight arrays
            
        Returns:
            dict: Feature stability analysis
        """
        if not feature_weights_list:
            return {}
        
        # Stack weights into matrix
        weights_matrix = np.stack(feature_weights_list)
        
        # Calculate statistics
        stability_analysis = {
            'mean_weights': np.mean(weights_matrix, axis=0),
            'std_weights': np.std(weights_matrix, axis=0),
            'min_weights': np.min(weights_matrix, axis=0),
            'max_weights': np.max(weights_matrix, axis=0),
            'coefficient_of_variation': np.std(weights_matrix, axis=0) / (np.abs(np.mean(weights_matrix, axis=0)) + 1e-10),
            'sign_consistency': np.mean(np.sign(weights_matrix), axis=0),
            'stable_features': [],
            'unstable_features': []
        }
        
        # Identify stable and unstable features
        cv_threshold = 0.5  # Coefficient of variation threshold
        sign_threshold = 0.8  # Sign consistency threshold
        
        for i, (cv, sign_cons) in enumerate(zip(stability_analysis['coefficient_of_variation'], 
                                              stability_analysis['sign_consistency'])):
            if cv < cv_threshold and abs(sign_cons) > sign_threshold:
                stability_analysis['stable_features'].append(i)
            elif cv > 1.0 or abs(sign_cons) < 0.2:
                stability_analysis['unstable_features'].append(i)
        
        logger.info(f"Feature stability analysis:")
        logger.info(f"  Stable features: {len(stability_analysis['stable_features'])}")
        logger.info(f"  Unstable features: {len(stability_analysis['unstable_features'])}")
        
        return stability_analysis
    
    def benchmark_against_baseline(self, model_metrics: Dict[str, Any], 
                                  baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare model performance against baseline.
        
        Parameters:
            model_metrics (dict): Model evaluation metrics
            baseline_metrics (dict): Baseline evaluation metrics
            
        Returns:
            dict: Comparison results
        """
        comparison = {}
        
        # Compare each metric
        for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'matthews_corrcoef']:
            if metric in model_metrics and metric in baseline_metrics:
                model_score = model_metrics[metric]
                baseline_score = baseline_metrics[metric]
                
                improvement = model_score - baseline_score
                relative_improvement = improvement / baseline_score if baseline_score != 0 else np.inf
                
                comparison[metric] = {
                    'model_score': model_score,
                    'baseline_score': baseline_score,
                    'improvement': improvement,
                    'relative_improvement': relative_improvement,
                    'better': improvement > 0
                }
        
        # Overall assessment
        improvements = [comp['improvement'] for comp in comparison.values()]
        comparison['overall'] = {
            'avg_improvement': np.mean(improvements),
            'significant_improvement': np.mean([imp > 0.05 for imp in improvements]) > 0.5
        }
        
        logger.info("Baseline comparison:")
        for metric, comp in comparison.items():
            if isinstance(comp, dict) and 'improvement' in comp:
                logger.info(f"  {metric}: {comp['improvement']:+.4f} ({comp['relative_improvement']:+.2%})")
        
        return comparison
    
    def get_model_insights(self, model, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract insights from a trained model.
        
        Parameters:
            model: Trained model
            feature_names (list, optional): Names of features
            
        Returns:
            dict: Model insights
        """
        insights = {
            'model_type': type(model).__name__,
            'model_params': model.get_params() if hasattr(model, 'get_params') else {}
        }
        
        # Extract feature importance for linear models
        if hasattr(model, 'coef_'):
            weights = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            
            # Top important features
            top_indices = np.argsort(np.abs(weights))[-10:][::-1]
            
            insights['feature_importance'] = {
                'weights': weights,
                'top_positive_features': np.argsort(weights)[-5:][::-1],
                'top_negative_features': np.argsort(weights)[:5],
                'most_important_features': top_indices,
                'sparsity': np.sum(weights == 0) / len(weights)
            }
            
            # Add feature names if provided
            if feature_names:
                insights['feature_importance']['top_positive_names'] = [
                    feature_names[i] for i in insights['feature_importance']['top_positive_features']
                ]
                insights['feature_importance']['top_negative_names'] = [
                    feature_names[i] for i in insights['feature_importance']['top_negative_features']
                ]
        
        return insights
    
    def save_evaluation_results(self, filepath: str):
        """
        Save all evaluation results to file.
        
        Parameters:
            filepath (str): Path to save results
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in self.evaluation_results.items():
                serializable_results[key] = self._make_serializable(value)
            
            import json
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Evaluation results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def apply_to_compound_library(self, model, fingerprints: np.ndarray, 
                                 compound_info: pd.DataFrame,
                                 model_name: str = "model",
                                 threshold: float = 0.5) -> Dict[str, Any]:
        """
        Apply a trained model to an external compound library.
        
        Parameters:
            model: Trained model
            fingerprints (numpy.ndarray): Fingerprints of compounds
            compound_info (pandas.DataFrame): Information about compounds
            model_name (str): Name of the model
            threshold (float): Probability threshold for classification
            
        Returns:
            dict: Results including predictions and filtered compounds
        """
        # Make predictions
        predictions = model.predict(fingerprints)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(fingerprints)
                # For binary classification, get probability of positive class
                if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                    prob_positive = probabilities[:, 1]
                else:
                    prob_positive = probabilities
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")
                prob_positive = None
        else:
            prob_positive = None
        
        # Create results dataframe
        results_df = compound_info.copy()
        results_df['prediction'] = predictions
        results_df['predicted_label'] = ['Blocker' if p == 1 else 'Non-blocker' for p in predictions]
        
        if prob_positive is not None:
            results_df['blocker_probability'] = prob_positive
            results_df['confidence'] = np.abs(prob_positive - 0.5) * 2  # Distance from decision boundary
        
        # Calculate statistics
        n_total = len(results_df)
        n_blockers = np.sum(predictions == 1)
        n_safe = np.sum(predictions == 0)
        
        # Filter compounds
        safe_compounds = results_df[results_df['prediction'] == 0]
        blockers = results_df[results_df['prediction'] == 1]
        
        # High confidence predictions
        if prob_positive is not None:
            high_conf_safe = results_df[
                (results_df['prediction'] == 0) & 
                (results_df['blocker_probability'] < 0.2)
            ]
            high_conf_blockers = results_df[
                (results_df['prediction'] == 1) & 
                (results_df['blocker_probability'] > 0.8)
            ]
        else:
            high_conf_safe = safe_compounds
            high_conf_blockers = blockers
        
        # Create summary
        summary = {
            'model_name': model_name,
            'n_total': n_total,
            'n_blockers': n_blockers,
            'n_safe': n_safe,
            'percent_blockers': (n_blockers / n_total * 100) if n_total > 0 else 0,
            'percent_safe': (n_safe / n_total * 100) if n_total > 0 else 0,
            'n_high_conf_safe': len(high_conf_safe),
            'n_high_conf_blockers': len(high_conf_blockers),
            'results_df': results_df,
            'safe_compounds': safe_compounds,
            'blockers': blockers,
            'high_conf_safe': high_conf_safe,
            'high_conf_blockers': high_conf_blockers
        }
        
        # Log summary
        logger.info(f"Applied {model_name} to {n_total} compounds:")
        logger.info(f"  Predicted blockers: {n_blockers} ({summary['percent_blockers']:.1f}%)")
        logger.info(f"  Predicted safe: {n_safe} ({summary['percent_safe']:.1f}%)")
        if prob_positive is not None:
            logger.info(f"  High confidence safe: {len(high_conf_safe)}")
            logger.info(f"  High confidence blockers: {len(high_conf_blockers)}")
        
        return summary