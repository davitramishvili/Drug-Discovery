"""
Visualization module for Chapter 3 ML screening.
Handles data exploration plots, molecular visualizations, and results analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any, Tuple
import logging
from pathlib import Path

# RDKit visualization
try:
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)

class VisualizationTools:
    """
    Visualization tools for Chapter 3 ML screening analysis.
    
    This class implements the visualization pipeline from Flynn's Chapter 3,
    including data exploration plots, model performance analysis, and molecular visualizations.
    """
    
    def __init__(self):
        """Initialize visualization tools."""
        self.setup_style()
        self.figures_dir = Path("figures/chapter3")
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        logger.info("VisualizationTools initialized")
    
    def setup_style(self):
        """Set up consistent visualization style."""
        colors = ["#A20025", "#6C8EBF"]
        sns.set_palette(sns.color_palette(colors))
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['legend.fontsize'] = 16   
        plt.rcParams['xtick.labelsize'] = 16   
        plt.rcParams['ytick.labelsize'] = 16
    
    def plot_activity_distribution(self, df: pd.DataFrame, 
                                 activity_col: str = "pIC50") -> plt.Figure:
        """Plot distribution of pIC50 values."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(
            df[activity_col], kde=True,
            stat="density", kde_kws=dict(cut=3),
            edgecolor=(1, 1, 1, .4),
            ax=ax
        )
        
        ax.set_title("Distribution of pIC50 Values", fontsize=16)
        ax.set_xlabel("pIC50", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        
        plt.tight_layout()
        self.save_figure(fig, "distribution_pic50")
        return fig
    
    def plot_activity_distribution_with_error(self, df: pd.DataFrame,
                                            activity_col: str = "pIC50", 
                                            error_value: float = 3.0,
                                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution showing the effect of annotation errors.
        
        Parameters:
            df (pandas.DataFrame): DataFrame with activity data
            activity_col (str): Column name with activity values
            error_value (float): Error value to simulate
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Simulate annotation error
        simulated_error = df[activity_col] + error_value
        combined_data = pd.concat([df[activity_col], simulated_error], ignore_index=True)
        
        sns.histplot(
            combined_data, kde=True,
            stat="density", kde_kws=dict(cut=3),
            edgecolor=(1, 1, 1, .4),
            ax=ax
        )
        
        ax.set_title("Distribution of pIC50 with Simulated Annotation Error", fontsize=16)
        ax.set_xlabel("pIC50", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            self.save_figure(fig, save_path)
        else:
            self.save_figure(fig, "distribution_pic50_error")
        
        return fig
    
    def visualize_extreme_molecules(self, df: pd.DataFrame, 
                                  activity_col: str = "pIC50", 
                                  name_col: str = "Name", 
                                  smiles_col: str = "SMILES", 
                                  n: int = 4,
                                  save_path: Optional[str] = None):
        """
        Visualize molecules with extreme activity values.
        
        Parameters:
            df (pandas.DataFrame): DataFrame containing the data
            activity_col (str): Column name with activity values
            name_col (str): Column name with compound names
            smiles_col (str): Column name with SMILES strings
            n (int): Number of molecules to show from each extreme
            save_path (str, optional): Path to save the visualization
            
        Returns:
            RDKit visualization or None
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - cannot visualize molecules")
            return None
        
        # Sort the dataframe by activity
        df_sorted = df.sort_values(activity_col, ascending=False)
        
        # Get the highest and lowest n molecules
        extremes = pd.concat([df_sorted.head(n), df_sorted.dropna().tail(n)])
        
        # Convert SMILES to molecules
        from rdkit import Chem
        mols = [Chem.MolFromSmiles(smi) for smi in extremes[smiles_col]]
        
        # Create legends with name and activity
        legends = [
            f"{name}: pIC50 = {activity:.2f}" 
            for name, activity in zip(extremes[name_col], extremes[activity_col])
        ]
        
        # Create the visualization
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=n,
            subImgSize=(250, 250),
            legends=legends,
            useSVG=True
        )
        
        # Save the image
        if save_path:
            save_file = save_path
        else:
            save_file = str(self.figures_dir / "rdkit_extremes.svg")
        
        with open(save_file, "w") as f:
            f.write(img.data)
        
        logger.info(f"Extreme molecules visualization saved to {save_file}")
        return img
    
    def explore_fingerprint_features(self, fingerprints: np.ndarray,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create exploratory plots for fingerprint features.
        
        Parameters:
            fingerprints (numpy.ndarray): Array of fingerprints
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Calculate statistics
        bits_per_mol = np.sum(fingerprints, axis=1)
        bit_frequency = np.sum(fingerprints, axis=0)
        
        # Plot 1: Distribution of bits per molecule
        axes[0, 0].hist(bits_per_mol, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribution of Bits per Molecule')
        axes[0, 0].set_xlabel('Number of Bits Set')
        axes[0, 0].set_ylabel('Frequency')
        
        # Plot 2: Bit frequency distribution
        axes[0, 1].hist(bit_frequency, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Bit Frequency Distribution')
        axes[0, 1].set_xlabel('Frequency (Number of Molecules)')
        axes[0, 1].set_ylabel('Number of Bits')
        axes[0, 1].set_yscale('log')
        
        # Plot 3: Sparsity visualization
        sparsity = 1 - np.mean(fingerprints)
        axes[1, 0].bar(['Set Bits', 'Unset Bits'], [1-sparsity, sparsity], 
                      color=['#A20025', '#6C8EBF'])
        axes[1, 0].set_title(f'Fingerprint Sparsity: {sparsity:.3f}')
        axes[1, 0].set_ylabel('Proportion')
        
        # Plot 4: Most common bits
        top_bits = np.argsort(bit_frequency)[-20:]
        axes[1, 1].bar(range(len(top_bits)), bit_frequency[top_bits])
        axes[1, 1].set_title('Top 20 Most Common Bits')
        axes[1, 1].set_xlabel('Bit Index (sorted)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            self.save_figure(fig, save_path)
        else:
            self.save_figure(fig, "fingerprint_eda")
        
        return fig
    
    def visualize_cv_results(self, cv_df: pd.DataFrame,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize cross-validation results.
        
        Parameters:
            cv_df (pandas.DataFrame): DataFrame with cross-validation results
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Extract metric names
        metrics = [col.split('_', 1)[1] for col in cv_df.columns if col.startswith('test_')]
        
        # Create a subplot for each metric
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 6), sharey=True)
        if len(metrics) == 1:
            axes = [axes]
        
        fig.suptitle("Cross-Validation Performance: Train vs Test", fontsize=16)
        
        # Set y-axis limits for all subplots
        ylim = (0.5, 1.0)
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Get train and test scores for this metric
            train_scores = cv_df[f'train_{metric}']
            test_scores = cv_df[f'test_{metric}']
            
            # Plot the scores
            x = range(1, len(train_scores) + 1)
            ax.plot(x, train_scores, 'o-', label='Training', 
                   color='#6C8EBF', linewidth=3, markersize=8)
            ax.plot(x, test_scores, 'o-', label='Validation', 
                   color='#A20025', linewidth=3, markersize=8)
            
            # Set labels and title
            ax.set_title(metric.replace('_', ' ').title(), fontsize=18)
            ax.set_xlabel('CV Fold', fontsize=18)
            if i == 0:
                ax.set_ylabel('Score', fontsize=18)
            
            # Set x-ticks
            ax.set_xticks(x)
            
            # Set y-axis limits
            ax.set_ylim(ylim)
            
            # Add legend only to the first subplot
            if i == 0:
                ax.legend(fontsize=16)
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.85)
        
        # Save if path provided
        if save_path:
            self.save_figure(fig, save_path)
        else:
            self.save_figure(fig, "cv_metrics_comparison")
        
        return fig
    
    def plot_model_weights(self, weights_dict: Dict[str, np.ndarray],
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot and compare model weights from different models.
        
        Parameters:
            weights_dict (dict): Dictionary with model names and their weights
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        n_models = len(weights_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, weights) in enumerate(weights_dict.items()):
            ax = axes[i]
            
            # Plot histogram of weights
            sns.histplot(
                weights, kde=True,
                stat="density", kde_kws=dict(cut=3),
                edgecolor=(1, 1, 1, .4),
                ax=ax
            )
            
            ax.set_title(f"{model_name} Weights")
            ax.set_xlabel("Weight Value")
            
            # Add statistics annotation
            stats_text = (
                f"Mean: {weights.mean():.4f}\n"
                f"Std: {weights.std():.4f}\n"
                f"Min: {weights.min():.4f}\n"
                f"Max: {weights.max():.4f}\n"
                f"Non-zero: {np.count_nonzero(weights)}/{len(weights)}"
            )
            
            ax.annotate(
                stats_text,
                xy=(0.95, 0.95),
                xycoords='axes fraction',
                ha='right',
                va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            self.save_figure(fig, save_path)
        else:
            self.save_figure(fig, "model_weights_comparison")
        
        return fig
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                            class_names: Optional[List[str]] = None,
                            title: str = "Confusion Matrix",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Parameters:
            cm (numpy.ndarray): Confusion matrix
            class_names (list, optional): Names of classes
            title (str): Title for the plot
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(cm.shape[0])]
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            self.save_figure(fig, save_path)
        else:
            self.save_figure(fig, "confusion_matrix")
        
        return fig
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc: float,
                      title: str = "ROC Curve",
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve.
        
        Parameters:
            fpr (numpy.ndarray): False positive rates
            tpr (numpy.ndarray): True positive rates
            auc (float): Area under the curve
            title (str): Title for the plot
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='#A20025', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            self.save_figure(fig, save_path)
        else:
            self.save_figure(fig, "roc_curve")
        
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str):
        """Save figure in multiple formats."""
        for fmt in ['svg', 'png']:
            filepath = self.figures_dir / f"{filename}.{fmt}"
            try:
                fig.savefig(filepath, bbox_inches='tight', dpi=600)
                logger.info(f"Figure saved: {filepath}")
            except Exception as e:
                logger.error(f"Error saving figure {filepath}: {e}")
    
    def create_model_comparison_plot(self, comparison_df: pd.DataFrame,
                                   primary_metric: str = 'F1_Macro',
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comparison plot of different models.
        
        Parameters:
            comparison_df (pandas.DataFrame): DataFrame with model comparison results
            primary_metric (str): Primary metric to highlight
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Select numeric columns for plotting
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Model']
        
        # Create grouped bar plot
        x = np.arange(len(comparison_df))
        width = 0.8 / len(numeric_cols)
        
        for i, col in enumerate(numeric_cols):
            offset = (i - len(numeric_cols)/2 + 0.5) * width
            bars = ax.bar(x + offset, comparison_df[col], width, 
                         label=col.replace('_', ' '), alpha=0.8)
            
            # Highlight primary metric
            if col == primary_metric:
                for bar in bars:
                    bar.set_edgecolor('#A20025')
                    bar.set_linewidth(2)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            self.save_figure(fig, save_path)
        else:
            self.save_figure(fig, "model_comparison")
        
        return fig 