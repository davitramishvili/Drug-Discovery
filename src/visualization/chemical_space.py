"""
Module for chemical space visualization and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ChemicalSpaceAnalyzer:
    """Class for analyzing and visualizing chemical space."""
    
    def __init__(self, output_dir: str = "results/plots"):
        """
        Initialize the ChemicalSpaceAnalyzer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting parameters
        self.figure_size = (12, 8)
        self.dpi = 300
        
    def prepare_descriptor_matrix(self, df: pd.DataFrame, 
                                descriptors: List[str] = None) -> np.ndarray:
        """
        Prepare descriptor matrix for dimensionality reduction.
        
        Args:
            df: DataFrame containing molecular descriptors
            descriptors: List of descriptors to use
            
        Returns:
            Standardized descriptor matrix
        """
        if descriptors is None:
            descriptors = ['MW', 'LogP', 'HBA', 'HBD', 'TPSA', 'RotBonds']
            
        # Filter available descriptors
        available_descriptors = [desc for desc in descriptors if desc in df.columns]
        
        if len(available_descriptors) < 2:
            raise ValueError("Need at least 2 descriptors for chemical space analysis")
            
        # Extract descriptor matrix
        X = df[available_descriptors].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, available_descriptors
        
    def plot_pca_space(self, df: pd.DataFrame, 
                      color_col: str = None,
                      descriptors: List[str] = None,
                      save_path: str = None) -> None:
        """
        Plot PCA chemical space.
        
        Args:
            df: DataFrame containing molecular data
            color_col: Column to use for coloring points
            descriptors: List of descriptors to use
            save_path: Path to save the plot
        """
        try:
            # Prepare data
            X_scaled, desc_names = self.prepare_descriptor_matrix(df, descriptors)
            
            # Perform PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create plot
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            if color_col and color_col in df.columns:
                # Color by category
                unique_categories = df[color_col].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
                
                for i, category in enumerate(unique_categories):
                    mask = df[color_col] == category
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                             c=[colors[i]], label=category, alpha=0.7, s=60)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=60)
                
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax.set_title('Chemical Space - PCA Projection')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"PCA plot saved to {save_path}")
            else:
                save_path = self.output_dir / "chemical_space_pca.png"
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"PCA plot saved to {save_path}")
                
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating PCA plot: {str(e)}")
            
    def plot_tsne_space(self, df: pd.DataFrame,
                       color_col: str = None,
                       descriptors: List[str] = None,
                       perplexity: int = 30,
                       save_path: str = None) -> None:
        """
        Plot t-SNE chemical space.
        
        Args:
            df: DataFrame containing molecular data
            color_col: Column to use for coloring points
            descriptors: List of descriptors to use
            perplexity: t-SNE perplexity parameter
            save_path: Path to save the plot
        """
        try:
            # Prepare data
            X_scaled, desc_names = self.prepare_descriptor_matrix(df, descriptors)
            
            # Adjust perplexity if dataset is small
            n_samples = X_scaled.shape[0]
            perplexity = min(perplexity, max(5, n_samples // 4))
            
            # Perform t-SNE
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            X_tsne = tsne.fit_transform(X_scaled)
            
            # Create plot
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            if color_col and color_col in df.columns:
                # Color by category
                unique_categories = df[color_col].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
                
                for i, category in enumerate(unique_categories):
                    mask = df[color_col] == category
                    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                             c=[colors[i]], label=category, alpha=0.7, s=60)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7, s=60)
                
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title(f'Chemical Space - t-SNE Projection (perplexity={perplexity})')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"t-SNE plot saved to {save_path}")
            else:
                save_path = self.output_dir / "chemical_space_tsne.png"
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"t-SNE plot saved to {save_path}")
                
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating t-SNE plot: {str(e)}")
            
    def plot_interactive_chemical_space(self, df: pd.DataFrame,
                                      method: str = "pca",
                                      color_col: str = None,
                                      size_col: str = None,
                                      hover_cols: List[str] = None,
                                      descriptors: List[str] = None,
                                      save_path: str = None) -> None:
        """
        Create interactive chemical space plot using Plotly.
        
        Args:
            df: DataFrame containing molecular data
            method: "pca" or "tsne"
            color_col: Column to use for coloring points
            size_col: Column to use for point sizes
            hover_cols: Additional columns to show on hover
            descriptors: List of descriptors to use
            save_path: Path to save the plot
        """
        try:
            # Prepare data
            X_scaled, desc_names = self.prepare_descriptor_matrix(df, descriptors)
            
            # Perform dimensionality reduction
            if method.lower() == "pca":
                reducer = PCA(n_components=2)
                X_reduced = reducer.fit_transform(X_scaled)
                x_label = f'PC1 ({reducer.explained_variance_ratio_[0]:.1%})'
                y_label = f'PC2 ({reducer.explained_variance_ratio_[1]:.1%})'
                title = "Interactive Chemical Space - PCA"
            elif method.lower() == "tsne":
                perplexity = min(30, max(5, X_scaled.shape[0] // 4))
                reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                X_reduced = reducer.fit_transform(X_scaled)
                x_label = "t-SNE 1"
                y_label = "t-SNE 2"
                title = f"Interactive Chemical Space - t-SNE (perplexity={perplexity})"
            else:
                raise ValueError("Method must be 'pca' or 'tsne'")
                
            # Create DataFrame for plotting
            plot_df = df.copy()
            plot_df['x'] = X_reduced[:, 0]
            plot_df['y'] = X_reduced[:, 1]
            
            # Prepare hover data
            hover_data = {}
            if hover_cols:
                for col in hover_cols:
                    if col in plot_df.columns:
                        hover_data[col] = True
                        
            # Create scatter plot
            fig = px.scatter(plot_df, x='x', y='y',
                           color=color_col, size=size_col,
                           hover_data=hover_data,
                           title=title)
            
            fig.update_layout(
                xaxis_title=x_label,
                yaxis_title=y_label,
                hovermode='closest'
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Interactive chemical space plot saved to {save_path}")
            else:
                save_path = self.output_dir / f"chemical_space_{method}_interactive.html"
                fig.write_html(save_path)
                logger.info(f"Interactive chemical space plot saved to {save_path}")
                
            fig.show()
            
        except Exception as e:
            logger.error(f"Error creating interactive chemical space plot: {str(e)}")
            
    def analyze_chemical_diversity(self, df: pd.DataFrame,
                                 descriptors: List[str] = None) -> Dict[str, float]:
        """
        Analyze chemical diversity of the dataset.
        
        Args:
            df: DataFrame containing molecular data
            descriptors: List of descriptors to use
            
        Returns:
            Dictionary with diversity metrics
        """
        try:
            X_scaled, desc_names = self.prepare_descriptor_matrix(df, descriptors)
            
            # Calculate diversity metrics
            diversity_metrics = {}
            
            # PCA analysis
            pca = PCA()
            pca.fit(X_scaled)
            
            # Cumulative variance explained
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            diversity_metrics['pca_2d_variance'] = cumvar[1]  # Variance in first 2 PCs
            diversity_metrics['pca_3d_variance'] = cumvar[2] if len(cumvar) > 2 else cumvar[-1]
            
            # Effective dimensionality (number of PCs needed for 90% variance)
            effective_dims = np.argmax(cumvar >= 0.9) + 1
            diversity_metrics['effective_dimensionality'] = effective_dims
            
            # Mean pairwise distance in descriptor space
            from scipy.spatial.distance import pdist
            distances = pdist(X_scaled, metric='euclidean')
            diversity_metrics['mean_pairwise_distance'] = np.mean(distances)
            diversity_metrics['std_pairwise_distance'] = np.std(distances)
            
            return diversity_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing chemical diversity: {str(e)}")
            return {}
            
    def create_diversity_report(self, df: pd.DataFrame,
                              descriptors: List[str] = None,
                              save_path: str = None) -> str:
        """
        Create a comprehensive diversity analysis report.
        
        Args:
            df: DataFrame containing molecular data
            descriptors: List of descriptors to use
            save_path: Path to save the report
            
        Returns:
            Report text
        """
        try:
            diversity_metrics = self.analyze_chemical_diversity(df, descriptors)
            
            report = []
            report.append("Chemical Diversity Analysis Report")
            report.append("=" * 40)
            report.append("")
            report.append(f"Dataset size: {len(df)} molecules")
            report.append("")
            
            if diversity_metrics:
                report.append("Diversity Metrics:")
                report.append("-" * 20)
                report.append(f"PCA 2D variance captured: {diversity_metrics.get('pca_2d_variance', 0):.1%}")
                report.append(f"PCA 3D variance captured: {diversity_metrics.get('pca_3d_variance', 0):.1%}")
                report.append(f"Effective dimensionality: {diversity_metrics.get('effective_dimensionality', 0)}")
                report.append(f"Mean pairwise distance: {diversity_metrics.get('mean_pairwise_distance', 0):.3f}")
                report.append(f"Std pairwise distance: {diversity_metrics.get('std_pairwise_distance', 0):.3f}")
                
            report_text = "\n".join(report)
            
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Diversity report saved to {save_path}")
            else:
                save_path = self.output_dir / "chemical_diversity_report.txt"
                with open(save_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Diversity report saved to {save_path}")
                
            return report_text
            
        except Exception as e:
            logger.error(f"Error creating diversity report: {str(e)}")
            return "Error generating diversity report" 