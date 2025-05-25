"""
Module for creating visualizations for virtual screening analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('default')
sns.set_palette("husl")

class VirtualScreeningPlotter:
    """Class for creating visualizations for virtual screening analysis."""
    
    def __init__(self, output_dir: str = "results/plots"):
        """
        Initialize the VirtualScreeningPlotter.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting parameters
        self.figure_size = (10, 6)
        self.dpi = 300
        
    def plot_descriptor_distributions(self, df: pd.DataFrame, 
                                    descriptors: List[str] = None,
                                    save_path: str = None) -> None:
        """
        Plot distributions of molecular descriptors.
        
        Args:
            df: DataFrame containing molecular descriptors
            descriptors: List of descriptors to plot (default: Lipinski descriptors)
            save_path: Path to save the plot
        """
        if descriptors is None:
            descriptors = ['MW', 'LogP', 'HBA', 'HBD', 'TPSA', 'RotBonds']
            
        # Filter descriptors that exist in the DataFrame
        available_descriptors = [desc for desc in descriptors if desc in df.columns]
        
        if not available_descriptors:
            logger.warning("No specified descriptors found in DataFrame")
            return
            
        # Create subplots
        n_descriptors = len(available_descriptors)
        n_cols = 3
        n_rows = (n_descriptors + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
            
        # Plot each descriptor
        for i, descriptor in enumerate(available_descriptors):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Remove NaN values for plotting
            data = df[descriptor].dropna()
            
            if len(data) > 0:
                # Histogram
                ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
                ax.set_xlabel(descriptor)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {descriptor}')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = data.mean()
                median_val = data.median()
                ax.axvline(mean_val, color='red', linestyle='--', 
                          label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='orange', linestyle='--', 
                          label=f'Median: {median_val:.2f}')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No data available', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{descriptor} - No Data')
                
        # Remove empty subplots
        for i in range(n_descriptors, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Descriptor distributions plot saved to {save_path}")
        else:
            save_path = self.output_dir / "descriptor_distributions.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Descriptor distributions plot saved to {save_path}")
            
        plt.show()
        
    def plot_lipinski_violations(self, df: pd.DataFrame, 
                               save_path: str = None) -> None:
        """
        Plot Lipinski rule violations analysis.
        
        Args:
            df: DataFrame with Lipinski analysis results
            save_path: Path to save the plot
        """
        if 'lipinski_violations' not in df.columns:
            logger.warning("No Lipinski violations data found in DataFrame")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Violations distribution
        violation_counts = df['lipinski_violations'].value_counts().sort_index()
        ax1.bar(violation_counts.index, violation_counts.values, 
                alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Number of Lipinski Violations')
        ax1.set_ylabel('Number of Molecules')
        ax1.set_title('Distribution of Lipinski Violations')
        ax1.grid(True, alpha=0.3)
        
        # Add percentages on bars
        total_molecules = len(df)
        for i, count in enumerate(violation_counts.values):
            percentage = count / total_molecules * 100
            ax1.text(violation_counts.index[i], count + total_molecules * 0.01, 
                    f'{percentage:.1f}%', ha='center', va='bottom')
        
        # Pass/fail pie chart
        if 'drug_like' in df.columns:
            pass_fail_counts = df['drug_like'].value_counts()
            labels = ['Drug-like', 'Non-drug-like']
            colors = ['lightgreen', 'lightcoral']
            
            ax2.pie(pass_fail_counts.values, labels=labels, colors=colors, 
                   autopct='%1.1f%%', startangle=90)
            ax2.set_title('Drug-likeness Classification')
        else:
            ax2.text(0.5, 0.5, 'No drug-likeness data available', 
                    transform=ax2.transAxes, ha='center', va='center')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Lipinski violations plot saved to {save_path}")
        else:
            save_path = self.output_dir / "lipinski_violations.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Lipinski violations plot saved to {save_path}")
            
        plt.show()
        
    def plot_similarity_distribution(self, df: pd.DataFrame, 
                                   save_path: str = None) -> None:
        """
        Plot similarity score distribution.
        
        Args:
            df: DataFrame containing similarity scores
            save_path: Path to save the plot
        """
        if 'similarity' not in df.columns:
            logger.warning("No similarity data found in DataFrame")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Similarity histogram
        similarities = df['similarity'].dropna()
        ax1.hist(similarities, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        ax1.set_xlabel('Similarity Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Similarity Scores')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_sim = similarities.mean()
        median_sim = similarities.median()
        ax1.axvline(mean_sim, color='red', linestyle='--', 
                   label=f'Mean: {mean_sim:.3f}')
        ax1.axvline(median_sim, color='orange', linestyle='--', 
                   label=f'Median: {median_sim:.3f}')
        ax1.legend()
        
        # Similarity ranges
        ranges = [(0.9, 1.0, 'Very High'), (0.8, 0.9, 'High'), 
                 (0.7, 0.8, 'Medium'), (0.6, 0.7, 'Low'), (0.0, 0.6, 'Very Low')]
        
        range_counts = []
        range_labels = []
        
        for min_sim, max_sim, label in ranges:
            count = len(similarities[(similarities >= min_sim) & (similarities < max_sim)])
            if count > 0:
                range_counts.append(count)
                range_labels.append(label)
                
        if range_counts:
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(range_counts)))
            ax2.pie(range_counts, labels=range_labels, colors=colors, 
                   autopct='%1.1f%%', startangle=90)
            ax2.set_title('Similarity Score Ranges')
        else:
            ax2.text(0.5, 0.5, 'No similarity data available', 
                    transform=ax2.transAxes, ha='center', va='center')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Similarity distribution plot saved to {save_path}")
        else:
            save_path = self.output_dir / "similarity_distribution.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Similarity distribution plot saved to {save_path}")
            
        plt.show()
        
    def plot_descriptor_correlation_matrix(self, df: pd.DataFrame, 
                                         descriptors: List[str] = None,
                                         save_path: str = None) -> None:
        """
        Plot correlation matrix of molecular descriptors.
        
        Args:
            df: DataFrame containing molecular descriptors
            descriptors: List of descriptors to include
            save_path: Path to save the plot
        """
        if descriptors is None:
            descriptors = ['MW', 'LogP', 'HBA', 'HBD', 'TPSA', 'RotBonds']
            
        # Filter descriptors that exist in the DataFrame
        available_descriptors = [desc for desc in descriptors if desc in df.columns]
        
        if len(available_descriptors) < 2:
            logger.warning("Need at least 2 descriptors for correlation matrix")
            return
            
        # Calculate correlation matrix
        corr_data = df[available_descriptors].corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, 
                   cbar_kws={"shrink": .8})
        
        plt.title('Molecular Descriptor Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Correlation matrix plot saved to {save_path}")
        else:
            save_path = self.output_dir / "descriptor_correlation_matrix.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Correlation matrix plot saved to {save_path}")
            
        plt.show()
        
    def plot_interactive_scatter(self, df: pd.DataFrame, 
                               x_col: str, y_col: str,
                               color_col: str = None,
                               size_col: str = None,
                               hover_cols: List[str] = None,
                               save_path: str = None) -> None:
        """
        Create interactive scatter plot using Plotly.
        
        Args:
            df: DataFrame containing data
            x_col: Column for x-axis
            y_col: Column for y-axis
            color_col: Column for color coding
            size_col: Column for point sizes
            hover_cols: Additional columns to show on hover
            save_path: Path to save the plot
        """
        if x_col not in df.columns or y_col not in df.columns:
            logger.warning(f"Columns {x_col} or {y_col} not found in DataFrame")
            return
            
        # Prepare hover data
        hover_data = {}
        if hover_cols:
            for col in hover_cols:
                if col in df.columns:
                    hover_data[col] = True
                    
        # Create scatter plot
        fig = px.scatter(df, x=x_col, y=y_col, 
                        color=color_col, size=size_col,
                        hover_data=hover_data,
                        title=f'{y_col} vs {x_col}')
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            hovermode='closest'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive scatter plot saved to {save_path}")
        else:
            save_path = self.output_dir / f"interactive_scatter_{x_col}_vs_{y_col}.html"
            fig.write_html(save_path)
            logger.info(f"Interactive scatter plot saved to {save_path}")
            
        fig.show()
        
    def create_screening_summary_dashboard(self, original_df: pd.DataFrame,
                                         filtered_df: pd.DataFrame,
                                         similarity_df: pd.DataFrame = None,
                                         save_path: str = None) -> None:
        """
        Create a comprehensive dashboard summarizing the screening results.
        
        Args:
            original_df: Original molecule library
            filtered_df: Filtered molecules
            similarity_df: Similarity search results
            save_path: Path to save the dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Library Size', 'Drug-likeness Filter', 
                          'Descriptor Distributions', 'Similarity Scores',
                          'Top Hits', 'Screening Funnel'),
            specs=[[{"type": "indicator"}, {"type": "pie"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "bar"}, {"type": "funnel"}]]
        )
        
        # Library size indicator
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=len(original_df),
                title={"text": "Total Molecules"},
                number={'font': {'size': 40}}
            ),
            row=1, col=1
        )
        
        # Drug-likeness pie chart
        if 'drug_like' in filtered_df.columns:
            drug_like_counts = filtered_df['drug_like'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=['Drug-like', 'Non-drug-like'],
                    values=drug_like_counts.values,
                    name="Drug-likeness"
                ),
                row=1, col=2
            )
        
        # Molecular weight distribution
        if 'MW' in original_df.columns:
            fig.add_trace(
                go.Histogram(
                    x=original_df['MW'].dropna(),
                    name="Molecular Weight",
                    nbinsx=30
                ),
                row=1, col=3
            )
        
        # Similarity scores
        if similarity_df is not None and 'similarity' in similarity_df.columns:
            fig.add_trace(
                go.Histogram(
                    x=similarity_df['similarity'].dropna(),
                    name="Similarity Scores",
                    nbinsx=20
                ),
                row=2, col=1
            )
        
        # Top hits by similarity
        if similarity_df is not None and len(similarity_df) > 0:
            top_hits = similarity_df.head(10)
            fig.add_trace(
                go.Bar(
                    x=list(range(1, len(top_hits) + 1)),
                    y=top_hits['similarity'],
                    name="Top 10 Hits"
                ),
                row=2, col=2
            )
        
        # Screening funnel
        funnel_data = [len(original_df)]
        funnel_labels = ['Original Library']
        
        if len(filtered_df) > 0:
            funnel_data.append(len(filtered_df))
            funnel_labels.append('Drug-like')
            
        if similarity_df is not None:
            funnel_data.append(len(similarity_df))
            funnel_labels.append('Similar to References')
            
        fig.add_trace(
            go.Funnel(
                y=funnel_labels,
                x=funnel_data,
                name="Screening Funnel"
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title_text="Virtual Screening Dashboard",
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Screening dashboard saved to {save_path}")
        else:
            save_path = self.output_dir / "screening_dashboard.html"
            fig.write_html(save_path)
            logger.info(f"Screening dashboard saved to {save_path}")
            
        fig.show() 