"""
Main pipeline for antimalarial drug discovery virtual screening.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any, Tuple
import time
from datetime import datetime

from .data_processing.loader import MoleculeLoader
from .filtering.drug_like import DrugLikeFilter
from .similarity.search import SimilaritySearcher
from .visualization.plots import VirtualScreeningPlotter
from .utils.config import ProjectConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AntimalarialScreeningPipeline:
    """Main pipeline for antimalarial drug discovery virtual screening."""
    
    def __init__(self, config: ProjectConfig = None):
        """
        Initialize the screening pipeline.
        
        Args:
            config: Project configuration object
        """
        self.config = config or ProjectConfig()
        
        # Initialize components
        self.loader = MoleculeLoader()
        self.filter = DrugLikeFilter(
            violations_allowed=self.config.filter_config.lipinski_violations_allowed
        )
        self.searcher = SimilaritySearcher(
            fingerprint_type=self.config.fingerprint_config.type,
            similarity_metric=self.config.similarity_config.metric,
            radius=self.config.fingerprint_config.radius,
            n_bits=self.config.fingerprint_config.n_bits
        )
        self.plotter = VirtualScreeningPlotter(
            output_dir=str(Path(self.config.output_dir) / "plots")
        )
        
        # Data storage
        self.library_data = None
        self.reference_data = None
        self.filtered_data = None
        self.similarity_results = None
        
        # Results tracking
        self.pipeline_stats = {}
        self.start_time = None
        
    def load_data(self, library_path: str = None, reference_path: str = None) -> None:
        """
        Load library and reference compound data.
        
        Args:
            library_path: Path to library SDF file
            reference_path: Path to reference compounds SDF file
        """
        logger.info("Starting data loading phase")
        
        # Use config paths if not provided
        library_path = library_path or self.config.input_library
        reference_path = reference_path or self.config.reference_compounds
        
        try:
            # Load library compounds
            logger.info(f"Loading library from {library_path}")
            self.library_data = self.loader.load_and_process(library_path)
            
            if self.library_data.empty:
                raise ValueError("No valid molecules found in library file")
                
            logger.info(f"Loaded {len(self.library_data)} library compounds")
            
            # Load reference compounds
            logger.info(f"Loading reference compounds from {reference_path}")
            reference_loader = MoleculeLoader()
            self.reference_data = reference_loader.load_and_process(reference_path)
            
            if self.reference_data.empty:
                raise ValueError("No valid molecules found in reference file")
                
            logger.info(f"Loaded {len(self.reference_data)} reference compounds")
            
            # Store loading stats
            self.pipeline_stats['library_size'] = len(self.library_data)
            self.pipeline_stats['reference_size'] = len(self.reference_data)
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def apply_filters(self, apply_lipinski: bool = True, 
                     apply_additional: bool = True) -> pd.DataFrame:
        """
        Apply drug-likeness filters to the library.
        
        Args:
            apply_lipinski: Whether to apply Lipinski's Rule of Five
            apply_additional: Whether to apply additional drug-like criteria
            
        Returns:
            Filtered DataFrame
        """
        if self.library_data is None:
            raise ValueError("No library data loaded. Call load_data() first.")
            
        logger.info("Starting filtering phase")
        
        try:
            # Apply filters
            self.filtered_data = self.filter.filter_dataframe(
                self.library_data,
                apply_lipinski=apply_lipinski,
                apply_additional=apply_additional
            )
            
            # Get filter statistics - need to use the data with filter columns
            # Apply filters to get the full data with filter columns for statistics
            full_data_with_filters = self.filter.filter_dataframe(
                self.library_data,
                apply_lipinski=apply_lipinski,
                apply_additional=apply_additional
            )
            filter_stats = self.filter.get_filter_statistics(full_data_with_filters)
            self.pipeline_stats.update(filter_stats)
            
            # Analyze violations
            violation_analysis = self.filter.analyze_violations(self.library_data)
            self.pipeline_stats['violation_analysis'] = violation_analysis
            
            logger.info(f"Filtering complete: {len(self.filtered_data)} molecules passed filters")
            
            return self.filtered_data
            
        except Exception as e:
            logger.error(f"Error during filtering: {str(e)}")
            raise
            
    def perform_similarity_search(self, threshold: float = None, 
                                max_results: int = None) -> pd.DataFrame:
        """
        Perform similarity search against reference compounds.
        
        Args:
            threshold: Similarity threshold (uses config default if None)
            max_results: Maximum results to return (uses config default if None)
            
        Returns:
            DataFrame with similarity search results
        """
        if self.filtered_data is None:
            raise ValueError("No filtered data available. Call apply_filters() first.")
            
        if self.reference_data is None:
            raise ValueError("No reference data loaded. Call load_data() first.")
            
        logger.info("Starting similarity search phase")
        
        # Use config defaults if not provided
        threshold = threshold or self.config.similarity_config.threshold
        max_results = max_results or self.config.similarity_config.max_results
        
        try:
            # Load reference compounds into searcher
            self.searcher.load_reference_compounds(self.reference_data)
            
            # Perform similarity search
            self.similarity_results = self.searcher.search_library(
                self.filtered_data,
                threshold=threshold,
                max_results=max_results
            )
            
            # Get search statistics
            search_stats = self.searcher.get_search_statistics(self.similarity_results)
            self.pipeline_stats.update(search_stats)
            
            logger.info(f"Similarity search complete: {len(self.similarity_results)} hits found")
            
            return self.similarity_results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise
            
    def generate_visualizations(self, create_dashboard: bool = True) -> None:
        """
        Generate visualizations for the screening results.
        
        Args:
            create_dashboard: Whether to create interactive dashboard
        """
        logger.info("Generating visualizations")
        
        try:
            # Descriptor distributions for original library
            if self.library_data is not None:
                self.plotter.plot_descriptor_distributions(
                    self.library_data,
                    save_path=str(Path(self.config.output_dir) / "plots" / "library_descriptors.png")
                )
                
                # Correlation matrix
                self.plotter.plot_descriptor_correlation_matrix(
                    self.library_data,
                    save_path=str(Path(self.config.output_dir) / "plots" / "descriptor_correlations.png")
                )
            
            # Lipinski violations analysis
            if self.filtered_data is not None:
                # Use the full library data with filter results for violation analysis
                full_data_with_filters = self.filter.filter_dataframe(
                    self.library_data, apply_lipinski=True, apply_additional=True
                )
                
                self.plotter.plot_lipinski_violations(
                    full_data_with_filters,
                    save_path=str(Path(self.config.output_dir) / "plots" / "lipinski_violations.png")
                )
            
            # Similarity distribution
            if self.similarity_results is not None and len(self.similarity_results) > 0:
                self.plotter.plot_similarity_distribution(
                    self.similarity_results,
                    save_path=str(Path(self.config.output_dir) / "plots" / "similarity_distribution.png")
                )
                
                # Interactive scatter plot
                if 'MW' in self.similarity_results.columns and 'LogP' in self.similarity_results.columns:
                    self.plotter.plot_interactive_scatter(
                        self.similarity_results,
                        x_col='MW',
                        y_col='LogP',
                        color_col='similarity',
                        hover_cols=['ID', 'NAME', 'similarity'],
                        save_path=str(Path(self.config.output_dir) / "plots" / "mw_vs_logp_interactive.html")
                    )
            
            # Create comprehensive dashboard
            if create_dashboard and all([
                self.library_data is not None,
                self.filtered_data is not None
            ]):
                self.plotter.create_screening_summary_dashboard(
                    original_df=self.library_data,
                    filtered_df=self.filtered_data,
                    similarity_df=self.similarity_results,
                    save_path=str(Path(self.config.output_dir) / "plots" / "screening_dashboard.html")
                )
                
            logger.info("Visualizations generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise
            
    def save_results(self) -> None:
        """Save all results to files."""
        logger.info("Saving results")
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save filtered molecules
            if self.filtered_data is not None:
                filtered_output = output_dir / "filtered_molecules.csv"
                # Remove ROMol column for CSV export
                filtered_export = self.filtered_data.drop(columns=['ROMol', 'fingerprint'], errors='ignore')
                filtered_export.to_csv(filtered_output, index=False)
                logger.info(f"Filtered molecules saved to {filtered_output}")
            
            # Save similarity results
            if self.similarity_results is not None and len(self.similarity_results) > 0:
                similarity_output = output_dir / "similarity_results.csv"
                # Remove ROMol column for CSV export
                similarity_export = self.similarity_results.drop(columns=['ROMol', 'fingerprint'], errors='ignore')
                similarity_export.to_csv(similarity_output, index=False)
                logger.info(f"Similarity results saved to {similarity_output}")
                
                # Save top hits
                top_hits = self.similarity_results.head(50)
                top_hits_output = output_dir / "top_50_hits.csv"
                top_hits_export = top_hits.drop(columns=['ROMol', 'fingerprint'], errors='ignore')
                top_hits_export.to_csv(top_hits_output, index=False)
                logger.info(f"Top 50 hits saved to {top_hits_output}")
            
            # Save pipeline statistics
            stats_output = output_dir / "pipeline_statistics.txt"
            with open(stats_output, 'w') as f:
                f.write("Antimalarial Virtual Screening Pipeline Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for key, value in self.pipeline_stats.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for subkey, subvalue in value.items():
                            f.write(f"  {subkey}: {subvalue}\n")
                        f.write("\n")
                    else:
                        f.write(f"{key}: {value}\n")
                        
            logger.info(f"Pipeline statistics saved to {stats_output}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
            
    def run_full_pipeline(self, library_path: str = None, 
                         reference_path: str = None,
                         save_results: bool = True,
                         generate_plots: bool = True) -> Dict[str, Any]:
        """
        Run the complete virtual screening pipeline.
        
        Args:
            library_path: Path to library SDF file
            reference_path: Path to reference compounds SDF file
            save_results: Whether to save results to files
            generate_plots: Whether to generate visualizations
            
        Returns:
            Dictionary containing pipeline results and statistics
        """
        self.start_time = time.time()
        logger.info("Starting antimalarial virtual screening pipeline")
        
        try:
            # Step 1: Load data
            self.load_data(library_path, reference_path)
            
            # Step 2: Apply filters
            self.apply_filters()
            
            # Step 3: Perform similarity search
            self.perform_similarity_search()
            
            # Step 4: Generate visualizations
            if generate_plots:
                self.generate_visualizations()
            
            # Step 5: Save results
            if save_results:
                self.save_results()
            
            # Calculate total runtime
            total_time = time.time() - self.start_time
            self.pipeline_stats['total_runtime_seconds'] = total_time
            self.pipeline_stats['total_runtime_minutes'] = total_time / 60
            
            logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            
            # Prepare results summary
            results = {
                'library_data': self.library_data,
                'filtered_data': self.filtered_data,
                'similarity_results': self.similarity_results,
                'statistics': self.pipeline_stats,
                'config': self.config
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
            
    def get_pipeline_summary(self) -> str:
        """
        Get a text summary of the pipeline results.
        
        Returns:
            Formatted summary string
        """
        if not self.pipeline_stats:
            return "No pipeline results available. Run the pipeline first."
            
        summary = []
        summary.append("Antimalarial Virtual Screening Pipeline Summary")
        summary.append("=" * 50)
        summary.append("")
        
        # Basic statistics
        if 'library_size' in self.pipeline_stats:
            summary.append(f"Original Library Size: {self.pipeline_stats['library_size']:,} molecules")
            
        if 'drug_like_molecules' in self.pipeline_stats:
            summary.append(f"Drug-like Molecules: {self.pipeline_stats['drug_like_molecules']:,}")
            summary.append(f"Drug-likeness Pass Rate: {self.pipeline_stats['pass_rate']:.1f}%")
            
        if 'total_hits' in self.pipeline_stats:
            summary.append(f"Similarity Search Hits: {self.pipeline_stats['total_hits']:,}")
            
        if 'mean_similarity' in self.pipeline_stats:
            summary.append(f"Mean Similarity Score: {self.pipeline_stats['mean_similarity']:.3f}")
            summary.append(f"Max Similarity Score: {self.pipeline_stats['max_similarity']:.3f}")
            
        if 'total_runtime_minutes' in self.pipeline_stats:
            summary.append(f"Total Runtime: {self.pipeline_stats['total_runtime_minutes']:.2f} minutes")
            
        return "\n".join(summary) 