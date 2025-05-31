#!/usr/bin/env python3
"""
Main script to run the antimalarial drug discovery virtual screening pipeline.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import AntimalarialScreeningPipeline
from utils.config import ProjectConfig
from utils.dataset_manager import DatasetManager

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Antimalarial Drug Discovery Virtual Screening Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --dataset small          # Run with small test dataset (5 molecules)
  python run_pipeline.py --dataset diverse        # Run with diverse dataset (20 molecules)
  python run_pipeline.py --dataset large          # Run with large dataset (50 molecules)
  python run_pipeline.py --dataset custom         # Run with custom dataset files
  python run_pipeline.py --list-datasets          # Show available datasets
  python run_pipeline.py --library my_lib.sdf --reference my_ref.sdf  # Custom files
        """
    )
    
    parser.add_argument(
        "--dataset", "-d",
        choices=["small", "diverse", "large", "custom"],
        default="diverse",
        help="Choose dataset to run (default: diverse)"
    )
    
    parser.add_argument(
        "--library", "-l",
        type=str,
        help="Path to custom library SDF file (overrides dataset selection)"
    )
    
    parser.add_argument(
        "--reference", "-r", 
        type=str,
        help="Path to custom reference SDF file (overrides dataset selection)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots and visualizations"
    )
    
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for screening (default: 0.5)"
    )
    
    parser.add_argument(
        "--lipinski-violations",
        type=int,
        default=1,
        help="Maximum allowed Lipinski violations (default: 1)"
    )
    
    # Add structural filter arguments
    parser.add_argument(
        "--disable-pains",
        action="store_true",
        help="Disable PAINS structural filter"
    )
    
    parser.add_argument(
        "--disable-brenk",
        action="store_true",
        help="Disable BRENK structural filter"
    )
    
    parser.add_argument(
        "--enable-nih",
        action="store_true",
        help="Enable NIH (NIH Molecular Libraries) structural filter"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the virtual screening pipeline."""
    
    args = parse_arguments()
    
    # Initialize dataset manager
    dataset_manager = DatasetManager()
    
    # Handle list datasets request
    if args.list_datasets:
        dataset_manager.list_datasets()
        return
    
    print("Antimalarial Drug Discovery Virtual Screening Pipeline")
    print("=" * 60)
    
    try:
        # Determine input files
        if args.library and args.reference:
            # Use custom files provided via command line
            library_path = args.library
            reference_path = args.reference
            dataset_name = "custom (command line)"
            print(f"Using custom files:")
            print(f"  Library: {library_path}")
            print(f"  Reference: {reference_path}")
        else:
            # Use predefined dataset
            dataset_name = args.dataset
            library_path, reference_path = dataset_manager.setup_dataset(dataset_name)
            dataset_info = dataset_manager.get_available_datasets()[dataset_name]
            print(f"Using dataset: {dataset_info['name']}")
            print(f"  Description: {dataset_info['description']}")
            print(f"  Expected molecules: {dataset_info['molecules']}")
            print(f"  Library: {library_path}")
            print(f"  Reference: {reference_path}")
        
        print()
        print("Pipeline Configuration:")
        print(f"  Output directory: {args.output}")
        print(f"  Similarity threshold: {args.similarity_threshold}")
        print(f"  Max Lipinski violations: {args.lipinski_violations}")
        print(f"  Generate plots: {not args.no_plots}")
        print()
        print("Starting pipeline execution...")
        
        # Initialize configuration with custom parameters
        config = ProjectConfig()
        config.input_library = library_path
        config.reference_compounds = reference_path
        config.output_dir = args.output
        config.similarity_config.threshold = args.similarity_threshold
        config.filter_config.lipinski_violations_allowed = args.lipinski_violations
        
        # Update structural filter settings
        config.filter_config.apply_pains = not args.disable_pains
        config.filter_config.apply_brenk = not args.disable_brenk
        config.filter_config.apply_nih = args.enable_nih
        
        # Print filter configuration
        print(f"  PAINS filter: {'enabled' if not args.disable_pains else 'disabled'}")
        print(f"  BRENK filter: {'enabled' if not args.disable_brenk else 'disabled'}")
        print(f"  NIH filter: {'enabled' if args.enable_nih else 'disabled'}")
        
        # Initialize and run pipeline
        pipeline = AntimalarialScreeningPipeline(config)
        
        # Run the complete pipeline
        results = pipeline.run_full_pipeline(
            library_path=library_path,
            reference_path=reference_path,
            generate_plots=not args.no_plots
        )
        
        # Print summary
        print()
        print("=" * 60)
        print("PIPELINE RESULTS SUMMARY")
        print("=" * 60)
        print(pipeline.get_pipeline_summary())
        print()
        print("=" * 60)
        print("Pipeline completed successfully!")
        print(f"Results saved to: {config.output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 