#!/usr/bin/env python3
"""
Script to run the antimalarial drug discovery virtual screening pipeline.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import AntimalarialScreeningPipeline
from src.utils.config import ProjectConfig

def main():
    """Run the antimalarial screening pipeline."""
    
    print("Antimalarial Drug Discovery Virtual Screening Pipeline")
    print("=" * 60)
    print()
    
    try:
        # Initialize configuration
        config = ProjectConfig()
        
        # Create pipeline
        pipeline = AntimalarialScreeningPipeline(config)
        
        # Run the full pipeline
        print("Starting pipeline execution...")
        results = pipeline.run_full_pipeline()
        
        # Print summary
        print("\n" + "=" * 60)
        print("PIPELINE RESULTS SUMMARY")
        print("=" * 60)
        print(pipeline.get_pipeline_summary())
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print(f"Results saved to: {config.output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")
        print("Please check the error message above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 