#!/usr/bin/env python3
"""
Generate Chapter 2 Hits - One-Time Setup
========================================

This script generates the top 1000 hits from Chapter 2 similarity screening
and saves them to a file for reuse in subsequent analyses.

This eliminates the need to repeat the expensive similarity search computation.

Usage:
    python generate_chapter2_hits.py
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List
import sys
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import pipeline components
from pipeline import AntimalarialScreeningPipeline
from utils.config import ProjectConfig
from data_processing.loader import MoleculeLoader

# Suppress warnings
warnings.filterwarnings('ignore')

class SpecsHitsGenerator:
    """Generates and saves the top 1000 hits from Chapter 2 screening."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the hits generator."""
        self.random_seed = random_seed
        self.output_file = "chapter2_top_1000_hits.csv"
        
        print("🎯 Chapter 2 Hits Generator Initialized")
        print(f"   📁 Will save results to: {self.output_file}")
    
    def check_existing_hits(self) -> bool:
        """Check if hits file already exists."""
        hits_path = Path(self.output_file)
        if hits_path.exists():
            try:
                existing_df = pd.read_csv(hits_path)
                print(f"✅ Found existing hits file: {hits_path}")
                print(f"   📊 Contains {len(existing_df)} compounds")
                
                # Show sample of existing data
                print("\n📋 Sample of existing hits:")
                sample_cols = ['ID', 'NAME', 'SMILES', 'MW', 'LogP', 'similarity']
                available_cols = [col for col in sample_cols if col in existing_df.columns]
                print(existing_df[available_cols].head())
                
                return True
            except Exception as e:
                print(f"⚠️  Error reading existing file: {e}")
                return False
        return False
    
    def generate_hits_from_pipeline(self, max_hits: int = 1000, max_library_size: int = 15000) -> pd.DataFrame:
        """Generate hits using the antimalarial screening pipeline."""
        print(f"\n🔍 Generating Top {max_hits} Hits from Chapter 2 Pipeline")
        print("="*60)
        
        # Initialize the antimalarial screening pipeline with relaxed filtering
        config = ProjectConfig()
        config.similarity_config.threshold = 0.58  # Based on pipeline stats
        config.similarity_config.max_results = max_hits
        
        # Use more permissive filtering
        config.filter_config.lipinski_violations_allowed = 2  # Allow up to 2 violations
        config.filter_config.apply_pains = False  # Skip PAINS for now
        config.filter_config.apply_brenk = False  # Skip BRENK for now
        config.filter_config.apply_nih = False   # Skip NIH alerts
        
        pipeline = AntimalarialScreeningPipeline(config)
        
        # Check data files
        specs_path = Path("data/raw/Specs.sdf")
        malaria_path = Path("data/reference/malaria_box_400.sdf")
        
        if not specs_path.exists():
            print(f"❌ Library file not found: {specs_path}")
            print("   Please ensure the Specs.sdf file is in the data/raw/ directory")
            return None
        
        if not malaria_path.exists():
            print(f"❌ Reference file not found: {malaria_path}")
            print("   Please ensure the malaria_box_400.sdf file is in the data/reference/ directory")
            return None
        
        print(f"✅ Found library file: {specs_path}")
        print(f"✅ Found reference file: {malaria_path}")
        
        try:
            start_time = time.time()
            
            print("\n📚 Loading molecular data...")
            pipeline.load_data(
                library_path=str(specs_path),
                reference_path=str(malaria_path)
            )
            
            # Optimize for performance by limiting library size
            original_size = len(pipeline.library_df) if hasattr(pipeline, 'library_df') else 0
            if hasattr(pipeline, 'library_df') and len(pipeline.library_df) > max_library_size:
                print(f"⚡ Optimizing: Using first {max_library_size} compounds from {original_size} total")
                print("   (For full screening, increase max_library_size parameter)")
                pipeline.library_df = pipeline.library_df.head(max_library_size)
            else:
                print(f"📊 Using all {original_size} compounds from library")
            
            print("\n🧪 Applying drug-likeness filters...")
            pipeline.apply_filters()
            
            filtered_size = len(pipeline.library_df) if hasattr(pipeline, 'library_df') else 0
            print(f"   Compounds after filtering: {filtered_size}")
            
            print(f"\n🔍 Performing similarity search for top {max_hits} hits...")
            print("   This step computes molecular fingerprints and similarities...")
            print("   ⏱️  Estimated time: 2-10 minutes depending on library size")
            
            hits_df = pipeline.perform_similarity_search(max_results=max_hits)
            
            elapsed_time = time.time() - start_time
            
            if hits_df is not None and not hits_df.empty:
                print(f"\n🎉 SUCCESS! Generated {len(hits_df)} hits in {elapsed_time:.1f} seconds")
                
                # Add some analysis
                if 'similarity' in hits_df.columns:
                    avg_sim = hits_df['similarity'].mean()
                    min_sim = hits_df['similarity'].min()
                    max_sim = hits_df['similarity'].max()
                    print(f"   📈 Similarity range: {min_sim:.3f} - {max_sim:.3f} (avg: {avg_sim:.3f})")
                
                return hits_df
            else:
                print(f"\n❌ No hits generated after {elapsed_time:.1f} seconds")
                print("   Try lowering the similarity threshold or check your data files")
                return None
                
        except Exception as e:
            print(f"\n❌ Error during pipeline execution: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_hits(self, hits_df: pd.DataFrame) -> bool:
        """Save hits to CSV file."""
        if hits_df is None or hits_df.empty:
            print("❌ No hits to save")
            return False
        
        try:
            output_path = Path(self.output_file)
            hits_df.to_csv(output_path, index=False)
            
            print(f"\n💾 Hits saved successfully!")
            print(f"   📁 File: {output_path.absolute()}")
            print(f"   📊 Compounds: {len(hits_df)}")
            print(f"   💽 File size: {output_path.stat().st_size / 1024:.1f} KB")
            
            # Show summary
            print(f"\n📋 Data Summary:")
            print(f"   Columns: {list(hits_df.columns)}")
            if 'MW' in hits_df.columns:
                print(f"   MW range: {hits_df['MW'].min():.1f} - {hits_df['MW'].max():.1f}")
            if 'LogP' in hits_df.columns:
                print(f"   LogP range: {hits_df['LogP'].min():.2f} - {hits_df['LogP'].max():.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving hits: {e}")
            return False
    
    def load_existing_hits(self) -> pd.DataFrame:
        """Load existing hits from file."""
        try:
            hits_path = Path(self.output_file)
            hits_df = pd.read_csv(hits_path)
            print(f"✅ Loaded {len(hits_df)} existing hits from {hits_path}")
            return hits_df
        except Exception as e:
            print(f"❌ Error loading existing hits: {e}")
            return None

def main():
    """Main execution function."""
    print("="*70)
    print("🎯 CHAPTER 2 HITS GENERATOR")
    print("One-time generation of top 1000 antimalarial hits for reuse")
    print("="*70)
    
    generator = SpecsHitsGenerator(random_seed=42)
    
    # Check if hits already exist
    if generator.check_existing_hits():
        user_input = input("\nDo you want to regenerate hits? (y/N): ").lower().strip()
        if user_input not in ['y', 'yes']:
            print("✅ Using existing hits file")
            existing_hits = generator.load_existing_hits()
            if existing_hits is not None:
                print("\n🎉 Ready for safety analysis!")
                print(f"   Use: python chapter2_hits_safety_analysis_optimized.py")
            return existing_hits
    
    # Generate new hits
    print("\n🚀 Starting hits generation...")
    hits_df = generator.generate_hits_from_pipeline(
        max_hits=1000,
        max_library_size=15000  # Adjust this based on your computational resources
    )
    
    if hits_df is not None:
        # Save hits
        if generator.save_hits(hits_df):
            print("\n🎉 Setup complete! Ready for safety analysis!")
            print(f"   Next step: python chapter2_hits_safety_analysis_optimized.py")
            return hits_df
        else:
            print("❌ Failed to save hits")
            return None
    else:
        print("❌ Failed to generate hits")
        return None

if __name__ == "__main__":
    hits = main() 