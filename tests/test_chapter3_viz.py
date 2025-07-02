#!/usr/bin/env python3
"""
Test script for Chapter 3 visualizations
"""

import sys
from pathlib import Path

# Add src to path - now that we're in tests/, we need to go up one level
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from chapter3_ml_screening.advanced_visualization import create_chapter3_visualizations, Chapter3Visualizer
    print("✅ Successfully imported Chapter3 visualization module")
    
    # Test basic functionality
    visualizer = Chapter3Visualizer(output_dir="test_viz")
    print("✅ Chapter3Visualizer initialized successfully")
    
    # Test with sample data
    sample_results = {
        'herg_model_comparison': {
            'sgd_results': {'accuracy': 0.85, 'f1_score': 0.80, 'matthews_cc': 0.65},
            'rf_results': {'accuracy': 0.88, 'f1_score': 0.83, 'matthews_cc': 0.70},
            'best_model': 'Random Forest',
            'best_mcc': 0.70
        },
        'data_info': {
            'total_molecules': 1500,
            'training_samples': 1000,
            'test_samples': 500,
            'fingerprint_dimensions': 2048
        },
        'compound_screening': {
            'specs': {
                'summary': {
                    'total_compounds': 1000,
                    'predicted_safe': 650,
                    'predicted_blockers': 350,
                    'blocker_percentage': 35.0,
                    'risk_distribution': {'LOW': 500, 'MEDIUM': 150, 'HIGH': 200}
                }
            },
            'malaria_box': {
                'summary': {
                    'total_compounds': 400,
                    'predicted_safe': 280,
                    'predicted_blockers': 120,
                    'blocker_percentage': 30.0,
                    'risk_distribution': {'LOW': 200, 'MEDIUM': 80, 'HIGH': 40}
                }
            }
        }
    }
    
    # Test visualization creation
    dashboard_path = visualizer.create_model_comparison_dashboard(sample_results)
    print(f"✅ Model comparison dashboard created: {dashboard_path}")
    
    screening_path = visualizer.create_screening_results_dashboard(sample_results['compound_screening'])
    print(f"✅ Screening results dashboard created: {screening_path}")
    
    print("\n🎉 All Chapter 3 visualization tests passed!")
    print("📊 Ready to enhance Chapter 3 exercises with beautiful visualizations!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc() 