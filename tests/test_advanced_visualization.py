#!/usr/bin/env python3
"""
Comprehensive test suite for Chapter 3 advanced visualization system.

This module provides thorough testing coverage for all visualization components
including dashboards, plots, data handling, and integration features.
"""

import unittest
import json
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import plotly.graph_objects as go
    from chapter3_ml_screening.advanced_visualization import (
        Chapter3Visualizer, 
        create_chapter3_visualizations
    )
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class TestChapter3Visualizer(unittest.TestCase):
    """Test suite for Chapter3Visualizer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        if not PLOTLY_AVAILABLE:
            self.skipTest("Plotly not available")
            
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.visualizer = Chapter3Visualizer(output_dir=self.test_dir)
        
        # Sample test data
        self.sample_results = {
            'herg_model_comparison': {
                'sgd_results': {
                    'accuracy': 0.7062,
                    'f1_score': 0.7635,
                    'matthews_cc': 0.4156
                },
                'rf_results': {
                    'accuracy': 0.7784,
                    'f1_score': 0.8481,
                    'matthews_cc': 0.4454
                },
                'best_model': 'Random Forest',
                'best_mcc': 0.4454
            },
            'data_info': {
                'total_molecules': 587,
                'training_samples': 393,
                'test_samples': 194,
                'fingerprint_dimensions': 2048
            },
            'compound_screening': {
                'screening_results': {
                    'specs': {
                        'summary': {
                            'total_compounds': 1000,
                            'predicted_safe': 283,
                            'predicted_blockers': 717,
                            'blocker_percentage': 71.7,
                            'risk_distribution': {'HIGH': 66, 'MEDIUM': 665, 'LOW': 269}
                        }
                    },
                    'malaria_box': {
                        'summary': {
                            'total_compounds': 400,
                            'predicted_safe': 12,
                            'predicted_blockers': 388,
                            'blocker_percentage': 97.0,
                            'risk_distribution': {'HIGH': 92, 'MEDIUM': 296, 'LOW': 12}
                        }
                    }
                }
            },
            'combined_safety_assessment': {
                'combined_screening_results': {
                    'specs': {
                        'summary': {
                            'total_compounds': 1000,
                            'combined_safe': 264,
                            'combined_safe_percentage': 26.4
                        }
                    }
                }
            }
        }
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'test_dir'):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_visualizer_initialization(self):
        """Test Chapter3Visualizer initialization."""
        # Test default initialization
        viz = Chapter3Visualizer()
        self.assertIsInstance(viz.output_dir, Path)
        # Use Path for cross-platform compatibility
        self.assertEqual(viz.output_dir, Path("results/visualizations"))
        
        # Test custom directory
        custom_dir = "custom_test_viz"
        viz_custom = Chapter3Visualizer(output_dir=custom_dir)
        self.assertEqual(viz_custom.output_dir, Path(custom_dir))
        
        # Test color schemes
        self.assertIn('primary', viz.colors)
        self.assertIn('secondary', viz.colors)
        self.assertIn('Random Forest', viz.model_colors)
    
    def test_model_comparison_dashboard(self):
        """Test model comparison dashboard creation."""
        dashboard_path = self.visualizer.create_model_comparison_dashboard(self.sample_results)
        
        # Check file creation
        self.assertTrue(Path(dashboard_path).exists())
        self.assertTrue(dashboard_path.endswith('.html'))
        
        # Check file content
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('Model Comparison', content)
            self.assertIn('Random Forest', content)
            self.assertIn('SGD Classifier', content)
            self.assertIn('plotly', content.lower())
    
    def test_screening_results_dashboard(self):
        """Test screening results dashboard creation."""
        dashboard_path = self.visualizer.create_screening_results_dashboard(
            self.sample_results['compound_screening']
        )
        
        # Check file creation
        self.assertTrue(Path(dashboard_path).exists())
        self.assertTrue(dashboard_path.endswith('.html'))
        
        # Check file content
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('Screening Results', content)
            self.assertIn('Specs', content)
            self.assertIn('Malaria Box', content)
    
    def test_safety_assessment_heatmap(self):
        """Test safety assessment heatmap creation."""
        heatmap_path = self.visualizer.create_safety_assessment_heatmap(
            self.sample_results['combined_safety_assessment']
        )
        
        # Check file creation
        self.assertTrue(Path(heatmap_path).exists())
        self.assertTrue(heatmap_path.endswith('.html'))
        
        # Check file content
        with open(heatmap_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('Safety', content)
            self.assertIn('Risk', content)
    
    def test_comprehensive_report(self):
        """Test comprehensive report creation."""
        report_path = self.visualizer.create_comprehensive_report(self.sample_results)
        
        # Check file creation
        self.assertTrue(Path(report_path).exists())
        self.assertTrue(report_path.endswith('.html'))
        
        # Check file content
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('Chapter 3', content)
            self.assertIn('Machine Learning', content)
            self.assertIn('Executive Summary', content)
            self.assertIn('Model Performance', content)
    
    def test_publication_plots(self):
        """Test publication plot creation."""
        plot_paths = self.visualizer.create_publication_plots(self.sample_results)
        
        # Check that plots were created
        self.assertIsInstance(plot_paths, list)
        self.assertGreater(len(plot_paths), 0)
        
        # Check first plot file
        if plot_paths:
            plot_path = plot_paths[0]
            self.assertTrue(Path(plot_path).exists())
            self.assertTrue(plot_path.endswith('.png'))
    
    def test_data_structure_handling(self):
        """Test handling of different data structures."""
        # Test with nested screening results
        nested_data = {
            'compound_screening': {
                'screening_results': {
                    'specs': {
                        'summary': {
                            'total_compounds': 100,
                            'predicted_safe': 30,
                            'predicted_blockers': 70,
                            'blocker_percentage': 70.0,
                            'risk_distribution': {'HIGH': 10, 'MEDIUM': 60, 'LOW': 30}
                        }
                    }
                }
            }
        }
        
        dashboard_path = self.visualizer.create_screening_results_dashboard(
            nested_data['compound_screening']
        )
        self.assertTrue(Path(dashboard_path).exists())
        
        # Test with direct screening results
        direct_data = {
            'specs': {
                'summary': {
                    'total_compounds': 100,
                    'predicted_safe': 30,
                    'predicted_blockers': 70,
                    'blocker_percentage': 70.0,
                    'risk_distribution': {'HIGH': 10, 'MEDIUM': 60, 'LOW': 30}
                }
            }
        }
        
        dashboard_path2 = self.visualizer.create_screening_results_dashboard(direct_data)
        self.assertTrue(Path(dashboard_path2).exists())
    
    def test_color_customization(self):
        """Test color scheme customization."""
        # Modify colors
        original_primary = self.visualizer.colors['primary']
        self.visualizer.colors['primary'] = '#FF0000'
        
        # Verify change
        self.assertEqual(self.visualizer.colors['primary'], '#FF0000')
        self.assertNotEqual(self.visualizer.colors['primary'], original_primary)
        
        # Test that dashboard still works with custom colors
        dashboard_path = self.visualizer.create_model_comparison_dashboard(self.sample_results)
        self.assertTrue(Path(dashboard_path).exists())
    
    def test_missing_data_handling(self):
        """Test handling of missing or incomplete data."""
        # Test with minimal data
        minimal_data = {
            'herg_model_comparison': {
                'sgd_results': {'accuracy': 0.5, 'f1_score': 0.5, 'matthews_cc': 0.1},
                'rf_results': {'accuracy': 0.6, 'f1_score': 0.6, 'matthews_cc': 0.2},
                'best_model': 'Random Forest',
                'best_mcc': 0.2
            }
        }
        
        # Should not raise errors
        dashboard_path = self.visualizer.create_model_comparison_dashboard(minimal_data)
        self.assertTrue(Path(dashboard_path).exists())
        
        # Test with empty screening data
        empty_screening = {'screening_results': {}}
        dashboard_path2 = self.visualizer.create_screening_results_dashboard(empty_screening)
        self.assertTrue(Path(dashboard_path2).exists())


class TestVisualizationIntegration(unittest.TestCase):
    """Test integration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PLOTLY_AVAILABLE:
            self.skipTest("Plotly not available")
            
        self.test_dir = tempfile.mkdtemp()
        self.sample_results = {
            'herg_model_comparison': {
                'sgd_results': {'accuracy': 0.7, 'f1_score': 0.75, 'matthews_cc': 0.4},
                'rf_results': {'accuracy': 0.8, 'f1_score': 0.85, 'matthews_cc': 0.5},
                'best_model': 'Random Forest',
                'best_mcc': 0.5
            },
            'data_info': {
                'total_molecules': 500,
                'training_samples': 300,
                'test_samples': 200,
                'fingerprint_dimensions': 1024
            }
        }
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_create_chapter3_visualizations(self):
        """Test the main visualization creation function."""
        outputs = create_chapter3_visualizations(
            self.sample_results, 
            output_dir=self.test_dir
        )
        
        # Check return structure
        self.assertIsInstance(outputs, dict)
        self.assertIn('comprehensive_report', outputs)
        
        # Check files were created
        for viz_type, path in outputs.items():
            self.assertTrue(Path(path).exists(), f"{viz_type} file not created: {path}")
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        # Create a results file
        results_file = Path(self.test_dir) / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.sample_results, f)
        
        # Load and process
        with open(results_file, 'r') as f:
            loaded_results = json.load(f)
        
        outputs = create_chapter3_visualizations(loaded_results, output_dir=self.test_dir)
        self.assertIsInstance(outputs, dict)
        self.assertGreater(len(outputs), 0)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with None input
        outputs = create_chapter3_visualizations(None, output_dir=self.test_dir)
        self.assertEqual(outputs, {})
        
        # Test with empty dict
        outputs = create_chapter3_visualizations({}, output_dir=self.test_dir)
        self.assertEqual(outputs, {})
        
        # Test with invalid directory (should create it)
        invalid_dir = Path(self.test_dir) / "nonexistent" / "deep" / "path"
        outputs = create_chapter3_visualizations(
            self.sample_results, 
            output_dir=str(invalid_dir)
        )
        self.assertTrue(invalid_dir.exists())


class TestDataValidation(unittest.TestCase):
    """Test data validation and processing."""
    
    def test_numeric_data_validation(self):
        """Test validation of numeric data."""
        if not PLOTLY_AVAILABLE:
            self.skipTest("Plotly not available")
            
        test_dir = tempfile.mkdtemp()
        visualizer = Chapter3Visualizer(output_dir=test_dir)
        
        try:
            # Test with valid numeric data
            valid_data = {
                'herg_model_comparison': {
                    'sgd_results': {'accuracy': 0.75, 'f1_score': 0.80, 'matthews_cc': 0.45},
                    'rf_results': {'accuracy': 0.85, 'f1_score': 0.90, 'matthews_cc': 0.55},
                    'best_model': 'Random Forest',
                    'best_mcc': 0.55
                }
            }
            
            dashboard_path = visualizer.create_model_comparison_dashboard(valid_data)
            self.assertTrue(Path(dashboard_path).exists())
            
            # Test with edge case values
            edge_data = {
                'herg_model_comparison': {
                    'sgd_results': {'accuracy': 0.0, 'f1_score': 0.0, 'matthews_cc': -1.0},
                    'rf_results': {'accuracy': 1.0, 'f1_score': 1.0, 'matthews_cc': 1.0},
                    'best_model': 'Random Forest',
                    'best_mcc': 1.0
                }
            }
            
            dashboard_path2 = visualizer.create_model_comparison_dashboard(edge_data)
            self.assertTrue(Path(dashboard_path2).exists())
            
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
    
    def test_string_data_validation(self):
        """Test validation of string data."""
        if not PLOTLY_AVAILABLE:
            self.skipTest("Plotly not available")
            
        test_dir = tempfile.mkdtemp()
        visualizer = Chapter3Visualizer(output_dir=test_dir)
        
        try:
            # Test with various model names
            test_cases = [
                'Random Forest',
                'SGD Classifier', 
                'Support Vector Machine',
                'Neural Network',
                'XGBoost'
            ]
            
            for model_name in test_cases:
                data = {
                    'herg_model_comparison': {
                        'sgd_results': {'accuracy': 0.7, 'f1_score': 0.75, 'matthews_cc': 0.4},
                        'rf_results': {'accuracy': 0.8, 'f1_score': 0.85, 'matthews_cc': 0.5},
                        'best_model': model_name,
                        'best_mcc': 0.5
                    }
                }
                
                dashboard_path = visualizer.create_model_comparison_dashboard(data)
                self.assertTrue(Path(dashboard_path).exists())
                
                # Verify model name appears in output
                with open(dashboard_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.assertIn(model_name, content)
        
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


class TestPerformance(unittest.TestCase):
    """Test performance aspects of the visualization system."""
    
    def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        if not PLOTLY_AVAILABLE:
            self.skipTest("Plotly not available")
            
        test_dir = tempfile.mkdtemp()
        visualizer = Chapter3Visualizer(output_dir=test_dir)
        
        try:
            # Create larger dataset
            large_data = {
                'compound_screening': {
                    'screening_results': {
                        'specs': {
                            'summary': {
                                'total_compounds': 100000,
                                'predicted_safe': 25000,
                                'predicted_blockers': 75000,
                                'blocker_percentage': 75.0,
                                'risk_distribution': {'HIGH': 15000, 'MEDIUM': 60000, 'LOW': 25000}
                            }
                        },
                        'malaria_box': {
                            'summary': {
                                'total_compounds': 50000,
                                'predicted_safe': 5000,
                                'predicted_blockers': 45000,
                                'blocker_percentage': 90.0,
                                'risk_distribution': {'HIGH': 20000, 'MEDIUM': 25000, 'LOW': 5000}
                            }
                        }
                    }
                }
            }
            
            # Should handle large numbers without issues
            dashboard_path = visualizer.create_screening_results_dashboard(
                large_data['compound_screening']
            )
            self.assertTrue(Path(dashboard_path).exists())
            
            # Check that large numbers are properly formatted
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn('100000', content)  # Should contain the large number
        
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
    
    def test_memory_usage(self):
        """Test memory usage patterns."""
        if not PLOTLY_AVAILABLE:
            self.skipTest("Plotly not available")
            
        # This is a basic test - in production you might use memory profilers
        test_dir = tempfile.mkdtemp()
        
        try:
            # Create multiple visualizers to test memory cleanup
            for i in range(10):
                visualizer = Chapter3Visualizer(output_dir=test_dir)
                data = {
                    'herg_model_comparison': {
                        'sgd_results': {'accuracy': 0.7, 'f1_score': 0.75, 'matthews_cc': 0.4},
                        'rf_results': {'accuracy': 0.8, 'f1_score': 0.85, 'matthews_cc': 0.5},
                        'best_model': 'Random Forest',
                        'best_mcc': 0.5
                    }
                }
                
                dashboard_path = visualizer.create_model_comparison_dashboard(data)
                self.assertTrue(Path(dashboard_path).exists())
                
                # Clean up visualizer
                del visualizer
        
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


def run_visualization_tests():
    """Run all visualization tests."""
    if not PLOTLY_AVAILABLE:
        print("⚠️  Plotly not available - skipping visualization tests")
        print("   Install with: pip install plotly")
        return False
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestChapter3Visualizer,
        TestVisualizationIntegration,
        TestDataValidation,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 Running Chapter 3 Advanced Visualization Tests")
    print("=" * 60)
    
    success = run_visualization_tests()
    
    if success:
        print("\n✅ All visualization tests passed!")
    else:
        print("\n❌ Some tests failed. Check output above for details.")
    
    exit(0 if success else 1) 