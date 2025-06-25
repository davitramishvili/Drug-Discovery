"""
Antimalarial Drug Discovery Pipeline

A comprehensive virtual screening pipeline for identifying potential antimalarial compounds.
"""

__version__ = "1.0.0"
__author__ = "Drug Discovery Pipeline Team"

# Main pipeline import
from .pipeline import AntimalarialScreeningPipeline

# Configuration
from .utils.config import ProjectConfig

# Core functionality exports
__all__ = [
    'AntimalarialScreeningPipeline',
    'ProjectConfig',
]
