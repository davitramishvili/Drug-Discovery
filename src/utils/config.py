from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os
from pathlib import Path

def find_project_root(current_path: Path = None) -> Path:
    """Find the project root by looking for key directories or files."""
    if current_path is None:
        current_path = Path(__file__).parent
    
    # Search up the directory tree
    for parent in [current_path] + list(current_path.parents):
        # Look for data directory or other project indicators
        if (parent / 'data').exists() and (parent / 'src').exists():
            return parent
        # Also check for specific files that indicate project root
        if (parent / 'requirements.txt').exists() or (parent / 'README.md').exists():
            if (parent / 'data').exists():
                return parent
    
    # Fallback to current directory
    return Path('.')

# Get project root once for use throughout
PROJECT_ROOT = find_project_root()

@dataclass
class FilterConfig:
    lipinski_violations_allowed: int = 1
    apply_pains: bool = True
    apply_brenk: bool = True
    apply_nih: bool = False  # NIH filter off by default
    custom_alerts_file: Optional[str] = None

@dataclass
class FingerprintConfig:
    type: str = "morgan"
    radius: int = 2
    n_bits: int = 2048

@dataclass
class SimilarityConfig:
    metric: str = "tanimoto"  # or "dice"
    threshold: float = 0.5  # Lowered for better demo results
    max_results: int = 1000

@dataclass
class ProjectConfig:
    # File paths
    input_library: str = "data/raw/test_library.sdf"
    reference_compounds: str = "data/reference/malaria_box.sdf"
    output_dir: str = "results/"
    
    # Processing configurations
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    fingerprint_config: FingerprintConfig = field(default_factory=FingerprintConfig)
    similarity_config: SimilarityConfig = field(default_factory=SimilarityConfig)
    
    def __post_init__(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

@dataclass
class DataPaths:
    """Centralized data path configuration."""
    
    # Base directories
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    reference_dir: str = "data/reference"
    processed_dir: str = "data/processed"
    chapter3_dir: str = "data/chapter3"
    
    # Common data files
    specs_sdf: str = "data/raw/Specs.sdf"
    malaria_box_sdf: str = "data/reference/malaria_box_400.sdf"
    enhanced_malaria_box_sdf: str = "data/reference/enhanced_malaria_box.sdf"
    extended_malaria_box_sdf: str = "data/reference/extended_malaria_box.sdf"
    
    # Chapter 3 files
    herg_data: str = "data/chapter3/hERG_blockers.xlsx"
    
    # Test files
    small_test_library: str = "data/raw/small_test_library.sdf"
    small_reference: str = "data/reference/small_reference.sdf"
    
    def get_path(self, relative_to_project_root: bool = True) -> 'DataPaths':
        """Get paths relative to project root or current directory."""
        if relative_to_project_root:
            return self
        else:
            # For examples in subdirectories, adjust paths
            adjusted = DataPaths()
            for field_name, field_value in self.__dict__.items():
                if isinstance(field_value, str) and field_value.startswith("data/"):
                    setattr(adjusted, field_name, f"../../../{field_value}")
                else:
                    setattr(adjusted, field_name, field_value)
            return adjusted
    
    def resolve_specs_path(self, from_examples: bool = False) -> str:
        """Resolve the Specs.sdf path based on context."""
        if from_examples:
            return "../../../data/raw/Specs.sdf"
        return self.specs_sdf
    
    def resolve_malaria_path(self, from_examples: bool = False) -> str:
        """Resolve the malaria box path based on context."""
        if from_examples:
            return "../../../data/reference/malaria_box_400.sdf"
        return self.malaria_box_sdf


# Global data paths instance
data_paths = DataPaths()