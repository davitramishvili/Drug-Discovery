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