from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os

@dataclass
class FilterConfig:
    lipinski_violations_allowed: int = 1
    apply_pains: bool = True
    apply_brenk: bool = True
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
    input_library: str = "data/raw/enhanced_library.sdf"
    reference_compounds: str = "data/reference/enhanced_malaria_box.sdf"
    output_dir: str = "results/"
    
    # Processing configurations
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    fingerprint_config: FingerprintConfig = field(default_factory=FingerprintConfig)
    similarity_config: SimilarityConfig = field(default_factory=SimilarityConfig)
    
    def __post_init__(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)