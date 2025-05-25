# Antimalarial Drug Discovery Pipeline

A comprehensive virtual screening pipeline for identifying potential antimalarial compounds using similarity searching and molecular property filtering.

## Overview

This pipeline implements a systematic approach to virtual drug discovery, specifically designed for antimalarial compound identification. It combines molecular descriptor calculation, drug-likeness filtering, and similarity-based virtual screening to identify promising compounds from large chemical libraries.

## Features

- **Molecular Data Processing**: Load and process SDF files with RDKit
- **Descriptor Calculation**: Compute Lipinski descriptors and molecular properties
- **Drug-likeness Filtering**: Apply Lipinski's Rule of Five and additional criteria
- **Similarity Searching**: Find compounds similar to known antimalarials using molecular fingerprints
- **Chemical Space Analysis**: PCA and t-SNE visualization of molecular diversity
- **Comprehensive Visualization**: Generate plots, charts, and interactive dashboards
- **Interactive Analysis**: Plotly-based interactive plots and dashboards
- **Diversity Analysis**: Chemical diversity metrics and reports
- **Flexible Configuration**: Easily customizable parameters and thresholds
- **Results Export**: Save filtered compounds and analysis results

## Project Structure

```
Drug-Discovery/
├── data/
│   ├── raw/                    # Raw input data
│   │   └── enhanced_library.sdf # Enhanced compound library (30 molecules)
│   ├── reference/              # Reference compounds
│   │   └── enhanced_malaria_box.sdf # Known antimalarial compounds (8 molecules)
│   └── processed/              # Processed data files
├── src/
│   ├── data_processing/        # Data loading and processing
│   │   ├── loader.py          # SDF file loading and descriptor calculation
│   │   └── descriptors.py     # Molecular descriptor functions
│   ├── filtering/              # Drug-likeness filtering
│   │   └── drug_like.py       # Lipinski and drug-like filters
│   ├── similarity/             # Similarity searching
│   │   ├── fingerprints.py    # Molecular fingerprint generation
│   │   └── search.py          # Similarity search algorithms
│   ├── visualization/          # Plotting and visualization
│   │   ├── plots.py           # Comprehensive plotting functions
│   │   └── chemical_space.py  # Chemical space analysis and PCA/t-SNE
│   ├── utils/                  # Utilities and configuration
│   │   └── config.py          # Configuration classes
│   └── pipeline.py            # Main pipeline orchestration
├── notebooks/                  # Jupyter notebooks
│   └── antimalarial_screening_demo.ipynb
├── tests/                      # Unit tests
├── results/                    # Output directory
├── requirements.txt           # Python dependencies
├── run_pipeline.py           # Main execution script
└── README.md                 # This file
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Drug-Discovery
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Running the Complete Pipeline

```bash
python run_pipeline.py
```

This will execute the entire pipeline using the default configuration and test data.

### Using the Jupyter Notebook

```bash
jupyter notebook notebooks/antimalarial_screening_demo.ipynb
```

The notebook provides an interactive environment for exploring the pipeline and analyzing results.

### Programmatic Usage

```python
from src.pipeline import AntimalarialScreeningPipeline
from src.utils.config import ProjectConfig

# Initialize pipeline
config = ProjectConfig()
pipeline = AntimalarialScreeningPipeline(config)

# Run complete pipeline
results = pipeline.run_full_pipeline()

# Access results
library_data = results['library_data']
filtered_data = results['filtered_data']
similarity_results = results['similarity_results']
```

## Usage

### Dataset Selection

The pipeline comes with multiple predefined datasets for different use cases:

#### List Available Datasets
```bash
python run_pipeline.py --list-datasets
# or
python list_datasets.py
```

#### Dataset Options

**1. Small Dataset (Quick Testing)**
```bash
python run_pipeline.py --dataset small
```
- **5 molecules**: Well-known drugs (Aspirin, Caffeine, Ibuprofen, Paracetamol, Benzene)
- **Purpose**: Quick testing and validation
- **Runtime**: ~30 seconds
- **Violations**: Minimal (0-1 violations per molecule)

**2. Diverse Dataset (Comprehensive Analysis) - DEFAULT**
```bash
python run_pipeline.py --dataset diverse
# or simply
python run_pipeline.py
```
- **20 molecules**: Wide range of drug-likeness properties
- **Purpose**: Comprehensive analysis with meaningful variance
- **Runtime**: ~1-2 minutes
- **Violations**: 0-3+ violations per molecule (45% drug-like)

**3. Large Dataset (Extensive Testing)**
```bash
python run_pipeline.py --dataset large
```
- **50 molecules**: Natural products, peptides, complex structures
- **Purpose**: Extensive testing with full spectrum of properties
- **Runtime**: ~3-5 minutes
- **Violations**: Full spectrum of drug-likeness

**4. Custom Dataset**
```bash
python run_pipeline.py --dataset custom
```
- **Your data**: Place `custom_library.sdf` and `custom_reference.sdf` in data directories
- **Purpose**: Analyze your own molecular data

### Command Line Options

#### Basic Usage
```bash
# Choose specific dataset
python run_pipeline.py --dataset small|diverse|large|custom

# Use custom files (overrides dataset selection)
python run_pipeline.py --library my_library.sdf --reference my_reference.sdf

# Change output directory
python run_pipeline.py --output my_results

# Skip generating plots (faster execution)
python run_pipeline.py --no-plots
```

#### Advanced Configuration
```bash
# Adjust similarity threshold (0.0-1.0)
python run_pipeline.py --similarity-threshold 0.7

# Change maximum allowed Lipinski violations
python run_pipeline.py --lipinski-violations 2

# Combine multiple options
python run_pipeline.py --dataset large --similarity-threshold 0.6 --output large_results
```

#### Help and Information
```bash
# Show all available options
python run_pipeline.py --help

# List available datasets
python run_pipeline.py --list-datasets
```

### Example Workflows

#### 1. Quick Validation
```bash
# Fast test with small dataset
python run_pipeline.py --dataset small --no-plots
```

#### 2. Standard Analysis
```bash
# Default comprehensive analysis
python run_pipeline.py --dataset diverse
```

#### 3. Extensive Research
```bash
# Large dataset with strict similarity
python run_pipeline.py --dataset large --similarity-threshold 0.8 --output research_results
```

#### 4. Custom Data Analysis
```bash
# Analyze your own data
python run_pipeline.py --library data/raw/my_compounds.sdf --reference data/reference/my_targets.sdf --output my_analysis
```

### Using Custom Datasets

To use your own data with the `custom` dataset option:

1. **Prepare your files**:
   - Library: `data/raw/custom_library.sdf`
   - Reference: `data/reference/custom_reference.sdf`

2. **Run the pipeline**:
   ```bash
   python run_pipeline.py --dataset custom
   ```

Alternatively, specify custom files directly:
```bash
python run_pipeline.py --library path/to/your/library.sdf --reference path/to/your/reference.sdf
```

### Understanding Results

#### Drug-likeness Analysis
- **Pass Rate**: Percentage of molecules meeting drug-likeness criteria
- **Violation Distribution**: Breakdown of Lipinski rule violations
- **Filter Statistics**: Detailed analysis of which criteria are most commonly violated

#### Similarity Search
- **Hit Rate**: Number of molecules similar to reference compounds
- **Score Distribution**: Range and distribution of similarity scores
- **Top Hits**: Most promising compounds for further investigation

#### Visualizations
- **Static Plots**: High-quality PNG images for publications
- **Interactive Plots**: HTML files for detailed exploration and analysis
- **Dashboard**: Comprehensive overview of entire screening process

### Output Files

The pipeline generates several output files in the specified results directory:

#### CSV Files
- `filtered_molecules.csv`: Molecules that passed drug-likeness filters
- `similarity_results.csv`: Similarity search results with scores
- `top_50_hits.csv`: Top 50 most similar compounds

#### Analysis Files
- `pipeline_statistics.txt`: Detailed statistics and violation analysis
- `chemical_diversity_report.txt`: Chemical space diversity analysis

#### Visualizations (`plots/` directory)
- `library_descriptors.png`: Molecular descriptor distributions
- `lipinski_violations.png`: Drug-likeness analysis with pie charts
- `similarity_distribution.png`: Similarity score distributions
- `descriptor_correlations.png`: Correlation matrix heatmap
- `chemical_space_pca.png`: PCA visualization of chemical space
- `screening_dashboard.html`: Interactive dashboard (open in browser)
- `mw_vs_logp_interactive.html`: Interactive molecular weight vs LogP plot
- `chemical_space_interactive.html`: Interactive chemical space exploration

## Pipeline Workflow

### 1. Data Loading
- Load compound library from SDF file
- Load reference antimalarial compounds
- Calculate molecular descriptors (MW, LogP, HBA, HBD, TPSA, RotBonds)

### 2. Drug-likeness Filtering
- Apply Lipinski's Rule of Five:
  - Molecular weight ≤ 500 Da
  - LogP ≤ 5
  - Hydrogen bond acceptors ≤ 10
  - Hydrogen bond donors ≤ 5
- Additional criteria:
  - TPSA ≤ 140 Ų
  - Rotatable bonds ≤ 10

### 3. Similarity Searching
- Generate molecular fingerprints (Morgan, RDKit, or MACCS)
- Calculate similarity to reference compounds
- Filter by similarity threshold
- Rank results by similarity score

### 4. Visualization and Analysis
- Descriptor distribution plots
- Lipinski violations analysis
- Similarity score distributions
- Interactive scatter plots
- Comprehensive dashboard

### 5. Results Export
- Filtered molecules (CSV)
- Similarity search results (CSV)
- Top hits (CSV)
- Pipeline statistics (TXT)
- Visualizations (PNG/HTML)

## Configuration

The pipeline behavior can be customized through the `ProjectConfig` class:

```python
from src.utils.config import ProjectConfig, FilterConfig, FingerprintConfig, SimilarityConfig

# Custom configuration
config = ProjectConfig(
    input_library="path/to/your/library.sdf",
    reference_compounds="path/to/references.sdf",
    output_dir="custom_results/",
    filter_config=FilterConfig(lipinski_violations_allowed=1),
    fingerprint_config=FingerprintConfig(type="morgan", radius=2, n_bits=2048),
    similarity_config=SimilarityConfig(metric="tanimoto", threshold=0.5)
)
```

### Available Parameters

**Filtering Options:**
- `lipinski_violations_allowed`: Number of Lipinski violations allowed (default: 1)
- `apply_pains`: Apply PAINS filters (default: True)
- `apply_brenk`: Apply Brenk filters (default: True)

**Fingerprint Options:**
- `type`: "morgan", "rdkit", or "maccs" (default: "morgan")
- `radius`: Radius for Morgan fingerprints (default: 2)
- `n_bits`: Number of bits for fingerprints (default: 2048)

**Similarity Options:**
- `metric`: "tanimoto", "dice", "cosine", or "jaccard" (default: "tanimoto")
- `threshold`: Minimum similarity threshold (default: 0.5)
- `max_results`: Maximum number of results (default: 1000)

## Output Files

The pipeline generates several output files in the `results/` directory:

- `filtered_molecules.csv`: Molecules passing drug-likeness filters
- `similarity_results.csv`: All similarity search hits
- `top_50_hits.csv`: Top 50 most similar compounds
- `pipeline_statistics.txt`: Detailed pipeline statistics
- `chemical_diversity_report.txt`: Chemical diversity analysis
- `plots/`: Directory containing all visualizations
  - `library_descriptors.png`: Descriptor distributions
  - `lipinski_violations.png`: Lipinski violations analysis
  - `similarity_distribution.png`: Similarity score distribution
  - `descriptor_correlations.png`: Descriptor correlation matrix
  - `chemical_space_pca.png`: PCA chemical space visualization
  - `chemical_space_interactive.html`: Interactive chemical space plot
  - `mw_vs_logp_interactive.html`: Interactive MW vs LogP scatter plot
  - `screening_dashboard.html`: Comprehensive interactive dashboard

## Testing

Run the test suite:

```bash
pytest tests/
```

## Dependencies

- **RDKit**: Cheminformatics toolkit for molecular processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Static plotting
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning utilities (PCA, t-SNE)
- **SciPy**: Scientific computing (distance calculations)
- **Jupyter**: Interactive notebooks

## Enhanced Dataset

The pipeline includes a carefully curated enhanced dataset designed to showcase realistic virtual screening results:

### Dataset Composition
- **Drug-like molecules** (15): FDA-approved drugs (aspirin, ibuprofen, statins, etc.)
- **Antimalarial-like molecules** (8): Structural analogs of known antimalarials
- **Non-drug-like molecules** (4): Compounds that fail Lipinski's Rule of Five
- **Natural products** (3): Bioactive compounds (quercetin, curcumin, resveratrol)

### Reference Compounds
- **8 known antimalarials**: Chloroquine, quinine, artemisinin, mefloquine, primaquine, doxycycline, atovaquone, pyrimethamine

## Example Results

With the enhanced dataset, you can expect:

- **Library Size**: 30 diverse molecules
- **Drug-like Molecules**: 25 molecules (83.3% pass rate)
- **Similarity Hits**: 7 compounds above 0.5 threshold
- **Mean Similarity**: 0.803 with perfect matches (1.0) for antimalarial analogs
- **Chemical Diversity**: 84% variance captured in 2D PCA space
- **Runtime**: ~2-3 minutes for enhanced dataset

## Extending the Pipeline

The modular design allows easy extension:

### Adding New Descriptors
```python
# In src/data_processing/descriptors.py
def calculate_custom_descriptors(mol):
    return {
        'custom_descriptor': calculate_custom_value(mol)
    }
```

### Adding New Filters
```python
# In src/filtering/drug_like.py
def custom_filter(self, df):
    # Implement custom filtering logic
    return filtered_df
```

### Adding New Similarity Metrics
```python
# In src/similarity/fingerprints.py
def custom_similarity(self, fp1, fp2):
    # Implement custom similarity calculation
    return similarity_score
```

### Adding Chemical Space Analysis
```python
# In src/visualization/chemical_space.py
from src.visualization.chemical_space import ChemicalSpaceAnalyzer

analyzer = ChemicalSpaceAnalyzer()
analyzer.plot_pca_space(df, color_col='CATEGORY')
analyzer.create_diversity_report(df)
```

## Performance Considerations

- **Memory Usage**: Scales with library size; consider chunking for very large libraries
- **Computation Time**: Fingerprint generation is the bottleneck; consider parallel processing
- **Storage**: SDF files can be large; consider compression for storage

## Troubleshooting

### Common Issues

1. **RDKit Import Error**: Ensure RDKit is properly installed
2. **Memory Error**: Reduce library size or increase system memory
3. **File Not Found**: Check file paths in configuration
4. **Empty Results**: Adjust similarity threshold or check reference compounds

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Acknowledgments

- RDKit community for the excellent cheminformatics toolkit
- Medicines for Malaria Venture (MMV) for antimalarial compound data
- Open source contributors to the scientific Python ecosystem
