# Dataset Selection Guide

This guide explains how to use the different datasets available in the Antimalarial Drug Discovery Pipeline.

## Quick Reference

| Dataset | Molecules | Runtime | Purpose | Command |
|---------|-----------|---------|---------|---------|
| **Small** | 5 | ~30s | Quick testing | `python run_pipeline.py --dataset small` |
| **Diverse** | 20 | ~1-2min | Comprehensive analysis | `python run_pipeline.py --dataset diverse` |
| **Large** | 50 | ~3-5min | Extensive testing | `python run_pipeline.py --dataset large` |
| **Custom** | Variable | Variable | Your own data | `python run_pipeline.py --dataset custom` |

## Dataset Details

### 1. Small Dataset (`--dataset small`)
**Perfect for: Quick validation and testing**

- **5 molecules**: Aspirin, Caffeine, Ibuprofen, Paracetamol, Benzene
- **Drug-likeness**: 100% pass rate (minimal violations)
- **Similarity hits**: Usually 0 (these are not antimalarial-like)
- **Use cases**:
  - Testing pipeline installation
  - Quick validation of changes
  - Learning the pipeline workflow
  - Debugging issues

```bash
# Quick test
python run_pipeline.py --dataset small --no-plots

# Test with plots
python run_pipeline.py --dataset small
```

### 2. Diverse Dataset (`--dataset diverse`) - DEFAULT
**Perfect for: Comprehensive analysis with meaningful variance**

- **20 molecules**: Wide range of drug-likeness properties
- **Drug-likeness**: ~45% pass rate (9/20 molecules)
- **Violations**: 0-3+ violations per molecule
- **Similarity hits**: 2 molecules at default threshold
- **Use cases**:
  - Standard analysis workflow
  - Demonstrating pipeline capabilities
  - Educational purposes
  - Publication-quality results

```bash
# Default run
python run_pipeline.py

# With custom threshold
python run_pipeline.py --dataset diverse --similarity-threshold 0.7
```

**Expected Results:**
- Original Library: 20 molecules
- Drug-like: 9 molecules (45% pass rate)
- Similarity Hits: 2 molecules
- Violation Distribution: 
  - 0 violations: 55%
  - 1 violation: 20%
  - 2 violations: 15%
  - 3+ violations: 10%

### 3. Large Dataset (`--dataset large`)
**Perfect for: Extensive testing and research**

- **50 molecules**: Natural products, peptides, complex structures, steroids
- **Drug-likeness**: ~77% pass rate (38/49 molecules)
- **Violations**: Full spectrum from 0 to 4+ violations
- **Similarity hits**: 6 molecules at default threshold
- **Use cases**:
  - Research projects
  - Performance testing
  - Comprehensive chemical space analysis
  - Method validation

```bash
# Standard large dataset run
python run_pipeline.py --dataset large

# Research configuration
python run_pipeline.py --dataset large --similarity-threshold 0.8 --output research_results
```

**Expected Results:**
- Original Library: 49 molecules
- Drug-like: 38 molecules (77.6% pass rate)
- Similarity Hits: 6 molecules
- Chemical Diversity: High (includes natural products, peptides, steroids)

### 4. Custom Dataset (`--dataset custom`)
**Perfect for: Analyzing your own molecular data**

- **Your molecules**: Place files in data directories
- **Flexible**: Any number of molecules
- **Use cases**:
  - Analyzing proprietary compound libraries
  - Research with specific datasets
  - Custom antimalarial targets

#### Setup Custom Dataset:
1. **Prepare your files**:
   ```
   data/raw/custom_library.sdf       # Your compound library
   data/reference/custom_reference.sdf  # Your reference compounds
   ```

2. **Run analysis**:
   ```bash
   python run_pipeline.py --dataset custom
   ```

#### Alternative: Direct file specification
```bash
python run_pipeline.py --library my_compounds.sdf --reference my_targets.sdf
```

## Command Line Options

### Basic Usage
```bash
# List all available datasets
python run_pipeline.py --list-datasets

# Choose specific dataset
python run_pipeline.py --dataset {small|diverse|large|custom}

# Custom output directory
python run_pipeline.py --dataset large --output my_results

# Skip plots for faster execution
python run_pipeline.py --dataset small --no-plots
```

### Advanced Configuration
```bash
# Adjust similarity threshold (0.0-1.0)
python run_pipeline.py --similarity-threshold 0.8

# Change Lipinski violations allowed
python run_pipeline.py --lipinski-violations 2

# Combine options
python run_pipeline.py --dataset large --similarity-threshold 0.6 --lipinski-violations 0 --output strict_results
```

## Workflow Examples

### 1. Development Workflow
```bash
# Quick test during development
python run_pipeline.py --dataset small --no-plots

# Full test with visualization
python run_pipeline.py --dataset diverse

# Performance test
python run_pipeline.py --dataset large --no-plots
```

### 2. Research Workflow
```bash
# Exploratory analysis
python run_pipeline.py --dataset diverse --output exploratory

# Strict filtering
python run_pipeline.py --dataset large --lipinski-violations 0 --similarity-threshold 0.8 --output strict

# Custom data analysis
python run_pipeline.py --library research_compounds.sdf --reference known_antimalarials.sdf --output research
```

### 3. Educational Workflow
```bash
# Start with small dataset
python run_pipeline.py --dataset small

# Show variance with diverse dataset
python run_pipeline.py --dataset diverse

# Compare different thresholds
python run_pipeline.py --dataset diverse --similarity-threshold 0.3 --output lenient
python run_pipeline.py --dataset diverse --similarity-threshold 0.8 --output strict
```

## Understanding Results

### Drug-likeness Analysis
- **Pass Rate**: Percentage meeting Lipinski criteria
- **Violation Distribution**: Breakdown by number of violations
- **Common Violations**: MW > 500, LogP > 5, etc.

### Similarity Search
- **Hit Rate**: Molecules similar to references
- **Score Distribution**: Range of similarity values
- **Top Hits**: Most promising candidates

### Visualizations
- **Static Plots**: PNG files for publications
- **Interactive Plots**: HTML files for exploration
- **Dashboard**: Comprehensive overview

## Performance Comparison

| Dataset | Load Time | Filter Time | Search Time | Plot Time | Total |
|---------|-----------|-------------|-------------|-----------|-------|
| Small | <1s | <1s | <1s | ~5s | ~30s |
| Diverse | ~1s | ~1s | ~2s | ~15s | ~1-2min |
| Large | ~2s | ~2s | ~5s | ~20s | ~3-5min |

## Troubleshooting

### Common Issues

1. **No similarity hits**: 
   - Lower similarity threshold: `--similarity-threshold 0.3`
   - Check reference compounds are appropriate

2. **All molecules pass filters**:
   - Use stricter criteria: `--lipinski-violations 0`
   - Try large dataset for more diversity

3. **Pipeline too slow**:
   - Use small dataset for testing
   - Skip plots: `--no-plots`
   - Use faster similarity threshold

4. **Custom dataset not found**:
   - Check file paths
   - Ensure SDF format is correct
   - Use absolute paths if needed

### Getting Help
```bash
# Show all options
python run_pipeline.py --help

# List datasets
python run_pipeline.py --list-datasets

# Simple dataset info
python list_datasets.py
```

## Best Practices

1. **Start Small**: Begin with small dataset to verify setup
2. **Use Diverse**: Default diverse dataset for most analyses
3. **Go Large**: Use large dataset for comprehensive studies
4. **Custom Data**: Prepare clean SDF files with proper formatting
5. **Save Results**: Use descriptive output directory names
6. **Document Settings**: Record command-line options used

## File Locations

After running with different datasets, you'll find:

```
data/
├── raw/
│   ├── small_test_library.sdf      # 5 molecules
│   ├── test_library.sdf            # 20 molecules (diverse)
│   ├── large_test_library.sdf      # 50 molecules
│   └── custom_library.sdf          # Your molecules
└── reference/
    ├── small_reference.sdf         # 2 antimalarials
    ├── malaria_box.sdf            # 2 antimalarials (diverse)
    ├── extended_malaria_box.sdf   # 5 antimalarials (large)
    └── custom_reference.sdf       # Your references
```

Results are saved to the specified output directory (default: `results/`). 