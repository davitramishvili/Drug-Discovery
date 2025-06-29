# Examples Directory

This directory contains learning examples and demonstrations organized by chapters from "Machine Learning for Drug Discovery".

## Directory Structure

```
examples/
├── chapter2_specs/          # Chapter 2: Specs compound generation and analysis
├── chapter3/                # Chapter 3: Machine Learning applications  
├── filter_examples.py       # Basic filtering demonstrations
├── random_forest_example.py # Random Forest model examples
└── diverse_results/         # Results from diversity analysis
```

## Chapter Organization

### 📚 **Chapter 2: Molecular Filtering and Similarity Search**
- **Location**: `chapter2_specs/`
- **Focus**: Compound generation, drug-like filtering, similarity search
- **Examples**: Specs library analysis, threading demonstrations, safety analysis

### 🧬 **Chapter 3: Machine Learning for Drug Discovery**
- **Location**: `chapter3/`  
- **Focus**: hERG blocker prediction, ML model comparison, feature engineering
- **Examples**: Complete ML pipeline, exercises, model evaluation

## Quick Start Examples

### Basic Concepts
```bash
# Basic filtering concepts
python filter_examples.py

# Random Forest model demonstration  
python random_forest_example.py
```

### Chapter-Specific Examples
```bash
# Chapter 2 examples
cd chapter2_specs/
python generate_Specs_hits_threaded_simple.py

# Chapter 3 examples  
cd chapter3/
python demo_chapter3.py
```

## Learning Path

1. **Start with Basics**
   - `filter_examples.py` - Learn molecular filtering
   - `random_forest_example.py` - Understand ML basics

2. **Chapter 2: Molecular Processing**
   - Work through `chapter2_specs/` examples
   - Learn threading and optimization techniques
   - Practice with real compound libraries

3. **Chapter 3: Machine Learning**
   - Progress to `chapter3/` examples
   - Build classification models
   - Learn feature engineering and evaluation

## Results and Artifacts

- **`diverse_results/`** - Contains results from chemical diversity analysis
  - Plots and visualizations
  - Performance metrics
  - Interactive dashboards

## Integration with Core

All examples use the project's core infrastructure:
- `src/filtering/` - Molecular filtering
- `src/similarity/` - Similarity search  
- `src/chapter3_ml_screening/` - Machine learning modules
- `src/visualization/` - Plotting and analysis

## Requirements

- All dependencies listed in main `requirements.txt`
- Some examples require specific datasets (see individual READMEs)
- Threading examples work best on multi-core systems

## Contributing

When adding new examples:
1. Place in appropriate chapter directory
2. Update the chapter's README
3. Follow existing naming conventions
4. Include usage instructions and expected outputs 