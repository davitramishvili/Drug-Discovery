# Chapter 3 Examples

This directory contains examples and exercises related to **Chapter 3: Machine Learning for Drug Discovery**.

## Contents

### Main Examples

- **`demo_chapter3.py`** - Complete Chapter 3 demonstration script
  - Shows hERG blocker prediction using machine learning
  - Demonstrates the full ML pipeline from data loading to evaluation
  - Includes visualization and model performance metrics

- **`chapter3_exercises_refactored.py`** - Refactored Chapter 3.6 exercises
  - Implements all exercises from Chapter 3.6 using existing infrastructure
  - Eliminates code duplication by leveraging the project's core modules
  - Demonstrates best practices for ML model comparison and evaluation

- **`legacy_random_forest_model.py`** - Legacy standalone Random Forest implementation
  - Complete standalone script with self-contained data processing
  - Original implementation with matplotlib/seaborn visualization
  - Educational reference for understanding the evolution to modular design
  - Shows how to implement ML models from scratch

## Key Concepts Demonstrated

1. **Machine Learning Pipeline**
   - Data loading and preprocessing
   - Feature engineering (molecular fingerprints)
   - Model training and evaluation
   - Cross-validation and hyperparameter tuning

2. **hERG Blocker Prediction**
   - Classification of molecules as hERG blockers or non-blockers
   - Feature importance analysis
   - Model performance metrics (MCC, accuracy, F1-score)

3. **Infrastructure Integration**
   - Uses `HERGDataProcessor` for data handling
   - Uses `MolecularFeaturizer` for fingerprint computation
   - Leverages existing evaluation and visualization modules

## Usage

```bash
# Run the main Chapter 3 demonstration
python demo_chapter3.py

# Run the refactored exercises
python chapter3_exercises_refactored.py
```

## Learning Path

1. **Start with** `demo_chapter3.py` to understand the basic ML pipeline
2. **Compare implementations**: Study `legacy_random_forest_model.py` vs the current modular approach
3. **Examine** `chapter3_exercises_refactored.py` to see infrastructure integration
4. **Compare** different ML approaches and model performance
5. **Experiment** with hyperparameters and feature engineering

### Implementation Comparison
- **Legacy approach**: Self-contained, standalone script (good for learning)
- **Current approach**: Modular, reusable infrastructure (good for production)
- **Refactored approach**: Best practices with existing components (optimal)

## Requirements

- All dependencies are listed in the main `requirements.txt`
- Requires hERG dataset in `data/chapter3/hERG_blockers.xlsx`
- Uses the project's core infrastructure from `src/`

## Related Documentation

- See `docs/README_Chapter3.md` for detailed Chapter 3 documentation
- See `docs/README_RandomForest_Integration.md` for ML model details 