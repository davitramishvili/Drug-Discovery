# Chapter 2 Specs Examples

This directory contains example and demonstration scripts for Chapter 2 antimalarial screening using the Specs database.

## 📁 Files Overview

### 📚 **Learning Examples**

#### `generate_Specs_hits_basic.py`
- **Purpose**: Basic implementation showing fundamental concepts
- **Features**: 
  - Simple pipeline-based approach using core infrastructure
  - Educational code structure (easy to understand)
  - Standard filtering and similarity search
  - Good starting point for learning the workflow
- **Use Case**: Understanding the basic drug discovery pipeline
- **Runtime**: ~5 minutes
- **Command**: `python generate_Specs_hits_basic.py`

### 🚀 **Threading Demonstrations**

#### `generate_Specs_hits_threaded_simple.py`
- **Purpose**: Multi-threading capabilities demonstration
- **Features**: 
  - Advanced parallel processing implementation
  - 6-thread parallel fingerprint computation
  - Educational progress tracking with thread-safe counters
  - Generates 1,000 hits from 5,000 compound subset
- **Use Case**: Learning optimization and threading concepts
- **Runtime**: ~2 minutes
- **Command**: `python generate_Specs_hits_threaded_simple.py`

### 🧪 **Safety Analysis Examples**

#### `Specs_hits_safety_analysis_quick.py`
- **Purpose**: Fast demonstration using synthetic compounds
- **Features**:
  - Uses 1,000 synthetic antimalarial-like compounds
  - Complete hERG and DILI toxicity analysis
  - Immediate results (no long computation)
  - Educational visualization and statistics
- **Use Case**: Quick testing and learning safety analysis workflow
- **Runtime**: <1 minute
- **Command**: `python Specs_hits_safety_analysis_quick.py`

#### `Specs_hits_safety_analysis.py`
- **Purpose**: Full safety analysis with real compound screening
- **Features**:
  - Complete Specs database screening (~200k compounds)
  - Real similarity search and hit generation
  - Comprehensive safety filtering (hERG, DILI, Lipinski)
  - Production-quality analysis workflow
- **Use Case**: Full-scale research screening (when you have time)
- **Runtime**: 10-30 minutes depending on library size
- **Command**: `python Specs_hits_safety_analysis.py`

## 🎯 **Recommended Learning Path**

1. **Start with**: `generate_Specs_hits_basic.py`
   - Learn the fundamental drug discovery pipeline
   - Understand core concepts without complexity

2. **Then try**: `generate_Specs_hits_threaded_simple.py`
   - Learn threading and optimization concepts
   - Understand parallel processing techniques

3. **Then try**: `Specs_hits_safety_analysis_quick.py`
   - Learn safety analysis concepts with fast results
   - Understand hERG and DILI toxicity filtering

4. **Finally run**: `Specs_hits_safety_analysis.py`
   - Apply to real large-scale screening
   - Get publication-quality results

## 📊 **Expected Outputs**

Each script generates:
- **CSV files** with hit compounds and properties
- **Console statistics** showing filtering results
- **Performance metrics** demonstrating threading benefits
- **Safety analysis reports** with toxicity predictions

## 🔧 **Configuration**

All scripts support command-line parameters:
- `--max-hits`: Number of hits to generate (default: 1000)
- `--library-size`: Maximum library compounds to process
- `--threads`: Number of threads to use (default: auto-detect)
- `--similarity-threshold`: Minimum similarity for hits (default: 0.58)

## 📚 **Educational Value**

These examples demonstrate:
- **Chapter 2.7 Exercises**: Different filtering approaches (BRENK, PAINS, custom)
- **Multi-threading**: Parallel processing for performance
- **Safety Assessment**: Real-world toxicity prediction workflows
- **Molecular Similarity**: Fingerprint-based virtual screening
- **Drug Discovery Pipeline**: End-to-end screening workflow

## 🏃‍♂️ **Quick Start**

```bash
# Quick demo (1-2 minutes)
python generate_Specs_hits_threaded_simple.py

# Safety analysis demo (30 seconds)
python Specs_hits_safety_analysis_quick.py

# Full pipeline (10+ minutes)
python Specs_hits_safety_analysis.py
```

## 📈 **Performance Comparison**

| Script | Compounds | Threading | Runtime | Output |
|--------|-----------|-----------|---------|---------|
| Simple | 5,000 | 6 threads | ~2 min | 1,000 hits |
| Quick | 1,000 | N/A | <1 min | Safety analysis |
| Full | 200,000+ | 6 threads | 10-30 min | Complete screening |

All scripts use the enhanced multi-threading capabilities documented in `../../THREADING_ENHANCEMENTS.md`. 