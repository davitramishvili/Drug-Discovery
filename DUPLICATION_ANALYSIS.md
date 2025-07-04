# Code Duplication Analysis

## Overview
This analysis identifies significant code duplication patterns across the Drug Discovery project and provides a plan for systematic resolution.

## Major Duplication Patterns Identified

### 1. **Fingerprint Computation** 🔢
**Duplicated across 6+ files:**

- `src/chapter3_ml_screening/molecular_features.py` - `compute_fingerprint()`, `compute_fingerprints_batch()`
- `src/similarity/fingerprints.py` - `generate_fingerprint()`, `generate_fingerprints_batch()`
- `examples/chapter3/legacy_random_forest_model.py` - `_compute_morgan_fingerprints()`
- `generate_Specs_hits.py` - `compute_fingerprints_batch()`
- `examples/chapter2_specs/generate_Specs_hits_threaded_simple.py` - `compute_fingerprints_batch()`

**Common Code Pattern:**
```python
def compute_fingerprints(self, molecules):
    fingerprints = []
    for mol in molecules:
        if mol is not None:
            fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros((2048,), dtype=int)
            ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
        else:
            fingerprints.append(np.zeros(2048, dtype=int))
    return np.array(fingerprints)
```

### 2. **SMILES Standardization** 🧪
**Duplicated across 4+ files:**

- `src/chapter3_ml_screening/data_processing.py` - `process_smiles()`
- `src/chapter3_ml_screening/molecular_features.py` - `_process_smiles()`
- `examples/chapter3/legacy_random_forest_model.py` - `_standardize_smiles()`

**Common Code Pattern:**
```python
def standardize_smiles(self, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Various standardization steps
    return Chem.MolToSmiles(mol)
```

### 3. **Threaded Batch Processing** ⚡
**Duplicated across 3+ files:**

- `generate_Specs_hits.py` - `threaded_fingerprint_computation()`, `compute_fingerprints_batch()`
- `examples/chapter2_specs/generate_Specs_hits_threaded_simple.py` - Same methods
- `src/similarity/fingerprints.py` - `generate_fingerprints_batch_threaded()`

**Common Code Pattern:**
```python
def threaded_processing(self, data, description):
    batch_size = max(1, len(data) // self.n_threads)
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
        futures = {executor.submit(self.process_batch, batch, i): i for i, batch in enumerate(batches)}
        # Collect results...
```

### 4. **Progress Tracking** 📊
**Duplicated across 3+ files:**

- `generate_Specs_hits.py` - `update_progress()`
- `examples/chapter2_specs/generate_Specs_hits_threaded_simple.py` - `update_progress()`

### 5. **Similarity Computation** 🔍
**Duplicated across 3+ files:**

- `generate_Specs_hits.py` - `compute_similarities_batch()`
- `examples/chapter2_specs/generate_Specs_hits_threaded_simple.py` - `compute_similarities_batch()`
- Similar patterns in multiple other files

## Impact Assessment

### Lines of Duplicated Code
- **Fingerprint computation**: ~150 lines duplicated across 6 files = **900 lines**
- **SMILES standardization**: ~40 lines duplicated across 4 files = **160 lines**
- **Threading utilities**: ~80 lines duplicated across 3 files = **240 lines**
- **Progress tracking**: ~20 lines duplicated across 3 files = **60 lines**
- **Similarity computation**: ~50 lines duplicated across 3 files = **150 lines**

**Total estimated duplicate code: ~1,510 lines**

## Resolution Plan

### Phase 1: Consolidate Core Molecular Processing
1. **Standardize on `MolecularFeaturizer`** in `src/chapter3_ml_screening/molecular_features.py`
2. **Remove duplicate fingerprint methods** from legacy files
3. **Update all imports** to use centralized featurizer

### Phase 2: Consolidate Threading Utilities
1. **Create `src/utils/threading.py`** with reusable threading patterns
2. **Move progress tracking** to utilities
3. **Refactor all threaded processing** to use common utilities

### Phase 3: Consolidate SMILES Processing
1. **Standardize on `HERGDataProcessor.process_smiles()`** 
2. **Remove duplicate standardization methods**
3. **Update all SMILES processing** to use centralized method

### Phase 4: Clean Up Legacy Files
1. **Remove or refactor** `legacy_random_forest_model.py`
2. **Clean up** chapter2 example files
3. **Update documentation** to reflect consolidated architecture

## Benefits of Resolution

### Code Quality
- ✅ **Single source of truth** for molecular processing
- ✅ **Consistent behavior** across all components
- ✅ **Easier maintenance** and bug fixes
- ✅ **Better testing coverage**

### Performance
- ✅ **Optimized implementations** in centralized locations
- ✅ **Consistent threading patterns**
- ✅ **Reduced memory footprint**

### Developer Experience
- ✅ **Clear API contracts**
- ✅ **Reduced cognitive load**
- ✅ **Easier to onboard new developers**
- ✅ **Better documentation focus**

## Implementation Priority

1. **HIGH**: Fingerprint computation consolidation (most duplicated)
2. **MEDIUM**: Threading utilities consolidation
3. **MEDIUM**: SMILES standardization consolidation  
4. **LOW**: Legacy file cleanup

## Next Steps

1. Execute Phase 1 (fingerprint consolidation)
2. Update all affected files to use centralized implementations
3. Add comprehensive tests for consolidated functions
4. Update documentation
5. Remove deprecated duplicate code
