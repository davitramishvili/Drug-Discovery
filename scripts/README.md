# Scripts Directory

This directory contains utility scripts for data processing, conversion, and system maintenance.

## Contents

### Data Conversion
- **`convert_csv_to_sdf.py`** - Convert malaria box compounds from CSV to SDF format
  - Converts SMILES strings to 3D molecular structures
  - Preserves compound properties and metadata
  - Used for creating standardized molecular datasets

## Usage

### CSV to SDF Conversion
```bash
# Convert malaria box data
python convert_csv_to_sdf.py
```

This script:
1. Reads CSV file containing SMILES and compound properties
2. Generates 3D molecular coordinates using RDKit
3. Exports to SDF format with all metadata preserved

## Input/Output

### Expected Input
- CSV file with columns: `Smiles`, `HEOS_COMPOUND_ID`, `Batch_No_March2012`, etc.
- Location: `data/reference/malaria_box_400_compounds.csv`

### Generated Output  
- SDF file with 3D molecular structures
- Location: `data/reference/malaria_box_400.sdf`

## Dependencies

- RDKit for molecular processing
- pandas for data handling
- All dependencies listed in main `requirements.txt`

## Error Handling

Scripts include error handling for:
- Invalid SMILES strings
- Failed 3D coordinate generation
- Missing compound properties
- File I/O errors

## Future Scripts

Additional utility scripts can be added here for:
- Database imports/exports
- Batch molecular property calculations
- Data validation and cleaning
- Performance benchmarking 