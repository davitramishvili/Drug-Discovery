# Antimalarial Drug Discovery Pipeline

A virtual screening pipeline for identifying potential antimalarial compounds using similarity searching and molecular property filtering.

## Project Structure

...
Drug-Discovery/
├── src/                    # Source code
│   ├── data_processing/    # Data loading and molecule processing
│   ├── filtering/          # Property and substructure filters
│   ├── similarity/         # Fingerprint generation and similarity search
│   ├── visualization/      # Plotting and visualization functions
│   └── utils/             # Configuration and utility functions
├── data/                   # Data files
│   ├── raw/               # Original SDF files
│   ├── processed/         # Processed datasets
│   └── reference/         # Reference compounds (Malaria Box)
├── tests/                 # Unit and integration tests
├── results/               # Output files and reports
├── notebooks/             # Jupyter notebooks for analysis
└── requirements.txt       # Python dependencies
...
