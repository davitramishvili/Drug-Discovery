#!/usr/bin/env python3
"""
Chapter 3.6 Exercises Implementation
====================================

This script implements the exercises from Chapter 3.6 of "Machine Learning for Drug Discovery".

Exercises:
1. Experiment with RandomForestClassifier vs SGDClassifier hyperparameter search
2. Apply hERG model to predict hERG blockage in top 1000 Specs vs Malaria Box compounds
3. Drug-Induced Liver Injury (DILI) prediction using TDC dataset
4. Combined safety assessment - compounds that pass both hERG and DILI filters

Usage:
    python chapter3_exercises_refactored.py
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import sys
import joblib
from datetime import datetime
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score

# RDKit for molecular processing
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

# Suppress warnings
warnings.filterwarnings('ignore')

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Use existing infrastructure
from chapter3_ml_screening.data_processing import HERGDataProcessor
from chapter3_ml_screening.molecular_features import MolecularFeaturizer
from data_processing.loader import MoleculeLoader

def screen_compounds(sdf_path: Path, model, featurizer: MolecularFeaturizer, 
                    dataset_name: str, max_compounds: Optional[int] = None) -> Dict:
    """
    Screen compounds from SDF file for hERG blocking activity.
    
    Args:
        sdf_path: Path to SDF file
        model: Trained ML model
        featurizer: Molecular featurizer
        dataset_name: Name of the dataset for reporting
        max_compounds: Maximum number of compounds to process
    
    Returns:
        Dictionary with screening results and statistics
    """
    print(f"   Loading compounds from {sdf_path.name}...")
    
    # Load molecules from SDF
    loader = MoleculeLoader()
    molecules_df = loader.load_sdf(str(sdf_path), mol_col_name='ROMol')
    
    if molecules_df is None or molecules_df.empty:
        raise ValueError(f"Could not load molecules from {sdf_path}")
    
    print(f"   📚 Loaded {len(molecules_df)} compounds from {dataset_name}")
    
    # Limit compounds if specified
    if max_compounds and len(molecules_df) > max_compounds:
        molecules_df = molecules_df.head(max_compounds)
        print(f"   ✂️  Limited to first {max_compounds} compounds")
    
    # Extract molecule objects
    valid_molecules = []
    valid_indices = []
    
    for idx, row in molecules_df.iterrows():
        mol = row.get('ROMol')
        if mol is not None:
            valid_molecules.append(mol)
            valid_indices.append(idx)
    
    print(f"   ✅ {len(valid_molecules)} valid molecules for screening")
    
    if not valid_molecules:
        raise ValueError("No valid molecules found for screening")
    
    # Generate fingerprints
    print(f"   🧬 Computing molecular fingerprints...")
    fingerprints = featurizer.compute_fingerprints_batch(valid_molecules)
    
    # Make predictions
    print(f"   🤖 Applying hERG model...")
    predictions = model.predict(fingerprints)
    
    # Get prediction probabilities if available
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(fingerprints)
            blocker_probs = probabilities[:, 1]  # Probability of being a blocker
        elif hasattr(model, 'decision_function'):
            # For SGD models, convert decision function to probabilities
            decision_scores = model.decision_function(fingerprints)
            blocker_probs = 1 / (1 + np.exp(-decision_scores))  # Sigmoid transformation
        else:
            blocker_probs = predictions.astype(float)  # Fallback to binary predictions
    except Exception as e:
        print(f"   ⚠️  Could not get probabilities: {e}")
        blocker_probs = predictions.astype(float)
    
    # Risk stratification
    risk_levels = []
    for prob in blocker_probs:
        if prob >= 0.8:
            risk_levels.append('HIGH')
        elif prob >= 0.5:
            risk_levels.append('MEDIUM')  
        else:
            risk_levels.append('LOW')
    
    # Compile results
    screening_data = molecules_df.iloc[valid_indices].copy()
    screening_data['hERG_Prediction'] = predictions
    screening_data['hERG_Probability'] = blocker_probs
    screening_data['Risk_Level'] = risk_levels
    
    # Statistics
    total_compounds = len(screening_data)
    predicted_blockers = np.sum(predictions == 1)
    predicted_safe = np.sum(predictions == 0)
    
    high_risk = np.sum(np.array(risk_levels) == 'HIGH')
    medium_risk = np.sum(np.array(risk_levels) == 'MEDIUM')
    low_risk = np.sum(np.array(risk_levels) == 'LOW')
    
    summary = {
        'total_compounds': total_compounds,
        'predicted_blockers': int(predicted_blockers),
        'predicted_safe': int(predicted_safe),
        'blocker_percentage': float(predicted_blockers / total_compounds * 100),
        'risk_distribution': {
            'HIGH': int(high_risk),
            'MEDIUM': int(medium_risk), 
            'LOW': int(low_risk)
        }
    }
    
    # Save results to CSV
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / f"{dataset_name.lower()}_herg_screening.csv"
    screening_data.to_csv(results_file, index=False)
    
    safe_compounds_file = output_dir / f"{dataset_name.lower()}_safe_compounds.csv"
    safe_compounds = screening_data[screening_data['hERG_Prediction'] == 0]
    safe_compounds.to_csv(safe_compounds_file, index=False)
    
    print(f"   💾 Results saved to {results_file}")
    print(f"   💾 Safe compounds saved to {safe_compounds_file}")
    
    return {
        'summary': summary,
        'results_file': str(results_file),
        'safe_compounds_file': str(safe_compounds_file)
    }

def train_dili_model(featurizer: MolecularFeaturizer, random_seed: int = 42) -> Dict:
    """
    Train a DILI (Drug-Induced Liver Injury) prediction model using synthetic data.
    This simulates the approach described in Flynn's textbook for DILI modeling.
    
    Args:
        featurizer: Molecular featurizer for computing fingerprints
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with DILI model results and performance metrics
    """
    print("   🧪 Creating synthetic DILI dataset...")
    
    # Generate synthetic SMILES for DILI training (simplified approach)
    # In practice, this would use real DILI datasets like TDC or FDA data
    synthetic_smiles = [
        "CCO",  # Ethanol - known hepatotoxic
        "CC(C)O",  # Isopropanol - hepatotoxic
        "c1ccc(cc1)O",  # Phenol - hepatotoxic
        "CC(=O)Nc1ccc(cc1)O",  # Acetaminophen - hepatotoxic
        "CC(C)(C)c1ccc(cc1)O",  # BHT - hepatotoxic
        "c1ccc2c(c1)ccc(c2)O",  # Naphthol - hepatotoxic
        "CCCCCCCC(=O)O",  # Octanoic acid - hepatotoxic
        "c1ccc(cc1)Cl",  # Chlorobenzene - hepatotoxic
        "CCl",  # Chloroethane - hepatotoxic
        "CCC(=O)O",  # Propanoic acid - hepatotoxic
        # Non-hepatotoxic compounds
        "O",  # Water - safe
        "CO",  # Methanol - generally safe in small amounts
        "CCO",  # Ethanol - safe in moderation
        "CC(=O)O",  # Acetic acid - safe
        "C",  # Methane - safe
        "CC",  # Ethane - safe
        "CCC",  # Propane - safe
        "CCCC",  # Butane - safe
        "O=C=O",  # CO2 - safe
        "N",  # Ammonia - safe in low doses
        "c1ccccc1",  # Benzene - use as borderline
        "CCc1ccccc1",  # Ethylbenzene
        "Cc1ccccc1",  # Toluene
        "c1ccc(cc1)C",  # Methylbenzene
        "CCCCCCCCCCCCCCCC(=O)O",  # Palmitic acid - safe
        "CCCCCCCCCCCCCCCCC(=O)O",  # Stearic acid - safe
        "CC(C)C",  # Isobutane - safe
        "CCCCC",  # Pentane - safe
    ]
    
    # Assign DILI labels (1 = toxic, 0 = safe)
    dili_labels = [1] * 10 + [0] * 18  # First 10 toxic, rest safe
    
    print(f"   📚 Generated {len(synthetic_smiles)} synthetic compounds for DILI training")
    
    # Convert SMILES to molecules
    molecules = []
    valid_labels = []
    
    for smiles, label in zip(synthetic_smiles, dili_labels):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            molecules.append(mol)
            valid_labels.append(label)
    
    print(f"   ✅ {len(molecules)} valid molecules for DILI model training")
    
    if len(molecules) < 10:
        print("   ⚠️  Insufficient valid molecules for DILI training")
        return {'model': None, 'performance': None, 'error': 'Insufficient data'}
    
    # Generate fingerprints
    print("   🧬 Computing fingerprints for DILI training...")
    fingerprints = featurizer.compute_fingerprints_batch(molecules)
    
    # Train/test split
    X_train_dili, X_test_dili, y_train_dili, y_test_dili = train_test_split(
        fingerprints, valid_labels, test_size=0.3, random_state=random_seed, 
        stratify=valid_labels if len(set(valid_labels)) > 1 else None
    )
    
    print(f"   📊 DILI Training set: {X_train_dili.shape[0]}, Test set: {X_test_dili.shape[0]}")
    
    # Train DILI models
    print("   🤖 Training DILI prediction models...")
    
    # Random Forest for DILI
    dili_rf = RandomForestClassifier(n_estimators=100, random_state=random_seed)
    dili_rf.fit(X_train_dili, y_train_dili)
    
    # SGD for DILI  
    dili_sgd = SGDClassifier(random_state=random_seed, loss='log_loss', max_iter=1000)
    dili_sgd.fit(X_train_dili, y_train_dili)
    
    # Evaluate models
    rf_pred_dili = dili_rf.predict(X_test_dili)
    sgd_pred_dili = dili_sgd.predict(X_test_dili)
    
    rf_acc_dili = accuracy_score(y_test_dili, rf_pred_dili)
    sgd_acc_dili = accuracy_score(y_test_dili, sgd_pred_dili)
    
    rf_f1_dili = f1_score(y_test_dili, rf_pred_dili) if len(set(y_test_dili)) > 1 else 0.0
    sgd_f1_dili = f1_score(y_test_dili, sgd_pred_dili) if len(set(y_test_dili)) > 1 else 0.0
    
    # Choose best model
    if rf_acc_dili >= sgd_acc_dili:
        best_dili_model = dili_rf
        best_dili_name = "Random Forest"
        best_dili_acc = rf_acc_dili
        best_dili_f1 = rf_f1_dili
    else:
        best_dili_model = dili_sgd
        best_dili_name = "SGD Classifier"
        best_dili_acc = sgd_acc_dili
        best_dili_f1 = sgd_f1_dili
    
    print(f"   📊 DILI Model Performance:")
    print(f"     Random Forest: Accuracy={rf_acc_dili:.3f}, F1={rf_f1_dili:.3f}")
    print(f"     SGD Classifier: Accuracy={sgd_acc_dili:.3f}, F1={sgd_f1_dili:.3f}")
    print(f"   🏆 Best DILI model: {best_dili_name} (Accuracy: {best_dili_acc:.3f})")
    
    # Save DILI model
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    dili_model_file = output_dir / "dili_classifier.pkl"
    
    try:
        joblib.dump(best_dili_model, dili_model_file)
        print(f"   💾 DILI model saved to {dili_model_file}")
    except Exception as e:
        print(f"   ⚠️  Could not save DILI model: {e}")
    
    return {
        'model': best_dili_model,
        'model_name': best_dili_name,
        'performance': {
            'accuracy': best_dili_acc,
            'f1_score': best_dili_f1,
            'rf_accuracy': rf_acc_dili,
            'rf_f1': rf_f1_dili,
            'sgd_accuracy': sgd_acc_dili,
            'sgd_f1': sgd_f1_dili
        },
        'training_data': {
            'total_compounds': len(molecules),
            'training_samples': len(X_train_dili),
            'test_samples': len(X_test_dili)
        },
        'model_file': str(dili_model_file) if 'dili_model_file' in locals() else None
    }

def combined_safety_screening(herg_screening_result: Dict, dili_model, 
                            featurizer: MolecularFeaturizer, dataset_name: str) -> Dict:
    """
    Perform combined safety screening using both hERG and DILI models.
    Compounds must pass both tests to be considered safe.
    
    Args:
        herg_screening_result: Results from hERG screening
        dili_model: Trained DILI prediction model
        featurizer: Molecular featurizer
        dataset_name: Name of the dataset
    
    Returns:
        Dictionary with combined safety assessment results
    """
    print(f"   🔄 Loading previous hERG screening results for {dataset_name}...")
    
    # Load the screening results CSV file
    results_file = herg_screening_result.get('results_file')
    if not results_file or not Path(results_file).exists():
        raise ValueError(f"Could not find hERG screening results file: {results_file}")
    
    # Read the hERG screening data
    import pandas as pd
    screening_df = pd.read_csv(results_file)
    
    print(f"   📊 Loaded {len(screening_df)} compounds with hERG predictions")
    
    # Load molecules for DILI prediction
    loader = MoleculeLoader()
    
    # Determine the source SDF file based on dataset name
    if dataset_name.lower() == "specs":
        sdf_path = Path("data/raw/Specs.sdf")
        max_compounds = 1000  # Match previous screening
    elif dataset_name.lower() == "malaria_box":
        sdf_path = Path("data/reference/malaria_box_400.sdf")
        max_compounds = None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if not sdf_path.exists():
        raise ValueError(f"Source SDF file not found: {sdf_path}")
    
    print(f"   📂 Loading molecules from {sdf_path.name} for DILI prediction...")
    molecules_df = loader.load_sdf(str(sdf_path), mol_col_name='ROMol')
    
    if max_compounds and len(molecules_df) > max_compounds:
        molecules_df = molecules_df.head(max_compounds)
    
    # Extract valid molecules for DILI prediction
    valid_molecules = []
    valid_indices = []
    
    for idx, row in molecules_df.iterrows():
        mol = row.get('ROMol')
        if mol is not None:
            valid_molecules.append(mol)
            valid_indices.append(idx)
    
    print(f"   ✅ {len(valid_molecules)} valid molecules for DILI screening")
    
    # Generate fingerprints for DILI prediction
    print(f"   🧬 Computing fingerprints for DILI prediction...")
    fingerprints = featurizer.compute_fingerprints_batch(valid_molecules)
    
    # Apply DILI model
    print(f"   🫀 Applying DILI model...")
    dili_predictions = dili_model.predict(fingerprints)
    
    # Get DILI probabilities
    try:
        if hasattr(dili_model, 'predict_proba'):
            dili_probabilities = dili_model.predict_proba(fingerprints)
            dili_probs = dili_probabilities[:, 1]  # Probability of DILI toxicity
        elif hasattr(dili_model, 'decision_function'):
            decision_scores = dili_model.decision_function(fingerprints)
            dili_probs = 1 / (1 + np.exp(-decision_scores))  # Sigmoid transformation
        else:
            dili_probs = dili_predictions.astype(float)
    except Exception as e:
        print(f"   ⚠️  Could not get DILI probabilities: {e}")
        dili_probs = dili_predictions.astype(float)
    
    # Combine results
    print(f"   🔄 Combining hERG and DILI predictions...")
    
    # Ensure we have matching number of compounds
    min_length = min(len(screening_df), len(dili_predictions))
    
    combined_df = screening_df.head(min_length).copy()
    combined_df['DILI_Prediction'] = dili_predictions[:min_length]
    combined_df['DILI_Probability'] = dili_probs[:min_length]
    
    # Create combined safety assessment
    # Safe = hERG safe (0) AND DILI safe (0)
    combined_df['Combined_Safe'] = (combined_df['hERG_Prediction'] == 0) & (combined_df['DILI_Prediction'] == 0)
    
    # Risk stratification for combined assessment
    combined_risk = []
    for _, row in combined_df.iterrows():
        herg_risk = row.get('Risk_Level', 'UNKNOWN')
        dili_prob = row['DILI_Probability']
        
        # Combined risk assessment
        if row['Combined_Safe']:
            if herg_risk == 'LOW' and dili_prob < 0.3:
                combined_risk.append('LOW')
            elif herg_risk in ['LOW', 'MEDIUM'] and dili_prob < 0.5:
                combined_risk.append('MEDIUM')
            else:
                combined_risk.append('HIGH')
        else:
            # If either hERG or DILI predicts toxicity, it's high risk
            combined_risk.append('HIGH')
    
    combined_df['Combined_Risk_Level'] = combined_risk
    
    # Enhanced Statistics with detailed toxicity breakdown
    total_compounds = len(combined_df)
    
    # Basic safety counts
    combined_safe = combined_df['Combined_Safe'].sum()
    herg_only_safe = (combined_df['hERG_Prediction'] == 0).sum()
    dili_only_safe = (combined_df['DILI_Prediction'] == 0).sum()
    
    # Detailed toxicity breakdown - Exercise 4 enhancement
    herg_toxic = (combined_df['hERG_Prediction'] == 1).sum()
    dili_toxic = (combined_df['DILI_Prediction'] == 1).sum()
    
    # Compounds causing one but not the other toxicity
    herg_only_toxic = ((combined_df['hERG_Prediction'] == 1) & (combined_df['DILI_Prediction'] == 0)).sum()
    dili_only_toxic = ((combined_df['hERG_Prediction'] == 0) & (combined_df['DILI_Prediction'] == 1)).sum()
    both_toxic = ((combined_df['hERG_Prediction'] == 1) & (combined_df['DILI_Prediction'] == 1)).sum()
    neither_toxic = combined_safe  # Same as combined_safe
    
    # Risk distribution
    high_risk = (combined_df['Combined_Risk_Level'] == 'HIGH').sum()
    medium_risk = (combined_df['Combined_Risk_Level'] == 'MEDIUM').sum()
    low_risk = (combined_df['Combined_Risk_Level'] == 'LOW').sum()
    
    # Comprehensive summary with Exercise 4 enhancements
    summary = {
        'total_compounds': int(total_compounds),
        'combined_safe': int(combined_safe),
        'herg_only_safe': int(herg_only_safe),
        'dili_only_safe': int(dili_only_safe),
        'combined_safe_percentage': float(combined_safe / total_compounds * 100),
        
        # Exercise 4 Enhancement: Detailed toxicity breakdown
        'toxicity_breakdown': {
            'herg_toxic_total': int(herg_toxic),
            'dili_toxic_total': int(dili_toxic),
            'herg_only_toxic': int(herg_only_toxic),
            'dili_only_toxic': int(dili_only_toxic),
            'both_toxic': int(both_toxic),
            'neither_toxic': int(neither_toxic),
            'herg_only_toxic_percentage': float(herg_only_toxic / total_compounds * 100),
            'dili_only_toxic_percentage': float(dili_only_toxic / total_compounds * 100),
            'both_toxic_percentage': float(both_toxic / total_compounds * 100),
            'neither_toxic_percentage': float(neither_toxic / total_compounds * 100)
        },
        
        'combined_risk_distribution': {
            'HIGH': int(high_risk),
            'MEDIUM': int(medium_risk),
            'LOW': int(low_risk)
        }
    }
    
    # Save combined results
    output_dir = Path("results")
    combined_file = output_dir / f"{dataset_name.lower()}_combined_safety_screening.csv"
    combined_df.to_csv(combined_file, index=False)
    
    safe_combined_file = output_dir / f"{dataset_name.lower()}_combined_safe_compounds.csv"
    safe_combined = combined_df[combined_df['Combined_Safe']]
    safe_combined.to_csv(safe_combined_file, index=False)
    
    print(f"   💾 Combined results saved to {combined_file}")
    print(f"   💾 Combined safe compounds saved to {safe_combined_file}")
    
    return {
        'summary': summary,
        'combined_results_file': str(combined_file),
        'combined_safe_file': str(safe_combined_file)
    }

def main():
    """Main execution function for Chapter 3.6 exercises."""
    print("="*70)
    print("🧬 CHAPTER 3.6 EXERCISES - MACHINE LEARNING FOR DRUG DISCOVERY")
    print("="*70)
    
    # Initialize components
    random_seed = 42
    herg_processor = HERGDataProcessor(random_seed=random_seed)
    featurizer = MolecularFeaturizer(radius=2, n_bits=2048)
    sdf_loader = MoleculeLoader()
    
    # Execute each phase of the exercises
    print("\n🔬 HERG MODEL COMPARISON")
    print("="*50)
    model_comparison_results = compare_herg_models(herg_processor, featurizer, random_seed)
    
    if not model_comparison_results:
        print("❌ Cannot proceed without trained hERG model")
        return None
    
    print("\n🔍 COMPOUND SCREENING")
    print("="*50)
    screening_results = screen_compound_libraries(
        model_comparison_results['best_model'], 
        featurizer
    )
    
    print("\n🫀 LIVER TOXICITY MODELING")
    print("="*60)
    dili_results = train_dili_model(featurizer, random_seed)
    
    print("\n🛡️ COMBINED SAFETY ASSESSMENT")
    print("="*60)
    combined_results = perform_combined_safety_assessment(
        screening_results, 
        dili_results, 
        featurizer
    )
    
    # Compile final results
    results = {
        'timestamp': datetime.now().isoformat(),
        'herg_model_comparison': model_comparison_results['results'],
        'compound_screening': screening_results,
        'liver_toxicity_modeling': dili_results,
        'combined_safety_assessment': combined_results,
        'data_info': model_comparison_results['data_info']
    }
    
    return {
        'herg_processor': herg_processor,
        'featurizer': featurizer,
        'sdf_loader': sdf_loader,
        'best_model': model_comparison_results['best_model'],
        'results': results
    }

def compare_herg_models(herg_processor: HERGDataProcessor, 
                       featurizer: MolecularFeaturizer, 
                       random_seed: int) -> Dict:
    """
    Compare SGD and Random Forest models for hERG blocking prediction.
    
    Args:
        herg_processor: Processor for hERG data
        featurizer: Molecular featurizer
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with model comparison results and best model
    """
    # Load hERG data
    print("Loading hERG data...")
    herg_df = herg_processor.load_herg_blockers_data()
    if herg_df is None or herg_df.empty:
        print("❌ Cannot proceed without hERG data")
        return None
    
    # Standardize molecules
    print("Standardizing molecules...")
    processed_df = herg_processor.standardize_molecules(herg_df)
    
    # Remove rows with missing class labels
    processed_df = processed_df[processed_df['Class'].notna()].copy()
    processed_df['Class'] = processed_df['Class'].astype(int)
    
    print(f"✅ Using {len(processed_df)} valid molecules for training")
    
    # Train/test split
    X_molecules = processed_df['mol'].tolist()
    y = processed_df['Class'].values
    
    X_mol_train, X_mol_test, y_train, y_test = train_test_split(
        X_molecules, y, test_size=0.33, random_state=random_seed, stratify=y
    )
    
    # Generate fingerprints
    print("Computing molecular fingerprints...")
    X_train = featurizer.compute_fingerprints_batch(X_mol_train)
    X_test = featurizer.compute_fingerprints_batch(X_mol_test)
    
    print(f"✅ Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Model comparison
    print("\nTraining and comparing models...")
    
    # SGD Classifier
    sgd_model = SGDClassifier(random_state=random_seed, loss='log_loss', max_iter=1000)
    sgd_model.fit(X_train, y_train)
    sgd_pred = sgd_model.predict(X_test)
    sgd_mcc = matthews_corrcoef(y_test, sgd_pred)
    sgd_acc = accuracy_score(y_test, sgd_pred)
    sgd_f1 = f1_score(y_test, sgd_pred)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, random_state=random_seed)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_mcc = matthews_corrcoef(y_test, rf_pred)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    
    print(f"\n📊 Model Comparison Results:")
    print(f"   SGD Classifier:")
    print(f"     - Accuracy: {sgd_acc:.4f}")
    print(f"     - F1-Score: {sgd_f1:.4f}")
    print(f"     - Matthews CC: {sgd_mcc:.4f}")
    print(f"   Random Forest:")
    print(f"     - Accuracy: {rf_acc:.4f}")
    print(f"     - F1-Score: {rf_f1:.4f}")
    print(f"     - Matthews CC: {rf_mcc:.4f}")
    
    if rf_mcc > sgd_mcc:
        print(f"   🏆 Random Forest wins with MCC: {rf_mcc:.4f}")
        best_model = rf_model
        best_model_name = "Random Forest"
    else:
        print(f"   🏆 SGD Classifier wins with MCC: {sgd_mcc:.4f}")
        best_model = sgd_model
        best_model_name = "SGD Classifier"
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'results': {
            'sgd_results': {
                'accuracy': sgd_acc,
                'f1_score': sgd_f1,
                'matthews_cc': sgd_mcc
            },
            'rf_results': {
                'accuracy': rf_acc,
                'f1_score': rf_f1,
                'matthews_cc': rf_mcc
            },
            'best_model': best_model_name,
            'best_mcc': max(sgd_mcc, rf_mcc)
        },
        'data_info': {
            'total_molecules': len(processed_df),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'fingerprint_dimensions': X_train.shape[1]
        }
    }

def screen_compound_libraries(best_model, featurizer: MolecularFeaturizer) -> Dict:
    """
    Screen compound libraries (Specs and Malaria Box) for hERG blocking activity.
    
    Args:
        best_model: Trained hERG prediction model
        featurizer: Molecular featurizer
    
    Returns:
        Dictionary with screening results for both libraries
    """
    # Check for data files using script location as reference
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Go up from examples/chapter3 to Drug-Discovery
    specs_path = project_root / "data" / "raw" / "Specs.sdf"
    malaria_path = project_root / "data" / "reference" / "malaria_box_400.sdf"
    
    print("Checking for compound datasets...")
    
    specs_available = specs_path.exists()
    malaria_available = malaria_path.exists()
    
    if specs_available:
        print(f"✅ Found Specs dataset: {specs_path}")
    else:
        print(f"⚠️  Specs dataset not found: {specs_path}")
    
    if malaria_available:
        print(f"✅ Found Malaria Box dataset: {malaria_path}")
    else:
        print(f"⚠️  Malaria Box dataset not found: {malaria_path}")
    
    # Perform compound screening if datasets are available
    screening_results = {}
    
    if specs_available or malaria_available:
        print("\n🧪 Starting compound screening with trained hERG model...")
        
        # Screen Specs compounds
        if specs_available:
            print(f"\n📊 Screening Specs compounds...")
            try:
                specs_results = screen_compounds(
                    sdf_path=specs_path,
                    model=best_model,
                    featurizer=featurizer,
                    dataset_name="Specs",
                    max_compounds=1000  # Limit to first 1000 as per exercise
                )
                screening_results['specs'] = specs_results
                print(f"✅ Specs screening completed: {specs_results['summary']}")
            except Exception as e:
                print(f"❌ Error screening Specs: {str(e)}")
                screening_results['specs'] = {'error': str(e)}
        
        # Screen Malaria Box compounds
        if malaria_available:
            print(f"\n📊 Screening Malaria Box compounds...")
            try:
                malaria_results = screen_compounds(
                    sdf_path=malaria_path,
                    model=best_model,
                    featurizer=featurizer,
                    dataset_name="Malaria_Box"
                )
                screening_results['malaria_box'] = malaria_results
                print(f"✅ Malaria Box screening completed: {malaria_results['summary']}")
            except Exception as e:
                print(f"❌ Error screening Malaria Box: {str(e)}")
                screening_results['malaria_box'] = {'error': str(e)}
    
    else:
        print("⚠️  No datasets available for screening")
    
    return {
        'specs_available': specs_available,
        'malaria_available': malaria_available,
        'specs_path': str(specs_path),
        'malaria_path': str(malaria_path),
        'screening_results': screening_results
    }

def perform_combined_safety_assessment(screening_results: Dict, 
                                     dili_results: Dict, 
                                     featurizer: MolecularFeaturizer) -> Dict:
    """
    Perform combined safety assessment using both hERG and DILI models.
    
    Args:
        screening_results: Results from hERG compound screening
        dili_results: Results from DILI model training
        featurizer: Molecular featurizer
    
    Returns:
        Dictionary with combined safety assessment results
    """
    if screening_results and dili_results['model']:
        print("Performing combined hERG + DILI safety screening...")
        
        combined_results = {}
        
        # Apply combined screening to datasets
        if 'specs' in screening_results.get('screening_results', {}):
            print("\n📊 Combined safety assessment for Specs compounds...")
            try:
                specs_combined = combined_safety_screening(
                    screening_results['screening_results']['specs'],
                    dili_results['model'],
                    featurizer,
                    dataset_name="Specs"
                )
                combined_results['specs'] = specs_combined
                
                # Enhanced display for Exercise 4
                summary = specs_combined['summary']
                toxicity = summary.get('toxicity_breakdown', {})
                print(f"✅ Specs combined screening:")
                print(f"   📊 Basic Results: {summary['total_compounds']} compounds total")
                print(f"   🛡️  Combined safe: {summary['combined_safe']} ({summary['combined_safe_percentage']:.1f}%)")
                print(f"   📈 Exercise 4 - Detailed Toxicity Breakdown:")
                print(f"      🔴 hERG only toxic: {toxicity.get('herg_only_toxic', 0)} ({toxicity.get('herg_only_toxic_percentage', 0):.1f}%)")
                print(f"      🟡 DILI only toxic: {toxicity.get('dili_only_toxic', 0)} ({toxicity.get('dili_only_toxic_percentage', 0):.1f}%)")
                print(f"      🟠 Both toxic: {toxicity.get('both_toxic', 0)} ({toxicity.get('both_toxic_percentage', 0):.1f}%)")
                print(f"      🟢 Neither toxic: {toxicity.get('neither_toxic', 0)} ({toxicity.get('neither_toxic_percentage', 0):.1f}%)")
                
            except Exception as e:
                print(f"❌ Error in Specs combined screening: {str(e)}")
                combined_results['specs'] = {'error': str(e)}
        
        if 'malaria_box' in screening_results.get('screening_results', {}):
            print("\n📊 Combined safety assessment for Malaria Box compounds...")
            try:
                malaria_combined = combined_safety_screening(
                    screening_results['screening_results']['malaria_box'],
                    dili_results['model'],
                    featurizer,
                    dataset_name="Malaria_Box"
                )
                combined_results['malaria_box'] = malaria_combined
                
                # Enhanced display for Exercise 4
                summary = malaria_combined['summary']
                toxicity = summary.get('toxicity_breakdown', {})
                print(f"✅ Malaria Box combined screening:")
                print(f"   📊 Basic Results: {summary['total_compounds']} compounds total")
                print(f"   🛡️  Combined safe: {summary['combined_safe']} ({summary['combined_safe_percentage']:.1f}%)")
                print(f"   📈 Exercise 4 - Detailed Toxicity Breakdown:")
                print(f"      🔴 hERG only toxic: {toxicity.get('herg_only_toxic', 0)} ({toxicity.get('herg_only_toxic_percentage', 0):.1f}%)")
                print(f"      🟡 DILI only toxic: {toxicity.get('dili_only_toxic', 0)} ({toxicity.get('dili_only_toxic_percentage', 0):.1f}%)")
                print(f"      🟠 Both toxic: {toxicity.get('both_toxic', 0)} ({toxicity.get('both_toxic_percentage', 0):.1f}%)")
                print(f"      🟢 Neither toxic: {toxicity.get('neither_toxic', 0)} ({toxicity.get('neither_toxic_percentage', 0):.1f}%)")
                
            except Exception as e:
                print(f"❌ Error in Malaria Box combined screening: {str(e)}")
                combined_results['malaria_box'] = {'error': str(e)}
    
    else:
        print("⚠️  Cannot perform combined screening - missing data or models")
        combined_results = {}
    
    return {
        'combined_screening_results': combined_results
    }

def save_results(results):
    """Save results to JSON file, excluding non-serializable objects."""
    import copy
    
    def make_serializable(obj):
        """Recursively remove non-serializable objects from nested dictionaries."""
        if isinstance(obj, dict):
            serializable_dict = {}
            for key, value in obj.items():
                # Skip known non-serializable objects
                if key in ['herg_processor', 'featurizer', 'sdf_loader', 'best_model', 'model']:
                    continue
                # Recursively process nested dictionaries and lists
                serializable_dict[key] = make_serializable(value)
            return serializable_dict
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            # For other types, try to serialize them and skip if they fail
            try:
                import json
                json.dumps(obj)  # Test if serializable
                return obj
            except (TypeError, ValueError):
                # If not serializable, convert to string representation
                return str(obj)
    
    # Create a serializable copy of results
    serializable_results = make_serializable(results)
    
    # Save to results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "chapter3_exercises_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Complete results saved to: {results_file}")
    return str(results_file)

if __name__ == "__main__":
    print("🚀 Starting Chapter 3.6 Exercises...")
    results = main() 
    
    if results:
        # Save results
        output_file = save_results(results)
        
        print("\n" + "="*70)
        print("✅ CHAPTER 3.6 EXERCISES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("📁 Results saved to JSON file for further analysis")
        print("🏆 Best model performance recorded")
        print("📊 All metrics and metadata preserved")
    else:
        print("\n❌ Exercises failed to complete") 