"""
Molecular features module for Chapter 3 ML screening.
Handles fingerprint generation and molecular feature transformations.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple
import logging
from sklearn.base import BaseEstimator, TransformerMixin

# RDKit imports
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdFingerprintGenerator import AdditionalOutput, GetMorganGenerator
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available - fingerprint functionality will be limited")

logger = logging.getLogger(__name__)

class MolecularFeaturizer:
    """
    Handles molecular fingerprint generation and feature extraction.
    
    This class implements the fingerprint generation pipeline from Flynn's Chapter 3,
    focusing on Morgan (ECFP) fingerprints for machine learning applications.
    """
    
    def __init__(self, radius: int = 2, n_bits: int = 2048, use_features: bool = False):
        """
        Initialize the molecular featurizer.
        
        Parameters:
            radius (int): Radius for Morgan fingerprints (default: 2)
            n_bits (int): Number of bits in the fingerprint (default: 2048)
            use_features (bool): Whether to use feature-based fingerprints
        """
        self.radius = radius
        self.n_bits = n_bits
        self.use_features = use_features
        
        if RDKIT_AVAILABLE:
            # Initialize Morgan fingerprint generator
            self.morgan_generator = GetMorganGenerator(
                radius=self.radius, 
                fpSize=self.n_bits,
                includeChirality=True
            )
        
        logger.info(f"MolecularFeaturizer initialized with radius={radius}, n_bits={n_bits}")
    
    def compute_fingerprint(self, mol, radius: Optional[int] = None, n_bits: Optional[int] = None) -> np.ndarray:
        """
        Compute Morgan fingerprint for a molecule.
        
        This implements the fingerprint computation from the notebook.
        
        Parameters:
            mol: RDKit molecule object or SMILES string
            radius (int, optional): Fingerprint radius (uses instance default if None)
            n_bits (int, optional): Number of bits (uses instance default if None)
            
        Returns:
            numpy.ndarray: Binary fingerprint array
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - returning dummy fingerprint")
            return np.zeros(self.n_bits)
        
        # Use instance defaults if not provided
        if radius is None:
            radius = self.radius
        if n_bits is None:
            n_bits = self.n_bits
        
        # Convert SMILES to molecule if needed
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        if mol is None:
            logger.warning("Invalid molecule - returning zero fingerprint")
            return np.zeros(n_bits)
        
        try:
            # Generate fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            
            # Convert to numpy array
            arr = np.zeros((n_bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            
            return arr
            
        except Exception as e:
            logger.warning(f"Error computing fingerprint: {e}")
            return np.zeros(n_bits)
    
    def compute_fingerprints_batch(self, molecules: List, 
                                   radius: Optional[int] = None, 
                                   n_bits: Optional[int] = None) -> np.ndarray:
        """
        Compute fingerprints for a batch of molecules.
        
        Parameters:
            molecules (List): List of RDKit molecule objects or SMILES strings
            radius (int, optional): Fingerprint radius
            n_bits (int, optional): Number of bits
            
        Returns:
            numpy.ndarray: Array of fingerprints (n_molecules, n_bits)
        """
        fingerprints = []
        
        for i, mol in enumerate(molecules):
            fp = self.compute_fingerprint(mol, radius, n_bits)
            fingerprints.append(fp)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(molecules)} molecules")
        
        return np.array(fingerprints)
    
    def explore_fingerprint_features(self, fingerprints: np.ndarray) -> dict:
        """
        Explore fingerprint features to understand the data distribution.
        
        Parameters:
            fingerprints (numpy.ndarray): Array of fingerprints
            
        Returns:
            dict: Dictionary with feature statistics
        """
        if fingerprints.size == 0:
            return {}
        
        stats = {
            'n_molecules': fingerprints.shape[0],
            'n_bits': fingerprints.shape[1],
            'mean_bits_per_molecule': np.mean(np.sum(fingerprints, axis=1)),
            'std_bits_per_molecule': np.std(np.sum(fingerprints, axis=1)),
            'min_bits_per_molecule': np.min(np.sum(fingerprints, axis=1)),
            'max_bits_per_molecule': np.max(np.sum(fingerprints, axis=1)),
            'bit_frequency': np.sum(fingerprints, axis=0),
            'sparsity': 1 - np.mean(fingerprints),
            'most_common_bits': np.argsort(np.sum(fingerprints, axis=0))[-10:][::-1],
            'least_common_bits': np.argsort(np.sum(fingerprints, axis=0))[:10]
        }
        
        logger.info(f"Fingerprint feature exploration:")
        logger.info(f"  Dataset: {stats['n_molecules']} molecules, {stats['n_bits']} bits")
        logger.info(f"  Average bits per molecule: {stats['mean_bits_per_molecule']:.1f} ± {stats['std_bits_per_molecule']:.1f}")
        logger.info(f"  Sparsity: {stats['sparsity']:.3f}")
        
        return stats
    
    def get_bit_info(self, mol, bit_number: int) -> dict:
        """
        Get information about what substructure a specific bit represents.
        
        Parameters:
            mol: RDKit molecule object
            bit_number (int): Bit number to analyze
            
        Returns:
            dict: Information about the bit
        """
        if not RDKIT_AVAILABLE:
            return {}
        
        try:
            # Generate fingerprint with additional output
            ao = AdditionalOutput()
            ao.AllocateBitInfoMap()
            fp = self.morgan_generator.GetFingerprint(mol, additionalOutput=ao)
            
            bit_info_map = ao.GetBitInfoMap()
            
            if bit_number in bit_info_map:
                # Convert environments to JSON-serializable format
                environments = []
                for env in bit_info_map[bit_number]:
                    environments.append({
                        'atom_id': int(env[0]) if len(env) > 0 else None,
                        'radius': int(env[1]) if len(env) > 1 else None
                    })
                
                return {
                    'bit_number': int(bit_number),
                    'environments': environments,
                    'found': True
                }
            else:
                return {'bit_number': int(bit_number), 'found': False}
                
        except Exception as e:
            logger.warning(f"Error getting bit info: {e}")
            return {}
    
    def draw_fragment_from_bit(self, mol, bit_number: int):
        """
        Draw the molecular fragment corresponding to a specific fingerprint bit.
        
        Parameters:
            mol: RDKit molecule object
            bit_number (int): Bit number to visualize
            
        Returns:
            SVG string or None
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - cannot draw fragments")
            return None
        
        try:
            from rdkit.Chem import Draw
            
            ao = AdditionalOutput()
            ao.AllocateBitInfoMap()
            fp = self.morgan_generator.GetFingerprint(mol, additionalOutput=ao)
            
            svg = Draw.DrawMorganBit(mol, bit_number, ao.GetBitInfoMap(), useSVG=True)
            return svg
            
        except Exception as e:
            logger.warning(f"Error drawing fragment for bit {bit_number}: {e}")
            return None
    
    def get_examples_for_bit(self, bit_number: int, molecules: List, fingerprints: np.ndarray) -> List:
        """
        Get visual examples of molecules that have a specific bit set.
        
        Parameters:
            bit_number (int): Bit number to find examples for
            molecules (List): List of RDKit molecule objects
            fingerprints (numpy.ndarray): Array of fingerprints
            
        Returns:
            List: List of SVG visualizations
        """
        if not RDKIT_AVAILABLE:
            return []
        
        # Find molecules with this bit set
        molecules_with_bit = np.where(fingerprints[:, bit_number] == 1)[0]
        
        if len(molecules_with_bit) == 0:
            logger.warning(f"No molecules found with bit {bit_number} set")
            return []
        
        examples = []
        for idx in molecules_with_bit[:5]:  # Limit to first 5 examples
            mol = molecules[idx]
            svg = self.draw_fragment_from_bit(mol, bit_number)
            if svg:
                examples.append(svg)
        
        logger.info(f"Found {len(examples)} examples for bit {bit_number}")
        return examples
    
    def interpret_feature_importance(self, feature_analysis: dict, molecules: List, 
                                   fingerprints: np.ndarray, top_n: int = 10) -> dict:
        """
        Interpret feature importance by connecting weights/importances to molecular substructures.
        
        This implements the missing Exercise 1 component for molecular interpretation.
        
        Parameters:
            feature_analysis (dict): Output from HERGClassifier.analyze_feature_importance()
            molecules (List): List of RDKit molecule objects
            fingerprints (numpy.ndarray): Array of fingerprints
            top_n (int): Number of top features to interpret
            
        Returns:
            dict: Molecular interpretation of important features
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - cannot interpret molecular features")
            return {}
        
        interpretation = {
            'model_type': feature_analysis.get('model_type', 'unknown'),
            'top_features': [],
            'substructure_examples': {},
            'molecular_insights': []
        }
        
        # Handle different model types
        if feature_analysis.get('model_type') == 'linear':
            # Linear model (SGD) - use coefficients
            weights = feature_analysis.get('weights', [])
            if len(weights) == 0:
                return interpretation
            
            # Get most important features (by absolute weight)
            top_indices = feature_analysis.get('most_important_indices', [])[:top_n]
            
            for i, bit_idx in enumerate(top_indices):
                bit_weight = weights[bit_idx]
                
                # Find molecules with this bit set
                molecules_with_bit = np.where(fingerprints[:, bit_idx] == 1)[0]
                
                if len(molecules_with_bit) > 0:
                    # Get examples of molecules with this bit
                    example_mols = [molecules[idx] for idx in molecules_with_bit[:3]]
                    
                    # Generate visualization for this bit
                    bit_info = self.get_bit_info(example_mols[0], bit_idx)
                    
                    feature_interpretation = {
                        'rank': i + 1,
                        'bit_index': int(bit_idx),
                        'weight': float(bit_weight),
                        'effect': 'promotes hERG blocking' if bit_weight > 0 else 'prevents hERG blocking',
                        'frequency': int(len(molecules_with_bit)),
                        'frequency_percent': float(len(molecules_with_bit) / len(molecules) * 100),
                        'example_molecules': len(example_mols),
                        'bit_info': bit_info
                    }
                    
                    interpretation['top_features'].append(feature_interpretation)
                    
                    # Store substructure examples
                    interpretation['substructure_examples'][int(bit_idx)] = {
                        'weight': float(bit_weight),
                        'molecules_count': int(len(molecules_with_bit)),
                        'example_indices': [int(x) for x in molecules_with_bit[:5]]
                    }
        
        elif feature_analysis.get('model_type') == 'tree_based':
            # Tree-based model (Random Forest) - use feature importances
            importances = feature_analysis.get('importances', [])
            if len(importances) == 0:
                return interpretation
            
            # Get most important features
            top_indices = feature_analysis.get('most_important_indices', [])[:top_n]
            
            for i, bit_idx in enumerate(top_indices):
                bit_importance = importances[bit_idx]
                
                # Find molecules with this bit set
                molecules_with_bit = np.where(fingerprints[:, bit_idx] == 1)[0]
                
                if len(molecules_with_bit) > 0:
                    # Get examples of molecules with this bit
                    example_mols = [molecules[idx] for idx in molecules_with_bit[:3]]
                    
                    # Generate visualization for this bit
                    bit_info = self.get_bit_info(example_mols[0], bit_idx)
                    
                    feature_interpretation = {
                        'rank': i + 1,
                        'bit_index': int(bit_idx),
                        'importance': float(bit_importance),
                        'effect': 'important for hERG prediction',
                        'frequency': int(len(molecules_with_bit)),
                        'frequency_percent': float(len(molecules_with_bit) / len(molecules) * 100),
                        'example_molecules': len(example_mols),
                        'bit_info': bit_info
                    }
                    
                    interpretation['top_features'].append(feature_interpretation)
                    
                    # Store substructure examples
                    interpretation['substructure_examples'][int(bit_idx)] = {
                        'importance': float(bit_importance),
                        'molecules_count': int(len(molecules_with_bit)),
                        'example_indices': [int(x) for x in molecules_with_bit[:5]]
                    }
        
        # Generate molecular insights
        interpretation['molecular_insights'] = self._generate_molecular_insights(interpretation)
        
        logger.info(f"Molecular interpretation completed for {len(interpretation['top_features'])} features")
        return interpretation
    
    def _generate_molecular_insights(self, interpretation: dict) -> List[str]:
        """Generate human-readable molecular insights from feature interpretation."""
        insights = []
        
        if interpretation['model_type'] == 'linear':
            positive_features = [f for f in interpretation['top_features'] if f.get('weight', 0) > 0]
            negative_features = [f for f in interpretation['top_features'] if f.get('weight', 0) < 0]
            
            if positive_features:
                insights.append(f"Found {len(positive_features)} molecular features that promote hERG blocking")
                most_positive = max(positive_features, key=lambda x: x['weight'])
                insights.append(f"Strongest hERG-promoting feature appears in {most_positive['frequency_percent']:.1f}% of molecules")
            
            if negative_features:
                insights.append(f"Found {len(negative_features)} molecular features that prevent hERG blocking")
                most_negative = min(negative_features, key=lambda x: x['weight'])
                insights.append(f"Strongest hERG-preventing feature appears in {most_negative['frequency_percent']:.1f}% of molecules")
        
        elif interpretation['model_type'] == 'tree_based':
            if interpretation['top_features']:
                insights.append(f"Random Forest identified {len(interpretation['top_features'])} key molecular features")
                most_important = max(interpretation['top_features'], key=lambda x: x['importance'])
                insights.append(f"Most discriminative feature appears in {most_important['frequency_percent']:.1f}% of molecules")
        
        return insights
    
    def visualize_important_substructures(self, interpretation: dict, molecules: List, 
                                        save_dir: str = "figures/chapter3") -> List[str]:
        """
        Create visualizations of important molecular substructures.
        
        This implements the missing Exercise 1 visualization component.
        
        Parameters:
            interpretation (dict): Output from interpret_feature_importance()
            molecules (List): List of RDKit molecule objects  
            save_dir (str): Directory to save visualizations
            
        Returns:
            List[str]: List of saved visualization file paths
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - cannot create molecular visualizations")
            return []
        
        from pathlib import Path
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        # Create visualizations for top features
        for feature in interpretation['top_features'][:5]:  # Top 5 features
            bit_idx = feature['bit_index']
            
            if bit_idx in interpretation['substructure_examples']:
                example_info = interpretation['substructure_examples'][bit_idx]
                example_indices = example_info['example_indices'][:4]  # Up to 4 examples
                
                # Get example molecules
                example_mols = [molecules[idx] for idx in example_indices if idx < len(molecules)]
                
                if example_mols:
                    try:
                        from rdkit.Chem import Draw
                        
                        # Create a grid of molecules showing this feature
                        legends = []
                        for idx, mol_idx in enumerate(example_indices[:len(example_mols)]):
                            if interpretation['model_type'] == 'linear':
                                weight = feature.get('weight', 0)
                                effect = 'Promotes' if weight > 0 else 'Prevents'
                                legends.append(f"Mol {mol_idx}: {effect} hERG")
                            else:
                                importance = feature.get('importance', 0)
                                legends.append(f"Mol {mol_idx}: Importance {importance:.3f}")
                        
                        # Generate grid image
                        img = Draw.MolsToGridImage(
                            example_mols,
                            molsPerRow=2,
                            subImgSize=(300, 300),
                            legends=legends,
                            useSVG=True
                        )
                        
                        # Save the image
                        filename = f"feature_bit_{bit_idx}_examples.svg"
                        filepath = save_path / filename
                        
                        with open(filepath, 'w') as f:
                            # Handle different RDKit versions
                            if hasattr(img, 'data'):
                                f.write(img.data)
                            else:
                                f.write(img)
                        
                        saved_files.append(str(filepath))
                        
                    except Exception as e:
                        logger.warning(f"Failed to create visualization for bit {bit_idx}: {e}")
        
        logger.info(f"Created {len(saved_files)} molecular substructure visualizations")
        return saved_files


class SmilesToMols(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer to convert SMILES strings to RDKit molecule objects.
    
    This transformer is used in ML pipelines to standardize the molecular representation step.
    """
    
    def __init__(self, standardize: bool = True):
        """
        Initialize the transformer.
        
        Parameters:
            standardize (bool): Whether to apply standardization to molecules
        """
        self.standardize = standardize
        
        if RDKIT_AVAILABLE and standardize:
            from rdkit.Chem.MolStandardize.rdMolStandardize import (
                Cleanup, LargestFragmentChooser, TautomerEnumerator, Uncharger
            )
            self.cleanup = Cleanup
            self.fragment_chooser = LargestFragmentChooser()
            self.uncharger = Uncharger()
            self.tautomer_enumerator = TautomerEnumerator()
    
    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """
        Transform SMILES strings to RDKit molecule objects.
        
        Parameters:
            X: Array-like of SMILES strings
            
        Returns:
            List of RDKit molecule objects
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - returning SMILES strings")
            return X
        
        molecules = []
        for smiles in X:
            mol = self._process_smiles(smiles)
            molecules.append(mol)
        
        return molecules
    
    def _process_smiles(self, smiles: str):
        """Process a single SMILES string."""
        if not isinstance(smiles, str):
            return None
        
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None or not self.standardize:
            return mol
        
        try:
            # Apply standardization
            mol = self.cleanup(mol)
            mol = self.fragment_chooser.choose(mol)
            mol = self.uncharger.uncharge(mol)
            mol = self.tautomer_enumerator.Canonicalize(mol)
            return mol
        except Exception:
            return mol


class FingerprintFeaturizer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer to convert molecules to fingerprint features.
    
    This transformer is used in ML pipelines for the feature extraction step.
    """
    
    def __init__(self, radius: int = 2, n_bits: int = 2048):
        """
        Initialize the featurizer.
        
        Parameters:
            radius (int): Morgan fingerprint radius
            n_bits (int): Number of bits in fingerprint
        """
        self.radius = radius
        self.n_bits = n_bits
        self.featurizer = MolecularFeaturizer(radius=radius, n_bits=n_bits)
    
    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """
        Transform molecules to fingerprint features.
        
        Parameters:
            X: List of RDKit molecule objects or SMILES strings
            
        Returns:
            numpy.ndarray: Array of fingerprints
        """
        if isinstance(X, (list, np.ndarray)):
            return self.featurizer.compute_fingerprints_batch(X, self.radius, self.n_bits)
        else:
            # Single molecule
            return self.featurizer.compute_fingerprint(X, self.radius, self.n_bits).reshape(1, -1) 