"""
Data validation utilities for METASIGHT.

This module provides functions to validate data integrity, feature dimensions,
and minimum sample sizes before training.
"""

import os
import glob
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


# Foundation model expected dimensions
FOUNDATION_MODEL_DIMS = {
    'CHIEF': 768,
    'UNI': 1024,
    'GIGAPATH': 1536,
    'VIRCHOW2': 2560
}

# Minimum sample sizes per cancer for reliable evaluation
MIN_SAMPLES_PER_CANCER = {
    'training': 30,    # Minimum slides for training
    'validation': 10,  # Minimum slides for validation
    'test': 30,        # Minimum slides for testing (per-cancer evaluation)
    'positive_class': 5,  # Minimum positive samples per class
}


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def validate_feature_dimension(feature_path: str, expected_dim: int, 
                               slide_id: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate that a feature file has the expected dimension.
    
    Args:
        feature_path: Path to .pt feature file
        expected_dim: Expected feature dimension
        slide_id: Optional slide ID for error messages
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        features = torch.load(feature_path)
        
        if not isinstance(features, torch.Tensor):
            return False, f"Feature file is not a tensor: {type(features)}"
        
        if len(features.shape) != 2:
            return False, f"Expected 2D tensor [n_patches, dim], got shape {features.shape}"
        
        n_patches, feature_dim = features.shape
        
        if feature_dim != expected_dim:
            return False, f"Expected dimension {expected_dim}, got {feature_dim}"
        
        if n_patches == 0:
            return False, "Feature file has 0 patches"
        
        # Check for NaN or Inf values
        if torch.isnan(features).any():
            return False, "Feature file contains NaN values"
        
        if torch.isinf(features).any():
            return False, "Feature file contains Inf values"
        
        return True, ""
        
    except Exception as e:
        return False, f"Error loading feature file: {str(e)}"


def validate_feature_directory(cancer: str, foundation_model: str, slide_type: str,
                               feature_root: str, verbose: bool = True) -> Dict:
    """
    Validate all feature files for a cancer type.
    
    Args:
        cancer: Cancer type (e.g., 'BRCA')
        foundation_model: Foundation model name
        slide_type: 'FS', 'PM', or 'MIX'
        feature_root: Root directory for features
        verbose: Print detailed validation results
    
    Returns:
        Dictionary with validation results
    """
    expected_dim = FOUNDATION_MODEL_DIMS.get(foundation_model)
    if expected_dim is None:
        raise ValueError(f"Unknown foundation model: {foundation_model}")
    
    # Build file paths
    if slide_type == "MIX":
        fs_path = f"{feature_root}TCGA-{cancer}-FS/{foundation_model}/20X/pt_files(stain_norm)/*.pt"
        pm_path = f"{feature_root}TCGA-{cancer}-PM/{foundation_model}/20X/pt_files(stain_norm)/*.pt"
        feature_paths = glob.glob(fs_path) + glob.glob(pm_path)
    else:
        path = f"{feature_root}TCGA-{cancer}-{slide_type}/{foundation_model}/20X/pt_files(stain_norm)/*.pt"
        feature_paths = glob.glob(path)
    
    if not feature_paths:
        raise DataValidationError(
            f"No feature files found for {cancer} {foundation_model} {slide_type}"
        )
    
    # Validate each file
    valid_files = []
    invalid_files = []
    total_patches = 0
    patch_counts = []
    
    for fpath in feature_paths:
        slide_id = os.path.basename(fpath).replace('.pt', '')
        is_valid, error_msg = validate_feature_dimension(fpath, expected_dim, slide_id)
        
        if is_valid:
            valid_files.append(fpath)
            features = torch.load(fpath)
            n_patches = features.shape[0]
            total_patches += n_patches
            patch_counts.append(n_patches)
        else:
            invalid_files.append((fpath, error_msg))
            if verbose:
                print(f"Invalid: {slide_id} - {error_msg}")
    
    # Compute statistics
    results = {
        'cancer': cancer,
        'foundation_model': foundation_model,
        'slide_type': slide_type,
        'n_valid': len(valid_files),
        'n_invalid': len(invalid_files),
        'total_files': len(feature_paths),
        'validation_rate': len(valid_files) / len(feature_paths) if feature_paths else 0,
        'invalid_files': invalid_files,
        'total_patches': total_patches,
        'mean_patches': np.mean(patch_counts) if patch_counts else 0,
        'median_patches': np.median(patch_counts) if patch_counts else 0,
        'min_patches': min(patch_counts) if patch_counts else 0,
        'max_patches': max(patch_counts) if patch_counts else 0,
    }
    
    if verbose:
        print(f"\nValidation Results for {cancer} {foundation_model} {slide_type}:")
        print(f"  Valid files: {results['n_valid']}/{results['total_files']} "
              f"({results['validation_rate']*100:.1f}%)")
        print(f"  Total patches: {results['total_patches']:,}")
        print(f"  Patches per slide: {results['mean_patches']:.0f} ± "
              f"{np.std(patch_counts):.0f} (median: {results['median_patches']:.0f})")
        print(f"  Range: [{results['min_patches']}, {results['max_patches']}]")
    
    return results


def validate_clinical_data(clinical_df: pd.DataFrame, required_cols: List[str],
                           task: str = 'status') -> Dict:
    """
    Validate clinical data completeness.
    
    Args:
        clinical_df: Clinical dataframe
        required_cols: List of required column names
        task: 'status' or 'trajectory'
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'n_total': len(clinical_df),
        'missing_columns': [],
        'missing_data': {},
        'valid': True
    }
    
    # Check required columns
    for col in required_cols:
        if col not in clinical_df.columns:
            results['missing_columns'].append(col)
            results['valid'] = False
    
    # Check for missing data in required columns
    for col in required_cols:
        if col in clinical_df.columns:
            n_missing = clinical_df[col].isna().sum()
            if n_missing > 0:
                results['missing_data'][col] = {
                    'n_missing': int(n_missing),
                    'pct_missing': float(n_missing / len(clinical_df) * 100)
                }
    
    # Task-specific validations
    if task == 'status' and 'metastasis_label' in clinical_df.columns:
        results['class_distribution'] = clinical_df['metastasis_label'].value_counts().to_dict()
        results['n_positive'] = int((clinical_df['metastasis_label'] == 1).sum())
        results['n_negative'] = int((clinical_df['metastasis_label'] == 0).sum())
        
    elif task == 'trajectory' and 'tumor_event_label' in clinical_df.columns:
        results['class_distribution'] = clinical_df['tumor_event_label'].value_counts().to_dict()
        results['n_censored'] = int(clinical_df.get('censored', pd.Series([0])).sum())
    
    return results


def check_minimum_samples(df: pd.DataFrame, label_col: str, 
                         cancer_col: Optional[str] = None,
                         min_train: int = MIN_SAMPLES_PER_CANCER['training'],
                         min_pos: int = MIN_SAMPLES_PER_CANCER['positive_class']) -> Dict:
    """
    Check if dataset meets minimum sample size requirements.
    
    Args:
        df: Dataset dataframe
        label_col: Name of label column
        cancer_col: Optional cancer type column for per-cancer checks
        min_train: Minimum total samples
        min_pos: Minimum positive samples per class
    
    Returns:
        Dictionary with validation results and warnings
    """
    results = {
        'meets_requirements': True,
        'warnings': [],
        'n_total': len(df)
    }
    
    # Check total samples
    if len(df) < min_train:
        results['meets_requirements'] = False
        results['warnings'].append(
            f"Total samples ({len(df)}) below minimum ({min_train})"
        )
    
    # Check class balance
    class_counts = df[label_col].value_counts()
    results['class_counts'] = class_counts.to_dict()
    
    for cls, count in class_counts.items():
        if count < min_pos:
            results['warnings'].append(
                f"Class {cls} has only {count} samples (minimum: {min_pos})"
            )
    
    # Per-cancer checks if applicable
    if cancer_col is not None and cancer_col in df.columns:
        per_cancer = {}
        for cancer in df[cancer_col].unique():
            cancer_df = df[df[cancer_col] == cancer]
            cancer_counts = cancer_df[label_col].value_counts()
            
            per_cancer[cancer] = {
                'n_total': len(cancer_df),
                'class_counts': cancer_counts.to_dict(),
                'sufficient': len(cancer_df) >= MIN_SAMPLES_PER_CANCER['test']
            }
            
            if len(cancer_df) < MIN_SAMPLES_PER_CANCER['test']:
                results['warnings'].append(
                    f"{cancer}: {len(cancer_df)} samples < {MIN_SAMPLES_PER_CANCER['test']} "
                    f"(may not be sufficient for per-cancer evaluation)"
                )
        
        results['per_cancer'] = per_cancer
    
    return results


def validate_features_match_clinical(feature_dict: Dict, clinical_df: pd.DataFrame,
                                     slide_id_col: str = 'folder_id') -> Dict:
    """
    Check that feature files match clinical data.
    
    Args:
        feature_dict: Dictionary of {slide_id: features}
        clinical_df: Clinical dataframe
        slide_id_col: Column name for slide IDs
    
    Returns:
        Dictionary with matching statistics
    """
    clinical_slides = set(clinical_df[slide_id_col].values)
    feature_slides = set(feature_dict.keys())
    
    matched = clinical_slides & feature_slides
    clinical_only = clinical_slides - feature_slides
    features_only = feature_slides - clinical_slides
    
    results = {
        'n_matched': len(matched),
        'n_clinical_only': len(clinical_only),
        'n_features_only': len(features_only),
        'match_rate': len(matched) / len(clinical_slides) if clinical_slides else 0,
        'clinical_only_ids': list(clinical_only),
        'features_only_ids': list(features_only)
    }
    
    if len(clinical_only) > 0:
        print(f"Warning: {len(clinical_only)} slides in clinical data have no features")
    
    if len(features_only) > 0:
        print(f"Note: {len(features_only)} feature files have no clinical data")
    
    return results


def validate_dataset_for_training(cancer: str, foundation_model: str, slide_type: str,
                                  feature_root: str, clinical_root: str, label_file: str,
                                  task: str = 'status', verbose: bool = True) -> bool:
    """
    Comprehensive validation of dataset before training.
    
    Args:
        cancer: Cancer type
        foundation_model: Foundation model name
        slide_type: Slide type
        feature_root: Feature directory root
        clinical_root: Clinical data root
        label_file: Path to label file
        task: 'status' or 'trajectory'
        verbose: Print detailed results
    
    Returns:
        True if all validations pass, False otherwise
    """
    all_valid = True
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Validating dataset: {cancer} {foundation_model} {slide_type}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")
    
    # 1. Validate feature files
    try:
        feature_results = validate_feature_directory(
            cancer, foundation_model, slide_type, feature_root, verbose
        )
        if feature_results['n_invalid'] > 0:
            all_valid = False
            if verbose:
                print(f"Warning: {feature_results['n_invalid']} invalid feature files")
    except Exception as e:
        if verbose:
            print(f"Error validating features: {e}")
        all_valid = False
    
    # 2. Load and validate clinical data
    try:
        from data import load_clinical_data, load_trajectory_clinical_data
        
        if task == 'status':
            clinical_df = load_clinical_data(cancer, clinical_root, label_file)
            required_cols = ['case_submitter_id', 'folder_id', 'metastasis_label']
            label_col = 'metastasis_label'
        else:
            clinical_df = load_trajectory_clinical_data(
                cancer, 365, clinical_root, label_file
            )
            required_cols = ['case_submitter_id', 'folder_id', 'tumor_event_label']
            label_col = 'tumor_event_label'
        
        if clinical_df is None or len(clinical_df) == 0:
            if verbose:
                print("Error: No clinical data loaded")
            return False
        
        clinical_results = validate_clinical_data(clinical_df, required_cols, task)
        
        if not clinical_results['valid']:
            all_valid = False
            if verbose:
                print(f"Clinical data validation failed:")
                if clinical_results['missing_columns']:
                    print(f"  Missing columns: {clinical_results['missing_columns']}")
        
        # 3. Check minimum samples
        sample_results = check_minimum_samples(clinical_df, label_col)
        
        if not sample_results['meets_requirements']:
            all_valid = False
            if verbose:
                print(f"\nSample size warnings:")
                for warning in sample_results['warnings']:
                    print(f"  - {warning}")
        
        if verbose and 'class_distribution' in clinical_results:
            print(f"\nClass distribution:")
            for cls, count in clinical_results['class_distribution'].items():
                print(f"  Class {cls}: {count}")
        
    except Exception as e:
        if verbose:
            print(f"Error validating clinical data: {e}")
        all_valid = False
    
    if verbose:
        print(f"\n{'='*60}")
        if all_valid:
            print("Validation PASSED: Dataset is ready for training")
        else:
            print("Validation FAILED: Please address issues before training")
        print(f"{'='*60}\n")
    
    return all_valid


if __name__ == "__main__":
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate METASIGHT dataset")
    parser.add_argument("--cancer", type=str, required=True)
    parser.add_argument("--foundation_model", type=str, default="CHIEF")
    parser.add_argument("--slide_type", type=str, default="FS")
    parser.add_argument("--task", type=str, default="status", choices=['status', 'trajectory'])
    parser.add_argument("--feature_root", type=str, 
                       default="/n/data2/hms/dbmi/kyu/lab/NCKU/foundation_model_features/WSI_features/")
    parser.add_argument("--clinical_root", type=str,
                       default="/n/data2/hms/dbmi/kyu/lab/pet200/clinical_gdc")
    parser.add_argument("--label_file", type=str,
                       default="/n/data2/hms/dbmi/kyu/lab/tik161/Metastasis_STpath/data/labels/TCGA_pancancer_label.csv")
    
    args = parser.parse_args()
    
    validate_dataset_for_training(
        args.cancer, args.foundation_model, args.slide_type,
        args.feature_root, args.clinical_root, args.label_file,
        args.task, verbose=True
    )

