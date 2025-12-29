"""Evaluation metrics for metastasis prediction."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss


def compute_fold_aware_metrics(fold_results, metric_names=['auroc', 'brier']):
    """
    Compute average metrics excluding folds with invalid values.
    
    Handles cases where some folds have only one class (undefined AUROC).
    
    Args:
        fold_results: List of dicts with metrics for each fold
        metric_names: List of metric names to aggregate
    
    Returns:
        Dict with mean, std, and n_valid_folds for each metric
    """
    results = {}
    
    for metric in metric_names:
        values = []
        for r in fold_results:
            val = r.get(metric)
            if val is not None and not np.isnan(val):
                values.append(val)
        
        if len(values) > 0:
            results[f'{metric}_mean'] = np.mean(values)
            results[f'{metric}_std'] = np.std(values) if len(values) > 1 else 0.0
            results[f'{metric}_n_folds'] = len(values)
        else:
            results[f'{metric}_mean'] = None
            results[f'{metric}_std'] = None
            results[f'{metric}_n_folds'] = 0
    
    return results


def evaluate_per_cancer(test_preds, test_labels, test_cancers, min_samples=30):
    """
    Evaluate performance separately for each cancer type.
    
    Args:
        test_preds: Array of prediction probabilities
        test_labels: Array of true labels
        test_cancers: Array of cancer type strings
        min_samples: Minimum samples required for evaluation
    
    Returns:
        DataFrame with per-cancer metrics
    """
    per_cancer_results = []
    unique_cancers = np.unique(test_cancers)
    
    for cancer in unique_cancers:
        cancer_mask = (test_cancers == cancer)
        n_samples = np.sum(cancer_mask)
        
        if n_samples >= min_samples:
            cancer_preds = np.array(test_preds)[cancer_mask]
            cancer_labels = np.array(test_labels)[cancer_mask]
            
            # Check if both classes present
            if len(np.unique(cancer_labels)) > 1:
                try:
                    auroc = roc_auc_score(cancer_labels, cancer_preds)
                    brier = brier_score_loss(cancer_labels, cancer_preds)
                except ValueError:
                    auroc = None
                    brier = None
                
                per_cancer_results.append({
                    'cancer': cancer,
                    'n_test': n_samples,
                    'auroc': auroc,
                    'brier': brier,
                    'n_positive': np.sum(cancer_labels == 1),
                    'n_negative': np.sum(cancer_labels == 0),
                    'positive_rate': np.mean(cancer_labels)
                })
    
    return pd.DataFrame(per_cancer_results)

