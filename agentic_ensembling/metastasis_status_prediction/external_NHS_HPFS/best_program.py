import numpy as np
from typing import Dict

def ensemble_predictions(
    model_preds: Dict[str, Dict],
    cancer_types: np.ndarray
) -> np.ndarray:
    
    # Weighted, cancer-aware aggregation using consolidated predictions
    models = ['CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2']
    preds = []
    present = []
    n = int(cancer_types.shape[0])

    for m in models:
        mp = model_preds.get(m, {})
        if 'consolidated' in mp:
            p = np.asarray(mp['consolidated'], dtype=float)
        else:
            fold_preds = [v for k, v in mp.items() if isinstance(k, int)]
            p = np.nanmean(fold_preds, axis=0) if fold_preds else None

        if p is None:
            continue

        p = np.asarray(p, dtype=float)
        # fix length mismatches conservatively
        if p.size != n:
            if p.size > n:
                p = p[:n]
            else:
                p = np.concatenate([p, np.full(n - p.size, np.nan, dtype=float)])

        # replace all-NaN with neutral 0.5, then fill remaining NaNs with model mean
        if np.all(np.isnan(p)):
            p = np.full(n, 0.5, dtype=float)
        mean_val = np.nanmean(p)
        if not np.isfinite(mean_val):
            mean_val = 0.5
        p = np.where(np.isnan(p), mean_val, p)

        preds.append(p)
        present.append(m)

    if len(preds) == 0:
        return np.full(n, 0.5, dtype=float)

    P = np.vstack(preds)  # shape (M, n)

    # Base weights (reflect overall strengths). Slightly favor GIGAPATH but boost UNI for COAD.
    base = {'CHIEF': 0.20, 'UNI': 0.30, 'GIGAPATH': 0.35, 'VIRCHOW2': 0.15}

    # Per-model confidence from fold-wise std (lower std -> higher confidence).
    # This lets models that are consistent across folds influence the ensemble more.
    M = len(present)
    conf_mat = np.ones((M, n), dtype=float)
    for i, m in enumerate(present):
        mp = model_preds.get(m, {})
        fold_vals = [np.asarray(v, dtype=float) for k, v in mp.items() if isinstance(k, int)]
        if fold_vals:
            try:
                # stack fold predictions (n_folds, n_patients) and compute per-sample std
                s = np.nanstd(np.vstack(fold_vals), axis=0)
                # replace NaNs with a sensible median fallback
                med = np.nanmedian(s) if np.isfinite(np.nanmedian(s)) else 0.15
                s = np.where(np.isnan(s), med, s)
                # convert std -> confidence in (0,1], smaller std -> higher confidence
                conf = 1.0 / (1.0 + s)
            except Exception:
                conf = np.ones(n, dtype=float)
        else:
            # no fold info -> neutral confidence
            conf = np.ones(n, dtype=float)
        conf_mat[i, :] = conf

    # Apply base weight modulated by per-sample confidence (shape (M,n))
    W_base = np.array([base[m] for m in present], dtype=float)[:, None]  # (M,1)
    # agreement factor: reward models close to the per-sample median prediction
    med = np.median(P, axis=0)
    distances = np.abs(P - med)  # (M, n)
    agreement = 1.0 / (1.0 + distances)  # closer -> closer to 1
    agreement = np.clip(agreement, 0.05, 1.0)

    # Compose raw weights: base * confidence * agreement
    W_raw = W_base * conf_mat * agreement  # (M, n)

    # Cancer-specific adjustments: UNI is best on COAD in this cohort; VIRCHOW2 is stable.
    co_mask = (cancer_types == 'COAD')
    if np.any(co_mask):
        if 'UNI' in present:
            W_raw[present.index('UNI'), co_mask] *= 1.6
        if 'VIRCHOW2' in present:
            W_raw[present.index('VIRCHOW2'), co_mask] *= 1.12

    # Normalize per-sample weights and compute weighted sum
    weight_sums = np.sum(W_raw, axis=0, keepdims=True)
    # avoid degenerate columns
    weight_sums = np.where(weight_sums <= 0, 1.0, weight_sums)
    W = W_raw / weight_sums

    # Weighted aggregation
    ensemble_pred = np.sum(W * P, axis=0)

    # If a single model dominates (>65% weight), prefer that model's raw prediction (avoid over-smoothing)
    max_w = np.max(W, axis=0)
    dom_mask = max_w > 0.65
    if np.any(dom_mask):
        dom_idx = np.argmax(W, axis=0)
        idxs = np.where(dom_mask)[0]
        # For dominated samples, trust dominant model directly
        ensemble_pred[idxs] = P[dom_idx[idxs], idxs]

    # Stability-based shrinkage toward 0.5 when models disagree, reducing overconfident outputs.
    per_sample_std = np.std(P, axis=0)
    shrink = np.clip(per_sample_std * 1.25, 0.0, 0.55)
    ensemble_pred = ensemble_pred * (1.0 - shrink) + 0.5 * shrink

    # Final safety: replace NaN/Inf, fallback to per-column mean, and clip to [0,1]
    col_mean = np.nanmean(P, axis=0)
    ensemble_pred = np.where(np.isfinite(ensemble_pred), ensemble_pred, np.where(np.isfinite(col_mean), col_mean, 0.5))
    ensemble_pred = np.nan_to_num(ensemble_pred, nan=0.5, posinf=1.0, neginf=0.0)
    ensemble_pred = np.clip(ensemble_pred, 0.0, 1.0)

    return ensemble_pred

def get_ensemble_function():
    return ensemble_predictions
