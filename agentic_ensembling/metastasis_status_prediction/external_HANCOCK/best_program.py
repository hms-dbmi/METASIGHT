import numpy as np
from typing import Dict

def ensemble_predictions(
    model_preds: Dict[str, Dict],
    cancer_types: np.ndarray
) -> np.ndarray:
    
    # Improved ensemble:
    # - Cancer-specific base weights using model specializations
    # - Per-sample confidence from fold-level agreement (if folds available)
    # - Weighted average (normalized per-sample) + small median blending for robustness
    models = ['CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2']

    # Collect consolidated predictions (fallback to fold average if needed)
    preds_list = []
    available_models = []
    fold_matrices = {}  # store fold-level predictions for confidence scoring
    n_patients = None

    for m in models:
        if m not in model_preds:
            continue
        mp = model_preds[m]
        # Prefer 'consolidated'
        if 'consolidated' in mp:
            arr = np.asarray(mp['consolidated'], dtype=float)
        else:
            fold_preds = [v for k, v in mp.items() if isinstance(k, int)]
            if fold_preds:
                arr = np.nanmean(np.stack(fold_preds, axis=0), axis=0)
            else:
                continue

        if n_patients is None:
            n_patients = arr.shape[0]
        else:
            # Safety: ensure same length
            if arr.shape[0] != n_patients:
                arr = np.resize(arr, n_patients)

        preds_list.append(arr)
        available_models.append(m)

        # Build fold matrix if fold-level data exists for confidence scoring
        fold_list = []
        for k, v in model_preds[m].items():
            if isinstance(k, int):
                fold_list.append(np.asarray(v, dtype=float))
        if fold_list:
            try:
                fold_matrices[m] = np.stack(fold_list, axis=0)  # shape: (n_folds, n_patients)
            except Exception:
                # if shapes mismatch, compute nanmean fallback (no fold confidence)
                fold_matrices[m] = None

    if not preds_list:
        # No available model predictions -> return uniform 0.5
        return np.full(shape=(len(cancer_types),), fill_value=0.5, dtype=float)

    preds_mat = np.vstack(preds_list)  # shape: (n_models_avail, n_patients)

    # Improved weighting pipeline:
    # - Stronger prior for GIGAPATH (best single model)
    # - Per-model decisiveness and fold-stability factors
    # - Cancer-specific multiplicative boosts for specialties
    # - Per-sample confidence from extremeness and fold agreement
    # - Per-sample agreement amplification for the top trusted model
    # - Calibration that regularizes low-mass (low-confidence) samples toward 0.5
    M, N = preds_mat.shape
    # Stronger base prior favoring GIGAPATH (empirical best)
    base_prior_map = {'GIGAPATH': 0.50, 'UNI': 0.22, 'CHIEF': 0.14, 'VIRCHOW2': 0.14}
    base_w = np.array([base_prior_map.get(m, 0.25) for m in available_models], dtype=float)

    # Per-model decisiveness (how far on average a model is from 0.5)
    decisiveness = np.nanmean(np.abs(preds_mat - 0.5), axis=1)  # (M,)
    max_dec = np.nanmax(decisiveness) if np.nanmax(decisiveness) > 0 else 1.0
    decisiveness_factor = 1.0 + 0.8 * (decisiveness / max_dec)  # in [1.0, 1.8]

    # Per-model fold-based stability penalty (more variable -> lower)
    fold_mean_std = []
    for m in available_models:
        if m in fold_matrices and fold_matrices[m] is not None:
            fm = fold_matrices[m]
            fold_mean_std.append(np.nanmean(np.nanstd(fm, axis=0)))
        else:
            fold_mean_std.append(0.0)
    fold_mean_std = np.array(fold_mean_std, dtype=float)
    max_fold = np.nanmax(fold_mean_std) if np.nanmax(fold_mean_std) > 0 else 1.0
    stability_penalty = 1.0 - 0.5 * (fold_mean_std / (max_fold + 1e-12))
    stability_penalty = np.clip(stability_penalty, 0.5, 1.0)  # in [0.5, 1.0]

    # Combined per-model factor
    model_factor = decisiveness_factor * stability_penalty
    base_w = base_w * model_factor
    # normalize base priors so they sum to 1 (keeps scale stable)
    base_w = base_w / (base_w.sum() + 1e-12)

    # Cancer-specialty sets (multiplicative boosts)
    cancer_boosts = {
        'GIGAPATH': set(['HNSC', 'KIRC', 'STAD', 'TGCT', 'ESCA']),
        'UNI': set(['KIRP', 'KIRC', 'READ', 'SKCM', 'THCA']),
        'CHIEF': set(['CHOL', 'PAAD', 'BRCA', 'MESO']),
        'VIRCHOW2': set(['ACC', 'CESC', 'KICH', 'LIHC', 'COAD'])
    }
    boost_factor = 1.20  # multiplicative boost for specialty cancers

    # Start building per-sample raw weights
    W = np.tile(base_w[:, None], (1, N))  # (M, N)

    # Apply cancer-specific multiplicative boosts
    for i, m in enumerate(available_models):
        specs = cancer_boosts.get(m, set())
        if specs:
            mask = np.isin(cancer_types, list(specs))
            if mask.any():
                W[i, mask] *= boost_factor

    # Per-model, per-sample confidence:
    # Prefer fold-based confidence when available, otherwise use extremeness proxy
    conf_matrix = np.ones_like(preds_mat, dtype=float)
    for i, m in enumerate(available_models):
        if m in fold_matrices and fold_matrices[m] is not None:
            fm = fold_matrices[m]  # (n_folds, N)
            std = np.nanstd(fm, axis=0)
            conf = 1.0 / (1.0 + std)  # higher when std small
            mn, mx = np.nanmin(conf), np.nanmax(conf)
            if mx - mn < 1e-12:
                conf_scaled = np.full_like(conf, 0.85)
            else:
                conf_scaled = 0.6 + 0.4 * (conf - mn) / (mx - mn)
            conf_matrix[i, :] = conf_scaled
        else:
            # extremeness proxy scaled to [0.6, 1.0]
            proxy = np.abs(preds_mat[i] - 0.5) * 2.0  # in [0,1]
            conf_matrix[i, :] = 0.6 + 0.4 * proxy

    # Apply confidence multiplier (favor extreme / fold-stable predictions)
    alpha = 0.85
    W = W * (1.0 + alpha * conf_matrix)

    # Per-sample agreement amplification: when models largely agree, softly boost the top-weighted model
    sample_std = np.nanstd(preds_mat, axis=0)
    agree = 1.0 - np.clip(sample_std / 0.5, 0.0, 1.0)  # higher -> more agreement
    top_idx = np.argmax(W, axis=0)
    # vectorized multiplication for speed
    idx = (top_idx.astype(int), np.arange(N))
    agree_boost = 0.50
    W[idx] = W[idx] * (1.0 + agree_boost * agree)

    # Handle missing/NaN model predictions: zero their weight and set predictions to 0.5 neutral
    nan_mask = np.isnan(preds_mat)
    if np.any(nan_mask):
        W[nan_mask] = 0.0
        preds_mat = np.where(nan_mask, 0.5, preds_mat)

    # Preserve total raw mass for calibration later, then normalize per-sample
    raw_mass = W.sum(axis=0)  # shape (N,)
    eps = 1e-12
    sumW = raw_mass.copy()
    zero_mask = (sumW <= eps)
    if zero_mask.any():
        # fallback: distribute uniformly among available models for these patients
        available_per_patient = (~np.isnan(preds_mat)).sum(axis=0)
        available_per_patient[available_per_patient == 0] = 1
        fallback = (1.0 / available_per_patient)
        # broadcast fallback to rows
        W[:, zero_mask] = fallback[zero_mask]
        sumW = W.sum(axis=0)
    # Normalize
    W = W / (sumW + eps)

    # Weighted aggregation
    ensemble_pred = (W * preds_mat).sum(axis=0)

    # Mild calibration: pull low-raw-mass (low-confidence) samples toward 0.5
    max_mass = np.max(raw_mass) + eps
    conf_norm = np.clip(raw_mass / max_mass, 0.0, 1.0)
    conf_sharp = conf_norm ** 1.25
    ensemble_pred = conf_sharp * ensemble_pred + (1.0 - conf_sharp) * 0.5

    # Final safety + clipping
    ensemble_pred = np.clip(np.nan_to_num(ensemble_pred, nan=0.5), 0.0, 1.0)
    return ensemble_pred

def get_ensemble_function():
    return ensemble_predictions
