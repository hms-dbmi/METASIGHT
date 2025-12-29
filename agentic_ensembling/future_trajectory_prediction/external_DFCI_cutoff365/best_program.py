import numpy as np
from typing import Dict

def ensemble_predictions(
    model_preds: Dict[str, Dict],
    cancer_types: np.ndarray,
    ipcw_weights: np.ndarray
) -> Dict[str, np.ndarray]:
    
    n_patients = len(cancer_types)
    # fallback uniform output if no patients
    if n_patients == 0:
        return {'class0': np.array([]), 'class1': np.array([]), 'class2': np.array([])}

    model_order = ['CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2']

    # Base performance priors (from cohort-level ranking). Sum to 1.
    base_priors = {
        'CHIEF': 0.32,
        'UNI': 0.28,
        'GIGAPATH': 0.24,
        'VIRCHOW2': 0.16
    }

    # Class multipliers per model (shape: [class0, class1, class2])
    # CHIEF is very strong on class1 (early event) per notes; adjust accordingly.
    class_factors = {
        'CHIEF': np.array([0.9, 1.4, 0.9]),
        'UNI': np.array([0.95, 0.9, 1.05]),
        'GIGAPATH': np.array([0.95, 0.9, 1.05]),
        'VIRCHOW2': np.array([1.0, 0.9, 1.0])
    }

    # Cancer-specific boosts: small multiplicative adjustments per model for some cancers
    cancer_boost_map = {
        'BRCA': {'CHIEF': 1.10},
        'LUAD': {'CHIEF': 1.07},
        'COAD': {'UNI': 1.10},
        'KIRC': {'UNI': 1.08},
        'OV': {'GIGAPATH': 1.08},
        'LUSC': {'GIGAPATH': 1.06}
    }

    # Collect present models and their prediction arrays
    present_models = []
    preds_list = []  # each entry shape (n_patients, 3)
    priors = []
    factors = []

    for m in model_order:
        if m not in model_preds:
            continue
        data = model_preds[m]
        arr = None
        if 'consolidated' in data:
            c = data['consolidated']
            if {'class0', 'class1', 'class2'}.issubset(c.keys()):
                arr = np.vstack([c['class0'], c['class1'], c['class2']]).T
        else:
            # fallback: average numeric folds if available
            fold_keys = [k for k in data.keys() if isinstance(k, int)]
            if fold_keys:
                fold_preds = []
                for k in fold_keys:
                    fk = data[k]
                    if {'class0', 'class1', 'class2'}.issubset(fk.keys()):
                        fold_preds.append(np.vstack([fk['class0'], fk['class1'], fk['class2']]).T)
                if fold_preds:
                    arr = np.nanmean(fold_preds, axis=0)

        if arr is None:
            continue

        # ensure shape correctness
        if arr.shape[0] != n_patients or arr.shape[1] != 3:
            # try to broadcast if model produced 1d arrays packaged differently
            try:
                arr = np.vstack([arr[:, 0], arr[:, 1], arr[:, 2]]).T
            except Exception:
                continue

        # Clip and renormalize to be safe
        arr = np.clip(arr, 1e-6, 1 - 1e-6)
        arr = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1e-10)

        present_models.append(m)
        preds_list.append(arr)
        priors.append(base_priors.get(m, 0.0))
        factors.append(class_factors.get(m, np.array([1.0, 1.0, 1.0])))

    if len(present_models) == 0:
        # No model predictions: return uniform probabilities
        uni = np.ones(n_patients) / 3.0
        return {'class0': uni.copy(), 'class1': uni.copy(), 'class2': uni.copy()}

    # Stack arrays: shape (n_models, n_patients, 3)
    stack = np.stack(preds_list, axis=0)
    priors = np.asarray(priors)  # (n_models,)
    factors = np.stack(factors, axis=0)  # (n_models, 3)

    # Normalize priors over present models
    if priors.sum() <= 0:
        priors = np.ones_like(priors) / len(priors)
    else:
        priors = priors / np.sum(priors)

    # Per-model per-sample confidence: max predicted probability (higher = model confident)
    max_prob = np.max(stack, axis=2)  # shape (n_models, n_patients)

    # Normalize ipcw weights to mean 1 to avoid scale issues
    ipcw = np.asarray(ipcw_weights, dtype=float)
    if ipcw.size != n_patients:
        ipcw = np.ones(n_patients)
    ipcw = np.maximum(ipcw, 1e-6)
    ipcw_norm = ipcw / np.maximum(np.mean(ipcw), 1e-10)

    # sample_conf: (n_models, n_patients)
    sample_conf = max_prob * ipcw_norm[np.newaxis, :]

    # base per-model per-class weights (broadcastable)
    model_class_weights = priors[:, np.newaxis] * factors  # shape (n_models, 3)

    # Cancer boosts: build boost matrix (n_models, n_patients)
    n_models = stack.shape[0]
    boosts = np.ones((n_models, n_patients), dtype=float)
    # Vectorized application of boosts
    for cancer, boosts_map in cancer_boost_map.items():
        mask = (cancer_types == cancer)
        if not np.any(mask):
            continue
        idxs = [i for i, m in enumerate(present_models) if m in boosts_map]
        for i in idxs:
            boosts[i, mask] = boosts_map[present_models[i]]

    # Final per-model per-sample per-class multipliers:
    # shape -> (n_models, n_patients, 3)
    multipliers = model_class_weights[:, np.newaxis, :] * (sample_conf[:, :, np.newaxis] * boosts[:, :, np.newaxis])

    # Weighted contributions
    contrib = stack * multipliers  # shape (n_models, n_patients, 3)
    agg = np.sum(contrib, axis=0)  # shape (n_patients, 3)

    # If aggregation could be zero (unlikely), fallback to simple mean
    row_sums = np.maximum(agg.sum(axis=1, keepdims=True), 1e-10)
    probs = agg / row_sums

    # Final light calibration: avoid extremal zeros/ones then renormalize
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

    # --- Risk-aware conservative recalibration to improve ranking (c_index) and AUROC ---
    # Build a risk score that emphasizes early-event probability (class1) while demoting high
    # censor probability (class0). Use the IPCW-normalized sample reliability (ipcw_norm)
    # to scale the conservative adjustment: more reliable samples get a slightly stronger boost.
    # Then apply a small, bounded multiplicative adjustment to class1 and re-normalize.
    try:
        risk = probs[:, 1] + 0.5 * (1.0 - probs[:, 0])  # higher => higher risk
        r_mean = np.mean(risk)
        r_std = np.std(risk) if np.std(risk) > 0 else 1.0
        risk_norm = (risk - r_mean) / r_std

        # Conservative scaling factor (gamma) tuned to nudge ranking without destabilizing calibration.
        gamma = 0.18
        # ipcw_norm was computed earlier; if missing, fall back to uniform scaling.
        ipcw_scale = ipcw_norm if ('ipcw_norm' in locals() or 'ipcw_norm' in globals()) else np.ones(probs.shape[0])
        ipcw_scale = np.clip(ipcw_scale, 0.7, 1.3)

        # Per-sample adjustment multiplier for class1, clipped to a safe range.
        adj = 1.0 + gamma * risk_norm * ipcw_scale
        adj = np.clip(adj, 0.7, 1.3)

        # Apply adjustment: boost class1, modestly reduce class0 and class2 proportionally,
        # then renormalize so probabilities sum to 1.
        probs[:, 1] = probs[:, 1] * adj
        shrink = 1.0 / np.maximum(adj, 1.0)
        probs[:, 0] = probs[:, 0] * shrink
        probs[:, 2] = probs[:, 2] * shrink

        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-10)
    except Exception:
        # If anything goes wrong, keep the original safe normalization
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-10)

    # Final clip to avoid numerical extremes
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

    return {
        'class0': probs[:, 0],
        'class1': probs[:, 1],
        'class2': probs[:, 2]
    }

def get_ensemble_function():
    return ensemble_predictions
