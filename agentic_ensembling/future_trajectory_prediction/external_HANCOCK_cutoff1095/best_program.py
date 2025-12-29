import numpy as np
from typing import Dict

def ensemble_predictions(
    model_preds: Dict[str, Dict],
    cancer_types: np.ndarray,
    ipcw_weights: np.ndarray
) -> Dict[str, np.ndarray]:
    
    # Initialize and collect per-model predictions
    n_patients = len(cancer_types)

    model_names = ['CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2']
    preds_list = []      # list of arrays shape (n_patients, 3)
    used_models = []     # names in same order as preds_list

    for model_name in model_names:
        if model_name not in model_preds:
            continue
        model_data = model_preds[model_name]

        # Prefer consolidated predictions; fallback to averaging folds
        pred_arr = None
        if 'consolidated' in model_data:
            cons = model_data['consolidated']
            if all(k in cons for k in ['class0', 'class1', 'class2']):
                pred_arr = np.vstack([cons['class0'], cons['class1'], cons['class2']]).T
        else:
            fold_keys = [k for k in model_data.keys() if isinstance(k, int)]
            if fold_keys:
                # stack and average across folds
                fold_preds = []
                for k in fold_keys:
                    fk = model_data[k]
                    if all(c in fk for c in ['class0', 'class1', 'class2']):
                        fold_preds.append(np.vstack([fk['class0'], fk['class1'], fk['class2']]).T)
                if fold_preds:
                    pred_arr = np.nanmean(np.stack(fold_preds, axis=0), axis=0)

        if pred_arr is None:
            continue

        # Defensive: ensure shape and numeric type
        pred_arr = np.asarray(pred_arr, dtype=float)
        if pred_arr.shape != (n_patients, 3):
            # try to handle transposed shapes
            if pred_arr.T.shape == (n_patients, 3):
                pred_arr = pred_arr.T
            else:
                # skip malformed predictions
                continue

        preds_list.append(pred_arr)
        used_models.append(model_name)

    # If no valid model predictions, return uniform probabilities
    if not preds_list:
        uniform = np.ones(n_patients) / 3.0
        return {'class0': uniform.copy(), 'class1': uniform.copy(), 'class2': uniform.copy()}

    preds = np.stack(preds_list, axis=0)  # shape (n_models, n_patients, 3)
    n_models = preds.shape[0]

    # Performance-based base weights (from cohort baseline). Fallback to 1.0 if unknown.
    default_scores = {'VIRCHOW2': 0.7148, 'UNI': 0.7115, 'CHIEF': 0.6823, 'GIGAPATH': 0.6792}
    base_weights = np.array([default_scores.get(m, 1.0) for m in used_models], dtype=float)
    # shift and normalize to avoid negatives (small eps to be robust)
    base_weights = base_weights - np.min(base_weights) + 1e-6
    base_weights = base_weights / (np.sum(base_weights) + 1e-12)
    # amplify contrast to favor stronger baseline models (conservative exponent)
    base_weights = base_weights ** 1.8
    # small model-specific nudges: favor VIRCHOW2 and UNI slightly (cohort baseline shows these are best)
    for i, m in enumerate(used_models):
        if m == 'VIRCHOW2':
            base_weights[i] *= 1.12
        elif m == 'UNI':
            base_weights[i] *= 1.05
    # renormalize
    base_weights = base_weights / (np.sum(base_weights) + 1e-12)

    # Class-specific multipliers: emphasize class1 (early event) where top models are stronger
    class_multipliers = np.ones((n_models, 3), dtype=float)
    for i, m in enumerate(used_models):
        # slightly stronger boosts for models known to be better at event detection
        if m == 'VIRCHOW2':
            class_multipliers[i, 1] *= 1.12  # stronger boost for early-event detection
        elif m == 'UNI':
            class_multipliers[i, 1] *= 1.06

    eps = 1e-12
    # preds shape: (n_models, n_patients, 3)
    # compute entropy and confidence per model/sample (higher = more confident)
    ent = -np.sum(preds * np.log(np.clip(preds, eps, 1.0)), axis=2)
    max_ent = np.log(3.0)
    conf = 1.0 - (ent / (max_ent + eps))
    conf = np.clip(conf, 0.01, 0.99)

    # Sharpen predictions where the model is confident to increase discrimination (improves AUROC)
    # exponent between ~1.01 and ~1.99 so confident predictions get amplified
    sharpen_power = 1.0 + 1.0 * conf
    sharpen_power = sharpen_power[:, :, np.newaxis]  # (n_models, n_patients, 1)
    preds_clipped = np.clip(preds, 1e-9, 1.0)
    sharpened = preds_clipped ** sharpen_power
    # renormalize per-model per-sample to keep proper probabilities
    sharpened = sharpened / (np.sum(sharpened, axis=2, keepdims=True) + eps)

    # apply class multipliers after sharpening (keeps class emphasis aligned with model confidence)
    adjusted_preds = sharpened * class_multipliers[:, np.newaxis, :]

    # Build adaptive per-sample per-model weights using confidence deviations from sample mean
    conf_mean = np.mean(conf, axis=0, keepdims=True)
    beta = 0.5  # slightly reduced to avoid over-modulating base weights
    bw = base_weights[:, np.newaxis]
    w_sample = bw * (1.0 + beta * (conf - conf_mean))
    # allow smaller floor so low-weight models still contribute some signal
    w_sample = np.maximum(w_sample, 1e-4)

    # Normalize per-sample so model weights sum to 1 for each patient
    w_norm = w_sample / (np.sum(w_sample, axis=0, keepdims=True) + eps)

    # Weighted aggregation using adaptive sample-wise weights on sharpened, adjusted predictions
    weighted_preds = np.sum(w_norm[:, :, np.newaxis] * adjusted_preds, axis=0)  # shape (n_patients, 3)

    # Cancer-specific tweak: prefer convex combination with best model for HNSC if present
    try:
        is_hnsc = (cancer_types == 'HNSC')
        if np.any(is_hnsc) and ('VIRCHOW2' in used_models):
            vi = used_models.index('VIRCHOW2')
            hnsc_boost = 0.12  # modest convex weight toward VIRCHOW2
            # convex combination per-sample so probabilities remain normalized later
            weighted_preds[is_hnsc] = (1.0 - hnsc_boost) * weighted_preds[is_hnsc] + hnsc_boost * preds[vi, is_hnsc]
    except Exception:
        pass

    # IPCW-aware adjustment: mild, monotonic scaling of class1 per sample using IPCW.
    sample_ipcw = np.asarray(ipcw_weights, dtype=float)
    if sample_ipcw.size == n_patients:
        # normalize to mean 1
        sample_ipcw = sample_ipcw / (np.mean(sample_ipcw) + eps)
        # apply small multiplicative adjustment centered at 1.0 (preserves scale on average)
        weighted_preds[:, 1] *= (1.0 + 0.25 * (sample_ipcw - 1.0))
        # avoid negative or zero probabilities before normalization
        weighted_preds = np.maximum(weighted_preds, 1e-8)

    # If models disagree a lot for a sample (high variance), dampen extreme probabilities to reduce overconfidence
    var_across = np.var(adjusted_preds, axis=0)  # shape (n_patients, 3)
    var_mean = np.mean(var_across, axis=1)  # shape (n_patients,)
    if np.any(var_mean > 0):
        # robust relative variance
        rel_var = var_mean / (np.mean(var_mean) + eps)
        gamma = 0.35
        damp = 1.0 + gamma * rel_var  # >1 reduces extremes
        weighted_preds = weighted_preds / damp[:, np.newaxis]

    # Ensemble-weighted risk (focus on class1) to improve ranking (c-index)
    try:
        model_class1 = preds[:, :, 1]  # (n_models, n_patients)
        # w_norm should exist (per-sample normalized model weights). If not, use base_weights as fallback.
        if 'w_norm' not in locals():
            w_norm = np.tile(base_weights[:, np.newaxis], (1, weighted_preds.shape[0]))
        weighted_risk = np.sum(w_norm * model_class1, axis=0)  # (n_patients,)
        wr_min, wr_max = float(np.min(weighted_risk)), float(np.max(weighted_risk))
        if (wr_max - wr_min) > eps:
            weighted_risk_norm = (weighted_risk - wr_min) / (wr_max - wr_min)
        else:
            weighted_risk_norm = np.full_like(weighted_risk, 0.5)
    except Exception:
        weighted_risk_norm = np.full(weighted_preds.shape[0], 0.5, dtype=float)

    # Moderate sharpening of class1 proportional to ensemble-weighted risk to improve ordering (helps c-index).
    # Keep sharpening conservative and bounded to avoid destroying calibration.
    alpha = 0.8
    s = 1.0 + alpha * (weighted_risk_norm - 0.5)
    s = np.clip(s, 0.85, 1.35)  # conservative bounds
    weighted_preds[:, 1] *= s
    # scale other classes down by sqrt(1/s) to keep overall magnitude roughly stable
    other_scale = np.sqrt(1.0 / s)
    weighted_preds[:, 0] *= other_scale
    weighted_preds[:, 2] *= other_scale

    # Per-sample temperature based on inter-model disagreement: low var -> temp < 1 (sharpen), high var -> temp > 1 (smooth)
    temp = np.clip(1.0 + 0.5 * (rel_var - 1.0), 0.8, 1.4)  # shape (n_patients,)

    # Ensure positivity and apply per-sample temperature softmax to obtain calibrated probabilities.
    weighted_preds = np.maximum(weighted_preds, eps)
    logits = np.log(weighted_preds)
    logits = logits / temp[:, np.newaxis]
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + eps)

    result = {
        'class0': probs[:, 0],
        'class1': probs[:, 1],
        'class2': probs[:, 2]
    }

    return result

def get_ensemble_function():
    return ensemble_predictions
