import numpy as np
from typing import Dict

def ensemble_predictions(
    model_preds: Dict[str, Dict],
    cancer_types: np.ndarray,
    ipcw_weights: np.ndarray
) -> Dict[str, np.ndarray]:
    
    n_patients = len(cancer_types)
    classes = ['class0', 'class1', 'class2']
    result = {c: np.zeros(n_patients) for c in classes}

    # Base model importance (based on prior ranking). Will be normalized for available models.
    base_weights_all = {'UNI': 0.33, 'VIRCHOW2': 0.28, 'CHIEF': 0.20, 'GIGAPATH': 0.19}
    # Class multipliers per model (hand-tuned from observed per-class strengths)
    class_mults_all = {
        'UNI':      {'class0': 1.00, 'class1': 1.10, 'class2': 0.95},
        'VIRCHOW2': {'class0': 1.00, 'class1': 0.95, 'class2': 1.05},
        'CHIEF':    {'class0': 1.00, 'class1': 1.00, 'class2': 1.00},
        'GIGAPATH': {'class0': 1.00, 'class1': 0.98, 'class2': 1.02}
    }

    # Collect available models and their consolidated predictions
    model_names = ['CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2']
    available = []
    preds_list = []  # each element shape (n_patients, 3)
    weights_list = []
    for m in model_names:
        if m not in model_preds:
            continue
        md = model_preds[m]
        consolidated = None
        if 'consolidated' in md:
            consolidated = md['consolidated']
        else:
            # fallback: average integer-fold entries if present
            fold_keys = [k for k in md.keys() if isinstance(k, int)]
            if fold_keys:
                # stack per-fold then mean
                fold_preds = []
                for k in fold_keys:
                    fk = md[k]
                    if all(cls in fk for cls in classes):
                        fold_preds.append(np.vstack([fk['class0'], fk['class1'], fk['class2']]).T)
                if fold_preds:
                    consolidated = np.nanmean(np.stack(fold_preds, axis=0), axis=0)
        if consolidated is None:
            continue
        # Ensure arrays are correct shape
        pred_arr = np.vstack([consolidated['class0'], consolidated['class1'], consolidated['class2']]).T
        if pred_arr.shape[0] != n_patients:
            # skip if length mismatch
            continue
        available.append(m)
        preds_list.append(pred_arr)  # (n_patients,3)
        weights_list.append(base_weights_all.get(m, 0.0))

    if not available:
        # fallback to uniform zeros->normalized small uniform
        result = {c: np.ones(n_patients) / 3.0 for c in classes}
        return result

    # Stack into arrays: (n_models, n_patients, 3)
    stack = np.stack(preds_list, axis=0)
    weights_arr = np.array(weights_list, dtype=float)  # (n_models,)

    # Normalize base weights among available models
    weights_arr = weights_arr / (np.sum(weights_arr) + 1e-12)

    # Build class multipliers array per model: shape (n_models, 3)
    mults = np.array([[class_mults_all[m][c] for c in classes] for m in available], dtype=float)

    # Build base per-model-per-sample weight matrix: (n_models, n_patients)
    # Use IPCW weights (higher => more reliable) normalized to mean 1 to avoid scale drift
    ipcw = np.array(ipcw_weights, dtype=float)
    if ipcw.shape[0] != n_patients:
        ipcw = np.ones(n_patients)
    ipcw = ipcw / (np.mean(ipcw) + 1e-12)
    base = weights_arr[:, None] * ipcw[None, :]  # (n_models, n_patients)

    # Cancer-specific boosts (per-sample). Keep modest to avoid overfitting.
    # LUSC: boost GIGAPATH; KIRC: boost UNI
    for model_idx, m in enumerate(available):
        if m == 'GIGAPATH':
            mask = (cancer_types == 'LUSC')
            if np.any(mask):
                base[model_idx, mask] *= 1.35
        if m == 'UNI':
            mask = (cancer_types == 'KIRC')
            if np.any(mask):
                base[model_idx, mask] *= 1.20

    # Enhanced per-sample weighting using model entropy (confidence) + clipping + LUSC smoothing.
    # Rationale:
    # - Use per-sample model entropy to boost models that make confident predictions.
    # - Clip per-model per-sample weights to avoid any single model dominating.
    # - For cancers with unstable baseline performance (e.g., LUSC) shrink toward the simple mean
    #   to reduce per-cancer AUROC variance.
    eps = 1e-12

    # Compute per-model, per-sample entropy as a proxy for confidence (lower entropy => higher confidence)
    # stack shape: (n_models, n_patients, 3)
    safe_stack = np.clip(stack, 1e-12, 1.0 - 1e-12)
    entropy = -np.sum(safe_stack * np.log(safe_stack), axis=2)  # (n_models, n_patients)

    # Convert entropy to a bounded confidence score in [0,1]
    ent_min = np.min(entropy)
    ent_max = np.max(entropy)
    if ent_max - ent_min < 1e-12:
        conf_norm = np.ones_like(entropy)
    else:
        conf_norm = (entropy - ent_min) / (ent_max - ent_min)
        # lower entropy => higher confidence, so invert
        conf_norm = 1.0 - conf_norm
        conf_norm = np.clip(conf_norm, 0.0, 1.0)

    # Modulate base (which already includes IPCW and cancer boosts) by confidence
    # map conf in [0,1] -> multiplier in [0.3, 1.0] (so very low confidence still has some weight)
    conf_mult = 0.3 + 0.7 * conf_norm  # (n_models, n_patients)
    base_conf = base * conf_mult

    # Prevent extreme per-model weights: normalize across models then clip and renormalize
    weight_sums = np.maximum(base_conf.sum(axis=0, keepdims=True), eps)
    weights_norm = base_conf / weight_sums  # (n_models, n_patients)

    # Clip to avoid dominance, then renormalize per sample
    weights_norm = np.clip(weights_norm, 0.01, 0.90)
    weights_norm /= np.maximum(weights_norm.sum(axis=0, keepdims=True), eps)

    # Aggregate using class-specific multipliers (mults) but with the stabilized weights_norm
    numerators = np.zeros((3, n_patients))
    denominators = np.zeros((3, n_patients))
    for c_idx in range(3):
        m_mult = mults[:, c_idx][:, None]  # (n_models,1)
        weighted_preds = weights_norm * m_mult * stack[:, :, c_idx]  # (n_models,n_patients)
        numerators[c_idx] = weighted_preds.sum(axis=0)
        denominators[c_idx] = (weights_norm * m_mult).sum(axis=0)

    # Safe division to get per-class probabilities (not yet normalized across classes)
    for idx, cls in enumerate(classes):
        result[cls] = numerators[idx] / (denominators[idx] + eps)

    # Simple mean across models (used to shrink unstable cancers like LUSC toward ensemble mean)
    simple_mean = np.mean(stack, axis=0)  # (n_patients, 3)
    lus_mask = (cancer_types == 'LUSC')
    if np.any(lus_mask):
        # blend factor: nudges LUSC toward simple mean to reduce per-cancer variance
        blend = 0.30
        for j, cls in enumerate(classes):
            result[cls][lus_mask] = (1.0 - blend) * result[cls][lus_mask] + blend * simple_mean[lus_mask, j]

    # Mild adaptive sharpening of class1 (early-event) to improve ranking / AUROC.
    # Use IPCW-normalized values: ipcw variable is already normalized to mean 1 earlier.
    try:
        ipcw_vec = ipcw if ipcw.shape[0] == n_patients else np.ones(n_patients)
    except Exception:
        ipcw_vec = np.ones(n_patients)
    ipcw_vec = np.clip(ipcw_vec, 0.5, 2.0)
    # exponent in [0.90, 0.98] smaller exponents sharpen probabilities >0.5
    exponent = np.clip(0.95 + 0.02 * (ipcw_vec - 1.0), 0.90, 0.98)
    cls1 = np.clip(result['class1'], 1e-12, 1.0 - 1e-12)
    # apply per-sample exponent (broadcast)
    result['class1'] = np.exp(np.log(cls1) * exponent)

    # Normalize across classes so probabilities sum to 1
    total = result['class0'] + result['class1'] + result['class2']
    total = np.maximum(total, 1e-10)
    for cls in classes:
        result[cls] = result[cls] / total

    # Clamping numerical issues
    for cls in classes:
        result[cls] = np.clip(result[cls], 0.0, 1.0)

    return result

def get_ensemble_function():
    return ensemble_predictions
