import numpy as np
from typing import Dict

def ensemble_predictions(
    model_preds: Dict[str, Dict],
    cancer_types: np.ndarray,
    ipcw_weights: np.ndarray
) -> Dict[str, np.ndarray]:
    
    n_patients = len(cancer_types)
    model_order = ['CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2']

    # collect model predictions (n_models, n_patients, 3)
    preds_list = []
    available = []
    for m in model_order:
        if m not in model_preds:
            continue
        md = model_preds[m]
        arr = None
        if 'consolidated' in md:
            c = md['consolidated']
            if all(k in c for k in ['class0', 'class1', 'class2']):
                arr = np.vstack([c['class0'], c['class1'], c['class2']]).T
        else:
            fold_keys = [k for k in md.keys() if isinstance(k, int)]
            if fold_keys:
                fold_arrs = []
                for k in fold_keys:
                    fk = md[k]
                    if all(x in fk for x in ['class0', 'class1', 'class2']):
                        fold_arrs.append(np.vstack([fk['class0'], fk['class1'], fk['class2']]).T)
                if fold_arrs:
                    arr = np.nanmean(np.stack(fold_arrs, axis=0), axis=0)
        if arr is not None:
            preds_list.append(arr)
            available.append(m)

    if not preds_list:
        u = np.ones(n_patients) / 3.0
        return {'class0': u.copy(), 'class1': u.copy(), 'class2': u.copy()}

    preds = np.stack(preds_list, axis=0)  # (M, n, 3)
    M = preds.shape[0]

    # stronger base strengths favouring VIRCHOW2 (empirical best)
    base_strength = {'CHIEF': 0.30, 'UNI': 0.12, 'GIGAPATH': 0.08, 'VIRCHOW2': 0.50}
    base_w = np.array([base_strength.get(m, 0.15) for m in available], dtype=float)  # (M,)

    # per-sample confidence via entropy (lower entropy => higher confidence)
    eps = 1e-12
    ent = -np.sum(preds * np.log(preds + eps), axis=2)  # (M, n)
    ent_norm = ent / (np.log(3.0) + eps)
    conf = 1.0 - ent_norm
    conf = np.clip(conf, 0.02, 1.0)  # (M, n)

    # IPCW per-sample scaling, safe and clipped
    ipcw = np.asarray(ipcw_weights, dtype=float)
    if ipcw.shape[0] != n_patients:
        ipcw = np.ones(n_patients, dtype=float)
    ipcw = np.nan_to_num(ipcw, nan=1.0, posinf=1.0, neginf=1.0)
    ipcw_scale = ipcw / (np.mean(ipcw) + eps)
    ipcw_scale = np.clip(ipcw_scale, 0.6, 1.4)  # avoid extremes

    # cancer-specific: small HNSC boost to VIRCHOW2
    cancer_types = np.asarray(cancer_types)
    hns_mask = (cancer_types == 'HNSC')
    boost = np.ones((M, n_patients), dtype=float)
    if hns_mask.any():
        for i, m in enumerate(available):
            if m == 'VIRCHOW2':
                boost[i, hns_mask] *= 1.12

    # raw per-model per-sample weights (before normalization)
    raw = base_w[:, np.newaxis] * conf * (ipcw_scale[np.newaxis, :] * boost)  # (M, n)

    # normalize weights per-sample across models
    sraw = np.sum(raw, axis=0)
    sraw = np.maximum(sraw, 1e-12)
    w_norm = raw / sraw[np.newaxis, :]  # (M, n)

    # per-class reliability (models with low variance for a class are more reliable)
    class_var = np.var(preds, axis=1)  # (M, 3)
    rel = 1.0 / (1.0 + class_var)  # (M,3)
    rel = rel / np.maximum(rel.mean(axis=0, keepdims=True), 1e-12)  # normalize per-class mean=1

    # aggregate per class using weights adjusted by class reliability
    ensemble = np.zeros((n_patients, 3), dtype=float)
    for c in range(3):
        w_c = w_norm * rel[:, c][:, np.newaxis]  # (M,n)
        # normalize across models per sample for this class
        sum_w_c = np.sum(w_c, axis=0)
        sum_w_c = np.maximum(sum_w_c, 1e-12)
        w_c = w_c / sum_w_c[np.newaxis, :]
        ensemble[:, c] = np.sum(preds[:, :, c] * w_c, axis=0)

    # agreement-based and IPCW-aware boost for class1 (early event)
    agreement = 1.0 / (1.0 + np.sum(np.var(preds, axis=0), axis=1))  # (n,)
    boost_factor = 1.0 + 0.15 * (ipcw_scale - 1.0) * agreement
    boost_factor = np.clip(boost_factor, 0.85, 1.25)
    ensemble[:, 1] *= boost_factor

    # improved logit-space sharpening and risk-aware calibration
    # Rationale:
    # - Operate in logit (log-probability) space to preserve ranking information
    #   and avoid overly aggressive elementwise power transforms that can harm AUROC.
    # - Apply a slightly stronger effective sharpening for class1 (early-event risk)
    #   in logit space (more monotonic effect on ordering -> helps c_index).
    # - Modulate sharpening by agreement and IPCW so we only strongly sharpen when
    #   predictions are reliable (consensus across models and reliable IPCW).
    eps = 1e-12
    ensemble = np.clip(ensemble, eps, 1.0 - eps)
    # convert to logits (log-probabilities)
    logits = np.log(ensemble)
    # class-specific temperature-like scaling (divide logits by class_temp)
    # Lower class_temp -> stronger sharpening in probability space for that class.
    class_temp = np.array([1.0, 0.92, 1.0], dtype=float)  # sharpen class1 moderately
    logits = logits / class_temp[np.newaxis, :]
    # risk-driven sharpening factor (per-sample) governed by agreement and IPCW
    # agreement and ipcw_scale are earlier computed; sharper when both indicate reliability
    sharpness = 1.0 + 0.5 * (agreement * (ipcw_scale - 1.0))
    sharpness = np.clip(sharpness, 0.8, 1.6)  # bound to avoid extremes
    # apply sharpening multiplicatively to the class1 logits (targeted ranking boost)
    logits[:, 1] = logits[:, 1] * sharpness
    # stable softmax back to probabilities
    mx = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - mx)
    ensemble = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + eps)
    # small prior smoothing to reduce overconfidence (mix with cohort prior)
    prior = np.mean(ensemble, axis=0, keepdims=True)
    mix = 0.03
    ensemble = (1.0 - mix) * ensemble + mix * prior
    # final renormalization for numerical safety
    ensemble /= np.maximum(ensemble.sum(axis=1, keepdims=True), eps)

    return {
        'class0': ensemble[:, 0],
        'class1': ensemble[:, 1],
        'class2': ensemble[:, 2]
    }

def get_ensemble_function():
    return ensemble_predictions
