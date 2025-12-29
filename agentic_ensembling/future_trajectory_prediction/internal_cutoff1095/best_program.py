import numpy as np
from typing import Dict

def ensemble_predictions(
    model_preds: Dict[str, Dict],
    cancer_types: np.ndarray,
    ipcw_weights: np.ndarray
) -> Dict[str, np.ndarray]:
    
    # Improved ensemble: performance + cancer-specific + IPCW-aware weighting
    n_patients = len(cancer_types)
    result = {
        'class0': np.zeros(n_patients),
        'class1': np.zeros(n_patients),
        'class2': np.zeros(n_patients)
    }

    model_names = ['CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2']

    # Base model weights (from internal AUROC ranking / small tuning to favor best model)
    # Slightly increase UNI weight (best single-model performer) and slightly reduce weakest
    base_w = {'UNI': 0.36, 'VIRCHOW2': 0.30, 'CHIEF': 0.19, 'GIGAPATH': 0.15}

    # Cancer groups where specific models historically perform better
    boost_map = {
        'VIRCHOW2': {'BRCA', 'LUAD', 'ESCA', 'SKCM'},
        'CHIEF': {'BLCA', 'HNSC', 'LUSC', 'STAD', 'THCA'},
        'GIGAPATH': {'MESO', 'PAAD'},
        'UNI': {'CESC', 'LIHC', 'PRAD', 'TGCT'}
    }
    # Preload consolidated preds (fallback to None for missing)
    preds = {}
    for m in model_names:
        if m in model_preds and 'consolidated' in model_preds[m]:
            preds[m] = model_preds[m]['consolidated']
        else:
            # try numeric folds as fallback (simple mean)
            model_data = model_preds.get(m, {})
            fold_keys = [k for k in model_data.keys() if isinstance(k, int)]
            if fold_keys:
                # compute mean across folds
                arr0 = np.nanmean([model_data[k]['class0'] for k in fold_keys if 'class0' in model_data[k]], axis=0)
                arr1 = np.nanmean([model_data[k]['class1'] for k in fold_keys if 'class1' in model_data[k]], axis=0)
                arr2 = np.nanmean([model_data[k]['class2'] for k in fold_keys if 'class2' in model_data[k]], axis=0)
                preds[m] = {'class0': arr0, 'class1': arr1, 'class2': arr2}
            else:
                preds[m] = None

    # Normalize IPCW to a stable scaling around 1.0 to avoid extreme multipliers
    ipcw_mean = float(np.mean(ipcw_weights)) if len(ipcw_weights) > 0 else 1.0
    ipcw_scale = np.clip(ipcw_weights / (ipcw_mean + 1e-12), 0.5, 1.5)  # keep within reasonable range

    # Precompute global calibration factors (reduce per-sample overhead)
    model_means = {}
    for m in model_names:
        if preds[m] is None:
            model_means[m] = {'class0': 1.0, 'class1': 1.0, 'class2': 1.0}
        else:
            model_means[m] = {
                'class0': float(np.mean(preds[m]['class0'])),
                'class1': float(np.mean(preds[m]['class1'])),
                'class2': float(np.mean(preds[m]['class2']))
            }
    global_mean = {
        'class0': np.mean([model_means[m]['class0'] for m in model_names if preds[m] is not None]) if any(preds[m] is not None for m in model_names) else 1.0,
        'class1': np.mean([model_means[m]['class1'] for m in model_names if preds[m] is not None]) if any(preds[m] is not None for m in model_names) else 1.0,
        'class2': np.mean([model_means[m]['class2'] for m in model_names if preds[m] is not None]) if any(preds[m] is not None for m in model_names) else 1.0
    }
    calib = {}
    for m in model_names:
        if preds[m] is None:
            calib[m] = {'class0': 1.0, 'class1': 1.0, 'class2': 1.0}
        else:
            calib[m] = {
                'class0': float(np.clip(global_mean['class0'] / (model_means[m]['class0'] + 1e-12), 0.87, 1.13)),
                'class1': float(np.clip(global_mean['class1'] / (model_means[m]['class1'] + 1e-12), 0.87, 1.13)),
                'class2': float(np.clip(global_mean['class2'] / (model_means[m]['class2'] + 1e-12), 0.87, 1.13)),
            }

    # Precompute per-model entropy-based confidences (more robust than max-prob)
    confidences = {}
    for m in model_names:
        if preds.get(m) is None:
            confidences[m] = None
        else:
            arr = np.stack([preds[m]['class0'], preds[m]['class1'], preds[m]['class2']], axis=1)
            arr_clipped = np.clip(arr, 1e-12, 1.0)
            entropy = -np.sum(arr_clipped * np.log(arr_clipped), axis=1)
            confidences[m] = 1.0 - entropy / (np.log(3.0) + 1e-12)  # 0..1

    # Two-stage aggregation:
    # Stage 1: compute conservative calibrated probabilities per-sample (store p0s/p1s/p2s)
    # Stage 2: inject a global rank signal (from risk=1-p0) into class1 using IPCW-aware blending,
    #          then apply a controlled logit + temperature softmax per-sample to finalize probabilities.
    eps = 1e-9
    p0s = np.zeros(n_patients, dtype=float)
    p1s = np.zeros(n_patients, dtype=float)
    p2s = np.zeros(n_patients, dtype=float)

    for i in range(n_patients):
        ctype = cancer_types[i]

        # Start from base weights and apply cancer-specific small boost
        w = np.array([base_w[m] for m in model_names], dtype=float)
        for j, m in enumerate(model_names):
            if preds[m] is None:
                w[j] = 0.0
                continue
            if ctype in boost_map.get(m, set()):
                w[j] += 0.06

            # Use entropy-based confidence to scale weight conservatively
            conf = float(confidences[m][i]) if confidences.get(m) is not None else 0.0
            # scale in [0.6, 1.4] to give more influence to confident models while keeping bounds
            w[j] *= (0.6 + 0.8 * conf)

        # Fallback to available models
        if w.sum() <= 0:
            avail = np.array([1.0 if preds[m] is not None else 0.0 for m in model_names], dtype=float)
            if avail.sum() == 0:
                w = np.ones(len(model_names)) / len(model_names)
            else:
                w = avail / avail.sum()
        else:
            w /= w.sum()

        # Aggregate calibrated predictions (stage 1)
        p0 = 0.0; p1 = 0.0; p2 = 0.0
        for j, m in enumerate(model_names):
            if preds[m] is None:
                continue
            p0 += w[j] * float(preds[m]['class0'][i]) * calib[m]['class0']
            p1 += w[j] * float(preds[m]['class1'][i]) * calib[m]['class1']
            p2 += w[j] * float(preds[m]['class2'][i]) * calib[m]['class2']

        # Normalize the per-sample triplet to preserve calibration
        s_local = p0 + p1 + p2 + eps
        p0s[i] = np.clip(p0 / s_local, eps, 1.0)
        p1s[i] = np.clip(p1 / s_local, eps, 1.0)
        p2s[i] = np.clip(p2 / s_local, eps, 1.0)

    # Stage 2: ranking injection to improve concordance (c_index)
    # risk = 1 - p0 (higher -> earlier event)
    risk = 1.0 - p0s
    order = np.argsort(risk)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    denom = float(max(1, n_patients - 1))
    norm_rank = ranks / (denom + 1e-12)  # 0..1

    # IPCW-aware boost to p1 then blend with rank signal (moderate blending keeps calibration)
    p1_boosted = p1s * (1.0 + 0.22 * (ipcw_scale - 1.0))
    blend_alpha = 0.34  # tuned moderate blend
    p1_final = (1.0 - blend_alpha) * p1_boosted + blend_alpha * norm_rank

    # mild sharpening to improve discrimination, bounded and numerically stable
    p1_final = np.clip(p1_final, eps, 1.0) ** 1.06

    # Final per-sample logit-space temperature softmax (keeps dynamic sharpening by IPCW)
    base_T = 0.80
    for i in range(n_patients):
        log0 = np.log(p0s[i] + eps)
        log1 = np.log(p1_final[i] + eps) + 0.30 * (ipcw_scale[i] - 1.0)
        log2 = np.log(p2s[i] + eps)

        # Dynamic temperature: sharpen more for reliable samples but keep bounds to avoid overconfidence
        T = float(np.clip(base_T - 0.08 * (ipcw_scale[i] - 1.0), 0.72, 0.88))
        mx = float(max(log0, log1, log2))
        e0 = np.exp((log0 - mx) / T)
        e1 = np.exp((log1 - mx) / T)
        e2 = np.exp((log2 - mx) / T)
        s = e0 + e1 + e2 + 1e-12

        result['class0'][i] = e0 / s
        result['class1'][i] = e1 / s
        result['class2'][i] = e2 / s

    return result

def get_ensemble_function():
    return ensemble_predictions
