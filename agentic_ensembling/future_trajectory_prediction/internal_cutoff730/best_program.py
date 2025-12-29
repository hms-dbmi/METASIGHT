import numpy as np
from typing import Dict

def ensemble_predictions(
    model_preds: Dict[str, Dict],
    cancer_types: np.ndarray,
    ipcw_weights: np.ndarray
) -> Dict[str, np.ndarray]:
    # Compact, performance-aware ensemble:
    model_names = ['CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2']
    n_patients = len(cancer_types)

    # Collect per-model consolidated predictions (fallback to fold-average)
    preds_list = []
    for m in model_names:
        arr = None
        if m in model_preds:
            md = model_preds[m]
            if 'consolidated' in md:
                c = md['consolidated']
                arr = np.vstack([c['class0'], c['class1'], c['class2']]).T
            else:
                folds = [k for k in md.keys() if isinstance(k, int)]
                if folds:
                    stack = np.stack([md[k]['class0'] for k in folds] + [md[k]['class1'] for k in folds] + [md[k]['class2'] for k in folds], axis=1)
                    # reshape fallback: simple mean per class if oddly structured
                    arr = np.column_stack([np.nanmean([md[k][cls] for k in folds], axis=0) for cls in ['class0','class1','class2']])
        if arr is None:
            arr = np.ones((n_patients, 3)) / 3.0
        preds_list.append(arr)
    stack = np.stack(preds_list, axis=0)  # shape (n_models, n_patients, 3)

    # Base class-specific model strengths (informed by prior ranking)
    base_w = np.array([
        [0.34, 0.20, 0.26],  # CHIEF prefers class0
        [0.15, 0.18, 0.17],  # UNI neutral/low
        [0.25, 0.35, 0.26],  # GIGAPATH stronger on class1
        [0.26, 0.27, 0.31],  # VIRCHOW2 slightly stronger on class2
    ])  # shape (4,3)
    base_w = base_w / (base_w.sum(axis=0, keepdims=True) + 1e-12)

    # Cancer-specific small boosts (from known strengths)
    # Use a milder, prevalence-aware boost to avoid overfitting small cancer groups.
    cancer_map = {
        'CHIEF': {'ESCA', 'HNSC', 'LIHC'},
        'GIGAPATH': {'BLCA', 'LUSC', 'MESO', 'PAAD', 'SKCM', 'THCA'},
        'VIRCHOW2': {'BRCA', 'CESC', 'PRAD'},
        'UNI': {'ACC', 'LUAD', 'STAD', 'TGCT'}
    }
    scale = np.ones((len(model_names), n_patients), dtype=float)
    nominal_boost = 1.08  # milder nominal boost
    for mi, m in enumerate(model_names):
        cancers = cancer_map.get(m, ())
        if cancers:
            mask = np.isin(cancer_types, list(cancers))
            if mask.any():
                # prevalence fraction of this cancer group in the batch
                prevalence = float(mask.mean())
                # attenuate boost if prevalence is small; scale in [1.0, nominal_boost]
                per_sample_boost = 1.0 + (nominal_boost - 1.0) * np.clip(prevalence * 3.0, 0.0, 1.0)
                scale[mi, mask] *= per_sample_boost

    # IPCW-aware per-sample scaling: emphasize samples with higher IPCW (more informative)
    s = ipcw_weights.astype(float)
    if s.size == 0:
        s = np.ones(n_patients)
    s = s / (np.mean(s) + 1e-12)
    s = np.clip(s, 0.6, 1.6)
    scale *= s[np.newaxis, :]

    # Form per-sample per-class model weights and normalize across models
    w = base_w[:, None, :] * scale[:, :, None]  # (models, patients, classes)
    w_sum = w.sum(axis=0, keepdims=True)
    w = w / (w_sum + 1e-12)

    # Weighted aggregation
    ensemble = (stack * w).sum(axis=0)  # (patients, classes)

    # IPCW-aware reliability + entropy-informed risk aggregation to improve ranking
    w_ipcw = ipcw_weights.astype(float)
    if w_ipcw.size == 0:
        w_ipcw = np.ones(n_patients)
    w_ipcw = w_ipcw / (np.mean(w_ipcw) + 1e-12)

    eps = 1e-12
    per_model_var = []
    entropy_conf = []
    for m in range(stack.shape[0]):
        p = np.clip(stack[m], eps, 1.0)
        vals1 = p[:, 1]
        mu = np.average(vals1, weights=w_ipcw)
        var = np.average((vals1 - mu) ** 2, weights=w_ipcw)
        per_model_var.append(var)
        H = - (p * np.log(p)).sum(axis=1)            # per-sample entropy
        entropy_conf.append((1.0 / (1.0 + H)).mean())  # higher => more decisive

    per_model_var = np.array(per_model_var)
    entropy_conf = np.array(entropy_conf)

    # combine low-variance and high-confidence as model-level weight (more robust blend)
    var_rel = 1.0 / (1.0 + per_model_var)
    # moderate entropy contribution to avoid over-emphasizing any single signal
    model_w = var_rel * (1.0 + 0.6 * entropy_conf)
    model_w = model_w / (model_w.sum() + 1e-12)

    # Build a robust per-model risk signal using log-odds (class1 vs rest), normalize per-model
    # Use a more robust scaling (smaller denominator) to magnify useful signal while resisting outliers
    logodds = np.log((stack[:, :, 1] + eps) / (stack[:, :, 0] + stack[:, :, 2] + eps))  # (models, patients)
    risk_norm = np.empty_like(logodds)
    for m in range(logodds.shape[0]):
        r = logodds[m]
        # weighted location/scale (IPCW) for robustness to censoring pattern
        rm = np.average(r, weights=w_ipcw)
        rs = np.sqrt(np.average((r - rm) ** 2, weights=w_ipcw) + eps)
        # robust min-max window around mean (rm ± 3*rs) then scale to [0,1]
        rmin = rm - 3.0 * rs
        rmax = rm + 3.0 * rs
        denom = (rmax - rmin) if (rmax - rmin) > eps else 1.0
        risk_norm[m] = np.clip((r - rmin) / denom, 0.0, 1.0)

    # aggregate model risks using model_w
    risk_model = (model_w[:, np.newaxis] * risk_norm).sum(axis=0)

    # compute a light cancer-specific scaling of the risk signal based on model-average class1 tendency
    # this modestly increases risk separation in cancers where models collectively predict higher class1
    simple_avg = stack.mean(axis=0)  # (patients, classes)
    global_mean_c1 = float(np.mean(simple_avg[:, 1]))
    # compute per-cancer mean of simple_avg class1 and convert to multiplicative factor
    cancer_scale = np.ones(len(cancer_types), dtype=float)
    unique_cancers = np.unique(cancer_types)
    for ct in unique_cancers:
        mask = (cancer_types == ct)
        if not mask.any():
            continue
        cm = float(np.mean(simple_avg[mask, 1]))
        # map deviation from global mean to a small scaling factor in [0.85, 1.25]
        factor = 1.0 + 0.25 * np.clip((cm - global_mean_c1) / (global_mean_c1 + eps), -1.0, 1.0)
        cancer_scale[mask] = np.clip(factor, 0.85, 1.25)
    risk_model = risk_model * cancer_scale

    # ensemble decisiveness -> adapt sharpening/blending per-sample
    ensemble_clipped = np.clip(ensemble, eps, 1.0)
    H_ens = - (ensemble_clipped * np.log(ensemble_clipped)).sum(axis=1)  # per-sample entropy
    conf = 1.0 / (1.0 + H_ens)  # higher => more decisive
    conf_min, conf_max = conf.min(), conf.max()
    conf_scaled = (conf - conf_min) / (conf_max - conf_min + eps)

    # adaptive sigmoid sharpening: less decisive -> milder sharpening, more decisive -> stronger separation
    slope_base = 2.8
    slope = slope_base + 3.0 * conf_scaled
    risk_sharp = 1.0 / (1.0 + np.exp(- (slope * (risk_model - 0.5))))

    # adaptive blending: trust risk_sharp more when ensemble is decisive
    # soften extremes by sqrt mapping of conf_scaled and lower max alpha to reduce overfitting
    conf_adj = np.sqrt(np.clip(conf_scaled, 0.0, 1.0))
    alpha_sample = 0.28 + 0.40 * conf_adj  # in ~[0.28, 0.68]
    new_class1 = alpha_sample * risk_sharp + (1.0 - alpha_sample) * ensemble[:, 1]

    # redistribute remaining mass to class0/class2 proportionally to the original ensemble split
    remaining = np.clip(1.0 - new_class1, 0.0, 1.0)
    denom = ensemble[:, 0] + ensemble[:, 2] + eps
    prop0 = ensemble[:, 0] / denom
    new_class0 = remaining * prop0
    new_class2 = remaining * (1.0 - prop0)

    # mild adaptive shrinkage toward simple average to reduce per-cancer variance (stabilizes AUROC across cancers)
    # Build ensemble matrix
    ensemble = np.column_stack([new_class0, new_class1, new_class2])
    # adapt shrinkage per-sample: apply more shrinkage when ensemble is less decisive (low conf_scaled)
    # This reduces per-cancer variance where predictions are noisy and preserves stronger signals.
    beta_min, beta_max = 0.02, 0.12
    beta = beta_min + (beta_max - beta_min) * (1.0 - conf_scaled)  # shape (n_patients,)
    ensemble = (1.0 - beta[:, np.newaxis]) * ensemble + beta[:, np.newaxis] * simple_avg
    # safety: clip small negatives and renormalize per-sample
    ensemble = np.clip(ensemble, 0.0, 1.0)
    row_sum = ensemble.sum(axis=1, keepdims=True)
    ensemble = ensemble / np.maximum(row_sum, 1e-12)

    # final normalization safeguard
    row_sum = ensemble.sum(axis=1, keepdims=True)
    ensemble = ensemble / np.maximum(row_sum, 1e-12)

    return {'class0': ensemble[:, 0], 'class1': ensemble[:, 1], 'class2': ensemble[:, 2]}

def get_ensemble_function():
    return ensemble_predictions
