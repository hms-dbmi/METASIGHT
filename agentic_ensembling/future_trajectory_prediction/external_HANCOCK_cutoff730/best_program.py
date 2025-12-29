import numpy as np
from typing import Dict

def ensemble_predictions(
    model_preds: Dict[str, Dict],
    cancer_types: np.ndarray,
    ipcw_weights: np.ndarray
) -> Dict[str, np.ndarray]:
    
    # Initialize result arrays
    n_patients = len(cancer_types)
    result = {
        'class0': np.zeros(n_patients),
        'class1': np.zeros(n_patients),
        'class2': np.zeros(n_patients)
    }

    # Improved weighted ensemble:
    # - performance-based base weights (UNI > GIGAPATH > VIRCHOW2 > CHIEF)
    # - small class-specific multipliers (favor models known to be stronger on a class)
    # - per-sample confidence scaling using prediction entropy (low entropy => higher confidence)
    # - IPCW per-sample scaling (use ipcw_weights to emphasize reliable samples)
    # - cancer-specific light boost (HNSC -> boost UNI)
    # - final mild sharpening (calibration) and normalization
    model_names = ['CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2']

    # Base weights derived from baseline ranking (sum to 1.0)
    # Rebalance to give more sensitivity to GIGAPATH for early events while keeping UNI strong for ranking.
    base_weights = {'UNI': 0.40, 'GIGAPATH': 0.33, 'VIRCHOW2': 0.20, 'CHIEF': 0.07}

    # Class-specific multipliers updated:
    # - Stronger boost for GIGAPATH on class1 (early events / rare class).
    # - Slightly favor UNI on class2 to aid ranking / c-index.
    class_multipliers = {
        'class0': {'UNI': 1.03, 'GIGAPATH': 0.94, 'VIRCHOW2': 0.96, 'CHIEF': 0.90},
        'class1': {'UNI': 0.75, 'GIGAPATH': 1.80, 'VIRCHOW2': 1.05, 'CHIEF': 0.95},
        'class2': {'UNI': 1.15, 'GIGAPATH': 0.90, 'VIRCHOW2': 0.98, 'CHIEF': 0.90},
    }

    # IPCW scale normalized to mean=1 and compressed to reduce extreme influence
    if ipcw_weights is None or len(ipcw_weights) == 0:
        ipcw_scale = np.ones(n_patients)
    else:
        iw = ipcw_weights.astype(float)
        # winsorize at 2.5/97.5 percentiles for robustness in small cohorts
        try:
            low_q, high_q = float(np.quantile(iw, 0.025)), float(np.quantile(iw, 0.975))
        except Exception:
            low_q, high_q = float(np.min(iw)), float(np.max(iw))
        iw_clipped = np.clip(iw, low_q, high_q)
        # compress extremes (sqrt) then renormalize to mean=1
        iw_s = np.sqrt(iw_clipped)
        mean_iw = np.mean(iw_s) if np.mean(iw_s) != 0 else 1.0
        ipcw_scale = iw_s / mean_iw

    # Cancer-specific mask (HANCOCK cohort is HNSC; boost UNI slightly for HNSC)
    hnsc_mask = (cancer_types == 'HNSC')

    # Accumulators: weighted numerators and denominators per class per sample
    numer = {c: np.zeros(n_patients) for c in ['class0', 'class1', 'class2']}
    denom = {c: np.zeros(n_patients) for c in ['class0', 'class1', 'class2']}

    # Collect model predictions first to compute consensus/agreement across models.
    # This enables per-sample agreement weighting: models closer to the cross-model median
    # are trusted more (reduces influence of outlier model predictions).
    any_model_used = False
    preds_by_model = {}
    for model_name in model_names:
        if model_name not in model_preds:
            continue
        model_data = model_preds[model_name]

        # obtain consolidated predictions or average folds
        if 'consolidated' in model_data:
            consolidated = model_data['consolidated']
            try:
                p0 = np.asarray(consolidated['class0'], dtype=float)
                p1 = np.asarray(consolidated['class1'], dtype=float)
                p2 = np.asarray(consolidated['class2'], dtype=float)
            except Exception:
                # skip malformed
                continue
        else:
            fold_keys = [k for k in model_data.keys() if isinstance(k, int)]
            if not fold_keys:
                continue
            # average fold predictions
            p_list = []
            for k in fold_keys:
                entry = model_data[k]
                p_list.append(np.vstack([entry['class0'], entry['class1'], entry['class2']]).T)
            stacked = np.nanmean(np.stack(p_list, axis=0), axis=0)
            p0, p1, p2 = stacked[:, 0], stacked[:, 1], stacked[:, 2]

        # ensure shapes match
        if p0.shape[0] != n_patients:
            # skip model if shapes mismatch
            continue

        preds_by_model[model_name] = np.vstack([p0, p1, p2]).T

    if len(preds_by_model) == 0:
        any_model_used = False
    else:
        any_model_used = True
        model_list = list(preds_by_model.keys())
        M = len(model_list)

        # stack shape: (M, n_patients, 3)
        stack = np.stack([preds_by_model[m] for m in model_list], axis=0)

        # median consensus per-sample/class and MAD-like distances
        median_preds = np.median(stack, axis=0)  # (n_patients, 3)
        dists = np.sum(np.abs(stack - median_preds[np.newaxis, :, :]), axis=2)  # (M, n_patients)

        # agreement score: models closer to median have higher agreement
        eps = 1e-12
        scale_dist = np.median(dists) + 1e-6
        agreement_raw = np.exp(-dists / (scale_dist + eps))  # (M, n_patients)

        # normalize agreement to [0.70, 0.95] to reward consensus while avoiding extreme scaling
        a_min = np.min(agreement_raw)
        a_max = np.max(agreement_raw)
        if a_max > a_min:
            agreement = 0.70 + 0.25 * (agreement_raw - a_min) / (a_max - a_min)
            agreement = np.clip(agreement, 0.70, 0.95)
        else:
            agreement = np.ones_like(agreement_raw)

        # Compress IPCW extremes, then renormalize to mean=1 to preserve scale
        if ipcw_weights is None or len(ipcw_weights) == 0:
            ipcw_scale = np.ones(n_patients)
        else:
            ipcw_scale = ipcw_weights.astype(float)
            mean_ipcw = np.mean(ipcw_scale) if np.mean(ipcw_scale) != 0 else 1.0
            ipcw_scale = ipcw_scale / mean_ipcw
            # compress extremes (sqrt) and renormalize so mean remains 1
            ipcw_scale = np.sqrt(ipcw_scale)
            ipcw_scale = ipcw_scale / np.mean(ipcw_scale)

        # iterate models and accumulate using agreement × confidence × ipcw × base_w
        for idx, model_name in enumerate(model_list):
            probs_stack = preds_by_model[model_name]  # (n_patients, 3)
            p0 = probs_stack[:, 0]
            p1 = probs_stack[:, 1]
            p2 = probs_stack[:, 2]

            # Use max class probability as a stable confidence proxy (less noisy than entropy)
            # Map to a conservative range [0.65, 0.98] and cap to avoid overconfidence.
            conf = np.max(probs_stack, axis=1)
            cmin = np.min(conf)
            cmax = np.max(conf)
            if cmax > cmin:
                conf = 0.65 + 0.33 * (conf - cmin) / (cmax - cmin)
                conf = np.minimum(conf, 0.98)
            else:
                conf = np.ones_like(conf)

            # per-sample base weight with cancer-specific boost for HNSC (slightly stronger)
            base_w = base_weights.get(model_name, 1.0 / max(1, len(model_names)))
            per_sample_w = base_w * (1.12 * hnsc_mask + 1.0 * (~hnsc_mask))

            # combine factors: base weight × agreement × confidence × ipcw
            per_sample_scale = per_sample_w * agreement[idx] * conf * ipcw_scale

            # accumulate weighted contributions and weight sums (class-specific multipliers)
            for cls, p_arr in zip(['class0', 'class1', 'class2'], [p0, p1, p2]):
                mult = class_multipliers.get(cls, {}).get(model_name, 1.0)
                w = per_sample_scale * float(mult)
                numer[cls] += w * p_arr
                denom[cls] += w

    if any_model_used:
        # Compute per-class weighted averages where available
        denom_safe0 = np.maximum(denom['class0'], 1e-10)
        denom_safe1 = np.maximum(denom['class1'], 1e-10)
        denom_safe2 = np.maximum(denom['class2'], 1e-10)
        p0_raw = numer['class0'] / denom_safe0
        p1_raw = numer['class1'] / denom_safe1
        p2_raw = numer['class2'] / denom_safe2

        # Clip to valid probability range
        p0_raw = np.clip(p0_raw, 0.0, 1.0)
        p1_raw = np.clip(p1_raw, 0.0, 1.0)
        p2_raw = np.clip(p2_raw, 0.0, 1.0)

        # Build risk signal and blend with p0 for better ranking (improve c-index)
        eps2 = 1e-12
        risk_raw = p1_raw + 0.6 * p2_raw
        rmin = np.min(risk_raw)
        rmax = np.max(risk_raw)
        if rmax > rmin:
            risk_norm = (risk_raw - rmin) / (rmax - rmin)
        else:
            risk_norm = risk_raw * 0.0

        cond_ratio = p1_raw / (p1_raw + p2_raw + eps2)
        cond_ratio = np.clip(cond_ratio, 0.0, 1.0)

        # Blend censor prob with (1 - risk) so higher-risk pts have lower class0 (helps c-index)
        result['class0'] = np.clip(0.56 * p0_raw + 0.44 * (1.0 - risk_norm), 0.0, 1.0)
        remaining = np.clip(1.0 - result['class0'], 0.0, 1.0)
        result['class1'] = remaining * cond_ratio
        result['class2'] = remaining * (1.0 - cond_ratio)

        # safety: if for some samples no model produced predictions (all denominators are zero),
        # fallback to uniform probabilities for those samples
        zero_mask = (np.sum([denom['class0'], denom['class1'], denom['class2']], axis=0) == 0)
        if np.any(zero_mask):
            result['class0'][zero_mask] = 1.0 / 3.0
            result['class1'][zero_mask] = 1.0 / 3.0
            result['class2'][zero_mask] = 1.0 / 3.0
    else:
        # fallback: equal probabilities
        result['class0'] = np.ones(n_patients) * (1.0 / 3.0)
        result['class1'] = np.ones(n_patients) * (1.0 / 3.0)
        result['class2'] = np.ones(n_patients) * (1.0 / 3.0)

    # Mild sharpening / calibration: raise to power slightly >1 to increase discrimination
    # Use slightly milder sharpening to improve calibration and avoid overconfident rare-class boosts
    sharpen = 1.08
    stacked = np.vstack([result['class0'], result['class1'], result['class2']]).T
    stacked = np.maximum(stacked, 1e-12)
    stacked = stacked ** sharpen
    row_sums = np.sum(stacked, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    stacked = stacked / row_sums

    result['class0'] = stacked[:, 0]
    result['class1'] = stacked[:, 1]
    result['class2'] = stacked[:, 2]

    return result

def get_ensemble_function():
    return ensemble_predictions
