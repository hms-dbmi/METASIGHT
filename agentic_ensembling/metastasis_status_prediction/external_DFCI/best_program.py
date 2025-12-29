import numpy as np
from typing import Dict

def ensemble_predictions(
    model_preds: Dict[str, Dict],
    cancer_types: np.ndarray
) -> np.ndarray:
    
    models = ['CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2']
    n_patients = cancer_types.shape[0]

    # 1) Obtain consolidated predictions (or compute from folds) and fold std (confidence)
    preds = {}
    fold_stds = {}
    for m in models:
        if m not in model_preds:
            continue
        # consolidated preferred
        if 'consolidated' in model_preds[m]:
            p = np.array(model_preds[m]['consolidated'], dtype=float)
        else:
            fold_list = [v for k, v in model_preds[m].items() if isinstance(k, int)]
            if fold_list:
                p = np.nanmean(np.stack(fold_list, axis=0), axis=0)
            else:
                # missing model -> skip
                continue
        preds[m] = np.nan_to_num(p, nan=np.nanmean(p) if np.any(~np.isnan(p)) else 0.5)
        # compute fold-wise std as confidence signal if available
        fold_list = [v for k, v in model_preds[m].items() if isinstance(k, int)]
        if fold_list:
            stacked = np.stack(fold_list, axis=0)
            fold_stds[m] = np.nanstd(stacked, axis=0)
        else:
            # fallback: small constant uncertainty
            fold_stds[m] = np.full(n_patients, 0.15, dtype=float)

    # If no models available, return 0.5s
    if not preds:
        return np.full(n_patients, 0.5, dtype=float)

    # 2) Base global weights (favor GIGAPATH & VIRCHOW2 more; smoothed to avoid overconfidence)
    # Use a stronger prior for the empirically best models but keep some mass for UNI/CHIEF.
    base_weights = {'GIGAPATH': 0.42, 'VIRCHOW2': 0.30, 'UNI': 0.16, 'CHIEF': 0.12}
    # Ensure only present models contribute; normalize base weights (safe if sum==0)
    present = [m for m in models if m in preds]
    bw = np.array([base_weights.get(m, 0.0) for m in present], dtype=float)
    s = bw.sum()
    if s <= 0:
        # fallback to uniform weights if something odd happens
        bw = np.ones_like(bw, dtype=float) / max(1, bw.size)
    else:
        # renormalize keeping proportions but allow later cancer boosts to be meaningful
        bw = bw / s

    # 3) Cancer-specific boosts (small adaptive additive boost to specialized models)
    # Map cancers -> model to boost (aligned with observed specializations)
    # - GIGAPATH: HNSC, STAD, TGCT, ESCA
    # - UNI: KIRC, KIRP, BRCA, SKCM, THCA
    # - CHIEF: CHOL, PAAD, MESO, LUSC
    # - VIRCHOW2: LUAD, READ, OV, COAD, ACC, CESC, KICH, LIHC
    cancer_boost_map = {
        'HNSC': 'GIGAPATH', 'STAD': 'GIGAPATH', 'TGCT': 'GIGAPATH', 'ESCA': 'GIGAPATH',
        'KIRC': 'UNI', 'KIRP': 'UNI', 'BRCA': 'UNI', 'SKCM': 'UNI', 'THCA': 'UNI',
        'CHOL': 'CHIEF', 'PAAD': 'CHIEF', 'MESO': 'CHIEF', 'LUSC': 'CHIEF',
        'LUAD': 'VIRCHOW2', 'READ': 'VIRCHOW2', 'OV': 'VIRCHOW2', 'COAD': 'VIRCHOW2',
        'ACC': 'VIRCHOW2', 'CESC': 'VIRCHOW2', 'KICH': 'VIRCHOW2', 'LIHC': 'VIRCHOW2'
    }
    # Adaptive boost: scale boost by rarity of the cancer (smaller cohorts -> larger boost)
    unique_cancers = np.unique(cancer_types)
    # compute counts per cancer once (robust to string types)
    counts = {c: int(np.sum(cancer_types == c)) for c in unique_cancers}
    cancer_weights = {}
    for c in unique_cancers:
        w = bw.copy()
        # which model to boost?
        boost_model = cancer_boost_map.get(c, None)
        if boost_model and boost_model in present:
            idx = present.index(boost_model)
            # proportion of dataset for this cancer (smaller -> larger boost)
            prop = float(counts[c]) / max(1, n_patients)
            # boost range tuned empirically: between 0.06 (large cohorts) and 0.20 (small cohorts)
            boost = 0.06 + 0.14 * (1.0 - prop)
            # safety clamp
            boost = float(np.clip(boost, 0.06, 0.20))
            w[idx] += boost
        # renormalize
        w = w / w.sum()
        cancer_weights[c] = w

    # 4) Confidence weighting per-sample: stabilized mapping of std -> confidence
    # Use an exponential decay mapping (exp(-std/tau)) which better separates low vs moderate uncertainty
    std_mat = np.vstack([fold_stds[m] for m in present])  # shape (M, N)
    # floor std to avoid extreme weights from near-zero variance and noisy tiny stds
    min_std = 0.02
    std_mat = np.maximum(std_mat, min_std)
    # tau controls sensitivity; tuned to be moderately discriminative
    tau = 0.08
    conf_mat = np.exp(-std_mat / tau)  # higher when std small, bounded (0,1]
    # rescale each model's confidence by its median to avoid a single model dominating globally
    model_median = np.median(conf_mat, axis=1, keepdims=True)
    model_median = np.maximum(model_median, 1e-3)
    conf_mat = conf_mat / model_median
    # normalize confidence across models per sample
    conf_norm = conf_mat / (np.sum(conf_mat, axis=0, keepdims=True) + 1e-12)

    # 5) Combine cancer_weights and conf_norm to get per-sample final weights
    # vectorize cancer weight expansion using unique cancers (faster and clearer)
    unique_cancers_list, inv_idx = np.unique(cancer_types, return_inverse=True)
    cw_arr = np.vstack([cancer_weights[c] for c in unique_cancers_list])  # shape (C_unique, M)
    cw_mat = cw_arr[inv_idx].T  # shape (M, N)
    # final raw weights per model per sample (elementwise)
    raw_w = cw_mat * conf_norm
    # normalize per sample
    samp_w = raw_w / (np.sum(raw_w, axis=0, keepdims=True) + 1e-12)

    # 6) Weighted prediction
    pred_mat = np.vstack([preds[m] for m in present])  # shape (M, N)
    # Per-model median-centering to reduce inter-model calibration bias while preserving ranking.
    model_medians = np.median(pred_mat, axis=1, keepdims=True)
    global_median = float(np.median(pred_mat))
    pred_mat = pred_mat - model_medians + global_median
    pred_mat = np.clip(pred_mat, 0.0, 1.0)
    weighted_pred = np.sum(samp_w * pred_mat, axis=0)

    # 7) Shrink toward robust central estimator (median) to improve robustness
    # Use a per-sample adaptive shrinkage: when the per-sample model-confidence is low,
    # move closer to the median; when one model strongly dominates confidence, trust the weighted_pred.
    simple_median = np.median(pred_mat, axis=0)
    # sample_confidence: how dominant is the top model's confidence for each sample
    sample_confidence = np.max(conf_norm, axis=0)
    # map sample_confidence -> alpha in [0.55, 0.95] (favor weighted_pred slightly more overall)
    alpha = 0.55 + 0.40 * sample_confidence
    alpha = np.clip(alpha, 0.55, 0.95)
    # ensemble_pred is now a vector blended per-sample
    ensemble_pred = alpha * weighted_pred + (1.0 - alpha) * simple_median

    # 8) Safety: clip to [0,1], replace NaN
    ensemble_pred = np.nan_to_num(ensemble_pred, nan=0.5)
    ensemble_pred = np.clip(ensemble_pred, 0.0, 1.0)

    return ensemble_pred

def get_ensemble_function():
    return ensemble_predictions
