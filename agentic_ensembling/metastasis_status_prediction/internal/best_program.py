import numpy as np
from typing import Dict

def ensemble_predictions(
    model_preds: Dict[str, Dict],
    cancer_types: np.ndarray
) -> np.ndarray:
    
    # Build consolidated predictions per model (fallback to fold averages)
    preds = {}
    for model_name in ['CHIEF','UNI','GIGAPATH','VIRCHOW2']:
        if model_name in model_preds:
            if 'consolidated' in model_preds[model_name]:
                arr = np.asarray(model_preds[model_name]['consolidated'])
            else:
                fold_preds = [v for k, v in model_preds[model_name].items() if isinstance(k, int)]
                arr = np.nanmean(fold_preds, axis=0) if fold_preds else None
            if arr is not None:
                preds[model_name] = arr

    # If no predictions available, return zeros of proper length
    if not preds:
        return np.zeros(len(cancer_types), dtype=float)

    # Keep consistent model order for arrays
    models_present = [m for m in ['CHIEF','UNI','GIGAPATH','VIRCHOW2'] if m in preds]
    preds_arr = np.vstack([preds[m] for m in models_present]).T  # shape (n_patients, n_models)
    n = preds_arr.shape[0]

    # Base global weights (favor GIGAPATH) — tuned to leverage its overall strength
    base_w = {'GIGAPATH':0.42,'UNI':0.26,'CHIEF':0.20,'VIRCHOW2':0.12}

    # Specialist mapping by cancer type (from prior analysis)
    specialist = {
        'HNSC':'GIGAPATH','KIRC':'GIGAPATH','STAD':'GIGAPATH','TGCT':'GIGAPATH','ESCA':'GIGAPATH',
        'KIRP':'UNI','READ':'UNI','THCA':'UNI','UVM':'UNI','BRCA':'UNI',
        'CHOL':'CHIEF','PAAD':'CHIEF','MESO':'CHIEF','BLCA':'CHIEF',
        'ACC':'VIRCHOW2','CESC':'VIRCHOW2','KICH':'VIRCHOW2','LIHC':'VIRCHOW2','COAD':'VIRCHOW2'
    }

    # Robust per-model trust estimated from fold-level variance, plus rank-based aggregation.
    # Aim: reduce calibration bias (via rank normalization) and produce stable softmax weights
    # that combine priors, per-sample confidence and per-model trust.

    # 1) Per-model trust (robust): median std across folds -> smaller std => higher trust.
    # Improve stability by slightly more aggressive mapping from fold-std -> trust,
    # a conservative default when fold info is missing, and clipping to avoid extreme multipliers.
    model_trust = {}
    for m in models_present:
        mp = model_preds.get(m, {})
        fold_vals = [np.asarray(v) for k, v in mp.items() if isinstance(k, int)]
        if fold_vals:
            stack = np.vstack(fold_vals)  # (n_folds, n_patients)
            std_per_patient = np.nanstd(stack, axis=0)
            median_std = float(np.nanmedian(std_per_patient))
            # Map median std into a trust multiplier (gentle but responsive)
            trust = 1.0 / (1.0 + median_std * 2.2)
        else:
            # If no fold info, assume slightly conservative neutral trust (not overly optimistic)
            trust = 0.95
        model_trust[m] = trust
    # normalize trust multipliers to mean 1.0 and clip to a safe range to avoid extreme influence
    trusts = np.array([model_trust[m] for m in models_present], dtype=float)
    if trusts.mean() > 0:
        trusts = trusts / trusts.mean()
    trusts = np.clip(trusts, 0.70, 1.30)
    for idx, m in enumerate(models_present):
        model_trust[m] = trusts[idx]

    # 2) Rank-normalize each model column to reduce calibration differences (preserve ordering)
    norm_preds = np.empty_like(preds_arr, dtype=float)
    for j in range(preds_arr.shape[1]):
        col = preds_arr[:, j]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            norm_preds[:, j] = 0.5
            continue
        order = np.argsort(col[valid], kind='stable')
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(order.size, dtype=float)
        denom = max(1.0, order.size - 1.0)
        col_ranks = np.full_like(col, np.nan, dtype=float)
        col_ranks[valid] = ranks / denom
        col_ranks[~valid] = np.nanmean(col_ranks[valid])
        norm_preds[:, j] = col_ranks

    # 3) Per-sample confidence & agreement signals
    model_std = np.nanstd(preds_arr, axis=1)                      # inter-model std per sample
    # Use an exponential mapping (smoother, less sensitive to outliers) and clip low bound
    conf = np.exp(-model_std * 1.25)
    conf = np.clip(conf, 0.03, 1.0)
    median_pred = np.nanmedian(preds_arr, axis=1)
    # per-model agreement multiplier: closer to median -> larger (>0)
    agreement = 1.0 / (1.0 + np.abs(preds_arr - median_pred[:, None]))

    # 4) Build softmax logits combining priors, specialist boost, confidence, agreement, and trust
    priors = np.array([base_w.get(m, 0.25) for m in models_present], dtype=float)  # (m,)
    boost = np.ones((n, len(models_present)), dtype=float)
    # Temper specialist boosts for volatile cancers, strengthen for reliable specialists.
    high_var = set(['HNSC', 'MESO', 'KICH'])
    for idx, m in enumerate(models_present):
        mask = np.vectorize(lambda x: specialist.get(x) == m)(cancer_types)
        if np.any(mask):
            # stronger boost for specialists, but tempered for high-variance cancer types
            boost[mask, idx] = np.where(np.isin(cancer_types[mask], list(high_var)), 1.20, 1.50)

    log_prior = np.log(priors + 1e-12)[None, :]                      # (1,m)
    conf_comp = (0.9 * conf)[:, None]                               # (n,1)
    agree_comp = 0.55 * agreement                                    # (n,m)
    trust_comp = np.log(np.array([model_trust[m] for m in models_present])[None, :] + 1e-12)

    logit = log_prior + conf_comp + agree_comp + trust_comp + np.log(boost + 1e-12)
    # stable softmax
    mx = np.max(logit, axis=1, keepdims=True)
    exp = np.exp(logit - mx)
    weights = exp / (np.sum(exp, axis=1, keepdims=True) + 1e-12)     # (n,m)

    # 5) Aggregate using rank-normalized predictions (reduces calibration sensitivity)
    ensemble = np.nansum(weights * norm_preds, axis=1)

    # 6) Blend back toward original mean to recover calibration (small anchoring)
    orig_mean = np.nanmean(preds_arr, axis=1)
    ensemble = 0.80 * ensemble + 0.20 * orig_mean

    # Soft specialist override for cancers where a single model historically dominates.
    # Gently pull ensemble toward the specialist's raw prediction for those cancers to leverage strengths.
    strong_specialists = set(['HNSC', 'KIRP', 'KICH', 'UVM', 'PAAD', 'LIHC'])
    model_index = {m: idx for idx, m in enumerate(models_present)}
    for i in range(n):
        ct = cancer_types[i]
        spec_model = specialist.get(ct)
        if spec_model and spec_model in model_index and ct in strong_specialists:
            spred = preds_arr[i, model_index[spec_model]]
            if not np.isnan(spred):
                ensemble[i] = 0.70 * spred + 0.30 * ensemble[i]

    # Final cleanup: replace any remaining NaNs with per-sample mean and clip to [0,1]
    per_sample_mean = np.nanmean(preds_arr, axis=1)
    isnan = np.isnan(ensemble)
    ensemble[isnan] = per_sample_mean[isnan]
    ensemble = np.clip(ensemble, 0.0, 1.0)
    return ensemble

def get_ensemble_function():
    return ensemble_predictions
