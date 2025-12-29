import numpy as np
from typing import Dict

def ensemble_predictions(
    model_preds: Dict[str, Dict],
    cancer_types: np.ndarray
) -> np.ndarray:
    
    # Improved ensemble:
    # - Use consolidated when available, otherwise mean across folds
    # - Per-model base weights favor GIGAPATH, then UNI, VIRCHOW2, CHIEF
    # - Apply cancer-specific multipliers based on known specializations
    # - Per-sample confidence from fold-wise std reduces weight for noisy models
    # - Normalize weights per sample, compute weighted mean, clip to [0,1]
    model_order = ['CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2']
    n = None
    preds = {}
    fold_stds = {}

    for m in model_order:
        if m not in model_preds:
            continue
        # get consolidated or compute mean across fold keys
        if 'consolidated' in model_preds[m]:
            p = np.array(model_preds[m]['consolidated'], dtype=float)
            # try to get per-sample std from fold keys if present
            fold_vals = [v for k, v in model_preds[m].items() if isinstance(k, int)]
            if fold_vals:
                fold_stds[m] = np.nanstd(fold_vals, axis=0)
            else:
                fold_stds[m] = np.zeros_like(p)
        else:
            fold_vals = [v for k, v in model_preds[m].items() if isinstance(k, int)]
            if fold_vals:
                p = np.nanmean(fold_vals, axis=0)
                fold_stds[m] = np.nanstd(fold_vals, axis=0)
            else:
                # if no data, skip model
                continue
        preds[m] = np.clip(p, 0.0, 1.0)
        n = preds[m].shape[0] if n is None else n

    if n is None:
        # no valid predictions -> return zeros
        return np.zeros(0, dtype=float)

    # Updated base weights to more strongly favor GIGAPATH (best overall),
    # but keep UNI and VIRCHOW2 relevant; reduce CHIEF baseline.
    base_w = {'GIGAPATH': 0.55, 'UNI': 0.25, 'VIRCHOW2': 0.15, 'CHIEF': 0.05}

    # cancer specialization sets (boost factors) - keep existing knowledge
    spec = {
        'GIGAPATH': {'HNSC', 'KIRC', 'STAD', 'TGCT', 'ESCA'},
        'UNI':     {'KIRP', 'KIRC', 'READ', 'SKCM', 'THCA'},
        'CHIEF':   {'CHOL', 'PAAD', 'BRCA', 'MESO'},
        'VIRCHOW2':{'ACC', 'CESC', 'KICH', 'LIHC', 'COAD'}
    }
    # stronger boost for specialized cancers, slightly larger penalty otherwise
    boost = 1.25
    non_spec_penalty = 0.90

    # Build per-model per-sample weights
    weights = np.zeros((len(model_order), n), dtype=float)
    model_idx = {m: i for i, m in enumerate(model_order)}

    cancers = np.array(cancer_types, copy=False) if cancer_types is not None else None

    # precompute cancer multipliers per model per patient
    for m in model_order:
        if m not in preds:
            continue
        idx = model_idx[m]
        # start from base weight
        w = np.full(n, base_w.get(m, 0.0), dtype=float)
        # cancer-specific adjustment
        if cancers is not None:
            mask = np.isin(cancers, list(spec.get(m, set())))
            # boost specialized cancers, apply modest penalty elsewhere
            w[mask] *= boost
            w[~mask] *= non_spec_penalty
        # confidence from fold std: higher std -> lower confidence
        std = fold_stds.get(m, np.ones(n) * 0.5)
        # scale std to [0,1], then conf = 1 - scaled_std (clip to avoid zeroing)
        scaled = np.clip(std, 0.0, 1.0)
        conf = np.clip(1.0 - scaled, 0.05, 1.0)
        w *= conf
        weights[idx] = w

    # Zero out rows for missing models
    valid_rows = [model_idx[m] for m in model_order if m in preds]
    # assemble prediction matrix (models x patients)
    pred_mat = np.vstack([preds[m] if m in preds else np.zeros(n) for m in model_order])

    # normalize weights per patient (avoid divide-by-zero)
    wsum = weights.sum(axis=0)
    zero_mask = (wsum == 0)
    if zero_mask.any():
        # distribute evenly across available models for those patients
        k = max(1, len(valid_rows))
        weights[:, zero_mask] = 1.0 / k
        wsum = weights.sum(axis=0)

    norm_w = weights / wsum[np.newaxis, :]

    # weighted ensemble prediction
    ensemble_pred = (norm_w * pred_mat).sum(axis=0)

    # If models disagree strongly on a sample, prefer a specialized/best model for that cancer
    # Compute inter-model std using only present models (avoid zeros for missing models)
    present_mask = np.array([m in preds for m in model_order])
    if present_mask.any():
        pred_sub = pred_mat[present_mask, :]
        inter_model_std = np.nanstd(pred_sub, axis=0)
    else:
        inter_model_std = np.zeros(n)

    # disagreement threshold empirically chosen to identify noisy samples
    disagree_thresh = 0.12

    # For samples with high disagreement, pick the most trusted model for that cancer:
    # preference order: any specialized model (highest base_w), otherwise GIGAPATH, otherwise highest-weight model
    preferred_models = ['GIGAPATH', 'UNI', 'VIRCHOW2', 'CHIEF']
    for i in np.where(inter_model_std > disagree_thresh)[0]:
        c = cancers[i] if cancers is not None else None
        chosen = None
        # find specialized models for this cancer among preds
        if c is not None:
            for m in sorted(base_w.keys(), key=lambda x: -base_w[x]):
                if m in preds and c in spec.get(m, set()):
                    chosen = m
                    break
        # fallback to GIGAPATH if present
        if chosen is None and 'GIGAPATH' in preds:
            chosen = 'GIGAPATH'
        # fallback to the model with highest per-sample normalized weight
        if chosen is None:
            row_idx = np.argmax(norm_w[:, i])
            chosen = model_order[row_idx]
        # blend: rely mostly on chosen model for this noisy sample
        ensemble_pred[i] = 0.80 * preds[chosen][i] + 0.20 * ensemble_pred[i]

    # final regularization: blend ensemble with a small bias toward GIGAPATH (best global)
    if 'GIGAPATH' in preds:
        gig = preds['GIGAPATH']
        ensemble_pred = 0.85 * ensemble_pred + 0.15 * gig

    # ensure valid probabilities
    ensemble_pred = np.clip(ensemble_pred, 0.0, 1.0)
    # ensure no NaN
    ensemble_pred = np.nan_to_num(ensemble_pred, nan=0.5, posinf=1.0, neginf=0.0)

    return ensemble_pred

def get_ensemble_function():
    return ensemble_predictions
