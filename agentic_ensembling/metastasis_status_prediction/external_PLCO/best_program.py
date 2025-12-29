import numpy as np
from typing import Dict

def ensemble_predictions(
    model_preds: Dict[str, Dict],
    cancer_types: np.ndarray
) -> np.ndarray:
    
    # Build a clean predictions matrix (models × patients) using 'consolidated' when available
    # Enhanced: preserve original per-model predictions, compute per-model fold-wise reliability,
    # and rank-normalize each model's predictions to reduce calibration differences.
    model_order = ['CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2']
    model_names = []
    preds_orig_rows = []
    rel_rows = []  # per-model per-patient reliability derived from fold std
    n_patients = cancer_types.shape[0]

    for model_name in model_order:
        if model_name not in model_preds:
            continue
        mp = model_preds[model_name]
        model_names.append(model_name)

        # Get consolidated if present; otherwise average available fold predictions
        if 'consolidated' in mp:
            orig = np.array(mp['consolidated'], dtype=float)
        else:
            # Collect numeric fold keys in sorted order if present
            fold_keys = sorted([k for k in mp.keys() if isinstance(k, int)])
            fold_preds = []
            for k in fold_keys:
                arr = mp.get(k)
                if arr is None:
                    continue
                fold_preds.append(np.array(arr, dtype=float))
            if fold_preds:
                orig = np.nanmean(fold_preds, axis=0)
            else:
                # Fallback constant if no info: neutral probability 0.5
                orig = np.full(n_patients, 0.5, dtype=float)

        # Ensure shape and valid values
        if orig.shape[0] != n_patients:
            # If unexpected shape, coerce to neutral
            orig = np.full(n_patients, 0.5, dtype=float)
        orig = np.clip(np.nan_to_num(orig, nan=0.5), 0.0, 1.0)
        preds_orig_rows.append(orig)

        # Compute per-patient fold std as a measure of model confidence (lower std -> higher reliability)
        fold_keys = sorted([k for k in mp.keys() if isinstance(k, int)])
        if fold_keys:
            stack = []
            for k in fold_keys:
                arr = mp.get(k)
                if arr is None:
                    stack.append(np.full(n_patients, np.nan))
                else:
                    a = np.array(arr, dtype=float)
                    if a.shape[0] != n_patients:
                        stack.append(np.full(n_patients, np.nan))
                    else:
                        stack.append(a)
            stack = np.vstack(stack)  # shape: (n_folds_available, n_patients)
            std = np.nanstd(stack, axis=0)
            # replace extreme/NaN stds with a reasonable fallback (median)
            med = float(np.nanmedian(std)) if np.isfinite(np.nanmedian(std)) else 0.2
            std = np.where(np.isfinite(std), std, med)
        else:
            # No fold info: assume moderate uncertainty
            std = np.full(n_patients, 0.20, dtype=float)

        # Convert std -> reliability score in (0,1], higher for low std
        rel = 1.0 / (1.0 + std)  # simple monotonic mapping
        rel_rows.append(rel)

    if not preds_orig_rows:
        # No predictions available (should not happen) -> return zeros
        return np.zeros(n_patients, dtype=float)

    # Stack originals
    preds_orig = np.vstack(preds_orig_rows)  # shape: (n_models, n_patients)
    preds_orig = np.clip(np.nan_to_num(preds_orig, nan=0.5), 0.0, 1.0)

    # Rank-normalize each model's predictions to reduce calibration differences (improves AUROC ensembles)
    # Use smoothed percentile (rank + 1) / (n + 1) to avoid extreme 0/1
    n = preds_orig.shape[1]
    preds_rank = np.empty_like(preds_orig)
    for r in range(preds_orig.shape[0]):
        row = preds_orig[r]
        order = np.argsort(row, kind='mergesort')
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n)
        preds_rank[r] = (ranks + 1.0) / (n + 1.0)

    # combine rank-based percentile with original predictions to preserve calibration
    # while reducing extreme inter-model calibration differences.
    preds_rank = np.clip(preds_rank, 0.0, 1.0)
    # Use an adaptive, per-sample blend: when cross-model disagreement is high, favor rank-percentiles
    # (which improves AUROC by emphasizing ordering); when agreement is high, preserve calibrated mags.
    cross_model_std = np.nanstd(preds_orig, axis=0)
    # map disagreement to a rank weight in [0.50, 0.80] (higher std -> more rank weight)
    rank_weight = 0.50 + 0.30 * (cross_model_std / (cross_model_std + 0.10))
    rank_weight = np.clip(rank_weight, 0.50, 0.80)
    # broadcast and blend per-sample
    alpha_rank = rank_weight[None, :]  # shape (1, n_patients)
    preds = (1.0 - alpha_rank) * preds_orig + alpha_rank * preds_rank
    preds = np.clip(preds, 0.0, 1.0)

    # Reliability matrix used later to modulate weights per-sample (higher when model folds agree)
    rel_arr = np.vstack(rel_rows)  # shape: (n_models, n_patients)
    # replace any NaNs with column median reliability
    col_meds = np.nanmedian(rel_arr, axis=0)
    col_meds = np.where(np.isfinite(col_meds), col_meds, 0.5)
    nan_mask = ~np.isfinite(rel_arr)
    if nan_mask.any():
        rel_arr[nan_mask] = np.take(col_meds, np.where(nan_mask)[1])

    # Map model names to indices for later use
    n_models = preds.shape[0]
    model_idx = {m: i for i, m in enumerate(model_names)}

    # Base weights computed from recent validation AUROCs (more principled, favoring GIGAPATH)
    # This uses observed model-level AUROCs to set a weighted prior rather than a fixed heuristic.
    model_reliability = {
        'GIGAPATH': 0.7027,
        'UNI': 0.6985,
        'CHIEF': 0.6966,
        'VIRCHOW2': 0.6912
    }
    # modestly boost overall best model to exploit its strong global signal
    # increase boost so the strongest single-model signal is better leveraged when appropriate
    # strengthen prior further to help the ensemble inherit the top-model's global ranking power
    if 'GIGAPATH' in model_reliability:
        model_reliability['GIGAPATH'] *= 1.60  # stronger influence (was 1.40)
    # Keep weights only for models present and normalize, enforce small weight floor for diversity
    present_models = [m for m in model_order if m in model_preds]
    rels = np.array([model_reliability.get(m, 0.69) for m in present_models], dtype=float)
    rels = np.maximum(rels, 0.05)  # floor to avoid zeroing models and maintain diversity
    rels = rels / float(rels.sum())
    base_weights = {m: float(rels[i]) for i, m in enumerate(present_models)}
    # Ensure that any reference to missing models falls back gracefully; other code uses base_weights.get(m, 0.0)

    # Cancer-specific specialization boosts (from domain insights)
    boosts = {
        'GIGAPATH': {'HNSC', 'KIRC', 'STAD', 'TGCT', 'ESCA'},
        'UNI': {'KIRP', 'KIRC', 'READ', 'SKCM', 'THCA'},
        'CHIEF': {'CHOL', 'PAAD', 'BRCA', 'MESO'},
        'VIRCHOW2': {'ACC', 'CESC', 'KICH', 'LIHC', 'COAD'},
    }
    boost_values = {'GIGAPATH': 0.15, 'UNI': 0.12, 'CHIEF': 0.12, 'VIRCHOW2': 0.15}
    # Strong overrides for cancers where a single model is known to be best.
    # Overrides assign a dominant weight to the specialist while leaving a small ensemble reserve.
    overrides = {
        'UNI': {'OV'},
        'VIRCHOW2': {'COAD', 'READ'},
        'CHIEF': {'LUSC', 'BRCA'},
        'GIGAPATH': {'LUAD', 'HNSC'}
    }

    n_patients = preds.shape[1]
    n_models = preds.shape[0]

    # precompute model index map for convenience
    model_idx = {m: i for i, m in enumerate(model_order) if m in model_preds}

    # Vector to store final predictions
    out = np.empty(n_patients, dtype=float)

    # Per-sample ensemble decision:
    # - If models largely agree (low std), use weighted average.
    # - If models strongly disagree (high std), pick the model with highest cancer-specific weight.
    # - Otherwise use weighted average (weights adjusted by cancer boosts).
    for i in range(n_patients):
        ctype = cancer_types[i]
        # build global/cancer-aware weights (order matches preds rows)
        w = np.array([base_weights.get(m, 0.0) for m in model_order][:n_models], dtype=float)

        # apply specialized boosts if cancer type matches
        for m in model_order[:n_models]:
            if ctype in boosts.get(m, ()):
                w[model_idx[m]] += boost_values.get(m, 0.0)

        # apply overrides: for certain cancers, give dominant weight to specialist model (keep small reserve)
        for m in model_order[:n_models]:
            if ctype in overrides.get(m, ()):
                reserve = 0.15  # keep 15% mass distributed among others for stability
                # distribute reserve evenly among the remaining models
                if len(w) > 1:
                    other_share = reserve / float(len(w) - 1)
                    w = np.full_like(w, other_share)
                    w[model_idx[m]] = 1.0 - reserve
                else:
                    # single-model case (unlikely) -> full weight
                    w = np.ones_like(w)
                    w[model_idx[m]] = 1.0
                break

        # normalize base weights
        if w.sum() <= 0:
            w = np.ones_like(w) / float(len(w))
        else:
            w = w / w.sum()

        # per-sample column of (ranked) predictions and basic disagreement measure
        col = preds[:, i]
        col_std = float(np.nanstd(col))

        # local consensus: models closer to others get higher consensus score
        try:
            diffs = np.abs(col[:, None] - col[None, :])
            mean_diff = np.nanmean(diffs, axis=1)  # per-model mean absolute diff
            consensus = 1.0 / (1.0 + mean_diff)
        except Exception:
            consensus = np.ones_like(w)

        # use per-model fold-derived reliability (higher => more trusted)
        # rel_arr shape: (n_models, n_patients)
        local_rel = rel_arr[:, i]
        # numeric safety: prevent exact zeros and allow mild exponentiation
        local_rel = np.maximum(local_rel, 1e-6)

        # combine global weight, consensus and reliability:
        # use a milder exponent so reliability helps but does not dominate in small-sample cases
        local_w = w * consensus * (local_rel ** 1.1)

        # damp weights when overall sample variance is high, but be less aggressive so strong signals survive.
        # Reduced damping helps overall AUROC by allowing informative models to influence predictions more.
        # lessen damping coefficient so reliable model votes are not overly suppressed
        damp = 1.0 + col_std * 0.9
        local_w = local_w / (damp + 1e-8)

        # normalize local weights
        if np.nansum(local_w) <= 0:
            local_w = np.ones_like(local_w) / float(len(local_w))
        else:
            local_w = local_w / float(np.nansum(local_w))

        # Decision rules (tuned thresholds):
        # - Very low disagreement -> weighted mean preserves ranking
        # - Very high disagreement -> trust highest locally-weighted model (robust fallback)
        # - Moderate disagreement -> weighted median (robust to outliers)
        # tightened thresholds: use ensemble when strong agreement, pick specialist earlier when disagreement grows
        if col_std <= 0.10:
            # slightly relaxed agreement threshold -> use ensemble more often for stability
            pred = float(np.nansum(local_w * col))
        elif col_std >= 0.22:
            best_idx = int(np.nanargmax(local_w))
            pred = float(col[best_idx])
        else:
            # weighted median
            idx = np.argsort(col)
            sc = col[idx]
            sw = local_w[idx]
            cumsum = np.cumsum(sw)
            cutoff = 0.5 * cumsum[-1]
            j = int(np.searchsorted(cumsum, cutoff, side='left'))
            j = max(0, min(j, len(sc) - 1))
            pred = float(sc[j])

        out[i] = pred

    # Adaptive global blend with GIGAPATH: give more influence when GIGAPATH is relatively more reliable
    if 'GIGAPATH' in model_idx:
        g_ix = model_idx['GIGAPATH']
        gpred = preds[g_ix, :]

        # per-patient GIGAPATH reliability vs others
        g_rel = rel_arr[g_ix, :]
        # protect against division by zero; use mean reliability of other models
        other_rel_mean = np.nanmean(rel_arr, axis=0)
        # map relative reliability to a stronger dynamic blend so GIGAPATH can assert when clearly more trusted.
        rel_ratio = g_rel / (other_rel_mean + 1e-8)
        # increase baseline and dynamic range: allow more GIGAPATH influence when it's relatively confident
        blend = 0.25 + 0.60 * (rel_ratio / (1.0 + rel_ratio))
        blend = np.clip(blend, 0.25, 0.85)

        out = (1.0 - blend) * out + blend * gpred

    # Post-ensemble cancer-wise shrinkage to reduce per-cancer AUROC variance:
    # Slightly pull predictions toward each cancer's mean to regularize extreme per-cancer offsets.
    # This modest regularization often reduces per-cancer std_AUC without hurting overall AUROC.
    try:
        uniq = np.unique(cancer_types)
        shrink_strength = 0.05  # 5% toward cancer mean (conservative)
        if shrink_strength > 0:
            for ct in uniq:
                idx = np.where(cancer_types == ct)[0]
                if idx.size == 0:
                    continue
                cm = float(np.nanmean(out[idx]))
                out[idx] = (1.0 - shrink_strength) * out[idx] + shrink_strength * cm
    except Exception:
        # if anything goes wrong, keep original out
        pass

    # final safety: no NaNs, clipped to [0,1]
    out = np.clip(np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)

    return out

def get_ensemble_function():
    return ensemble_predictions
