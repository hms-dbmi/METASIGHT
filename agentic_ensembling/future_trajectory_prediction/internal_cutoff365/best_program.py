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

    # Smarter weighted ensemble:
    # - use consolidated / fold-averaged preds per model
    # - apply fixed base model weights (performance prior)
    # - apply class-specific multipliers to favor models for rare class (class1)
    # - apply cancer-type adjustments for models known to be stronger on some cancers
    # - incorporate ipcw_weights per-sample to emphasize reliable samples
    model_names = ['CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2']

    # collect model prediction arrays (n_models may be <4 if missing)
    model_preds_arr = {}
    for model_name in model_names:
        if model_name not in model_preds:
            continue
        model_data = model_preds[model_name]
        if 'consolidated' in model_data:
            model_preds_arr[model_name] = {
                'class0': np.asarray(model_data['consolidated']['class0']),
                'class1': np.asarray(model_data['consolidated']['class1']),
                'class2': np.asarray(model_data['consolidated']['class2'])
            }
        else:
            fold_keys = [k for k in model_data.keys() if isinstance(k, int)]
            if fold_keys:
                # average folds
                for cls in ['class0', 'class1', 'class2']:
                    preds = [model_data[k][cls] for k in fold_keys if cls in model_data[k]]
                    model_preds_arr.setdefault(model_name, {})[cls] = np.nanmean(preds, axis=0)

    available_models = list(model_preds_arr.keys())
    if not available_models:
        # fallback to uniform random-safe output
        for cls in ['class0', 'class1', 'class2']:
            result[cls] = np.ones(n_patients) / 3.0
    else:
        # Base weights (prior from validation ranks) - normalize across available models
        # Normalizing avoids unintended global scaling when a model is missing and stabilizes per-sample reweighting.
        _raw_base = {'GIGAPATH': 1.00, 'UNI': 0.95, 'VIRCHOW2': 0.90, 'CHIEF': 0.85}
        total = sum(_raw_base[m] for m in _raw_base if m in available_models)
        if total <= 0:
            total = 1.0
        base_w = {m: (_raw_base.get(m, 0.0) / total) for m in _raw_base}

        # Class-specific multipliers (boost models believed better for rare/early events)
        class_mult = {
            'class0': {'GIGAPATH': 1.00, 'UNI': 1.00, 'VIRCHOW2': 0.98, 'CHIEF': 0.97},
            'class1': {'GIGAPATH': 0.95, 'UNI': 1.08, 'VIRCHOW2': 1.10, 'CHIEF': 0.90},  # emphasize UNI/VIRCHOW2 for early events
            'class2': {'GIGAPATH': 1.05, 'UNI': 1.02, 'VIRCHOW2': 0.98, 'CHIEF': 0.95}
        }

        # Cancer-specific small adjustments (vectorized)
        cancer_boost = {
            # Known strengths from metadata: give +15% to the best model for that cancer
            'LUSC': ('GIGAPATH', 0.15), 'MESO': ('GIGAPATH', 0.15), 'SKCM': ('GIGAPATH', 0.10),
            'STAD': ('GIGAPATH', 0.10), 'TGCT': ('GIGAPATH', 0.10),
            'ACC': ('UNI', 0.12), 'CESC': ('UNI', 0.12), 'LIHC': ('UNI', 0.10),
            'PAAD': ('UNI', 0.08), 'THCA': ('UNI', 0.08),
            'BRCA': ('VIRCHOW2', 0.12), 'HNSC': ('VIRCHOW2', 0.10),
            'BLCA': ('CHIEF', 0.10), 'CHOL': ('CHIEF', 0.08), 'ESCA': ('CHIEF', 0.08), 'LUAD': ('CHIEF', 0.08)
        }

        # Precompute per-model cancer multiplier arrays (shape n_patients)
        cancer_mult = {m: np.ones(n_patients) for m in available_models}
        for ctype, (mname, adj) in cancer_boost.items():
            if mname not in cancer_mult:
                continue
            mask = (cancer_types == ctype)
            if np.any(mask):
                cancer_mult[mname][mask] *= (1.0 + adj)

        # Ensure ipcw_weights is positive, normalize to mean 1, smooth extremes and clip to avoid extreme influence
        ipcw = np.asarray(ipcw_weights, dtype=float)
        ipcw = np.maximum(ipcw, 1e-6)
        mean_ipcw = float(np.mean(ipcw)) if ipcw.size else 1.0
        if mean_ipcw <= 0:
            mean_ipcw = 1.0
        # normalize so IPCW only reweights samples relatively (mean == 1.0)
        ipcw = ipcw / mean_ipcw
        # smooth extremes via sqrt to reduce dominance of outliers while preserving ordering
        ipcw = np.sqrt(ipcw)
        # clip to limit per-sample influence
        ipcw = np.clip(ipcw, 0.70, 1.30)

        # Compute weighted ensemble per class (variance-aware, consensus-boosted)
        # Stack predictions: shape (n_models, n_patients, 3)
        models = available_models
        n_models = len(models)
        preds_stack = np.zeros((n_models, n_patients, 3))
        for i, m in enumerate(models):
            preds_stack[i, :, 0] = model_preds_arr[m]['class0']
            preds_stack[i, :, 1] = model_preds_arr[m]['class1']
            preds_stack[i, :, 2] = model_preds_arr[m]['class2']

        # base weight vector for quick access
        base_vec = np.array([base_w.get(m, 0.9) for m in models], dtype=float)

        # Per-model, per-sample confidence multiplier (entropy-based).
        # Lower entropy -> higher confidence. Map into a conservative multiplier range so no model can dominate.
        eps = 1e-12
        max_ent = np.log(3.0)
        conf_mult = {}
        for i, m in enumerate(models):
            ps = np.clip(preds_stack[i], eps, 1.0)  # (n_patients, 3)
            ent = -np.sum(ps * np.log(ps), axis=1)
            ent_norm = np.clip(ent / (max_ent + 1e-12), 0.0, 1.0)
            # map to multiplier in [0.85, 1.15] (confident -> closer to 1.15)
            conf_mult[m] = 0.85 + 0.30 * (1.0 - ent_norm)

        # For each class compute variance across models (per-sample) and upweight consensus
        for ci, cls in enumerate(['class0', 'class1', 'class2']):
            preds_cls = preds_stack[:, :, ci]  # (n_models, n_patients)
            var_cls = np.var(preds_cls, axis=0)
            vmax = np.max(var_cls)
            if vmax <= 0:
                vmax = 1.0
            # consensus in [0,1] (higher when variance is low => models agree)
            consensus = 1.0 - (var_cls / (vmax + 1e-12))

            weighted_num = np.zeros(n_patients)
            weighted_den = np.zeros(n_patients)
            for i, m in enumerate(models):
                w_m = base_vec[i] * class_mult[cls].get(m, 1.0)
                # amplify weight when models agree on this sample/class
                cons_factor = 0.5 + 0.5 * consensus  # between 0.5 and 1.0
                # include per-sample confidence multiplier so confident model predictions get a modest boost
                w_sample = w_m * cancer_mult[m] * ipcw * cons_factor * conf_mult.get(m, 1.0)
                weighted_num += preds_cls[i] * w_sample
                weighted_den += w_sample
            weighted_den = np.where(weighted_den <= 0, 1.0, weighted_den)
            result[cls] = weighted_num / weighted_den

        # Mild per-class sharpening to adjust sensitivity for the rare early-event class
        # Increase class1 (early-event) probability slightly to improve AUROC for the rare/important class,
        # keep class0/class2 near neutral to avoid destabilizing calibration.
        sharpen = {'class0': 0.99, 'class1': 1.10, 'class2': 1.01}
        for cls in ['class0', 'class1', 'class2']:
            result[cls] = np.clip(result[cls], 1e-8, 1.0 - 1e-8)
            result[cls] = result[cls] ** sharpen[cls]

        # Risk-based mild boosting for class1 to improve ranking (c_index) while preserving calibration.
        # Compute ensemble risk and a smooth sigmoid-based boost; blend conservatively.
        ensemble_risk = result['class1'] + 0.5 * result['class2']
        r_med = np.median(ensemble_risk)
        r_scale = max(np.std(ensemble_risk), 1e-6)
        class1_boost = 1.0 / (1.0 + np.exp(-(ensemble_risk - r_med) / (0.8 * r_scale)))
        # blend original and learned boost (conservative)
        new_c1 = 0.80 * result['class1'] + 0.20 * class1_boost
        # redistribute remaining mass proportionally to class0 and class2 to preserve simplex
        remaining = 1.0 - new_c1
        orig_share = (result['class0'] + result['class2']) + 1e-12
        result['class0'] = remaining * (result['class0'] / orig_share)
        result['class1'] = new_c1
        result['class2'] = remaining * (result['class2'] / orig_share)

        # Per-cancer shrinkage for class1 to reduce per-cancer AUROC variance, then normalize
        unique_cancers, inv = np.unique(cancer_types, return_inverse=True)
        global_mean_c1 = float(np.mean(result['class1']))
        shrink_strength = 0.30  # modest shrinkage toward global mean
        for i in range(len(unique_cancers)):
            idx = np.where(inv == i)[0]
            if idx.size:
                result['class1'][idx] = result['class1'][idx] * (1.0 - shrink_strength) + global_mean_c1 * shrink_strength

        # Final normalization to ensure valid probability simplex per sample
        total = result['class0'] + result['class1'] + result['class2']
        total = np.maximum(total, 1e-12)
        for cls in ['class0', 'class1', 'class2']:
            result[cls] = result[cls] / total

        return result

def get_ensemble_function():
    return ensemble_predictions
