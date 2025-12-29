import numpy as np
from typing import Dict

def ensemble_predictions(
    model_preds: Dict[str, Dict],
    cancer_types: np.ndarray,
    ipcw_weights: np.ndarray
) -> Dict[str, np.ndarray]:
    
    eps = 1e-12
    n_patients = len(cancer_types)
    classes = ['class0', 'class1', 'class2']
    model_names = ['CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2']

    # Collect per-model consolidated predictions (fall back to uniform)
    preds = {}
    for m in model_names:
        preds[m] = {}
        if m in model_preds and 'consolidated' in model_preds[m]:
            c = model_preds[m]['consolidated']
            for cls in classes:
                preds[m][cls] = np.asarray(c.get(cls, np.ones(n_patients) / 3.0), dtype=float)
        else:
            # uniform fallback
            for cls in classes:
                preds[m][cls] = np.ones(n_patients) / 3.0

    # Base global weights reflecting relative model quality (sum to 1)
    base_weights = {
        'CHIEF': 0.35,
        'UNI': 0.35,
        'GIGAPATH': 0.18,
        'VIRCHOW2': 0.12
    }

    # Class-specific multipliers (increase influence where model is strong)
    class_mult = {
        'class0': {'CHIEF': 1.0, 'UNI': 1.0, 'GIGAPATH': 1.05, 'VIRCHOW2': 1.05},
        'class1': {'CHIEF': 1.25, 'UNI': 1.25, 'GIGAPATH': 0.85, 'VIRCHOW2': 0.75},
        'class2': {'CHIEF': 0.95, 'UNI': 0.95, 'GIGAPATH': 1.10, 'VIRCHOW2': 1.05}
    }

    # Cancer-specific single-model boosts (observed from prior analysis)
    cancer_boost_map = {
        'COAD': 'CHIEF', 'KIRC': 'UNI',
        'BRCA': 'GIGAPATH', 'LUSC': 'GIGAPATH',
        'LUAD': 'VIRCHOW2', 'OV': 'VIRCHOW2'
    }
    boost_factor = 1.20  # modest boost for mapped best model

    # Normalize ipcw weights to mean 1 and clip to avoid extreme influence
    if ipcw_weights is None or len(ipcw_weights) != n_patients:
        iw = np.ones(n_patients)
    else:
        iw = np.asarray(ipcw_weights, dtype=float)
        mean_iw = float(np.mean(iw)) if iw.size else 1.0
        iw = iw / (mean_iw + eps)
        iw = np.clip(iw, 0.5, 1.5)

    # Build ensemble per-sample using weighted sum with IPCW scaling
    out = {c: np.zeros(n_patients, dtype=float) for c in classes}
    for i in range(n_patients):
        cancer = str(cancer_types[i]) if cancer_types is not None else ''
        # per-class weighted sum and collect model-wise class1 for consensus
        per_class_scores = []
        class1_vals = []
        class1_w = []
        for cls in classes:
            numer = 0.0
            for m in model_names:
                w = base_weights.get(m, 0.0) * class_mult.get(cls, {}).get(m, 1.0)
                # apply cancer boost if applicable
                if cancer in cancer_boost_map and cancer_boost_map[cancer] == m:
                    w *= boost_factor
                pm = preds[m][cls][i]
                numer += w * pm
                # collect class1 stats for consensus-based sharpening
                if cls == 'class1':
                    class1_vals.append(float(pm))
                    class1_w.append(float(w))
            # apply IPCW scaling to numer (more reliable samples get slight boost)
            numer *= iw[i]
            per_class_scores.append(numer + eps)

        # compute consensus on class1 (weighted std / mean -> relative dispersion)
        cw = np.array(class1_w, dtype=float)
        if cw.sum() <= 0:
            cw = np.ones_like(cw)
        cw = cw / (cw.sum() + eps)
        vals = np.array(class1_vals, dtype=float)
        wmean = float((cw * vals).sum())
        wvar = float((cw * (vals - wmean) ** 2).sum())
        wstd = np.sqrt(max(wvar, 0.0))
        std_norm = wstd / (wmean + eps)

        # confidence: low relative std -> high confidence
        conf = 1.0 / (1.0 + std_norm * 4.0)
        # sharpening exponent scales with consensus and IPCW (more reliable -> stronger sharpening)
        exponent = 1.0 + 0.35 * conf * float(iw[i])
        exponent = min(max(exponent, 1.0), 1.6)

        # apply adaptive sharpening
        scores = np.array(per_class_scores)
        scores = scores / (scores.sum() + eps)
        scores = scores ** exponent
        scores = scores / (scores.sum() + eps)

        # risk-based nudge to improve ranking (boost class1 when risk estimate suggests)
        risk = scores[1] + 0.45 * (1.0 - scores[2])
        adj = 1.0 + 0.28 * (risk - 0.5)
        # clamp adjustment to avoid extreme distortions
        if adj < 0.8:
            adj = 0.8
        elif adj > 1.6:
            adj = 1.6
        scores[1] = scores[1] * adj
        scores = scores / (scores.sum() + eps)

        out['class0'][i] = scores[0]
        out['class1'][i] = scores[1]
        out['class2'][i] = scores[2]

    # Final sanity normalization (row-wise)
    total = out['class0'] + out['class1'] + out['class2']
    total = np.maximum(total, eps)
    for cls in classes:
        out[cls] = out[cls] / total

    return out

def get_ensemble_function():
    return ensemble_predictions
