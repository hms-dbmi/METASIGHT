"""Initial program (Model_1, binary metastasis) for the METASIGHT ensemble search.

OpenEvolve mutates the function inside the EVOLVE-BLOCK. The starting point is a NaN-aware coverage mean over the
foundation models, the same as the ENSEMBLE_Sim reference ensemble; the search looks for programs that improve on
the three reference ensembles (single best FM, ENSEMBLE_Sim, ENSEMBLE_Val).

CONTRACT (a pure transform; the function never receives labels):

    ensemble_blend(
        fm_probs: np.ndarray,      # shape (K, N): P(metastasis) per FM per slide; NaN where
                                   #   FM k does not cover slide n (partial coverage is normal)
        fm_names: list[str],       # length K, e.g. ["CHIEF","GIGAPATH","KEEP","MUSK"] (2 for slide)
        cancer_types: np.ndarray,  # (N,) str cancer label per slide
        context: dict,             # hints computed on the training cohort (see below); not labels
    ) -> np.ndarray                # (N,) P(metastasis) in [0,1]; NaN allowed only where no FM covers

`context` keys (optional hints for evolved constants):
    model, dataset, level, cutoff, fm_order,
    single_best_fm,            # the FM with the highest AUROC on the training cohort for this task
    fm_auroc,                  # {fm: AUROC on the training cohort}, a reliability prior per FM
    per_cancer_floor,          # {cancer: highest single-FM AUROC, which the blend should not fall below}
    baseline_targets,          # {"single_best","ENSEMBLE_Sim","ENSEMBLE_Val": AUROC} of the reference ensembles
"""

import numpy as np
from typing import Dict, List

# EVOLVE-BLOCK-START
def ensemble_blend(
    fm_probs: np.ndarray,
    fm_names: List[str],
    cancer_types: np.ndarray,
    context: Dict,
) -> np.ndarray:
    """Combine per-FM P(metastasis) into one probability per slide.

    {{TASK_CONTEXT}}

    Default = NaN-aware coverage mean (== ENSEMBLE_Sim). Aim for a higher overall AUROC
    with a lower across-fold AUROC std and Brier, without any cancer falling below
    its single-best AUROC (the floor). Ideas to explore: per-FM reliability
    weights (see context['fm_auroc']), per-cancer dispatch, logit-space
    stacking, product-of-experts, rank averaging, confidence weighting, power-mean.
    """
    arr = np.asarray(fm_probs, dtype=float)        # (K, N)
    if arr.ndim == 1:
        arr = arr[None, :]
    with np.errstate(invalid="ignore"):
        out = np.nanmean(arr, axis=0)              # (N,), NaN where no FM covers
    return np.clip(out, 0.0, 1.0)


def get_ensemble_function():
    """Entry point called by the Stage 1 evaluator. Returns the evolved blend callable."""
    return ensemble_blend
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, N = 4, 50
    fm = rng.uniform(0, 1, (K, N))
    fm[0, ::5] = np.nan
    out = get_ensemble_function()(fm, ["CHIEF", "GIGAPATH", "KEEP", "MUSK"],
                                  np.array(["BRCA"] * N), {})
    assert out.shape == (N,), out.shape
    finite = out[np.isfinite(out)]
    assert np.all((finite >= 0) & (finite <= 1))
    print("SUCCESS", np.round(out[:5], 3))
