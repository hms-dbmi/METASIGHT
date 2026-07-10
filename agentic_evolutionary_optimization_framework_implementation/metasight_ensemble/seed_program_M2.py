"""Initial program (Model_2, 3-class future trajectory) for the METASIGHT ensemble search.

OpenEvolve mutates the function inside the EVOLVE-BLOCK. The default is a NaN-aware
coverage mean over foundation models, row-normalised — i.e. identical to ENSEMBLE_Sim.

CONTRACT (label-free pure transform — never receives eval labels, so no leakage):

    ensemble_blend(
        fm_probs: np.ndarray,      # shape (K, N, 3): per-FM class probabilities; an FM that
                                   #   does not cover slide n has NaN across all 3 classes
        fm_names: list[str],       # length K (4 for patch)
        cancer_types: np.ndarray,  # (N,) str
        context: dict,             # informational hints (see seed_program_M1 docstring); NOT labels
    ) -> np.ndarray                # (N, 3) row-normalised; NaN row allowed only where no FM covers

Metric = macro one-vs-rest AUROC (mean over the 3 classes), across-fold std, macro Brier.
Classes: 0 = stable disease, 1 = locoregional recurrence, 2 = distant metastasis.
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
    """Combine per-FM 3-class probabilities into one distribution per slide.

    {{TASK_CONTEXT}}

    Default = NaN-aware coverage mean then row-normalise (== ENSEMBLE_Sim). Beat it
    on macro-OvR AUROC while lowering across-fold AUROC std and macro Brier, and
    never dropping any cancer below its single-best macro-AUROC (hard floor). Ideas:
    per-FM reliability weights (context['fm_auroc']), per-class logit stacking,
    product-of-experts (log-space mean + softmax), rank averaging, power-mean.
    """
    arr = np.asarray(fm_probs, dtype=float)            # (K, N, 3)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    with np.errstate(invalid="ignore"):
        avg = np.nanmean(arr, axis=0)                  # (N, 3); NaN row where no FM covers
    s = avg.sum(axis=1, keepdims=True)
    mask = np.isfinite(s).flatten() & (s.flatten() > 1e-12)
    out = avg.copy()
    out[mask] = avg[mask] / s[mask]
    return out


def get_ensemble_function():
    """Entry point called by the evaluator. Returns the evolved blend callable."""
    return ensemble_blend
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, N = 4, 50
    fm = rng.uniform(0, 1, (K, N, 3))
    fm /= fm.sum(axis=2, keepdims=True)
    fm[0, ::5, :] = np.nan
    out = get_ensemble_function()(fm, ["CHIEF", "GIGAPATH", "KEEP", "MUSK"],
                                  np.array(["BRCA"] * N), {})
    assert out.shape == (N, 3), out.shape
    rs = out.sum(axis=1)
    finite = np.isfinite(rs)
    assert np.allclose(rs[finite], 1.0, atol=1e-6)
    print("SUCCESS", np.round(out[0], 3))
