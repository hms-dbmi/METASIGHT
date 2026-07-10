# Input format — per-model prediction CSVs

The ensemble search operates on the **per-slide probability outputs** of the base
foundation-model (FM) predictors — not on slides or features. Each `(task, cohort, FM)`
pair is one CSV. No data is shipped with this repo — **you provide these files**: put your FM prediction
CSVs under `data/real/` (git-ignored) following the schema below.

## File naming

```
M1_{COHORT}_{FM}_patch.csv                 # metastasis status (binary)
M2_{COHORT}_{FM}_patch_cut{365|730|1095}.csv   # trajectory (3-class), per horizon
```

Patch-level FMs: `CHIEF, GIGAPATH, KEEP, MUSK`. Cohorts (named in `metasight_ensemble/task_registry.py`): `TCGA` (internal, 5-fold)
and `DFCI` (external, single inference set).

## Columns

**Model_1 (binary metastasis)**

| column | meaning |
|---|---|
| `patient_id` | patient identifier |
| `slide_id` | slide identifier (the alignment key across FMs) |
| `fold_id` | 0..4 for internal CV; `-1` for external single set |
| `true_label` | 0 = M0, 1 = M1 |
| `pred_prob_metastasis` | P(M1) for this slide from this FM |
| `pred_prob_no_metastasis` | P(M0) = 1 − P(M1) |
| `cancer_type` | e.g. BRCA, LUAD, COAD |

**Model_2 (3-class trajectory)** — replaces the two prob columns with:

| column | meaning |
|---|---|
| `true_label` | 0 = stable, 1 = locoregional recurrence, 2 = distant metastasis |
| `pred_prob_class0/1/2` | per-class probabilities (row sums ≈ 1) |

## Conventions

- **Slide-level** rows; AUROC is computed directly on slide rows.
- **Partial coverage is normal**: an FM may be missing some slides — those rows are simply
  absent from that FM's CSV and become `NaN` after alignment. Blends must be NaN-aware.
- Ground truth (`true_label`, `cancer_type`, `fold_id`) is shared across FMs for the same
  `slide_id`; the loader takes it from whichever FM covers the slide.
