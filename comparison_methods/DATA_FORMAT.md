# Input data format

The package reads one folder, set with `METASIGHTCMP_DATA` or `--data` (default: `data/` inside the package folder).
It never writes there. The layout is the one the METASIGHT two-stage search reads, with one optional column more
(`patient_id` in the files of an external cohort), so both packages can read the same folder.

```
data/
  tasks.csv
  <task_id>/
    cohort.csv      TCGA tasks only
    <FM>.csv        one file per foundation model of the task
```

Foundation models (FMs): `CHIEF`, `GIGAPATH`, `KEEP` and `MUSK` for patch-level tasks; `CHIEF` and `GIGAPATH` for
slide-level tasks. A missing FM file means that FM covers no row of the task.

`python examples/make_synthetic_data.py data/` writes a small synthetic folder in this format.

## tasks.csv

| Column | Values |
|---|---|
| `task_id` | `M1_<dataset>_<level>` or `M2_<dataset>_<level>_cutoff<cutoff>` (checked against the other columns) |
| `dataset` | `TCGA` for the training cohort; any other name is an external cohort |
| `model` | `Model_1`: metastatic status at diagnosis (0 M0, 1 M1). `Model_2`: clinical trajectory at `cutoff` days (0 stable, 1 locoregional recurrence, 2 distant metastasis) |
| `level` | `patch` or `slide` (which FMs the task uses) |
| `cutoff` | empty for `Model_1`; 365, 730 or 1095 for `Model_2` |

Every external task needs a TCGA task with the same model, level and cutoff: the methods are fitted on that TCGA
cohort and applied unchanged to the external cohort.

## TCGA task

`cohort.csv`: one row per slide of the cohort.

| Column | Meaning |
|---|---|
| `row_id` | optional; `patient_id|slide_id` when absent |
| `patient_id`, `slide_id` | identifiers |
| `fold_id` | cross-validation fold, 0 to 4; all slides of a patient in one fold (checked) |
| `true_label` | outcome (see `model`) |
| `cancer_type` | cancer type code |

`<FM>.csv`: the FM's cross-validated predictions for the TCGA slides (each slide predicted by the FM model trained
without that slide's fold), with `patient_id`, `slide_id` and the probability columns: `pred_prob_metastasis` for
`Model_1`; `pred_prob_class0`, `pred_prob_class1`, `pred_prob_class2` for `Model_2`. Other columns are ignored.

- Rows are matched to `cohort.csv` by `patient_id|slide_id`; a slide listed more than once is averaged.
- If fewer than half of an FM's rows match a cohort slide, the FM's predictions are averaged per patient and given to
  every slide of that patient (for an FM whose slide ids follow another scheme).
- A cohort slide the FM file does not list is a slide that FM does not cover.

## External task

`<FM>.csv`: one or more rows per sample, with

| Column | Meaning |
|---|---|
| `slide_id` (`Model_1`) or `case_id` (`Model_2`) | sample identifier |
| `true_label`, `cancer_type` | as above |
| probability columns | as above |
| `patient_id` | optional; the patient of the sample, for a cohort with several slides per patient |
| `fold_id` | optional; when present, the pooled task AUROC used inside the searches is averaged over these folds |

Rows with the same identifier are averaged. The cohort is the union of the identifiers across the task's FM files;
each sample's label, cancer type and patient come from the first FM file that lists it.

Without a `patient_id` column, every sample is taken as one patient. With it, every sample needs a patient, and all
samples of a patient must have the same label.

## How the folds and the patients are used

- A TCGA task is cross-fitted: for each fold k, a method is fitted on the other four folds and predicts fold k.
- Every method is scored on one prediction per patient (see the README, "Unit of analysis").
