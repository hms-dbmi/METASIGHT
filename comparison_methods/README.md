# METASIGHT comparison methods

METASIGHT combines the per-slide predictions of several pathology foundation models (FMs) into one prediction per
slide, with a separate ensemble program for each task. This package holds the methods METASIGHT is compared with,
and the code that scores them:

| Group | Method (`--methods` name) | What it is |
|---|---|---|
| Baselines | `single_best_fm` | The best single FM (highest AUROC on the training data) |
| | `standard_ensemble` | Mean of the FMs that cover a slide (ENSEMBLE_Sim) |
| | `validation_weighted_ensemble` | Weighted mean with Caruana greedy ensemble selection weights (ENSEMBLE_Val) |
| Supervised stacking | `ridge` | L2-penalized logistic stacking (multinomial for three classes) |
| | `elasticnet` | Elastic-net logistic stacking |
| | `logistic` | Unpenalized logistic stacking |
| | `gbm` | Histogram gradient boosting with early stopping |
| | `nnls` | Nonnegative weights on the simplex, minimizing the log loss |
| | `bma` | Pseudo Bayesian model averaging (Yao et al. 2018), tau = 1 |
| Search without an LLM | `random_search` | Programs drawn independently from the METASIGHT program grammar |
| | `evolutionary_search` | Population 20, tournament selection, subtree crossover and point mutation, same grammar |

Tasks: `M1_*` predicts metastatic status at diagnosis (M0 vs M1); `M2_*_cutoff<d>` predicts the clinical trajectory
at d days (stable, locoregional recurrence, distant metastasis). `patch` tasks combine CHIEF, GigaPath, KEEP and MUSK;
`slide` tasks combine CHIEF and GigaPath. TCGA is the training cohort; other cohorts are external.

## Partitions

Every method makes each choice that uses outcomes (weights, the best FM, fitted models, hyperparameters, the selected
program) on TCGA training data only (`metasightcmp/data/cohorts.py`):

- **TCGA tasks** are cross-fitted over five folds. For held-out fold k, a method is fitted on the other four folds
  and predicts fold k. The five held-out predictions together are the task's predictions.
- **External tasks** use the TCGA task with the same model, level and cutoff. A method is fitted on that full TCGA
  cohort, frozen, and applied unchanged to the external cohort, with no refitting and no recalibration.

## Unit of analysis

The patient. A cohort with several slides per patient names the patient of each slide (TCGA: `cohort.csv`; an
external cohort: the `patient_id` column of its FM files); a cohort without patient identifiers has one row per
patient.

- The baselines and the searches predict slides; a patient's prediction is the mean of the predictions of the
  patient's slides.
- The stacking methods are fitted on patients: the FM probabilities are averaged within each patient before fitting
  and before prediction.

## Methods

**Baselines** (`metasightcmp/baselines.py`). The validation-weighted ensemble uses Caruana greedy ensemble selection
(100 iterations, selection by AUROC) on the training data; its weights are renormalized per slide over the FMs that
cover it. Where the best single FM does not cover a slide, the standard ensemble's prediction is used.

**Supervised stacking** (`metasightcmp/stacking/`). Each method gets up to 300 hyperparameter configurations per
training set (`--budget`), drawn without replacement from its grid with a fixed seed, selected by inner stratified
cross-validation on the training set (pooled AUROC, ties broken by log loss) and refitted on the whole training set.

| Method | Grid | Configurations |
|---|---|---:|
| `ridge` | C: 81 values, 1e-4 to 1e4 (log scale) | 81 |
| `elasticnet` | C: 33 values, 1e-4 to 1e4; l1_ratio: 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1 | 231 |
| `gbm` | max_depth 2, 3, 4, 6, none; learning_rate 0.01, 0.03, 0.05, 0.1, 0.2; max_iter 100, 200, 400; min_samples_leaf 5, 10, 20, 50; l2_regularization 0, 0.1, 1 | 300 of 900 |
| `logistic`, `nnls`, `bma` | one configuration | 1 |

With more than 8,000 training units, the selection runs on a stratified subsample of 8,000 with two inner folds
(three otherwise); the selected configuration is always refitted on every training unit. Inputs are the logits of the
FM probabilities. Missing FM coverage: `ridge`, `elasticnet` and `logistic` get mean imputation (means from the
training set) and one indicator column per FM; `gbm` takes missing values as they are and gets the indicator columns;
`nnls` and `bma` renormalize their weights over the FMs that cover each unit.

**Search without an LLM** (`metasightcmp/search/`). Both searches draw programs from the METASIGHT program grammar
(`search/interp.py` is the interpreter of the METASIGHT two-stage search, unchanged; `search/space.py` is the program
space), evaluate exactly the task's candidate budget of distinct programs on the training cohort, starting from the
standard ensemble, and keep the best one by the selection rule of `search/scoring.py`: all three margins over the
training cohort's reference ensembles (AUROC, across-fold standard deviation, Brier) and every per-cancer floor met,
then zero per-cancer floor deficit, then the smallest margin, then the sum of the margins. Learners inside a program
and its calibration are fitted on the training cohort.

- `random_search`: every candidate is drawn independently.
- `evolutionary_search`: population 20 (the standard ensemble and 19 random programs), 20 children per generation,
  tournament size 3, crossover probability 0.6, mutation probability 0.8, the best 20 of parents and children kept.

The candidate budget per task is in `config/search_budget.csv` (207 to 291 candidate programs per search; it is the
`candidate_budget` of the METASIGHT two-stage search, the most candidates one of its searches evaluates for the
task). A TCGA task is searched five times (one per held-out fold) and an external task once (on the full TCGA
cohort), each with the task's budget.

**Seeds.** The stacking methods and the searches take a seed (`--seeds`, default 0 to 4). For a stacking method the
seed sets the draw of configurations, the inner folds and the model's random state. Each search has its own random
stream, set by the task, the training split, the method and the seed. The baselines have no random element.

## Installation

Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Input data

The input folder layout is described in [DATA_FORMAT.md](DATA_FORMAT.md): a `tasks.csv`, and per task the FM
predictions (and, for TCGA tasks, the cohort with its folds). It is the layout the METASIGHT two-stage search reads.
Set `METASIGHTCMP_DATA` to the folder, or pass `--data`.

To try the package without data, write a synthetic folder:

```bash
.venv/bin/python examples/make_synthetic_data.py data/
```

It holds seven tasks (three TCGA, four external, one of them with several slides per patient) with random labels and
FM probabilities; results on it have no meaning.

## Running

```bash
.venv/bin/python -m metasightcmp.run --methods baselines
.venv/bin/python -m metasightcmp.run --methods search --seeds 0,1,2,3,4 --workers 4
.venv/bin/python -m metasightcmp.run --methods stacking --seeds 0,1,2,3,4
.venv/bin/python -m metasightcmp.run --methods ridge,random_search --tasks M1_TCGA_slide,M1_HANCOCK_slide --seeds 0
```

`--methods` takes method names or the groups `baselines`, `stacking`, `search` and `all`; `--tasks` limits the run to
some tasks (default: every task in `tasks.csv`). Outputs go under `runs/` (or `METASIGHTCMP_RUNS`, or `--out`):

```
runs/
  predictions/slide_level/<task>__<method>[__s<seed>].csv     baselines and searches: one row per slide (or case)
  predictions/patient_level/<task>__<method>[__s<seed>].csv   every method: one row per patient
  baselines/<task>.json                                        the best FM and the weights per training split
  search/<task>__<split>__<method>__s<seed>.json               one record per search, with the selected program
  stacking/<task>__<method>__s<seed>.json                      the selected configuration per training set
```

A finished prediction file or search record is not computed again, so an interrupted run continues where it stopped.
`--workers` sets the number of processes for the searches; the stacking methods use `METASIGHTCMP_JOBS` processes for
their inner selection (default: all cores). Each process uses one numerical-library thread.

Quick run on the synthetic data with small budgets (a few minutes):

```bash
export METASIGHTCMP_DATA=data
.venv/bin/python -m metasightcmp.run --methods all --seeds 0,1 --budget 6 --search-budget 12 --workers 2
.venv/bin/python -m metasightcmp.evaluate --predictions runs/predictions/patient_level \
    --reference-method standard_ensemble --out runs/tables
```

Running time with the full budgets on the 36 METASIGHT tasks, 4 cores: the baselines take minutes; the searches about
11 CPU hours for five seeds (68 training splits, 680 searches); the stacking methods with a grid (`ridge`,
`elasticnet`, `gbm`) take the longest.

## Scores and tests

```bash
.venv/bin/python -m metasightcmp.evaluate --predictions runs/predictions/patient_level \
    --reference <folder with METASIGHT's patient-level files, <task>.csv> --out runs/tables
```

- A task's score is the mean of the AUROCs computed within each cancer type, on one prediction per patient. A cancer
  type enters with at least 10 patients and both outcome classes. The Brier score is averaged the same way.
  Three-class tasks: mean of the one-vs-rest AUROCs (and Brier scores) of the classes present.
- A task enters the comparison when its smallest outcome class has at least 5 patients.
- A method run with several seeds is reported at the seed most favorable to the method (the smallest difference in
  AUROC between the reference and the method). Seeds are never pooled.
- Within a task: stratified paired bootstrap, 2,000 resamples, patients resampled within cancer type and outcome
  class; Benjamini-Hochberg across tasks within each method.
- Across tasks: Wilcoxon signed-rank test on the per-task differences, with the Hodges-Lehmann estimate and its 95%
  interval; Benjamini-Hochberg across the methods.
- Differences are the reference minus the method.

The reference files have the layout and row order of the patient-level prediction files (first column the patient
or row identifier, then `cancer_type`, `true_label`, `n_slides` and the probability columns). `--reference-method`
uses one of the methods in `--predictions` as the reference instead. Tables written: `per_task_all_seeds.csv`,
`per_task.csv`, `summary_auroc.csv`, `summary_brier.csv`, `tasks_not_compared.csv`.

## Reproducibility

Every run is deterministic: on one machine and environment, a rerun gives the same files. The package versions are
pinned in `requirements.txt`. Methods that use iterative solvers (the logistic fits inside the stacking methods and
inside the searched programs, and the constrained optimizer of `nnls`) can differ slightly between machines with
different numerical libraries; unpenalized logistic stacking (`logistic`) is the most sensitive of them.

## Tests

```bash
.venv/bin/python tests/test_methods_synthetic.py   # every method end to end on synthetic data, small budgets
```

## Layout

```
config/            candidate budget of the searches per task
metasightcmp/
  data/            input loading, cohorts and folds, reference ensembles, metrics
  units.py         the patient as the unit of analysis
  baselines.py     single best FM, standard ensemble, validation-weighted ensemble
  stacking/        the six stacking methods, their grids, selection and fitting
  search/          program grammar and interpreter, program space, random and evolutionary search, selection rule
  predictions.py   prediction files
  run.py           command line: run the methods
  evaluate.py      command line: scores and tests
examples/          synthetic data generator
tests/             test on synthetic data
```

## License

This project is licensed under the GNU Affero General Public License v3.0; see [LICENSE](../LICENSE).
