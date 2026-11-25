# METASIGHT two-stage agentic ensemble search

Implementation of the two-stage agentic ensemble optimization described in the manuscript Methods ("An Agentic
Evolutionary Framework for METASIGHT Ensemble Optimization").

A runnable implementation of the METASIGHT ensemble search in two LLM-guided stages, with locked evaluation. Stage 1
writes ensemble programs by LLM-guided evolution (OpenEvolve). Stage 2 has a model refine the best Stage 1 programs
through recipes that the code compiles, fits and scores within a fixed candidate budget. Every choice that uses
outcomes is made on TCGA training data only, and each program is then applied, frozen, to data that never entered its
making.

## Requirements

- Python 3.12 and the packages in `requirements.txt`:

  ```bash
  cd two_stage_impl
  uv venv -p python3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt
  ```

- Input data: `vendor/`, next to `two_stage_impl/`, holds the files the cohort builder reads. This repository ships
  only the cohort builder; the per-slide foundation-model predictions are patient-level data and are not distributed.
  Place them in the layout the builder expects:

  ```
  vendor/
    Rev_analyses/8_ensemble/build_ensemble_results.py   the cohort builder, baselines and metrics
    0_METASIGHT_ensemble_final/scoreboard_controlled.csv
    0_METASIGHT_ensemble_final/predictions_ensemble/     ensemble prediction files (TCGA folds, cohort membership)
    results_rev/best_results_summary_local.csv           FM result summary
    results_rev/TCGA_training/, results_rev/ExtVal_<cohort>/   FM predictions per cohort
  ```

  The package only reads this folder, and it stops at start with a message
  if a required input is missing. The paths in `best_results_summary_local.csv` start at `/vendor/`; the package
  resolves them to this folder. To use another copy of the inputs, set `METASIGHT2S_VENDOR` to that folder.

- API keys, for Stage 1, Stage 2 and the connection test only:
  - `OPENAI_API_KEY`: Stage 1 (OpenEvolve), and Stage 2 when `--provider openai` is chosen.
  - `METASIGHT2S_ANTHROPIC_API_KEY` (or `ANTHROPIC_API_KEY`): Stage 2 with Claude, the default. Use the first name
    where `ANTHROPIC_API_KEY` is reserved; Claude Code cloud environments, for example, do not pass it on to sessions.

## API switch

Stage 1, the GPT and Claude providers and the connection test call external APIs. They run only when the command
has `--allow-api` and the environment has `METASIGHT2S_ALLOW_API=1`; otherwise they stop before any request. Keys are
never printed or written. `calls.jsonl` records which model answered each Stage 2 call.

## Layout

```
config/
  stage1_openevolve_template.yaml   OpenEvolve settings (model gpt-6-luna)
  stage2.yaml                       Stage 2 settings: provider (default Claude Opus 5.5), carry_top_k, batch_size, bootstraps
  stage2_budget.csv                 per-task budget K, Stage 1 limits and Stage 2 budget B2
metasight2s/
  paths.py                          input and run locations, the write guard
  data/collab_eval.py, task_registry.py   cohort construction and scoring on top of the vendored builder
  data/cohorts.py                   training cohorts (one TCGA fold held out) and the locked design
  stage1/initial_program_M1.py, _M2.py    blend contract and starting program
  stage1/evaluator.py               Stage 1 fitness for OpenEvolve (scores on a training cohort)
  stage1/render.py                  per-run initial program and config with the system message
  stage1/run.py                     runs OpenEvolve for a TCGA task and training split (API)
  stage1/llm_compat.py              OpenEvolve's OpenAI client, with reasoning-model parameters for gpt-6 names
  stage1/export.py                  hands the top-k Stage 1 programs of each split to Stage 2
  stage2/interp.py                  SPEC interpreter, including the `stage1` node
  stage2/fitters.py                 learner, router and calibration fits
  stage2/grammar.py                 the proposal language: checks and compilation to a SPEC
  stage2/scoring.py                 evaluator and selection rule
  stage2/budget.py                  the budget rule and per-search plan
  stage2/prompts.py                 system prompt, task block, progress block
  stage2/llm.py                     model interface, API switch, offline mock
  stage2/llm_openai.py              GPT provider (openai SDK)
  stage2/llm_anthropic.py           Claude provider (anthropic SDK)
  stage2/refine.py                  one search; a task's searches, locked predictions and score
  stage2/emit.py                    the emitted program
  check_api.py                      connection test, one short request per model
  pipeline.py                       command line
tests/
  test_guards.py                    API switch, write guard, budget arithmetic, export, splits, providers
../vendor/                          the cohort builder and the input data (see "Requirements")
```

## Locked evaluation

Every step that uses outcomes, Stage 1 evolution and Stage 2 fitting and selection, runs on TCGA training data only
(`data/cohorts.py`):

- **TCGA tasks** are cross-fitted over the five author folds. For held-out fold k, Stage 1 and Stage 2 run on the
  other four folds, rebuilt with the vendored cohort builder so the baselines, Caruana weights and per-cancer floors
  also come from those folds. The program they produce predicts fold k. The five held-out predictions together are
  the task's locked predictions.
- **External tasks** (CPTAC, DFCI, HANCOCK, HKU, HPFS, NHS, PLCO) use the TCGA task with the same model, level and
  cutoff. Stage 1 and Stage 2 run on that full TCGA cohort; the program is frozen and applied unchanged to the
  external cohort, with no refitting and no recalibration.

When a frozen program is applied, the hints it may read (`context`: FM AUROCs, per-cancer floors and best FMs,
reference ensemble AUROCs) come from its training cohort. The Stage 1 evaluator refuses an external task. The locked predictions are scored on the task's own cohort with the
vendored evaluator (1,000 bootstrap resamples) and written to `locked_predictions.csv` for scoring at the patient
level elsewhere.

## Stage 1: LLM-guided program evolution

For each TCGA task and training split (h0 to h4: that fold held out; full: all folds), OpenEvolve evolves the function
`ensemble_blend(fm_probs, fm_names, cancer_types, context)`, starting from the coverage-aware mean (identical to
ENSEMBLE_Sim). One model call per iteration, at least 100 and at most K - 20 iterations (Budget below), early stopping after iteration 100 once 30 iterations bring no gain of 0.0005 in `combined_score`; model gpt-6-luna,
8,000 output tokens, 5 islands, population 60. The template's temperature, 0.7, is not sent to reasoning models. Fitness is the Stage 1 evaluator's `combined_score` on
the split's training cohort: margins over the single best FM, ENSEMBLE_Sim and ENSEMBLE_Val on AUROC, across-fold std
and Brier, minus the per-cancer floor deficit.

OpenEvolve 0.4.0 sends reasoning-model parameters (`max_completion_tokens`, no temperature) only to model names
starting with o1, o3, o4, gpt-5 or gpt-oss, and the API rejects `max_tokens` for gpt-6-luna. `stage1/llm_compat.py`
gives every Stage 1 model a client (OpenEvolve's `init_client` hook) that sends gpt-6 names as OpenEvolve sends gpt-5
names and leaves every other name to OpenEvolve's own client.

`stage1/render.py` writes each run's initial program and config, with a system message built from the template, the
evaluator's contract and fitness, and the training cohort's facts. `stage1-export` reads the last OpenEvolve
checkpoint of a split, drops programs with evaluator errors, ranks the rest by strict pass and `combined_score`,
removes identical code, and writes the top k as that split's Stage 2 inputs. For the accounting it records
`stage1_calls` and their tokens from the split's call log, and the checkpoint's `last_iteration`.

Each part of a Stage 1 run (`stage1/run.py`) runs through OpenEvolve's controller, and its end state is saved as a
checkpoint at the database's last iteration when OpenEvolve has not saved one: OpenEvolve 0.4.0 checkpoints every 10
iterations and at the end only when the planned last iteration is a multiple of 10, also after an early stop, so its
own last checkpoint can miss up to 9 iterations. Every answered Stage 1 call is logged (`stage1/llm_compat.py`,
`runs/<run>/stage1/<task>/<split>/calls.jsonl`: model requested and served, request id, finish reason, prompt, cached,
completion and reasoning tokens, latency).

## Stage 2: model-guided refinement at a fixed budget

One search runs on one training cohort. Its inputs are the top k Stage 1 programs of the same split (default k = 3)
with their Stage 1 scores and source code, the training cohort's facts (FM AUROCs, baselines, per-cancer floors), and
every Stage 2 candidate scored so far.

Proposals are written in a recipe language (`stage2/grammar.py`): means, rank and weighted means, single FMs and FM
blends, logistic, polynomial and random-feature learners, the output of a Stage 1 program (`stage1` node), mixes,
per-cancer routing (explicit or fitted by the code), floor protection, and calibration (none, global, per cancer).
The model chooses structure and hyperparameters; the code fits every coefficient on the training cohort with fixed
fitting code and fixed seeds (`stage2/fitters.py`), so each proposal maps to one label-free SPEC. Limits per proposal:
depth 6, 40 nodes, 8 fitted nodes.

Each round is one model call with three parts: the system prompt (fixed for all tasks), the task block (fixed for a
search) and the progress block (the 8 best candidates with their margins and the cancers under their floor, the
previous round's results, recent failures, slots left). The reply is JSON with exactly the requested number of
proposals.

Candidates are scored on the training cohort with the vendored evaluator (100 bootstrap resamples while searching)
and ranked by strict pass, then zero floor deficit, then the smallest of the three margins, then their sum, the same
rule for every task. The best candidate over the whole budget is emitted as a self-contained program (interpreter,
SPEC and, if used, the Stage 1 source), checked to reproduce the search's blend exactly, and applied unchanged to the
held-out fold or the external cohort.

Model: Claude Opus 5.5 by default (`config/stage2.yaml`: `effort: high`; no sampling parameters; server-side refusal
fallback on, so a declined request is rerun on a fallback model within the same call and `calls.jsonl` records the
model that answered). The code was tested with Claude Haiku 5.5 (`stage2 --model claude-haiku-5-5`; Haiku models
have no refusal fallback, so it is off for them). `--provider openai` uses gpt-6-luna.

## Budget (fixed before any run)

- Each task has a candidate budget K per search (`candidate_budget` in `config/stage2_budget.csv`): 291 (M1 TCGA),
  285 (M2 TCGA), 215 (other M1), 207 to 212 (other M2).
- Stage 1: at least 100 iterations, one model call each, and at most K - 20 (`stage1_min_iterations`,
  `stage1_max_iterations`). The first 100 run with early stopping off; the run then resumes from that checkpoint with
  early stopping on (30 iterations without a gain of 0.0005 in `combined_score`, counted from iteration 101) up to the
  maximum. The five held-out-fold runs of a TCGA task use its own limits (100 to 271 for M1, 100 to 265 for M2); the
  run on the full TCGA cohort serves every external task with the same model, level and cutoff and uses the smallest
  maximum among them (187, 189 or 195). Six runs per TCGA task, 48 in all: 4,800 to 12,182 Stage 1 calls.
- Stage 2: B2 = 20 slots per search (`stage2_budget`), so Stage 1 plus Stage 2 never exceeds K. A TCGA task runs five
  searches (one per held-out fold), an external task one, on the full TCGA cohort.
- Every slot counts once, whether its candidate is scored, fails a check, repeats an earlier proposal or was not
  returned. A search always uses exactly B2 slots; there is no early stopping.
- Slots 1 to 2k: the k Stage 1 programs, raw and with a global calibration.
- The remaining B2 - 2k slots are model proposals, at most 8 per call, spread evenly over ceil((B2 - 2k) / 8) calls.
  With k = 3: 14 proposals in 2 calls of 7 per search; 68 searches over the 36 tasks, 136 Stage 2 calls
  (`python -m metasight2s.pipeline plan all`).
- Transport retries after an API error are not counted as calls; in both stages each call's log keeps tokens, the
  model that answered, request id, latency and stop reason.

Outputs per task, `runs/<run>/stage2/<task>/`: `searches/<split>/` (`budget.json`, `ledger.jsonl` with one row per
slot, `calls.jsonl`, `prompts/` with the system prompt, task block, every round and every reply, `program.py`,
`selected.json`, `verify.json`, `summary.json`), `locked_predictions.csv` and `summary.json` with the locked metrics.
Per Stage 1 run, `runs/<run>/stage1/<TCGA task>/<split>/`: the rendered `config.yaml`, `calls.jsonl`,
`stage1_schedule.json` (its parts, calls per part, last checkpoint) and `openevolve_output/`; the export writes
`runs/<run>/stage1_candidates/`. `accounting` writes `runs/<run>/accounting.csv` (Stage 1 calls and tokens, Stage 2
slots, calls and tokens per task).

## How to run

```bash
cd two_stage_impl
.venv/bin/python -m metasight2s.pipeline plan all                                  # budget plan, no data needed
.venv/bin/python -m metasight2s.pipeline stage1-render M1_TCGA_slide --run run1      # all six splits
METASIGHT2S_ALLOW_API=1 .venv/bin/python -m metasight2s.pipeline stage1-run M1_TCGA_slide --run run1 --allow-api
.venv/bin/python -m metasight2s.pipeline stage1-export M1_TCGA_slide --run run1
METASIGHT2S_ALLOW_API=1 .venv/bin/python -m metasight2s.pipeline stage2 M1_TCGA_slide M1_HANCOCK_slide --run run1 \
    --allow-api
.venv/bin/python -m metasight2s.pipeline accounting --run run1
```

`--split h0..h4|full` limits a Stage 1 command to one split. `stage2 ... --provider mock` runs offline with a mock
model that draws proposals at random, to check the pipeline; its scores are not results. `--budget-override N` is for
test runs only. `stage1-import TASK --run R --from DIR` copies existing programs into a run as Stage 1 candidates,
for tests; they are recorded as not locked.

Runs are written under `runs/<run>/` (or `METASIGHT2S_RUNS`). The write guard refuses the input data and, apart from
the package's own folder, the folder that holds the package.

### Connection test

```bash
METASIGHT2S_ALLOW_API=1 .venv/bin/python -m metasight2s.pipeline check-api --run T1 --allow-api \
    --stage1 gpt-6-luna --openai gpt-6-luna --anthropic claude-opus-5-5 claude-haiku-5-5
```

One short request per model: `--stage1` goes through OpenEvolve's own client with the Stage 1 settings, so it shows
whether a Stage 1 run with that model name can call the API; `--openai` and `--anthropic` go through the Stage 2
providers and parse the reply as Stage 2 would. Results in `runs/T1/api_check.json`.

Checked on 2026-10-12: Stage 1 with gpt-6-luna, Stage 2 with gpt-6-luna and Claude Haiku 5.5, all
passing. The default Stage 2 model, Claude Opus 5.5, uses the same provider code; `--anthropic claude-opus-5-5`
checks it with your key.

## Selected programs

`selected_programs/` holds the ensemble program finalized on the TCGA development cohort for each of the eight tasks:
metastatic status (`M1`) and 1-, 2- and 3-year trajectory (`M2`, cutoffs 365, 730 and 1095 days), at patch and slide
level. Each file is self-contained (numpy only): `get_ensemble_function()` returns
`blend(fm_probs, fm_names, cancer_types)`.

## Tests (offline)

`.venv/bin/python tests/test_guards.py` (needs the input data in `vendor/`): the API switch (including the connection test), the write guard, the budget
arithmetic on all 36 tasks, reply parsing, the Stage 1 export on a synthetic checkpoint and call log, split names,
the Stage 1 client for gpt-6 names, the Stage 1 call log, the end-state checkpoint (with OpenEvolve stubbed out), and
the Claude provider's key variable and defaults. No network needed; it uses `vendor/`, because the data layer loads the vendored builder on import.
