# METASIGHT — Agentic Evolutionary Optimization Framework (reference implementation)

A public reference implementation of the METASIGHT agentic evolutionary optimization framework: a two-stage
evolutionary program search that blends the per-slide probabilities of pathology foundation
models (patch level: **CHIEF, GigaPath, MUSK, KEEP**; slide level: **CHIEF + GigaPath**) into
one calibrated ensemble prediction per task, exactly as described in the manuscript.

```
seed = coverage-aware mean
      │
Stage 1  broad exploration      (gpt-5.4-mini)  ─▶  diverse Pareto-optimal programs
      │
reflect  distil scored outcomes into lessons  ─▶  memory injected into the next prompt
      │
Stage 2  targeted refinement    (Claude API)  ─▶  refine each retained program
      │
select   lexicographic winner (per-cancer floor respected)  ─▶  freeze ONE program / task
      │
transfer apply the frozen program unchanged to internal test + external cohorts
```

Two outcomes: **Model_1** (binary metastasis, M0/M1) and **Model_2** (3-class trajectory
at 365/730/1095 days). Scope here is **patch level only**; one frozen program per task,
transferred to all cohorts. Slide-level is covered in "Slide-level vs patch-level ensembles" below.

## Slide-level vs patch-level ensembles

METASIGHT builds **two ensembles per outcome**. They differ only in the foundation-model
(FM) lineup and the granularity of the features behind each FM's per-slide prediction — the
**agentic evolutionary search itself is identical** (same blend contract, coverage-aware-mean
seed, read-only context, two-stage GPT-5.4-mini -> Claude search, strict 3-baseline /
3-metric evaluator with the per-cancer floor, and freeze-then-transfer):

- **Patch-level ensemble** — FM lineup **CHIEF, GigaPath, MUSK, KEEP**. Each FM's per-slide
  probability is produced from patch/tile-level embeddings aggregated to the slide (AB-MIL).
  This is the instantiation shipped as the runnable reference in this repo.
- **Slide-level ensemble** — FM lineup **CHIEF + GigaPath**, the two FMs with native
  whole-slide encoders. Each per-slide probability comes from slide-level features (AB-MIL
  aggregation), and those two streams are blended by the *same* search.

Both ensembles operate on **per-slide probabilities** (AUROC is computed on slide rows in
either case); "patch-level" vs "slide-level" refers to the FM feature granularity, not to
the level at which the ensemble runs. To reproduce the slide-level ensemble with this code,
supply slide-level prediction CSVs for **CHIEF and GigaPath** (same schema as
`data/INPUT_FORMAT.md`, with `K = 2` FMs) and run the identical
Stage 1 -> Stage 2 -> select -> freeze -> transfer pipeline — no changes to the search,
evaluator, or selection logic are needed. This public reference implements the patch-level
instantiation.

> This repo ships **no data** — you provide your own FM prediction CSVs (see
> `data/INPUT_FORMAT.md`). The metrics are a **clean-room** reimplementation; no patient
> data or private scoring code is included.

## Install

```bash
pip install -r requirements.txt        # core: numpy, pandas, scikit-learn, pyyaml
# (openevolve, openai, anthropic are needed to run the search — see "Running the pipeline")
```

## The blend contract

A candidate program is a **label-free pure transform** (it never sees labels):

```python
def ensemble_blend(fm_probs, fm_names, cancer_types, context):
    # Model_1:  fm_probs (K, N)     -> (N,)     P(metastasis) in [0,1]
    # Model_2:  fm_probs (K, N, 3)  -> (N, 3)   row-normalised class probabilities
    ...
def get_ensemble_function():
    return ensemble_blend
```

`fm_probs` is `NaN` where an FM does not cover a slide (handle it). `context` carries
hints only — per-FM AUROC, the single-best FM, the per-cancer floor and its FM — never
labels. See `metasight_ensemble/seed_program_M{1,2}.py`.

## Evaluation (the strict bar)

The evaluator scores a blend against three baselines — the single-best FM, `ENSEMBLE_Sim`
(NaN-aware coverage mean), and `ENSEMBLE_Val` (Caruana greedy) — on three metrics:
fold-averaged **pooled AUROC** (↑), **across-fold std** (↓), **Brier** (↓); plus a hard
**per-cancer floor** (a cancer's ensemble AUROC may not fall below its single-best FM,
enforced where n ≥ 30). `combined_score` (what the search maximises) and `strict_pass`
encode this. See `metasight_ensemble/evaluator.py`.

## Reflection (lessons memory)

The search runs a **propose -> evaluate -> reflect** loop. After each batch of candidates is
scored, the reflection step (`metasight_ensemble/reflection.py`) distils the outcomes into a
few human-readable **lessons** — which combiner families gained, which collapsed to a
degenerate single-model passthrough, which violated a per-cancer floor, which improved
calibration — and appends them to an accumulating memory at `results/reflection/<task>.md`.
That memory is rendered into a compact block and **injected into the next round's evolving
prompt** (see `results/reflection/<task>_next_prompt.txt` for the exact prompt), so the search
avoids exhausted strategies and builds on promising ones. Each candidate's note is also
returned to OpenEvolve as an artifact (`include_artifacts: true`) so it feeds back within the
search loop.

## Running the pipeline

First prepare your input: place the per-model prediction CSVs under `data/real/`
(git-ignored) following `data/INPUT_FORMAT.md`. Then:

```bash
cp .env.example .env                    # fill OPENAI_API_KEY (Stage 1) and ANTHROPIC_API_KEY (Stage 2)
python -m driver.run_all --tasks M1_TCGA_patch --stage1-iters 100 --stage2-iters 40
```

`run_all` runs the full two-stage pipeline per task: Stage 1 (gpt-5.4-mini) → reflection
lessons → Stage 2 (Claude API, seeded from the Stage-1 best with the lessons injected) →
select + freeze + transfer. You can also run the stages individually via
`driver/run_stage1.py` and `driver/run_stage2.py`.

Keys are read from the environment only (via `.env`, which is git-ignored) and are never
written into a rendered config or logged. **Do not commit API keys.**

## Layout

```
metasight_ensemble/   contract seeds, metrics, baselines, cohort loader, context,
                      evaluator, reflection, selection, llm_config
driver/               run_stage1 (gpt-5.4-mini), run_stage2 (Claude API), select_freeze,
                      transfer_eval, run_all (online orchestrator), oe_evaluator
configs/              openevolve_stage1_gpt.yaml, openevolve_stage2_claude.yaml
data/                 INPUT_FORMAT.md   (input schema — put your CSVs in data/real/)
```

See `data/INPUT_FORMAT.md` for the prediction-CSV schema.
