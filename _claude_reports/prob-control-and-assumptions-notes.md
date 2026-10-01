# prob: experiment control and assumptions — review notes

Date: 2026-10-01. Branch `probing` at `e5db211`.
Read: every pipeline module from extraction to statistics, the cluster and local run scripts,
one checkpoint `config.json`, the extraction manifests on disk, and the target-defining lines
of `interactions/hb_scores.ipynb` and `mw.ipynb`.
Not read: `explorer/`, `db/` internals, plotting modules, tests, the other notebooks,
and the kinodata model code.

Sections are ordered by how much they can change a conclusion. Section 1 needs your decisions.
Section 2 is what to state in the thesis. Sections 3–5 are the edits.

---

## 1. Decide these first (they affect validity)

### 1.1 Extraction uses test molecules only, but the pipeline assumes test + val  ⚠️

| Place | What it says |
|---|---|
| `cluster/generate_prob_dataset.sh` | passes `--include_val 0` |
| all 27 current `manifest.json` | `include_val: false` |
| `data_checks/test_reprs.py` | **requires** `include_val: true`, otherwise "split types hold different molecules" |
| `run_probe_per_checkpoint` docstring | claims test rows "are the same idents for every model, split type and layer at a cutoff" |
| `ProbingJobSpec` | default `True`, but its comment says "test split only" |
| `build_kd_ds` | default `False` |

Measured on disk at RMSD ≤ 2:

| | count |
|---|---|
| cutoff dataset size | 41,238 |
| molecules extracted per condition | 20,620 |
| shared between random and scaffold | 10,342 |
| shared between random and pocket | 10,333 |

**Consequence:** comparing models or layers within one split type is paired and fine.
Comparing random against scaffold against pocket uses different molecules, so those
comparisons cannot be paired.

**Your call:**
- *Test + val (re-extract all 27 conditions).* Every split type holds the same molecules, and the comparisons become paired. The cost is that validation molecules influenced checkpoint selection, though not the weights' gradients.
- *Test only (keep the data).* Representations are fully held out. You then drop the split-type pairing claim and fix the data check and docstring.

Whichever you pick, set it in one place and make the other five agree.

### 1.2 Probe split file is missing locally and would be silently recreated

`data/probing/probe_split.csv` does not exist on this machine. `load_probe_split` builds a
new one on first use from the catalogue with seed 0. A rebuild is deterministic, so it is
probably identical to the cluster's file. Nothing local can confirm that, because none of the
2,500 archived summaries record the split's hash.
**Edit:** never auto-create the file. Commit it, or commit its sha256, and fail on mismatch.

### 1.3 Tuned parameters are reused across re-runs without checking the representations

A per-checkpoint probe reuses `reports/<probe>_best_params.json` whenever the parameter grid
matches. After a re-extraction, for example after fixing 1.1, the new representations would
be fit with parameters tuned on the old ones. `per_fold.params_source = "saved"` records that
it happened but nothing prevents it. Pooled probes behave differently and always retune
unless the reuse flag is set.
**Edit:** store the extraction manifest's hash in the best-params file and reuse only when it
matches.

### 1.4 No code version is recorded

`git_commit` is `null` in all 28 manifests, probably because the container has no git.
Probe runs record no commit at all.
**Edit:** have the submit script export the commit, for example
`GIT_COMMIT=$(git rev-parse HEAD)`, and have both stages write it.

---

## 2. Assumptions to state in the thesis

These are deliberate and sound, but each one needs to be written down.

**Representations**
- They are graph-level, pooled with the model's own readout (`gnn.aggr`). DTI-soft is the DTI weights re-pooled with softmax.
- `layer_0` is the embedding before any message passing.
- Each molecule's representation comes from the checkpoint whose cross-validation fold held it out.
- Extraction applies no coordinate noise. Training used Gaussian noise of 0.1 on complex positions for CGNN and CGNN-3D, and none for DTI.
- The checkpoint is the one selected on minimum validation MAE.

**Probe split**
- It is a fixed random 10% of all complexes, keyed by ident with seed 0. The same split applies to every model, split type, cutoff and layer.
- It is not grouped by ligand. At RMSD ≤ 2, 5–6% of probe-test complexes share a ligand with probe-train, which is small.
- Within each checkpoint, the test fraction must stay within ±1.5 points of the file's fraction, or the run fails.

**Probes**
- One probe is trained per checkpoint. Headline metrics pool all folds' test rows, and `across_checkpoints` reports the mean ± sd over folds.
- A pooled probe across all folds also runs by default, written as `<probe>_pooled`.
- StandardScaler sits inside the pipeline. Tuning uses an inner 3-fold KFold on probe-train only, shuffled with seed 96, and refits on R².
- The Ridge alpha grid is 1e-5 to 1e4, with 10 values on a log scale.
- The MLP uses an internal 10% early-stopping split of probe-train.

**Control**
- The shuffled-ident control is **one** permutation with seed 96, identical for every target and layer.
- The permutation crosses GNN folds.
- The control is fit with the real target's tuned parameters and is never tuned itself.
- Because it is a single draw, its own variance is not estimated.

**Uncertainty**
- Confidence intervals come from 1,000 bootstrap resamples of probe-test rows with seed 96, so conditions with the same n get the same resample indices.
- The intervals capture test-sample noise only. Checkpoint variance is the separate `across_checkpoints` sd. Probe-training variance is not estimated.

**Targets**
- A missing target becomes NaN and its row is dropped. Only the final `n_samples` is reported, not the number dropped.
- Docking-score sentinels are converted to NaN.
- `hb_score` is a sigmoid bond score per activity.
- `hb_score_mw_weighted` is that score divided by `scale·√MW`.
- `hb_score_mw_subtracted` is the residual of a linear fit on MW. That fit uses all molecules with a score, not the molecules of each cutoff.
- These definitions live only in `interactions/hb_scores.ipynb`.

**Seeds**

| What | Seed |
|---|---|
| extraction | 96 |
| probe models | 96 |
| inner CV | 96 |
| bootstrap | 96 |
| baseline permutation | 96 |
| probe split | 0 |

The value 96 is defined separately in `prob_models.py` and `prob_orchestrate.py`.

**Device-dependent algorithm** ⚠️
- On CUDA, `random_forest` is xgboost's random-forest mode: 0.8 row and column subsampling, and `max_depth=None` becomes 20. On CPU it is sklearn's random forest.
- Cluster and local runs of this probe are therefore different algorithms.
- No current queue runs `random_forest`, only `mlp`. The device is not recorded in the summary.

---

## 3. Where control lives today

A probe run's conditions come from five places, and only the first one is logged, as a printed line.

| Source | Knobs |
|---|---|
| CLI of `prob_orchestrate` | gnn, split, rmsd, target, device, baseline_tag |
| Environment variables | `PROB_RUN_LINEAR_MODELS`, `PROB_RUN_NON_LINEAR_MODELS`, `PROB_RUN_SHUFFLED_BASELINE`, `PROB_NONLINEAR_MODELS`, `PROB_PER_CKPT`, `PROB_POOLED`, `PROB_REUSE_BEST_PARAMS`, `PROB_BEST_PARAMS_CACHE_DIR`, `PROB_LAYERS`, plus `NSLOTS`/`SLURM_CPUS_PER_TASK`/`OMP_NUM_THREADS` for n_jobs |
| Code constants | all seeds, inner CV folds, probe test size, grids, bootstrap n and confidence, test-fraction tolerances |
| Files on disk | which `layer_*.pt` exist decides the layers; `probe_split.csv`; saved best params |
| Checkpoint `config.json` | merged *over* the extraction config, including `split_index`, `seed`, `k_fold`, `split_type`, `filter_rmsd_max_value` |

Inconsistencies that hide control:
- **Toggles cannot be set from the CLI.** `get_ds_load_config` rejects any key it does not list, so the run toggles can only be set through environment variables.
- **Precedence is inconsistent.** Environment variables beat config for the toggles. For `baseline_tag` the config wins, so `PROB_BASELINE_TAG` is dead.
- **Notebook defaults differ from the scripts.** In a notebook with no environment variables, the baseline is off, because the config default is 0 while the code default is on. The scripts always set it.
- **Defaults are duplicated.** Inner CV folds are 3 in `prob_orchestrate` but 5 in `prob_run`. `include_val` has three different defaults (see 1.1). The extraction defaults exist twice, in `ProbingJobSpec` and `set_probing_config`, plus the argparse defaults. The allowed split types and RMSD values are listed in three places.
- **The checkpoint config can override the request.** It agrees today because directories are organized by these keys. A misplaced checkpoint would be followed silently, and the fold output folder is named from the checkpoint's own `split_index`.
- **Two project-root rules exist.** Writers use `HOME_PROJ_DIR` or the package location. Readers use `HOME_PROJ_DIR` or walk up from the current directory. Running from elsewhere without the variable can read a different tree.

---

## 4. Minimal edits, in order

Each edit is small and local, and none changes results except item 1.

1. **Settle `include_val`.** Apply 1.1: one default, script and data check aligned, docstrings fixed, and re-extraction if you choose test + val.
2. ✅ **Done 2026-10-01 (`prob/run_manifest.py`, commit 1f9d192).** **Add a probe run manifest.** Write the resolved settings to `<target>/experiments/run_manifest.json` and ingest it into prob.db. It should hold every knob from section 3 after resolution, the seeds, git commit, probe-split sha256, extraction-manifest sha256, layers, device and backend, n_jobs, and package versions. That is one function of about 30 lines. This gives you monitoring of the experimental conditions.
3. ✅ **Done 2026-10-01 as `ProbingExperimentSpec` in `prob/prob_config.py`.** **Use one probe settings object.** A frozen dataclass in `prob_config.py` holds every knob with its single default and is filled from the CLI. Environment variables stay as aliases read in one function with one precedence, CLI > env > default, and it fails on unknown `PROB_*` variables. It replaces `_flag`, removes the dead `PROB_BASELINE_TAG` path, and lets notebooks and scripts share defaults. The cluster and local scripts keep working unchanged.
4. **Freeze the probe split.** Apply 1.2: fail when the file is missing and check its hash against a constant.
5. **Guard parameter reuse.** Apply 1.3 using the extraction-manifest hash.
6. **Add three cheap assertions.** Require exactly one `.ckpt` per model folder; today `get_model_ckpt` takes whichever the glob returns first, and every local folder has one. Require the checkpoint `config.json` to agree with the requested split, fold, cutoff and k. Pass `GIT_COMMIT` into cluster jobs.
7. **Use one project-root rule.**
8. **Document targets outside notebooks.** Write a `targets/README.md`, or a sidecar JSON per target, that gives each target's definition, source notebook and date. Moving the code into a script is optional.

## 5. Cleanup (low priority, no behaviour change)

- **Unused modules.** `_deprecated_runners.py` and `drafts/` are not imported.
- **Obsolete layer-0 path.** `extract_layer0.py`, `aggregate_layer0.py` and `cluster/layer0_standalone.sub` are obsolete by `run_extraction`'s own docstring once older runs are backfilled.
- **Unused code in `prob_orchestrate`.** `linear_models_shuffled_ident_baseline` is never called, because `main` builds the baseline itself.
- **Unused branch in `run_probe`.** Its self-splitting branch is unused, because every caller passes the split.
- **Misplaced and misnamed code.** `resloves_and_transforms.py` has a misspelled name, a second import block and nitrogen-count code duplicated from `prob_targets.py`.
- **Cluster file read locally.** `_cpu_budget` parses `cluster/run_prob.sub` even for local runs.
- **Windows download files.** The `*Zone.Identifier` files in `prob/` and `data/` are leftovers.

## Status of results on disk

All local probe results are in `data/probing_archive/20260929_160507`. They predate the fixed
probe split, per-checkpoint probing and per-fold records, and they carry no split hash. Do not
pair them with new runs.
