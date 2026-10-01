# Audit: scaffold vs random split in the affinity probe

Date: 2026-09-10. Source: `data/probing/*/rmsd_cutoff_*/*-k-fold/affinity/*/[0-9]/reports/*_summary.json`
plus `manifest.json`, `predictions.csv`, and `kinodata/model/complex_transformer.py` in
`~/Thesis/kinodata-3D-affinity-prediction`.

## CONFIRMED: at the last layer the ordering is not reversed

MLP probe, target `affinity`, final layer, n_test = 2062 / 2915 / 4671:

| model | rmsd | layer | random | scaffold | pocket |
|---|---|---|---|---|---|
| CGNN-3D | 2 | L3 | **0.617** | 0.680 | 0.735 |
| CGNN-3D | 4 | L3 | **0.619** | 0.657 | 0.792 |
| CGNN-3D | 6 | L3 | **0.621** | 0.698 | 0.819 |
| CGNN | 2 | L3 | **0.715** | 0.726 | 0.760 |
| CGNN | 4 | L3 | **0.739** | 0.743 | 0.830 |
| DTI | 2 | L4 | 0.686 | 0.683 | 0.776 |

(MAE, lower is better.) random < scaffold < pocket — the paper's split-hardness
ordering. The same holds for ridge at the final layer.

The reversal the question was about appears only at **layers 0-2**, e.g. CGNN-3D
rmsd 2: L0 ridge 0.916 / 0.880 / 0.857, L1 mlp 0.896 / 0.876 / 0.861,
L2 ridge 0.859 / 0.843 / 0.816 (random / scaffold / pocket).

## CONFIRMED: the last-layer probe is a refit of the model's own readout

`ComplexTransformer.forward` computes `graph_repr = self.aggr(node_repr, batch)`
after the last attention block and returns `self.out(graph_repr)`. The pooled
`layer_3` representation *is* `graph_repr`, i.e. exactly the input to `self.out`
(`Dropout -> BatchNorm1d -> FF x decoder_hidden_layers -> Linear(256, 1)`).

So an MLP probe on the last layer re-trains the readout on a frozen encoder. It
inherits the encoder's generalisation gap, which is why it reproduces the paper's
ordering there.

Caveat for earlier layers: `self.aggr` (learned SoftmaxAggregation) was trained for
last-block features. Pooling layers 0-2 with it is out of domain for the aggregator,
so early-layer comparisons are not a clean "same readout, different depth" contrast.

## CONFIRMED: the probe's own split is random regardless of `split_type`

`prob_run.run_probe` falls back to
`train_test_split(X, y, test_size=0.1, random_state=96)` over the pooled, fold-
concatenated rows. Fold assignment (and therefore scaffold disjointness) is not
respected by that split.

Consequence: the probe evaluation is always in-distribution. `split_type` changes
the *encoder* only, never the difficulty of the probe's test set. The probe number
is decodability, the paper's number is out-of-distribution generalisation — there is
no reason the two must order the same way, and neither number is comparable to the
other in absolute terms.

## CONFIRMED: the probe test rows differ between split conditions

Each split type's aggregated dataset covers the same 20,620 idents (n_samples 18,558
+ n_test 2,062) with near-identical label distribution (y mean 7.424 vs 7.447,
sd 1.179 vs 1.198), but the fold concatenation order differs, so the positional
`train_test_split` selects different rows. Checked on
`affinity/mlp/3/artifacts/mlp_predictions.csv`: y_true vectors are not equal, not
equal as multisets, and only 397 distinct values overlap (of 915 / 978).

Every cross-split comparison therefore carries an uncontrolled row-sampling term.
Fix: derive the probe split from a deterministic hash of `ident`, shared across all
conditions, so conditions become paired on identical test rows.

## CONFIRMED: layer 0 gives the between-condition noise floor

Layer 0 is the pre-message-passing embedding, so it should depend on `split_type`
only weakly. It already spans MAE 0.857-0.950 across conditions. The L1/L2 "flips"
(0.02-0.04) sit inside that spread; the L3 gap (0.06-0.08) sits outside it and is
directionally consistent across CGNN, CGNN-3D, DTI and all three RMSD cutoffs.
Use `prob_stats` pairwise comparison + Holm-Bonferroni rather than eyeballing.

## CONFIRMED: mixed-run contamination in the result directories

- `CGNN-3D/rmsd_cutoff_2/random-k-fold/predictions.csv`: 41,238 rows, y mean 7.007,
  sd 1.314, model MAE 0.769, R2 0.362. Its manifest (2026-07-24) has
  `save_representations: false` and no `include_val` key.
- The `layer_*.pt` in the same directory correspond to a 20,620-row set with
  y mean ~7.43 (from the probe artifacts above). **Different run.**
- `CGNN-3D/rmsd_cutoff_2/scaffold-k-fold/predictions.csv`: 20,620 rows, model MAE
  0.637 — but that manifest (2026-08-18) has `save_predictions: false`, so the CSV
  is a leftover from an older extraction.

Comparing model MAE from these CSVs against probe MAE, or across split types, is
invalid. Use the per-fold `preds_<k>.pt` / `y_true_<k>.pt` in each fold directory,
which are aligned with the extracted representations.

## ASSUMPTION / not yet controlled: training length differs by split

Checkpoint versions: random folds `v57, v48, v31, v40, v43`; scaffold folds
`v17, v21, v32, v24, v25`. Until epochs-at-best-checkpoint and val MAE are reported
per fold, any "scaffold representations are better organised" claim is confounded
with how long each model trained and where early stopping fired.

## Next steps

1. Ident-keyed deterministic probe split shared across all conditions; re-run the
   affinity sweep for the three split types.
2. Report probe MAE against the model's own MAE on the *same* rows, from
   `preds_<k>.pt`, per split type.
3. Report the layer-0 spread as the noise floor in any cross-split figure.
4. Re-extract so `predictions.csv` and `layer_*.pt` come from one run; delete or
   archive the stale CSVs.
5. Log epochs / val MAE per fold alongside each manifest.
6. If a *generalisation* statement is wanted, that needs a separate condition:
   probe trained on fold k's train-side embeddings and tested on fold k's test side.
   Representations are currently extracted for test+val only, so this requires
   extending extraction to train rows.
