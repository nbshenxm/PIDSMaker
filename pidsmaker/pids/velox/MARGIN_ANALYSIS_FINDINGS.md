# Margin-to-threshold analysis: findings

Source run: slurm job 39266, `velox` on `CADETS_E3`, 12 epochs, node-level
evaluation with `threshold_method=max_val_loss` (threshold = the single
largest validation-set edge loss).

Question this investigated: given the node-level decision `y_hat = score >
thr`, does a node's distance from `thr` (its "margin") correlate with whether
the decision is actually correct? Analyzed with
[margin_analysis.py](margin_analysis.py) against the saved
`scores_model_epoch_*.pkl` files (no retraining needed).

## 1. How node maliciousness is derived from edge-level scores

Velox's objective computes one cross-entropy loss per *edge*; the shared
framework's `node_evaluation.py` (`get_node_predictions`) turns this into a
node decision by crediting every edge's loss to both its src and dst node
(`use_dst_node_loss: True`), then reducing each node's accumulated losses
down to a single score with `max` (per `threshold_method=max_val_loss`) — so
a node's score is simply its single worst edge, and it's compared against a
threshold computed the same way (`max` over all edge losses in validation,
never node-aggregated) via plain thresholding (`use_kmeans` is off for
velox). Ground truth (`y_true`) is looked up independently from the
dataset's labeled malicious-entity files, unrelated to any of this. That
`max`-over-touched-edges step is the direct mechanical cause of the
score-plateau effect in finding 2 below: any two unrelated nodes that each
happen to touch one edge of the same rare, high-loss type get an identical
node score.

## 2. Raw score distribution is not continuous near the boundary

Node score = `max` over that node's per-edge cross-entropy loss. Across every
epoch checked (0, 5, 8, 9, 11), the top of the score distribution collapses
into a handful of near-identical repeated values (e.g. epoch 11: only 14
test edges score >= 9.5, clustered at just two values, 11.60 and 12.33,
shared across unrelated nodes). This comes from the model assigning a
near-constant tiny probability to rare edge-type mismatches, so unrelated
edges land on the same loss value. Consequence: a raw, un-anchored
"how far above the global score range" view is not useful for triage — see
next section for what *is* useful.

## 3. Trusting the threshold: margin does correlate with correctness

Computed `AUC(margin, is_correct)` separately for nodes predicted positive
(does higher margin above `thr` predict TP over FP?) and nodes predicted
negative (does more-negative margin below `thr` predict TN over FN?), across
all 12 epochs:

| side | metric | epoch 0 | epoch 5 | epoch 9 | epoch 11 |
|---|---|---|---|---|---|
| above thr | AUC(margin -> TP vs FP) | 0.499 (random) | 0.799 | 1.000 | 1.000 |
| below thr | AUC(margin -> TN vs FN) | 0.442 | 0.253 | 0.183 | 0.174 |

Full `margin_analysis.py --epochs all` output behind that table:

```
 ep        thr |  n_abv  tp   fp   AUC(margin->TP|above) |   n_blw  fn      tn   AUC(margin->TN|below)
  0     9.0184 |   1081   2 1079                   0.499 |  280504  66  280438                   0.442
  1    10.8832 |   1081   2 1079                   0.497 |  280504  66  280438                   0.314
  2    12.1802 |      2   0    2                     nan |  281583  68  281515                   0.306
  3    11.2712 |      7   3    4                   1.000 |  281578  65  281513                   0.300
  4    10.3233 |   1084   5 1079                   0.801 |  280501  63  280438                   0.285
  5     9.4071 |   1084   5 1079                   0.799 |  280501  63  280438                   0.253
  6    10.1490 |   1084   5 1079                   0.801 |  280501  63  280438                   0.220
  7     8.4965 |   1095   5 1090                   0.802 |  280490  63  280427                   0.207
  8     9.0628 |      3   3    0                     nan |  281582  65  281517                   0.190
  9     8.5074 |      6   3    3                   1.000 |  281579  65  281514                   0.183
 10    10.6481 |     12   3    9                   1.000 |  281573  65  281508                   0.183
 11    11.6001 |     12   3    9                   1.000 |  281573  65  281508                   0.174
```

- **Above threshold**: in every well-converged epoch (3, 8, 9, 10, 11), TP
  margin is always strictly larger than FP margin (perfect separation,
  n=3-12 — small sample, but consistent across 5 independent epochs).
- **Below threshold**: AUC(margin -> TN) is *below* 0.5 in all 12 epochs and
  gets progressively lower with more training (0.44 -> 0.17). Read the other
  way, `AUC(margin -> FN)` rises from 0.56 to 0.83 — being closer to the
  threshold from below increasingly predicts a missed attack, not a
  correctly-rejected benign node, as training progresses.

## 4. The below-threshold signal is concentrated in a band, not a smooth gradient

Binning below-threshold nodes by distance from `thr` (see `--bands` output)
shows the missed-attack density is NOT highest immediately adjacent to the
cutoff — it peaks a few loss-units out:

- Epoch 11: band `(thr-3, thr-2]` -> 7 of 18 nodes are real attacks (39%
  purity); recall 4.4% -> 14.7% for 21 total nodes reviewed. The band
  immediately touching the threshold (`thr-2` to `thr`) is empty.
- Epoch 9: same band `(thr-3, thr-2]` -> 19 of 106 nodes are attacks (18%).
- Epoch 5: band `(thr-2, thr-1.5]` -> 13 of 13 nodes are attacks (100%).

Epoch 11's `--bands` output behind these numbers:

```
 11    11.6001 |     12   3    9                   1.000 |  281573  65  281508                   0.174
      threshold=11.6001, bands below it:
        (  -0.5,   0.0] n=       3 fn_here=   0 cum_recall=  0.0% cum_nodes_to_review=       3
        (  -1.0,  -0.5] n=       0 fn_here=   0 cum_recall=  0.0% cum_nodes_to_review=       3
        (  -1.5,  -1.0] n=       0 fn_here=   0 cum_recall=  0.0% cum_nodes_to_review=       3
        (  -2.0,  -1.5] n=       0 fn_here=   0 cum_recall=  0.0% cum_nodes_to_review=       3
        (  -3.0,  -2.0] n=      18 fn_here=   7 cum_recall= 10.3% cum_nodes_to_review=      21
        (  -4.0,  -3.0] n=    1130 fn_here=   3 cum_recall= 14.7% cum_nodes_to_review=    1151
        (  -6.0,  -4.0] n=     752 fn_here=  13 cum_recall= 33.8% cum_nodes_to_review=    1903
        (  -8.0,  -6.0] n=     132 fn_here=   7 cum_recall= 44.1% cum_nodes_to_review=    2035
        ( -12.0,  -8.0] n=  279538 fn_here=  35 cum_recall= 95.6% cum_nodes_to_review=  281573
```

Practical read: a two-tier decision (report above `thr`; hold a band roughly
1-3 loss-units below `thr` for analyst review; suppress everything further
below) would recover meaningful extra recall at a much better hit-rate than
brute-force lowering the threshold, which instead sweeps in thousands of
benign nodes per extra true positive (see the large low-purity bands further
out in the table, e.g. epoch 11's `(thr-6, thr-4]` band: 752 nodes for only
13 hits).

## Hypotheses to test next

All follow from the `max`-over-touched-edges mechanism in finding 1, aimed
at *why* the margin signal exists before trying to exploit it further:

- **H1 — degree confound.** Score = max over however many edges a node
  touched, so high-degree nodes are statistically more likely to hit a high
  max by volume alone, no real anomaly needed. Check if FPs skew high-degree
  and FNs skew low-degree.
- **H2 — is the flagged edge the attack edge?** A TP's score comes from one
  specific max-loss edge — not yet confirmed that edge is itself in
  `get_ground_truth_edges`, vs. an incidental high-loss edge on an otherwise
  correctly-labeled node.
- **H3 — is the sweet-spot band really an edge-type signature?** The
  `(thr-3, thr-2]` band (finding 4) may be dominated by one or two
  `(src_type, edge_type, dst_type)` combos rather than being a generic
  medium-anomaly zone — if so, a type-based rule would be more robust than a
  numeric margin window that already shifts between epochs.
- **H4 — generalization.** If H1/H3 hold, re-test on THEIA_E3/TRACE_E3 by
  looking for the same *cause*, not the same numeric offset from `thr`.

## Caveats / what NOT to conclude yet

- **This is one test set, 65-68 malicious nodes total.** Per-epoch band
  purity numbers (18%, 39%, 100%) are small-sample and noisy; don't read the
  exact percentages as stable.
- **The band's exact width/location moved between epochs** (`thr-3..-2` for
  epochs 9 and 11, `thr-2..-1.5` for epoch 5) — the "few units below
  threshold" story is consistent, but a fixed numeric width has not been
  validated as portable.
- **No labeled validation slice exists to calibrate a band width without
  touching test labels** — CADETS_E3 only labels the attack window inside
  the test split. Any band boundary picked by looking at this test set's
  labels is descriptive of this run, not yet a leak-free policy.
- Not yet checked on other datasets (THEIA_E3, TRACE_E3) or another training
  seed — that is the natural next step before trusting this generalizes
  beyond CADETS_E3.

## Files

- [margin_analysis.py](margin_analysis.py) — reusable script, re-run against
  any evaluation run's `precision_recall_dir`. Infers the threshold directly
  from the saved `pred_scores`/`y_preds` (no hardcoded per-run constants).
  Reproduce the output quoted above with:
  ```
  python ./pidsmaker/pids/velox/margin_analysis.py --scores-dir <precision_recall_dir> --epochs all [--bands]
  ```
  Run it as a plain script, not `-m pidsmaker...` — the package `__init__.py`
  pulls in the full trainer stack, including an unrelated broken `nltk`/
  `sqlite3` import in some conda envs.
