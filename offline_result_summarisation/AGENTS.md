# Offline Result Summarisation Stack — Agent Reference

## File layout

```
offline_result_summarisation/
├── call_summarisation_stack_crossfold.sh   ← (NEW) Top-level: loop folds → Phase 1 → Phase 2
├── call_summarisation_stack_phase1.sh      ← (NEW) Per-fold computation (Phase 1, no stat tests/tables)
├── call_summarisation_stack_adaptive.sh    ← Original orchestrator (kept for standalone per-split use)
├── call_summarisation_stack_adaptive_zeroshot.sh
├── utils.py
├── cross_fold_summarisation/               ← (NEW) Phase 2 module
│   ├── cross_fold_sig_and_ranking.py       ← Cross-fold stat tests + betterness + variability
│   ├── cross_fold_trajectory_plots.py      ← Cross-fold trajectory plots + NoS
│   └── cross_fold_run.sh                   ← Phase 2 orchestrator
├── result_summarisation/                   ← unchanged
├── trajectory_metrics/                     ← pseudotime_auc_calc.py stripped of per-run variability
├── ranking_and_stat_tests/                 ← generate_algo_rankings.py reused; others deprecated
├── table_generation/                       ← table_generator.py updated with --cross_fold_mode
├── oracle_metrics/                         ← unchanged
└── deprecated/                             ← unchanged
```

## Two-Phase Architecture

The summarisation stack operates in two phases:

1. **Phase 1 — Per-fold**: The existing pipeline runs independently for each fold (e.g., `fold_0`, `fold_1`, ...). Each fold is treated as its own split. No changes to the core aggregation/summarisation logic.
2. **Phase 2 — Cross-fold**: All folds are brought together for statistical testing, ranking, and final reporting. This is where significance and betterness are determined.

---

## Phase 1 — Per-fold Pipeline (unchanged internals)

Called once per fold with `MASTER_SPLIT=fold_N`. Outputs go to `Results_Summary/fold_N/` and `Results_Pseudotime/fold_N/`.

```
call_summarisation_stack_adaptive.sh
  │
  ├── 1. result_summarisation/results_aggregation_run.sh
  │     └── result_aggregation.py          # groupby('Case_Name').mean() across runs
  │
  ├── 2. result_summarisation/standard_metric_summary_run.sh
  │     └── standard_metric_summarisation.py   # mean/median/std/AUC/peak per iteration
  │
  ├── 3. oracle_metrics/num_of_interactions_run.sh
  │     └── num_of_interaction.py          # first iteration exceeding nnUNet threshold
  │
  ├── 4. result_summarisation/ (repeat for adaptive apps)
  ├── 5. oracle_metrics/ (repeat for adaptive apps)
  │
  ├── 6. result_summarisation/results_aggregation_run_final_episode.sh
  │
  ├── 7. trajectory_metrics/merge_episodes_run.sh
  │     └── merge_episodes.py              # spread episodic metrics across pseudotime samples (PER RUN)
  │
  ├── 8. trajectory_metrics/pseudotime_results_aggregation_run.sh
  │     └── pseudotime_result_aggregation.py    # average per-run trajectories → run-aggregated
  │
  ├── 9. trajectory_metrics/pseudotime_auc_run.sh
  │     └── pseudotime_auc_calc.py         # trapezoidal AUC on aggregated trajectory
  │
  ├──10. trajectory_metrics/plot_pseudotime_auc_run.sh
  │     └── plot_pseudotime_auc.py         # per-fold trajectory plots
  │
  └── (Phase 2 scripts are NOT sourced here anymore)
```

### Key properties of Phase 1

- **Snapshot aggregation** (`result_aggregation.py`): within each fold, runs are averaged (`groupby('Case_Name').mean()`), preserving per-case granularity.
- **Trajectory spreading** (`merge_episodes.py`): done per-run first, then trajectories are averaged across runs. This is because trajectory length may vary across runs (async episode scheduling).
- **Uniform episode assertion**: within a single fold, all runs must have the same episode count. This is enforced.
- **Per-run variability**: removed. Phase 1 no longer computes run-level std/variability — that is now a cross-fold concern.
- **Outputs per fold**: `Results_Summary/fold_N/` (per-case aggregated metrics), `Results_Pseudotime/fold_N/` (per-fold trajectories + AUCs).

---

## Phase 2 — Cross-fold Pipeline (new)

Called after all folds have completed Phase 1.

```
call_summarisation_stack_crossfold.sh
  │
  ├── (Phase 1 loop over MASTER_FOLDS)
  │
  └── cross_fold_summarisation/cross_fold_run.sh
        │
        ├── 1. cross_fold_sig_and_ranking.py
        │     └── Snapshot: pooled per-case Wilcoxon on concatenated fold data
        │     └── Trajectory: fold-paired Wilcoxon on K AUC scalars
        │     └── Betterness: mean of fold-level representatives
        │     └── Variability: std of fold-level representatives
        │
        ├── 2. ranking_and_stat_tests/generate_algo_rankings.py
        │     └── Reads cross-fold sig matrices from Results_CrossSplit/
        │
        ├── 3. cross_fold_summarisation/cross_fold_trajectory_plots.py
        │     └── NoS is computed HERE (not per-fold)
        │     └── Normalise per-fold trajectories to proportional time [0,1]
        │     └── Interpolate to common grid, average, std band
        │     └── NoS on averaged trajectory + failure fraction annotation
        │
        └── 4. table_generation/table_generator.py (--cross_fold_mode, default on)
              └── Reads metric values from cross_fold_summary.csv
              └── Reads significance from Results_CrossSplit/
```

---

## Cross-fold Statistical Tests — Detailed Logic

### Snapshot (episodic) metrics

| Step | Detail |
|---|---|
| **Input** | Per-case aggregated CSVs from each fold (`Results_Summary/fold_N/<APP>/<DATASET>/pointsonly/run-aggregated/metrics/<Metric>/cross_class_scores.csv`, `casewise_aucs.csv`, `casewise_noi.csv`) |
| **Stat test** | Row-concatenate all folds' DataFrames (cases are disjoint per fold, indexed by `Case_Name` → no collisions) → case-wise paired Wilcoxon on the combined set. Same logic as current `compute_algo_wise_significance_score`. McNemar for NoF. |
| **Betterness** | Per fold: compute fold-level representative using the configured statistic (`mean`, `median`, etc. from `MEASURES_OF_BETTERNESS`). Cross-fold: mean of K fold-level representatives. Compare across algorithms. |
| **Variability** | `std` of K fold-level representatives |

### Trajectory metrics

| Step | Detail |
|---|---|
| **Input** | Fold-level AUC scalar from each fold (`Results_Pseudotime/fold_N/<APP>/<DATASET>/pointsonly/run-aggregated/pseudotime_metrics/all_trajectory_aucs.csv` — one scalar per metric per algorithm) |
| **Stat test** | Build K-valued series per algorithm (paired by fold index). Paired Wilcoxon on K values. |
| **Betterness** | Mean of K fold-level AUCs |
| **Variability** | `std` of K fold-level AUCs |
| **NoS** | Computed on the averaged normalised trajectory only (not per-fold). Failure fraction reported as `count(folds_never_crossed) / K`. |

---

## Cross-fold Trajectory Plot

1. For each fold: read `all_pseudotime_metrics.csv`, normalise x-axis to proportional time [0, 1]
2. Interpolate all folds to a common N-point grid (e.g., 1000 points)
3. Compute mean trajectory ± 1 std band at each grid point across folds
4. Compute NoS on the averaged trajectory (crossing of nnUNet threshold). If no crossing, NoS = NaN.
5. Annotate: `NoS = X, failure fraction = Y/Z folds`

---

## Output Directory Structure

```
Results_CrossSplit/<CONFIG_NAME>/
  <DATASET>/
    Dice_Init_significance.csv
    Dice_Init_bettername.csv
    Dice_Interactive_Edit_Iter_100_significance.csv
    ...
    Dice_AUC_significance.csv
    Dice_AUC_bettername.csv
    NSD_AUC_significance.csv
    NSD_AUC_bettername.csv
    NOI_significance.csv
    NOI_bettername.csv
    NoF_significance.csv
    NoF_bettername.csv
    ...
  plots/
    <APP>_<DATASET>_trajectory.png
    <APP>_<DATASET>_trajectory.pdf
  cross_fold_summary.csv              # mean ± std across folds, per metric per algorithm
  rankings.csv                        # cross-fold MSD-style ranking
  tables/                             # from table_generator.py
    <APP>_summary.csv
    <APP>_summary.tex

Results_Summary/
  fold_0/ ...                    # unchanged per-fold summaries
  fold_1/ ...

Results_Pseudotime/
  fold_0/ ...                    # unchanged per-fold trajectories
  fold_1/ ...
```

---

## Key Design Decisions

1. **Per-fold vs cross-fold separation**: Phase 1 produces intermediate metrics per fold. Phase 2 is the only place where significance, ranking, and final tables are determined. Per-fold stat testing is meaningless on its own and is never done.

2. **Run-level variability is stripped**: Phase 1 no longer computes per-run std or run-to-run variability. All variability reporting is cross-fold (std of K fold-level representatives).

3. **Asymmetric trajectory pipeline**: Trajectories must be spread per-run (because episode scheduling is asynchronous) before cross-run averaging. Snapshot metrics are i.i.d. and can be aggregated first.

4. **Cross-fold mode is the default**: The table generator defaults to `--cross_fold_mode`. Using per-fold data for tables is only for debugging.

5. **Episode count assertion kept per-fold**: Within a fold, all runs must have the same episode count. Cross-fold, each fold is independent.

---

## Deprecated / Unused Code

The following files remain in the repo but are no longer called by the pipeline:

- `ranking_and_stat_tests/algo_wise_sig_score.py` — replaced by `cross_fold_sig_and_ranking.py`
- `ranking_and_stat_tests/trajectory_algo_wise_sig_score.py` — replaced by `cross_fold_sig_and_ranking.py`
- `ranking_and_stat_tests/execute_sig_and_ranking.sh` — per-fold sig/ranking no longer run
- `ranking_and_stat_tests/execute_sig_and_ranking_traj.sh` — per-fold trajectory sig/ranking no longer run

- `trajectory_metrics/num_of_samples_run.sh` — NoS moved to cross-fold trajectory plot script
- `trajectory_metrics/num_of_samples.py` — NoS moved to cross-fold trajectory plot script

`ranking_and_stat_tests/generate_algo_rankings.py` is kept and reused — it reads cross-fold significance from `Results_CrossSplit/` instead of per-fold `Results_StatSig/`.
