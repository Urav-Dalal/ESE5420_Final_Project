# Which Fraud-Detection Strategy Works Best Under Severe Class Imbalance?

## Group
Canvas: [fill in group number]
Members: Sumit, Urav, Mahalakshmi

## Changes since the proposal
Six changes from the feedback. (1) Headline metric is now expected cost at `c_FN/c_FP = 10` and `100`, not AP: AP depends only on score ranking, so a moved threshold has exactly the AP of no repair, and class weights in LR mostly shift the intercept. AP stays as a secondary panel. (2) Thresholds now include the formula `tau* = c_FP/(c_FP+c_FN)` and a cost-minimizing threshold chosen inside CV (selected on each training fold, scored on the validation fold). (3) Gradient boosting will be sklearn's `HistGradientBoostingClassifier` for speed (next milestone). (4) Bootstrap intervals will use the test set, stratified by class and paired across methods; **a fold-level standard error / 95% CI is reported now** (see Result), in addition to fold min/max. (5) Raw `Time` (seconds since first transaction) was replaced with `Time_of_day = Time % 86400` so a random split cannot use absolute position in the two-day file. (6) **Exact duplicate rows are now dropped before splitting** — this dataset contains 1,081 exact duplicate transactions (see Issues found); left in place they can leak across the train/CV/test split.

## Data
OpenML CreditCardFraudDetection (ID 42175), cached locally as `data/creditcard.csv` (same ULB content: N = 284,807 raw rows; 30 inputs after the Time transform; `Class` with 492 frauds, 0.173%, before dedup). No missing values. **1,081 exact duplicate rows were dropped** (1,062 legitimate, 19 fraud), leaving **N = 283,726** rows and **473 frauds (0.167%)**. Features used: `Time_of_day`, `Amount`, `V1`–`V28`.

## Split
Stratified 80/20 train/test on the deduplicated data, `random_state = 42`. Train n = 226,980 (378 frauds); test n = 56,746 (95 frauds). Test set set aside and not used in this checkpoint. Tuning uses stratified 5-fold CV on the training half. Scaling is fit inside each CV training fold via a sklearn `Pipeline`.

## Baseline
Logistic Regression with L2. Log-spaced grid: `C` in `{0.001, 0.01, 0.1, 1, 10, 100, 1000}`. `C` is selected by mean validation Average Precision (ranking quality), because expected cost at threshold 0.5 is nearly flat for `C >= 0.1`. Selected `C` (no repair) = 0.1; class_weight='balanced' retunes `C` by AP separately; selected `C` (balanced) = 1000. **Both selections are near-ties** — the mean-AP curve varies by about 0.005 across the whole grid for `C >= 0.1`, so the argmax is sensitive to numerical noise and is not stable across reruns or package versions (confirmed by rerunning the pipeline independently twice: the selected `C` for no-repair came out as 100, then 1, then 0.1 across three runs with the same code and data, all within ~0.001 mean AP of each other).

Strategies compared (unstacked):
- no repair at `tau = 0.5`
- formula `tau* = 1/(1+ratio)` ≈ 0.091 (ratio 10) and 0.010 (ratio 100)
- CV-chosen `tau` (minimize training-fold cost on a threshold grid; evaluate on validation fold)
- `class_weight='balanced'` at `tau = 0.5`

## Result
Headline: mean expected cost per transaction on stratified 5-fold validation predictions, with a **95% CI from the fold-level standard error** (`t`-distribution, 4 df; `mean ± t(0.975,4) × s/√5`), alongside the fold (min, max). Naive always-legitimate costs are 0.01665 (ratio 10) and 0.16653 (ratio 100).

| Strategy | cost r=10 (mean, 95% CI) | cost r=100 (mean, 95% CI) | AP |
|---|---|---|---|
| no repair (0.5) | 0.00677 (0.00555, 0.00799) | 0.06625 (0.05361, 0.07888) | 0.756 (0.712, 0.808) |
| formula tau* | 0.00401 (0.00266, 0.00537) | 0.02454 (0.01694, 0.03214) | 0.756 (same scores) |
| CV-chosen tau | 0.00360 (0.00235, 0.00485) | 0.02453 (0.01561, 0.03344) | 0.756 (same scores) |
| class weight balanced | 0.02659 (0.02311, 0.03007) | 0.04047 (0.03475, 0.04618) | 0.744 (0.691, 0.776) |

No-repair precision@0.5 = 0.863, recall@0.5 = 0.603. Formula tau* at ratio 100 raises recall to 0.865 (precision 0.415). Balanced weights at 0.5 get recall 0.907 but precision only 0.057, so they lose at ratio 10 to false positives. At ratio 100, CV-chosen tau and formula tau* are statistically indistinguishable (their 95% CIs overlap almost entirely: [0.0156, 0.0334] vs [0.0169, 0.0321]) — call this a tie, not a win for CV-tau. At ratio 10 the gap between the threshold repairs and no-repair/balanced is real (CIs don't overlap). Stratified paired bootstrap intervals will be computed on the held-out test set once all methods are run.

**Versus the pre-dedup numbers**: the ranking of strategies is unchanged (CV-tau ≈ formula-tau cheapest, no-repair worst at ratio 100, balanced worst at ratio 10), but the magnitudes shifted a few percent (e.g. no-repair at r=100 moved from 0.0634 to 0.0663) now that the ~1,100 duplicate rows — including 19 duplicate frauds — are no longer inflating the fraud-detection numbers with rows the model may have partially memorized across folds.

## Issues found
- **Duplicates**: the raw OpenML/ULB extract contains 1,081 exact duplicate rows (1,062 legitimate, 19 fraud) — identical across every column including `Class`. Left in place, a random split can place one copy in training and its twin in validation/test, which is a leakage path (the model effectively sees the exact row it's evaluated on). Now dropped before any split; see Data/Result for the effect size.
- AP cannot separate threshold repairs from no repair (identical 0.756), which is exactly why cost is the headline.
- `C` selection is noise-sensitive: cost at 0.5 is flat across `C >= 0.1`; AP picks a different argmax across reruns (100, then 1, then 0.1 across three independent runs of the identical code) — treat the whole `C >= 0.1` region as a tie, not a specific "best" value.
- CV-chosen `tau` is selected using in-sample training-fold probabilities (not a nested/inner split). The reported **cost** is still honestly computed on the held-out validation fold, so the headline number is not inflated, but the chosen `tau` itself is a mildly optimistic point estimate — worth a caveat, not a fix, for this checkpoint.
- Class weights without a moved threshold are too aggressive at ratio 10.
- CV-chosen tau and formula tau* are statistically tied at ratio 100 (CIs overlap); CV-chosen is slightly better at ratio 10, with non-overlapping CIs.
- Test set still untouched.

## Next steps
Before the 16 November draft:
- Urav: `HistGradientBoostingClassifier`, same CV and cost objective, on the deduplicated data.
- Urav: Random undersampling inside folds (still unstacked).
- Mahalakshmi: Paired stratified test-set bootstrap (1,000 resamples) on the deduplicated test set; calibration diagnostics (reliability diagram, Brier score on out-of-fold probabilities).
- Sumit: Key figure for the paper (cost per strategy with bootstrap CIs; AP as panel 2); methods and results sections.

## Code
[https://github.com/Urav-Dalal/ESE5420_Final_Project](https://github.com/Urav-Dalal/ESE5420_Final_Project) — `baseline.py` / `baseline.ipynb` (run from the repo root; reads `data/creditcard.csv`, fetched automatically from OpenML if absent; writes `results/baseline.png`, `results/baseline_results.json`, `results/baseline_folds.csv`).
