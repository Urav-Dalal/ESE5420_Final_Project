# Proposal vs feedback

**Decision: Go, with changes**

ESE 5420 Topic 21 — credit-card fraud under 0.173% positives.
Baseline checkpoint Monday 9 November. One member submits for the team from this checkpoint on.

## What you are actually solving

Which repair gives the lowest expected cost at `c_FN / c_FP = 10` and `100`, for logistic regression and histogram gradient boosting? Average precision stays as a second panel. It cannot tell a moved threshold, class weights, or undersampling apart on logistic regression, because those repairs barely change how scores rank.

| | |
|---|---|
| Proposal decision | **Go** |
| Test-set frauds | **98** (wide CIs — report ties as ties) |
| Cost ratios (frozen) | **10** and **100** |

## The question, sharpened

The proposal asked which of undersampling, oversampling, class weighting, or threshold adjustment offers the best precision–recall trade-off. That is too vague, and AP is the wrong headline for it. AP is the area under the precision–recall curve: it depends only on the ranking of scores, not on where you cut.

### Repairs AP cannot see

A moved threshold is a different point on the same curve, so AP is identical to no repair. Class weights or undersampling in logistic regression mostly shift the intercept (a constant added to every log-odds). Rankings stay almost the same, so AP stays almost the same.

### What cost can see

Expected cost, `(c_FP × FP + c_FN × FN) / n`, changes as soon as the operating point moves. At ratio 10 the Bayes threshold is `τ* = 1/11 ≈ 0.091`; at 100 it is `1/101 ≈ 0.010`. Those are far from 0.5, which is why the baseline recall near 0.63 is the regime the repairs are meant to fix.

**Frozen wording:** for logistic regression and HistGradientBoosting, which of no repair, formula threshold, CV-chosen threshold, class weights, or undersampling has the lowest expected cost at the two stated ratios on a test set used once?

## What to change (from proposal to baseline)

1. **Freeze Time now, do not revisit later** — raw `Time` is seconds since the first transaction in a two-day file; replaced with `Time_of_day = Time % 86400` so a random split cannot use absolute position in the two-day file.
2. **Stop selecting C by cost at 0.5** — cost at 0.5 is nearly flat once `C ≥ 0.1`; select `C` by mean validation AP instead, and treat the flat region as a tie.
3. **Do not stack repairs** — treat no repair (0.5), formula `τ*`, CV-chosen `τ`, class weights at 0.5, and undersampling at 0.5 as independent, unstacked strategies.
4. **Keep paths in-repo** — data and outputs live under this project; fill the code link before submitting.

## Experiment to run (both models)

| Strategy | What changes | Threshold | LR effect | Boosting effect |
|---|---|---|---|---|
| No repair | Nothing | 0.5 | Current baseline | Same protocol, no scaler needed |
| Formula threshold | Operating point only | `τ* = 1/(1+r)` → 0.091 and 0.010 | AP identical to no repair | AP identical to no repair |
| CV-chosen threshold | `τ` picked to min cost on the fold | Searched inside each training fold | Needed if probabilities are miscalibrated | More likely to beat `τ*` (trees miscalibrate) |
| Class weights | `class_weight='balanced'` | 0.5 (isolates the weight) | Mostly intercept shift; AP ≈ no repair | Can change splits, not just the intercept |
| Undersampling | Random under-sample majority inside the fold | 0.5 | Mostly prior shift; AP ≈ no repair | Changes the sample trees see; can move AP |

Scaling and resampling stay inside each CV training fold via a Pipeline. Never fit them on the validation fold or the test set.

## Required by feedback

- Paired stratified bootstrap on the test set (~1,000 resamples), reported once every method is locked.
- Two cost ratios: **10** and **100** (frozen).
- Fold-level mean/min/max (and a fold-level standard error) reported now, ahead of the test-set bootstrap.
- Two dumb comparators: always-legitimate cost = `r × π`, and always-fraud cost ≈ 1.

> **Do not touch the test set until every method is locked.** 98 frauds is the entire confirmation sample. Peeking to choose a repair, a threshold, or whether to drop Time spends that sample. Fold metrics and fold-chosen thresholds only. Bootstrap the test set once, for all methods together.

## Key figure

Panel 1: expected cost per strategy, two clusters (ratio 10 and 100), bars for LR (and later HistGB), error bars from fold spread/SE now, paired stratified bootstrap later.
Panel 2: AP for the same strategies — LR threshold bars are expected to be equal. Precision, recall, chosen `τ`, and cost go in a table, not the headline plot.

Sources: team proposal, instructor feedback, baseline artifacts.
