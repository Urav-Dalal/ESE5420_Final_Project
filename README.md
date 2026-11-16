# Which Fraud-Detection Strategy Works Best Under Severe Class Imbalance?

ESE 5420 Final Project — Sumit Pradeep Kolpe, Urav Dalal, Mahalakshmi Phaldesai (Canvas Group Sigma)

## Question (locked)

For logistic regression and `HistGradientBoostingClassifier`, which of no repair, formula threshold $\tau^\star$, CV-chosen threshold, class weighting, or random undersampling has the lowest **expected cost** at `c_FN / c_FP ∈ {10, 100}` on a test set used once? Average Precision is a secondary panel.

## Why it matters

Banks need to catch fraud without declining too many legitimate purchases. Missed fraud and false alarms have asymmetric costs, so the operating point matters—not only score ranking.

## Data

[OpenML CreditCardFraudDetection (ID 42175)](https://www.openml.org/d/42175) — 284,807 transactions, 492 frauds. Exact duplicates dropped before splitting (→ 283,726 rows, 473 frauds). Raw `Time` replaced by `Time_of_day = Time % 86400`.

## Status

- **Baseline checkpoint (done):** logistic repairs only, CV fold metrics — see `baseline.txt`, `baseline.py` / `baseline.ipynb`.
- **Beyond baseline (done for early draft):** LR + HistGB × five unstacked strategies; test scored once; paired stratified bootstrap ($B=1000$); calibration plots — see `experiments_draft.py` and `results/`.
- **Paper early draft:** NeurIPS-style LaTeX in [`paper/`](paper/).

## Setup

```bash
pip install -r requirements.txt
python baseline.py            # baseline checkpoint
python experiments_draft.py   # full draft experiments (writes results/)
```

The first run caches the dataset to `data/creditcard.csv` (gitignored). Outputs land in `results/`.

### Paper

```bash
cd paper
pdflatex draft.tex
pdflatex draft.tex
```

## Repo layout

| Path | Role |
|---|---|
| `baseline.py` / `baseline.ipynb` | Baseline checkpoint |
| `baseline.txt` | Baseline write-up |
| `experiments_draft.py` | Beyond-baseline + test bootstrap |
| `results/` | Figures, JSON, CSV metrics |
| `paper/` | NeurIPS-style early draft |
| `docs/` | Proposal and feedback notes |

## Headline test results (bootstrap means)

| Best at ratio 10 | Best at ratio 100 |
|---|---|
| LR CV-chosen τ ≈ 0.0043 (tied with formula τ\* and LR undersample) | HistGB class weight ≈ 0.0318 |
