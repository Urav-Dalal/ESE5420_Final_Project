# ESE 5420 Final Project — Fraud Detection Under Class Imbalance

Topic 21: which repair (threshold adjustment, class weighting, undersampling) works best for detecting credit-card fraud when fraud is <1% of transactions.

## Setup

```bash
pip install -r requirements.txt
python baseline.py
```

The first run fetches the dataset from OpenML (ID 42175) and caches it to `data/creditcard.csv`. Outputs are written to `results/`.

## Layout

- `baseline.py` / `baseline.ipynb` — pipeline (logistic regression, no repair / formula threshold / CV-chosen threshold / class weighting)
- `results/` — key figure and metrics
- `docs/` — proposal and instructor feedback
- `baseline.txt` — checkpoint write-up (submission)

## Status

Baseline checkpoint (logistic regression) done — see `baseline.txt`. Next: HistGradientBoostingClassifier, undersampling, test-set bootstrap CIs.
