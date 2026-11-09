# Which Fraud-Detection Strategy Works Best Under Severe Class Imbalance?

ESE 5420 Final Project — Sumit Pradeep Kolpe, Urav Dalal, Mahalakshmi Phaldesai

## Question

Credit-card fraud is extremely rare (under 1% of transactions). When fraud is this rare, does undersampling, oversampling, class weighting, or threshold adjustment give the best trade-off between catching fraud and avoiding false alarms?

## Why it matters

Banks need to catch fraud without declining too many legitimate purchases. Missed fraud costs customers money; false alarms annoy customers and cost investigation time. This project compares strategies for that trade-off.

## Data

[OpenML CreditCardFraudDetection (ID 42175)](https://www.openml.org/d/42175) — 284,807 transactions, 492 of them fraud (about 0.17%), 30 numeric inputs (`Time`, `Amount`, `V1`–`V28`). No missing values.

## Approach

- Baseline: Logistic Regression, no resampling, regularization tuned with 5-fold CV.
- Compare no correction, threshold adjustment (formula and CV-chosen), and class weighting; undersampling and Gradient Boosting are next.
- Stratified 80/20 train/test split; the test set is touched only once, at the end.
- Report Average Precision, precision, recall, and expected cost, not just accuracy.

## Setup

```bash
pip install -r requirements.txt
python baseline.py
```

The first run fetches the dataset from OpenML and caches it to `data/creditcard.csv`. Outputs land in `results/`.

## Repo layout

- `baseline.py` / `baseline.ipynb` — the pipeline
- `results/` — key figure and metrics
- `docs/` — proposal and instructor feedback
- `baseline.txt` — checkpoint write-up (submission)

## Status

Baseline checkpoint (logistic regression) is done — see `baseline.txt`. Next: HistGradientBoostingClassifier, undersampling, test-set bootstrap CIs.
