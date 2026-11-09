"""
ESE 5420 Final Project — Baseline checkpoint pipeline.

Loads OpenML CreditCardFraudDetection (ID 42175), drops exact duplicate rows,
replaces raw Time with Time_of_day, holds out a stratified 20% test set
(unused here), tunes L2 logistic regression C by mean validation Average
Precision in stratified 5-fold CV, and compares four threshold/weighting
strategies on expected cost at two cost ratios (c_FN/c_FP = 10, 100).

Writes results/baseline.png, results/baseline_results.json,
results/baseline_folds.csv.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "creditcard.csv"
RESULTS_DIR = ROOT / "results"
DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
COST_RATIOS = [10, 100]
THRESHOLD_GRID = np.unique(
    np.concatenate(
        [
            np.array([0.5]),
            np.array([1.0 / (1.0 + r) for r in COST_RATIOS]),
            np.geomspace(1e-4, 0.5, 40),
        ]
    )
)


def expected_cost(y_true, proba, ratio, threshold=0.5, c_fp=1.0):
    c_fn = c_fp * ratio
    pred = (proba >= threshold).astype(int)
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return (c_fp * fp + c_fn * fn) / len(y_true)


def formula_threshold(ratio: float) -> float:
    return 1.0 / (1.0 + ratio)


def choose_threshold(y_true, proba, ratio, grid=THRESHOLD_GRID):
    costs = [expected_cost(y_true, proba, ratio, t) for t in grid]
    return float(grid[int(np.argmin(costs))])


def make_lr(C: float, class_weight=None) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=C,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight=class_weight,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def summarize_metric(df: pd.DataFrame, col: str) -> dict:
    """Mean, fold range, and a fold-level standard error / 95% CI (t, df=K-1).

    The SE/CI here describes how much the mean would move if CV were rerun
    with a different fold split. It is NOT the bootstrap CI planned for the
    held-out test set later (that one captures per-transaction noise, which
    matters given how few fraud cases are in the test set).
    """
    vals = df[col].to_numpy(dtype=float)
    k = len(vals)
    mean = float(vals.mean())
    lo, hi = float(vals.min()), float(vals.max())
    se = float(vals.std(ddof=1) / np.sqrt(k)) if k > 1 else 0.0
    t_crit = float(stats.t.ppf(0.975, df=k - 1)) if k > 1 else float("nan")
    half = t_crit * se
    return {
        "mean": mean,
        "min": lo,
        "max": hi,
        "se": se,
        "ci95_lo": mean - half,
        "ci95_hi": mean + half,
    }


def collect_oof_proba(C, X_tr, y_tr, class_weight=None):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    for fold_i, (tr, va) in enumerate(skf.split(X_tr, y_tr)):
        pipe = make_lr(C, class_weight=class_weight)
        pipe.fit(X_tr.iloc[tr], y_tr[tr])
        proba_tr = pipe.predict_proba(X_tr.iloc[tr])[:, 1]
        proba_va = pipe.predict_proba(X_tr.iloc[va])[:, 1]
        row = {
            "fold": fold_i,
            "C": C,
            "class_weight": "balanced" if class_weight == "balanced" else "none",
            "ap": average_precision_score(y_tr[va], proba_va),
            "precision@0.5": precision_score(
                y_tr[va], (proba_va >= 0.5).astype(int), zero_division=0
            ),
            "recall@0.5": recall_score(
                y_tr[va], (proba_va >= 0.5).astype(int), zero_division=0
            ),
        }
        for r in COST_RATIOS:
            tau_star = formula_threshold(r)
            tau_cv = choose_threshold(y_tr[tr], proba_tr, r)
            row[f"cost_norepair_r{r}"] = expected_cost(y_tr[va], proba_va, r, 0.5)
            row[f"cost_formula_r{r}"] = expected_cost(y_tr[va], proba_va, r, tau_star)
            row[f"cost_cvtau_r{r}"] = expected_cost(y_tr[va], proba_va, r, tau_cv)
            row[f"tau_formula_r{r}"] = tau_star
            row[f"tau_cv_r{r}"] = tau_cv
            row[f"precision_formula_r{r}"] = precision_score(
                y_tr[va], (proba_va >= tau_star).astype(int), zero_division=0
            )
            row[f"recall_formula_r{r}"] = recall_score(
                y_tr[va], (proba_va >= tau_star).astype(int), zero_division=0
            )
            row[f"precision_cvtau_r{r}"] = precision_score(
                y_tr[va], (proba_va >= tau_cv).astype(int), zero_division=0
            )
            row[f"recall_cvtau_r{r}"] = recall_score(
                y_tr[va], (proba_va >= tau_cv).astype(int), zero_division=0
            )
        fold_rows.append(row)
    return pd.DataFrame(fold_rows)


def load_data() -> tuple[pd.DataFrame, int]:
    """Load the raw OpenML data and drop exact duplicate rows.

    This dataset is known to contain exact duplicate transactions (same
    Time, Amount, V1-V28, and Class). Left in place, a random train/test or
    CV split can place one copy of a duplicate pair in training and its
    twin in validation/test, which leaks that exact row into evaluation and
    inflates recall/AP for a model that has effectively already seen it.
    """
    if DATA_CSV.exists():
        print(f"Loading local CSV: {DATA_CSV}")
        df = pd.read_csv(DATA_CSV)
    else:
        print("Fetching OpenML CreditCardFraudDetection (ID 42175)...")
        ds = fetch_openml(data_id=42175, as_frame=True, parser="auto")
        df = ds.data.copy()
        df["Class"] = ds.target.astype(int)
        for c in df.columns:
            if c != "Class":
                df[c] = df[c].astype(float)
        df["Class"] = df["Class"].astype(int)
        df.to_csv(DATA_CSV, index=False)
        print(f"Saved {DATA_CSV}")

    assert set(["Time", "Amount", "Class"]).issubset(df.columns)
    assert df.isna().sum().sum() == 0

    n_before = len(df)
    frauds_before = int(df["Class"].sum())
    dup_mask = df.duplicated()
    n_dupes = int(dup_mask.sum())
    dupe_frauds = int(df.loc[dup_mask, "Class"].sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(
        f"Dropped {n_dupes} exact duplicate rows "
        f"({dupe_frauds} of them fraud) out of {n_before} "
        f"-> {len(df)} rows, {int(df['Class'].sum())} frauds "
        f"(was {frauds_before})"
    )
    return df, n_dupes


def main():
    df, n_dupes = load_data()
    y = df["Class"].astype(int).to_numpy()
    X = df.drop(columns=["Class"]).astype(float).copy()
    X["Time_of_day"] = X["Time"] % 86400.0
    X = X.drop(columns=["Time"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: n={len(y_train)}, frauds={int(y_train.sum())}")
    print(f"Test : n={len(y_test)},  frauds={int(y_test.sum())}  (held out)")
    del X_test, y_test

    print("\nGrid search over C (no repair) — select by mean AP...")
    fold_tables = {}
    grid_rows = []
    for C in C_GRID:
        fold_df = collect_oof_proba(C, X_train, y_train, class_weight=None)
        mean_ap = fold_df["ap"].mean()
        mean_cost100 = fold_df["cost_norepair_r100"].mean()
        grid_rows.append(
            {"C": C, "mean_ap": mean_ap, "mean_cost_r100@0.5": mean_cost100}
        )
        fold_tables[C] = fold_df
        print(
            f"  C={C:>8g}  mean AP={mean_ap:.4f}  "
            f"mean cost@0.5 (r=100)={mean_cost100:.6f}"
        )
    grid_df = pd.DataFrame(grid_rows)
    best_C = float(grid_df.loc[grid_df["mean_ap"].idxmax(), "C"])
    best_df = fold_tables[best_C]
    ap_spread = grid_df["mean_ap"].max() - grid_df["mean_ap"].min()
    print(f"Selected C = {best_C}")
    print(
        f"Note: mean AP ranges only {ap_spread:.4f} across the whole C grid "
        "(C >= 0.1 is a near-tie) -- the argmax C is not stable across "
        "reruns/environments and should not be over-interpreted."
    )

    print("\nGrid search over C (class_weight='balanced')...")
    bal_fold_tables = {}
    bal_grid_rows = []
    for C in C_GRID:
        fold_df = collect_oof_proba(C, X_train, y_train, class_weight="balanced")
        mean_ap = fold_df["ap"].mean()
        mean_cost100 = fold_df["cost_norepair_r100"].mean()
        bal_grid_rows.append(
            {"C": C, "mean_ap": mean_ap, "mean_cost_r100@0.5": mean_cost100}
        )
        bal_fold_tables[C] = fold_df
        print(
            f"  C={C:>8g}  mean AP={mean_ap:.4f}  "
            f"mean cost@0.5 (r=100)={mean_cost100:.6f}"
        )
    bal_grid_df = pd.DataFrame(bal_grid_rows)
    best_C_bal = float(bal_grid_df.loc[bal_grid_df["mean_ap"].idxmax(), "C"])
    bal_df = bal_fold_tables[best_C_bal]
    print(f"Selected C (balanced) = {best_C_bal}")

    strategies = []
    for name, cost_key in [
        ("no_repair_0.5", "cost_norepair"),
        ("formula_tau", "cost_formula"),
        ("cv_tau", "cost_cvtau"),
    ]:
        entry = {
            "strategy": name,
            "model": "logistic_regression",
            "C": best_C,
            "class_weight": "none",
            "average_precision": summarize_metric(best_df, "ap"),
        }
        for r in COST_RATIOS:
            entry[f"cost_ratio{r}"] = summarize_metric(best_df, f"{cost_key}_r{r}")
            if name == "no_repair_0.5":
                entry[f"threshold_ratio{r}"] = 0.5
            elif name == "formula_tau":
                entry[f"threshold_ratio{r}"] = formula_threshold(r)
                entry[f"precision_ratio{r}"] = summarize_metric(
                    best_df, f"precision_formula_r{r}"
                )
                entry[f"recall_ratio{r}"] = summarize_metric(
                    best_df, f"recall_formula_r{r}"
                )
            else:
                entry[f"threshold_ratio{r}_mean"] = float(
                    best_df[f"tau_cv_r{r}"].mean()
                )
                entry[f"precision_ratio{r}"] = summarize_metric(
                    best_df, f"precision_cvtau_r{r}"
                )
                entry[f"recall_ratio{r}"] = summarize_metric(
                    best_df, f"recall_cvtau_r{r}"
                )
        if name == "no_repair_0.5":
            entry["precision_at_0.5"] = summarize_metric(best_df, "precision@0.5")
            entry["recall_at_0.5"] = summarize_metric(best_df, "recall@0.5")
        strategies.append(entry)

    bal_entry = {
        "strategy": "class_weight_balanced_0.5",
        "model": "logistic_regression",
        "C": best_C_bal,
        "class_weight": "balanced",
        "average_precision": summarize_metric(bal_df, "ap"),
        "precision_at_0.5": summarize_metric(bal_df, "precision@0.5"),
        "recall_at_0.5": summarize_metric(bal_df, "recall@0.5"),
    }
    for r in COST_RATIOS:
        bal_entry[f"cost_ratio{r}"] = summarize_metric(bal_df, f"cost_norepair_r{r}")
        bal_entry[f"threshold_ratio{r}"] = 0.5
    strategies.append(bal_entry)

    pi = float(y_train.mean())
    summary = {
        "selected_C_norepair": best_C,
        "selected_C_balanced": best_C_bal,
        "feature_note": "Dropped raw Time; added Time_of_day = Time % 86400",
        "duplicates_dropped": n_dupes,
        "n_train": int(len(y_train)),
        "frauds_train": int(y_train.sum()),
        "fraud_rate_train": pi,
        "naive_baselines": {
            "always_legitimate": {f"cost_ratio{r}": r * pi for r in COST_RATIOS},
            "always_fraud": {f"cost_ratio{r}": (1.0 - pi) for r in COST_RATIOS},
        },
        "strategies": strategies,
        "c_grid_norepair": grid_rows,
        "c_grid_balanced": bal_grid_rows,
    }

    rows = []
    for s in strategies:
        for r in COST_RATIOS:
            c = s[f"cost_ratio{r}"]
            rows.append(
                {
                    "strategy": s["strategy"],
                    "ratio": r,
                    "cost_mean": c["mean"],
                    "cost_min": c["min"],
                    "cost_max": c["max"],
                    "cost_se": c["se"],
                    "ap_mean": s["average_precision"]["mean"],
                    "ap_min": s["average_precision"]["min"],
                    "ap_max": s["average_precision"]["max"],
                }
            )
    result_table = pd.DataFrame(rows)

    strategy_order = [
        "no_repair_0.5",
        "formula_tau",
        "cv_tau",
        "class_weight_balanced_0.5",
    ]
    labels = {
        "no_repair_0.5": "No repair\n(τ=0.5)",
        "formula_tau": "Formula τ*",
        "cv_tau": "CV-chosen τ",
        "class_weight_balanced_0.5": "Class weight\nbalanced",
    }
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=120)
    x = np.arange(len(COST_RATIOS))
    width = 0.18
    for i, strat in enumerate(strategy_order):
        sub = result_table[result_table["strategy"] == strat].set_index("ratio").loc[
            COST_RATIOS
        ]
        means = sub["cost_mean"].to_numpy()
        ses = sub["cost_se"].to_numpy()
        axes[0].bar(
            x + (i - 1.5) * width,
            means,
            width,
            yerr=1.96 * ses,
            capsize=4,
            color=colors[i],
            edgecolor="black",
            linewidth=0.4,
            label=labels[strat].replace("\n", " "),
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"c_FN/c_FP = {r}" for r in COST_RATIOS])
    axes[0].set_ylabel("Expected cost per transaction")
    axes[0].set_title("Expected cost by strategy\n(mean ± 1.96×SE across 5 folds)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)

    ap_means, ap_mins, ap_maxs, ap_labels = [], [], [], []
    for strat in strategy_order:
        s = next(s for s in strategies if s["strategy"] == strat)
        ap_means.append(s["average_precision"]["mean"])
        ap_mins.append(s["average_precision"]["min"])
        ap_maxs.append(s["average_precision"]["max"])
        ap_labels.append(labels[strat])
    xpos = np.arange(len(strategy_order))
    ap_means = np.array(ap_means)
    ap_mins = np.array(ap_mins)
    ap_maxs = np.array(ap_maxs)
    axes[1].bar(
        xpos,
        ap_means,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        yerr=[ap_means - ap_mins, ap_maxs - ap_means],
        capsize=4,
    )
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(ap_labels, fontsize=8)
    axes[1].set_ylabel("Average Precision")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("AP by strategy (secondary)\nthreshold moves share no-repair AP")
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle(
        f"LR baseline checkpoint (duplicates dropped) — "
        f"C_norepair={best_C:g}, C_balanced={best_C_bal:g}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "baseline.png", bbox_inches="tight")

    best_df.assign(strategy_family="norepair").to_csv(
        RESULTS_DIR / "baseline_folds.csv", index=False
    )
    bal_df.assign(strategy_family="balanced").to_csv(
        RESULTS_DIR / "baseline_folds.csv", mode="a", header=False, index=False
    )
    with open(RESULTS_DIR / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {RESULTS_DIR / 'baseline.png'}")
    print(f"Saved {RESULTS_DIR / 'baseline_results.json'}")
    print(f"Saved {RESULTS_DIR / 'baseline_folds.csv'}")
    print("\nHeadline (mean expected cost, 95% CI from fold SE):")
    for s in strategies:
        r10, r100 = s["cost_ratio10"], s["cost_ratio100"]
        print(
            f"  {s['strategy']:28s}  "
            f"r10={r10['mean']:.5f} [{r10['ci95_lo']:.5f},{r10['ci95_hi']:.5f}]  "
            f"r100={r100['mean']:.5f} [{r100['ci95_lo']:.5f},{r100['ci95_hi']:.5f}]  "
            f"AP={s['average_precision']['mean']:.3f}"
        )


if __name__ == "__main__":
    main()
