"""
ESE 5420 — beyond-baseline draft experiments.

Extends the locked baseline protocol:
  dedup -> Time_of_day -> stratified 80/20 (seed 42) -> tune on train CV
  -> refit on full train -> score test once -> paired stratified bootstrap.

Models: LogisticRegression, HistGradientBoostingClassifier
Strategies (unstacked): no repair @0.5, formula tau*, CV-chosen tau,
  class_weight='balanced' @0.5, random 1:1 undersampling @0.5.

Writes under results/:
  draft_results.json, draft_bootstrap.json, draft_test_predictions.npz,
  table_test_metrics.csv, fig_cost_ap.png, fig_calibration.png
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
from sklearn.calibration import calibration_curve
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score
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
N_BOOT = 1000
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
COST_RATIOS = [10, 100]
HGB_GRID = [
    {"max_depth": d, "learning_rate": lr, "max_iter": m}
    for d in (3, 6)
    for lr in (0.05, 0.1)
    for m in (100, 200)
]
THRESHOLD_GRID = np.unique(
    np.concatenate(
        [
            np.array([0.5]),
            np.array([1.0 / (1.0 + r) for r in COST_RATIOS]),
            np.geomspace(1e-4, 0.5, 40),
        ]
    )
)

STRATEGY_ORDER = [
    "no_repair_0.5",
    "formula_tau",
    "cv_tau",
    "class_weight_balanced_0.5",
    "undersample_0.5",
]
MODEL_ORDER = ["logistic_regression", "hist_gradient_boosting"]


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


def undersample_indices(y: np.ndarray, random_state: int) -> np.ndarray:
    """Keep all positives; sample negatives 1:1 without replacement."""
    rng = np.random.RandomState(random_state)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_keep = min(len(pos), len(neg))
    neg_keep = rng.choice(neg, size=n_keep, replace=False)
    idx = np.concatenate([pos, neg_keep])
    rng.shuffle(idx)
    return idx


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


def make_hgb(params: dict, class_weight=None) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )


def load_data() -> tuple[pd.DataFrame, int]:
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

    n_before = len(df)
    dup_mask = df.duplicated()
    n_dupes = int(dup_mask.sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(
        f"Dropped {n_dupes} duplicates from {n_before} -> "
        f"{len(df)} rows, {int(df['Class'].sum())} frauds"
    )
    return df, n_dupes


def fit_predict_proba(model_name, params, X_fit, y_fit, X_score, class_weight=None, under=False):
    if under:
        idx = undersample_indices(y_fit, RANDOM_STATE)
        X_fit = X_fit.iloc[idx] if hasattr(X_fit, "iloc") else X_fit[idx]
        y_fit = y_fit[idx]
        class_weight = None

    if model_name == "logistic_regression":
        model = make_lr(params["C"], class_weight=class_weight)
    else:
        model = make_hgb(params, class_weight=class_weight)
    model.fit(X_fit, y_fit)
    return model.predict_proba(X_score)[:, 1], model


def cv_mean_ap(model_name, params, X, y, class_weight=None, under=False):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aps = []
    for tr, va in skf.split(X, y):
        proba, _ = fit_predict_proba(
            model_name, params, X.iloc[tr], y[tr], X.iloc[va],
            class_weight=class_weight, under=under,
        )
        aps.append(average_precision_score(y[va], proba))
    return float(np.mean(aps))


def oof_proba(model_name, params, X, y, class_weight=None, under=False):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=float)
    for tr, va in skf.split(X, y):
        proba, _ = fit_predict_proba(
            model_name, params, X.iloc[tr], y[tr], X.iloc[va],
            class_weight=class_weight, under=under,
        )
        oof[va] = proba
    return oof


def tune_lr(X, y, class_weight=None, under=False, label=""):
    rows = []
    best_C, best_ap = None, -1.0
    print(f"\nTune LR {label}...")
    for C in C_GRID:
        ap = cv_mean_ap("logistic_regression", {"C": C}, X, y, class_weight, under)
        rows.append({"C": C, "mean_ap": ap})
        print(f"  C={C:>8g}  mean AP={ap:.4f}")
        if ap > best_ap:
            best_ap, best_C = ap, C
    print(f"  selected C={best_C} (AP={best_ap:.4f})")
    return {"C": best_C}, rows


def tune_hgb(X, y, class_weight=None, under=False, label=""):
    rows = []
    best_params, best_ap = None, -1.0
    print(f"\nTune HistGB {label}...")
    for params in HGB_GRID:
        ap = cv_mean_ap("hist_gradient_boosting", params, X, y, class_weight, under)
        row = {**params, "mean_ap": ap}
        rows.append(row)
        print(
            f"  depth={params['max_depth']} lr={params['learning_rate']} "
            f"iter={params['max_iter']}  mean AP={ap:.4f}"
        )
        if ap > best_ap:
            best_ap, best_params = ap, dict(params)
    print(f"  selected {best_params} (AP={best_ap:.4f})")
    return best_params, rows


def metrics_at_threshold(y, proba, threshold, ratios=COST_RATIOS):
    out = {
        "ap": float(average_precision_score(y, proba)),
        "precision": float(
            precision_score(y, (proba >= threshold).astype(int), zero_division=0)
        ),
        "recall": float(
            recall_score(y, (proba >= threshold).astype(int), zero_division=0)
        ),
        "threshold": float(threshold),
    }
    for r in ratios:
        out[f"cost_ratio{r}"] = float(expected_cost(y, proba, r, threshold))
    return out


def paired_stratified_bootstrap(y_test, score_dict, n_boot=N_BOOT, seed=RANDOM_STATE):
    """score_dict: name -> (proba, threshold_by_ratio_or_scalar).

    threshold can be a float (same for all ratios) or dict {ratio: tau}.
    """
    rng = np.random.RandomState(seed)
    pos = np.flatnonzero(y_test == 1)
    neg = np.flatnonzero(y_test == 0)
    results = {name: {f"cost_ratio{r}": [] for r in COST_RATIOS} | {"ap": []} for name in score_dict}

    for _ in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            ]
        )
        y_b = y_test[idx]
        for name, (proba, thr) in score_dict.items():
            p_b = proba[idx]
            results[name]["ap"].append(float(average_precision_score(y_b, p_b)))
            for r in COST_RATIOS:
                t = thr[r] if isinstance(thr, dict) else thr
                results[name][f"cost_ratio{r}"].append(
                    float(expected_cost(y_b, p_b, r, t))
                )

    summary = {}
    for name, metrics in results.items():
        summary[name] = {}
        for key, vals in metrics.items():
            arr = np.asarray(vals, dtype=float)
            summary[name][key] = {
                "mean": float(arr.mean()),
                "ci95_lo": float(np.percentile(arr, 2.5)),
                "ci95_hi": float(np.percentile(arr, 97.5)),
            }
    return summary


def reliability_bins(y, proba, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos


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
    print(f"Test : n={len(y_test)},  frauds={int(y_test.sum())}")

    # ---- hyperparameter selection on train CV only ----
    lr_none_params, lr_none_grid = tune_lr(X_train, y_train, label="no-repair")
    lr_bal_params, lr_bal_grid = tune_lr(
        X_train, y_train, class_weight="balanced", label="balanced"
    )
    lr_under_params, lr_under_grid = tune_lr(
        X_train, y_train, under=True, label="undersample"
    )

    hgb_none_params, hgb_none_grid = tune_hgb(X_train, y_train, label="no-repair")
    hgb_bal_params, hgb_bal_grid = tune_hgb(
        X_train, y_train, class_weight="balanced", label="balanced"
    )
    hgb_under_params, hgb_under_grid = tune_hgb(
        X_train, y_train, under=True, label="undersample"
    )

    # ---- OOF calibration for no-repair models ----
    print("\nOOF probabilities for calibration...")
    oof_lr = oof_proba("logistic_regression", lr_none_params, X_train, y_train)
    oof_hgb = oof_proba("hist_gradient_boosting", hgb_none_params, X_train, y_train)
    calib = {
        "logistic_regression": {
            "brier": float(brier_score_loss(y_train, oof_lr)),
            "ap": float(average_precision_score(y_train, oof_lr)),
        },
        "hist_gradient_boosting": {
            "brier": float(brier_score_loss(y_train, oof_hgb)),
            "ap": float(average_precision_score(y_train, oof_hgb)),
        },
    }
    print(f"  LR Brier={calib['logistic_regression']['brier']:.6f}")
    print(f"  HGB Brier={calib['hist_gradient_boosting']['brier']:.6f}")

    # ---- refit on full train; score test once ----
    print("\nRefit on full train and score test once...")
    configs = {
        ("logistic_regression", "base"): (lr_none_params, None, False),
        ("logistic_regression", "balanced"): (lr_bal_params, "balanced", False),
        ("logistic_regression", "undersample"): (lr_under_params, None, True),
        ("hist_gradient_boosting", "base"): (hgb_none_params, None, False),
        ("hist_gradient_boosting", "balanced"): (hgb_bal_params, "balanced", False),
        ("hist_gradient_boosting", "undersample"): (hgb_under_params, None, True),
    }

    train_proba = {}
    test_proba = {}
    for (model_name, kind), (params, cw, under) in configs.items():
        key = f"{model_name}:{kind}"
        p_tr, model = fit_predict_proba(
            model_name, params, X_train, y_train, X_train,
            class_weight=cw, under=under,
        )
        # re-score train without under for threshold selection on full train preds
        # For undersampled/balanced models, use the fitted model on full X_train
        p_tr = model.predict_proba(X_train)[:, 1]
        p_te = model.predict_proba(X_test)[:, 1]
        train_proba[key] = p_tr
        test_proba[key] = p_te
        print(f"  fitted {key}")

    # Build strategy -> (test_proba, threshold spec)
    methods = {}
    point_metrics = []

    for model_name in MODEL_ORDER:
        base_key = f"{model_name}:base"
        bal_key = f"{model_name}:balanced"
        und_key = f"{model_name}:undersample"

        # no repair
        name = f"{model_name}/no_repair_0.5"
        methods[name] = (test_proba[base_key], 0.5)
        m = metrics_at_threshold(y_test, test_proba[base_key], 0.5)
        m.update({"model": model_name, "strategy": "no_repair_0.5"})
        point_metrics.append(m)

        # formula tau* — same scores, ratio-specific thresholds
        name = f"{model_name}/formula_tau"
        thr_f = {r: formula_threshold(r) for r in COST_RATIOS}
        methods[name] = (test_proba[base_key], thr_f)
        # store per-ratio metrics in long form below
        for r in COST_RATIOS:
            m = metrics_at_threshold(y_test, test_proba[base_key], thr_f[r], ratios=[r])
            point_metrics.append(
                {
                    "model": model_name,
                    "strategy": "formula_tau",
                    "ratio": r,
                    "ap": m["ap"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "threshold": m["threshold"],
                    f"cost_ratio{r}": m[f"cost_ratio{r}"],
                }
            )

        # CV-chosen tau from full-train probabilities of base model
        name = f"{model_name}/cv_tau"
        thr_cv = {
            r: choose_threshold(y_train, train_proba[base_key], r) for r in COST_RATIOS
        }
        methods[name] = (test_proba[base_key], thr_cv)
        for r in COST_RATIOS:
            m = metrics_at_threshold(y_test, test_proba[base_key], thr_cv[r], ratios=[r])
            point_metrics.append(
                {
                    "model": model_name,
                    "strategy": "cv_tau",
                    "ratio": r,
                    "ap": m["ap"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "threshold": m["threshold"],
                    f"cost_ratio{r}": m[f"cost_ratio{r}"],
                }
            )

        # class weights @0.5
        name = f"{model_name}/class_weight_balanced_0.5"
        methods[name] = (test_proba[bal_key], 0.5)
        m = metrics_at_threshold(y_test, test_proba[bal_key], 0.5)
        m.update({"model": model_name, "strategy": "class_weight_balanced_0.5"})
        point_metrics.append(m)

        # undersample @0.5
        name = f"{model_name}/undersample_0.5"
        methods[name] = (test_proba[und_key], 0.5)
        m = metrics_at_threshold(y_test, test_proba[und_key], 0.5)
        m.update({"model": model_name, "strategy": "undersample_0.5"})
        point_metrics.append(m)

    # Compact test table: one row per model x strategy x ratio
    rows = []
    for model_name in MODEL_ORDER:
        for strat in STRATEGY_ORDER:
            key = f"{model_name}/{strat}"
            proba, thr = methods[key]
            ap = float(average_precision_score(y_test, proba))
            for r in COST_RATIOS:
                t = thr[r] if isinstance(thr, dict) else thr
                pred = (proba >= t).astype(int)
                rows.append(
                    {
                        "model": model_name,
                        "strategy": strat,
                        "ratio": r,
                        "threshold": float(t),
                        "ap": ap,
                        "precision": float(
                            precision_score(y_test, pred, zero_division=0)
                        ),
                        "recall": float(recall_score(y_test, pred, zero_division=0)),
                        "cost": float(expected_cost(y_test, proba, r, t)),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "table_test_metrics.csv", index=False)
    print("\nTest point estimates:")
    print(table.to_string(index=False))

    # ---- paired stratified bootstrap ----
    print(f"\nPaired stratified bootstrap (n={N_BOOT})...")
    boot = paired_stratified_bootstrap(y_test, methods, n_boot=N_BOOT)
    with open(RESULTS_DIR / "draft_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump(boot, f, indent=2)

    pi_test = float(y_test.mean())
    naive = {
        "always_legitimate": {f"cost_ratio{r}": r * pi_test for r in COST_RATIOS},
        "always_fraud": {f"cost_ratio{r}": float(1.0 - pi_test) for r in COST_RATIOS},
    }

    summary = {
        "duplicates_dropped": n_dupes,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "frauds_train": int(y_train.sum()),
        "frauds_test": int(y_test.sum()),
        "feature_note": "Dropped raw Time; added Time_of_day = Time % 86400",
        "selected_params": {
            "lr_none": lr_none_params,
            "lr_balanced": lr_bal_params,
            "lr_undersample": lr_under_params,
            "hgb_none": hgb_none_params,
            "hgb_balanced": hgb_bal_params,
            "hgb_undersample": hgb_under_params,
        },
        "grids": {
            "lr_none": lr_none_grid,
            "lr_balanced": lr_bal_grid,
            "lr_undersample": lr_under_grid,
            "hgb_none": hgb_none_grid,
            "hgb_balanced": hgb_bal_grid,
            "hgb_undersample": hgb_under_grid,
        },
        "calibration_oof": calib,
        "cv_tau_selected": {
            model: {
                str(r): choose_threshold(y_train, train_proba[f"{model}:base"], r)
                for r in COST_RATIOS
            }
            for model in MODEL_ORDER
        },
        "naive_baselines_test": naive,
        "test_point_estimates": rows,
        "bootstrap": boot,
    }
    with open(RESULTS_DIR / "draft_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        RESULTS_DIR / "draft_test_predictions.npz",
        y_test=y_test,
        oof_lr=oof_lr,
        oof_hgb=oof_hgb,
        y_train=y_train,
        **{k.replace("/", "__").replace(":", "_"): v for k, v in test_proba.items()},
    )

    # ---- figures ----
    make_cost_ap_figure(table, boot, naive)
    make_calibration_figure(y_train, oof_lr, oof_hgb, calib)

    print(f"\nSaved artifacts under {RESULTS_DIR}")


def make_cost_ap_figure(table: pd.DataFrame, boot: dict, naive: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=130)
    x = np.arange(len(COST_RATIOS))
    n_strat = len(STRATEGY_ORDER)
    width = 0.075
    colors_lr = plt.cm.Blues(np.linspace(0.35, 0.85, n_strat))
    colors_hgb = plt.cm.Oranges(np.linspace(0.35, 0.85, n_strat))

    for mi, model in enumerate(MODEL_ORDER):
        colors = colors_lr if model == "logistic_regression" else colors_hgb
        short = "LR" if model == "logistic_regression" else "HGB"
        for si, strat in enumerate(STRATEGY_ORDER):
            means, los, his = [], [], []
            for r in COST_RATIOS:
                key = f"{model}/{strat}"
                b = boot[key][f"cost_ratio{r}"]
                means.append(b["mean"])
                los.append(b["mean"] - b["ci95_lo"])
                his.append(b["ci95_hi"] - b["mean"])
            # cluster by ratio; within each ratio: 5 LR then 5 HGB bars
            xpos = (
                x
                + (si - (n_strat - 1) / 2) * width
                + (mi - 0.5) * (n_strat * width + 0.02)
            )
            axes[0].bar(
                xpos,
                means,
                width,
                yerr=[los, his],
                capsize=2,
                color=colors[si],
                edgecolor="black",
                linewidth=0.3,
                label=f"{short}: {strat}",
            )

    axes[0].axhline(
        naive["always_legitimate"]["cost_ratio10"],
        color="gray",
        ls=":",
        lw=1,
        label="always-legit (r=10)",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"c_FN/c_FP = {r}" for r in COST_RATIOS])
    axes[0].set_ylabel("Expected cost per transaction")
    axes[0].set_title("Test expected cost (paired stratified bootstrap 95% CI)")
    axes[0].legend(fontsize=5.5, ncol=2, loc="upper left")
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)

    ap_labels, ap_means, ap_los, ap_his, ap_colors = [], [], [], [], []
    for model, cmap in [
        ("logistic_regression", colors_lr),
        ("hist_gradient_boosting", colors_hgb),
    ]:
        for si, strat in enumerate(STRATEGY_ORDER):
            key = f"{model}/{strat}"
            b = boot[key]["ap"]
            short = "LR" if model == "logistic_regression" else "HGB"
            ap_labels.append(f"{short}\n{strat.replace('_', ' ')}")
            ap_means.append(b["mean"])
            ap_los.append(b["mean"] - b["ci95_lo"])
            ap_his.append(b["ci95_hi"] - b["mean"])
            ap_colors.append(cmap[si])
    xpos = np.arange(len(ap_labels))
    axes[1].bar(
        xpos,
        ap_means,
        color=ap_colors,
        edgecolor="black",
        linewidth=0.3,
        yerr=[ap_los, ap_his],
        capsize=2,
    )
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(ap_labels, fontsize=6, rotation=90)
    axes[1].set_ylabel("Average Precision")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Test AP (secondary; threshold moves share scores)")
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        "Fraud repairs under class imbalance — test-set paired bootstrap",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_cost_ap.png", bbox_inches="tight")
    plt.close(fig)


def make_calibration_figure(y_train, oof_lr, oof_hgb, calib):
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=130)
    for name, proba, color in [
        ("Logistic regression", oof_lr, "#4C72B0"),
        ("HistGradientBoosting", oof_hgb, "#DD8452"),
    ]:
        mp, fp = reliability_bins(y_train, proba, n_bins=10)
        brier = calib[
            "logistic_regression" if "Logistic" in name else "hist_gradient_boosting"
        ]["brier"]
        ax.plot(mp, fp, marker="o", color=color, label=f"{name} (Brier={brier:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("OOF reliability (no-repair models)")
    ax.legend(fontsize=8)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_calibration.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
