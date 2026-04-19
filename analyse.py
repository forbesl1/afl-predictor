"""
analyse.py — on-demand model performance analysis.

Fetches data, builds features, runs cross-validation, and generates:
  - Printed summary: per-fold accuracy, AUC, margin MAE
  - analysis/feature_importance.png
  - analysis/calibration.png
  - analysis/season_accuracy.png
  - analysis/confidence_tiers.png
  - analysis/margin_scatter.png

All accuracy metrics use out-of-fold (OOF) predictions to avoid optimistic
bias from evaluating on training data. Feature importance uses the final
model trained on the full dataset.

Usage:
    python analyse.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from xgboost import XGBClassifier, XGBRegressor

from fetch_data import fetch_training_games, fetch_all_tips
from features import FEATURE_COLS, build_training_features, compute_elo, to_df
from pipeline import START_YEAR, END_YEAR, _build_tips_lookup
from train import XGB_BASE, train, train_margin
from afl_tables import build_stats_lookup

ANALYSIS_DIR = "analysis"
CV_FOLDS     = 5

BLUE = "#0055A5"
NAVY = "#002B5C"


def _cv_split():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)


def _oof_clf_probs(X, y):
    clf = XGBClassifier(**XGB_BASE, eval_metric="logloss")
    probs = cross_val_predict(clf, X, y, cv=_cv_split(), method="predict_proba")
    return probs[:, 1]  # P(home win)


def _oof_margin_preds(X, y_margin, cv):
    reg = XGBRegressor(**XGB_BASE)
    return cross_val_predict(reg, X, y_margin, cv=cv)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_feature_importance(model, out_dir):
    raw = model.get_booster().get_score(importance_type="gain")
    # When trained on numpy arrays XGBoost names features f0, f1, ...
    importance = pd.Series(
        {FEATURE_COLS[int(k[1:])]: v for k, v in raw.items()},
    ).sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(8, 7))
    importance.plot.barh(ax=ax, color=BLUE)
    ax.set_title("Feature Importance — top 20 by gain", fontweight="bold")
    ax.set_xlabel("Gain")
    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_calibration(y_true, y_prob, out_dir):
    bins = np.linspace(0, 1, 11)
    labels = np.arange(10)
    bin_idx = pd.cut(y_prob, bins=bins, labels=labels, include_lowest=True).astype(float)

    centers, actuals = [], []
    for i in range(10):
        mask = bin_idx == i
        if mask.sum() >= 10:
            centers.append(bins[i] + 0.05)
            actuals.append(float(y_true[mask].mean()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(centers, actuals, "o-", color=BLUE, linewidth=2, markersize=7, label="Model")
    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Actual home win rate")
    ax.set_title("Calibration Curve", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "calibration.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_season_accuracy(y_true, y_prob, years, out_dir):
    df = pd.DataFrame({"correct": (y_prob >= 0.5).astype(int) == y_true, "year": years})
    acc = df.groupby("year")["correct"].mean()
    mean_acc = acc.mean()

    colors = [BLUE if v >= mean_acc else "#c0392b" for v in acc]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(acc.index.astype(str), acc.values, color=colors, edgecolor="white")
    ax.axhline(mean_acc, color="black", linestyle="--", linewidth=1,
               label=f"Mean {mean_acc:.1%}")
    for bar, v in zip(bars, acc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Season-by-Season Accuracy (OOF)", fontweight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 0.85)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "season_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confidence_tiers(y_true, y_prob, out_dir):
    confidence = np.maximum(y_prob, 1 - y_prob)
    predicted  = (y_prob >= 0.5).astype(int)
    correct    = (predicted == y_true).astype(int)

    tiers = [
        ("High\n(≥70%)",     confidence >= 0.70,                          "#1e7e34"),
        ("Medium\n(58–69%)", (confidence >= 0.58) & (confidence < 0.70), "#856404"),
        ("Low\n(<58%)",       confidence < 0.58,                          "#c0392b"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (label, mask, color) in enumerate(tiers):
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        bar = ax.bar(label, acc, color=color, edgecolor="white", width=0.5)
        ax.text(i, acc + 0.01, f"{acc:.1%}\n(n={mask.sum():,})",
                ha="center", va="bottom", fontsize=10)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Coin flip (50%)")
    ax.set_title("Accuracy by Confidence Tier (OOF)", fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "confidence_tiers.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_margin_scatter(y_margin_true, y_margin_pred, out_dir):
    mae = mean_absolute_error(y_margin_true, y_margin_pred)
    lim = int(max(np.abs(y_margin_true).max(), np.abs(y_margin_pred).max())) + 10

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_margin_true, y_margin_pred, alpha=0.15, s=8, color=BLUE)
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, label="Perfect prediction")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"Margin Prediction — OOF (MAE = {mae:.1f} pts)", fontweight="bold")
    ax.set_xlabel("Actual margin (home team perspective)")
    ax.set_ylabel("Predicted margin (home team perspective)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "margin_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(y_true, y_prob_oof, cv_scores, y_margin_true, y_margin_pred):
    acc  = ((y_prob_oof >= 0.5).astype(int) == y_true).mean()
    auc  = roc_auc_score(y_true, y_prob_oof)
    mae  = mean_absolute_error(y_margin_true, y_margin_pred)

    print("\n=== Model Performance Summary ===\n")
    print(f"  Win classifier ({CV_FOLDS}-fold OOF):")
    print(f"    Accuracy  : {acc:.1%}")
    print(f"    AUC       : {auc:.3f}")
    print(f"\n  Per-fold accuracy:")
    for i, s in enumerate(cv_scores, 1):
        print(f"    Fold {i}: {s:.1%}")
    print(f"    Mean ± SD : {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    print(f"\n  Margin regressor (OOF):")
    print(f"    MAE       : {mae:.1f} pts")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    print("=== AFL Predictor — Analysis ===\n")

    print(f"[1/4] Fetching data ({START_YEAR}–{END_YEAR})...")
    raw          = fetch_training_games(START_YEAR, END_YEAR)
    df           = to_df(raw)
    all_tips     = fetch_all_tips(START_YEAR, END_YEAR)
    tips_lookup  = _build_tips_lookup(all_tips)
    stats_lookup = build_stats_lookup(START_YEAR, END_YEAR)
    print(f"  Completed games: {len(df)}")

    print("\n[2/4] Building features...")
    elo_lookup, _ = compute_elo(df)
    feat_df = build_training_features(
        df, tips_lookup=tips_lookup,
        elo_lookup=elo_lookup, stats_lookup=stats_lookup,
    )
    print(f"  Feature rows: {len(feat_df)}")

    X        = feat_df[FEATURE_COLS].values
    y        = feat_df["home_win"].values
    y_margin = feat_df["home_margin"].values
    years    = df["date"].dt.year.values

    print("\n[3/4] Running cross-validation (this takes a few minutes)...")
    cv           = _cv_split()
    y_prob_oof   = _oof_clf_probs(X, y)
    cv_scores    = cross_val_score(XGBClassifier(**XGB_BASE, eval_metric="logloss"),
                                   X, y, cv=cv, scoring="accuracy")
    y_margin_oof = _oof_margin_preds(X, y_margin, cv)

    print("\n[4/4] Training final model + generating plots...")
    model, _ = train(feat_df)

    print_summary(y, y_prob_oof, cv_scores, y_margin, y_margin_oof)

    print("Saving plots...")
    plot_feature_importance(model, ANALYSIS_DIR)
    plot_calibration(y, y_prob_oof, ANALYSIS_DIR)
    plot_season_accuracy(y, y_prob_oof, years, ANALYSIS_DIR)
    plot_confidence_tiers(y, y_prob_oof, ANALYSIS_DIR)
    plot_margin_scatter(y_margin, y_margin_oof, ANALYSIS_DIR)

    print(f"\nDone. All plots saved to {ANALYSIS_DIR}/")


if __name__ == "__main__":
    run()
