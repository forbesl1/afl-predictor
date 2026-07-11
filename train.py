"""
train.py — trains an XGBoost classifier on the feature matrix.

XGBoost handles non-linear interactions between features (e.g. rest advantage
matters more when form is also close) and generally outperforms logistic
regression on tabular sports prediction tasks by 3–5%.

No feature scaling needed — tree-based models are scale-invariant.
"""
import pickle
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score

from features import FEATURE_COLS

MODEL_PATH = "model.pkl"

# Shared hyperparameters — imported by analyse.py to keep CV consistent with training
XGB_BASE = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)


def train(feature_df):
    X = feature_df[FEATURE_COLS].values
    y = feature_df["home_win"].values

    model = XGBClassifier(**XGB_BASE, eval_metric="logloss")

    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    accuracy = float(scores.mean())
    print(f"  CV accuracy: {accuracy:.1%}  (+/- {scores.std():.1%})")

    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "accuracy": accuracy}, f)

    return model, accuracy


def train_margin(feature_df):
    """
    Train an XGBRegressor to predict the home team's final point margin.
    Uses the same feature set as the classifier.
    Returns the fitted regressor.
    """
    X = feature_df[FEATURE_COLS].values
    y = feature_df["home_margin"].values

    regressor = XGBRegressor(**XGB_BASE)
    regressor.fit(X, y)
    return regressor


def stacked_oof_probs(clf_probs, margins, y, cv=5):
    """
    Out-of-fold probabilities of the stacked ensemble: a logistic regression
    over [P(home win), predicted margin], evaluated with a second-level CV so
    the stacker is never scored on rows it was fitted on.
    """
    Z = np.column_stack([clf_probs, margins])
    return cross_val_predict(LogisticRegression(), Z, y, cv=cv, method="predict_proba")[:, 1]


def train_ensemble(feature_df):
    """
    Train the full prediction stack:
      1. XGBClassifier (win) + XGBRegressor (margin), each fitted on all data
      2. A logistic-regression stacker fitted on their out-of-fold predictions,
         blending P(home win) and predicted margin into the final probability

    The margin regressor carries win signal the classifier misses (OOF 2015–2025:
    stacking lifts accuracy ~64.0% → ~64.7% and improves logloss/Brier; see
    docs/decisions/ensemble_stacked_probability.md).

    Returns (model, margin_model, stacker, accuracy) — accuracy is the stacked
    OOF accuracy, which is what the page displays.
    """
    X  = feature_df[FEATURE_COLS].values
    y  = feature_df["home_win"].values
    ym = feature_df["home_margin"].values

    # OOF predictions from both base models — no row sees its own training fold
    clf_probs = cross_val_predict(XGBClassifier(**XGB_BASE, eval_metric="logloss"),
                                  X, y, cv=5, method="predict_proba")[:, 1]
    margins   = cross_val_predict(XGBRegressor(**XGB_BASE), X, ym, cv=5)

    stack_oof = stacked_oof_probs(clf_probs, margins, y)
    accuracy  = float(((stack_oof >= 0.5).astype(int) == y).mean())
    clf_acc   = float(((clf_probs >= 0.5).astype(int) == y).mean())
    print(f"  OOF accuracy: classifier {clf_acc:.1%} -> stacked {accuracy:.1%}")

    stacker      = LogisticRegression().fit(np.column_stack([clf_probs, margins]), y)
    model        = XGBClassifier(**XGB_BASE, eval_metric="logloss").fit(X, y)
    margin_model = XGBRegressor(**XGB_BASE).fit(X, ym)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "margin_model": margin_model,
                     "stacker": stacker, "accuracy": accuracy}, f)

    return model, margin_model, stacker, accuracy


def load_model():
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["accuracy"]
