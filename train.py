"""
train.py — trains an XGBoost classifier on the feature matrix.

XGBoost handles non-linear interactions between features (e.g. rest advantage
matters more when form is also close) and generally outperforms logistic
regression on tabular sports prediction tasks by 3–5%.

No feature scaling needed — tree-based models are scale-invariant.
"""
import pickle
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import cross_val_score

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


def load_model():
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["accuracy"]
