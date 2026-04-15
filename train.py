"""
train.py — trains an XGBoost classifier on the feature matrix.

XGBoost handles non-linear interactions between features (e.g. rest advantage
matters more when form is also close) and generally outperforms logistic
regression on tabular sports prediction tasks by 3–5%.

No feature scaling needed — tree-based models are scale-invariant.
"""
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

from features import FEATURE_COLS

MODEL_PATH = "model.pkl"


def train(feature_df):
    X = feature_df[FEATURE_COLS].values
    y = feature_df["home_win"].values

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    accuracy = float(scores.mean())
    print(f"  CV accuracy: {accuracy:.1%}  (+/- {scores.std():.1%})")

    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "accuracy": accuracy}, f)

    return model, accuracy


def load_model():
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["accuracy"]
