"""
train.py — trains a logistic regression model on the feature matrix.

The model is a sklearn Pipeline: StandardScaler → LogisticRegression.
Scaling is important because feature ranges differ (probabilities 0–1 vs raw counts).

Cross-validation accuracy is printed and saved alongside the model so the
HTML output can display it as a trust indicator.
"""
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLS

MODEL_PATH = "model.pkl"


def train(feature_df):
    X = feature_df[FEATURE_COLS].values
    y = feature_df["home_win"].values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
    ])

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
