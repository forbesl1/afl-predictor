"""
predict.py — applies a trained model to upcoming game features.
"""
from features import FEATURE_COLS


def predict(pred_df, model, margin_model=None):
    """
    Returns pred_df with added columns:
      home_prob        — probability the home team wins
      away_prob        — probability the away team wins
      predicted_winner — team name of predicted winner
      confidence       — max(home_prob, away_prob)
      predicted_margin — predicted point margin from winner's perspective (if margin_model given)
    """
    if pred_df.empty:
        return pred_df

    X = pred_df[FEATURE_COLS].values
    # predict_proba returns [[P(class=0), P(class=1)], ...]
    # class 0 = away win, class 1 = home win
    probs = model.predict_proba(X)

    out = pred_df.copy()
    out["home_prob"] = probs[:, 1]
    out["away_prob"] = probs[:, 0]
    out["predicted_winner"] = out.apply(
        lambda r: r["home_team"] if r["home_prob"] >= 0.5 else r["away_team"],
        axis=1,
    )
    out["confidence"] = out[["home_prob", "away_prob"]].max(axis=1)

    if margin_model is not None:
        import numpy as np
        raw_margin = margin_model.predict(X)
        # raw_home_margin: positive = home team winning, negative = away team winning
        out["raw_home_margin"] = np.round(raw_margin).astype(int)
        # predicted_margin: magnitude of the regressor's margin
        out["predicted_margin"] = out["raw_home_margin"].abs()
        # margin_team: who the margin model favours. Disagrees with predicted_winner
        # (the classifier's pick) in ~15% of games (OOF) — display must say whose
        # margin this is rather than implying it belongs to the predicted winner.
        out["margin_team"] = np.where(
            out["raw_home_margin"] >= 0, out["home_team"], out["away_team"]
        )

    return out
