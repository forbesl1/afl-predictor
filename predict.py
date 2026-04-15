"""
predict.py — applies a trained model to upcoming game features.
"""
from features import FEATURE_COLS


def predict(pred_df, model):
    """
    Returns pred_df with added columns:
      home_prob        — probability the home team wins
      away_prob        — probability the away team wins
      predicted_winner — team name of predicted winner
      confidence       — max(home_prob, away_prob)
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
    return out
