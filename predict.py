"""
predict.py — applies the trained models to upcoming game features.
"""
import numpy as np

from features import FEATURE_COLS


def predict(pred_df, model, margin_model=None, stacker=None):
    """
    Returns pred_df with added columns:
      home_prob        — probability the home team wins (stacked if stacker given)
      away_prob        — probability the away team wins
      predicted_winner — team name of predicted winner
      confidence       — max(home_prob, away_prob)
      predicted_margin — magnitude of the margin regressor's prediction (if margin_model given)
      margin_team      — team the margin model favours (may differ from predicted_winner)
    """
    if pred_df.empty:
        return pred_df

    X = pred_df[FEATURE_COLS].values
    # predict_proba returns [[P(class=0), P(class=1)], ...]
    # class 0 = away win, class 1 = home win
    home_prob = model.predict_proba(X)[:, 1]

    out = pred_df.copy()

    if margin_model is not None:
        raw_margin = margin_model.predict(X)
        if stacker is not None:
            # Final probability blends the classifier with the margin regressor —
            # the regressor carries win signal the classifier misses (see
            # docs/decisions/ensemble_stacked_probability.md)
            home_prob = stacker.predict_proba(np.column_stack([home_prob, raw_margin]))[:, 1]
        # raw_home_margin: positive = home team winning, negative = away team winning
        out["raw_home_margin"] = np.round(raw_margin).astype(int)
        # predicted_margin: magnitude of the regressor's margin
        out["predicted_margin"] = out["raw_home_margin"].abs()
        # margin_team: who the margin model favours. Can still disagree with the
        # stacked winner — display must say whose margin this is rather than
        # implying it belongs to the predicted winner.
        out["margin_team"] = np.where(
            out["raw_home_margin"] >= 0, out["home_team"], out["away_team"]
        )

    out["home_prob"] = home_prob
    out["away_prob"] = 1 - home_prob
    out["predicted_winner"] = np.where(home_prob >= 0.5, out["home_team"], out["away_team"])
    out["confidence"] = out[["home_prob", "away_prob"]].max(axis=1)

    return out
