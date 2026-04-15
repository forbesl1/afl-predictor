"""
features.py — converts raw game data into a feature matrix for modelling.

Features computed for each game:
  home_form   — home team win rate in last 5 games
  away_form   — away team win rate in last 5 games
  form_diff   — home_form - away_form (composite signal)
  h2h         — home team win rate vs this specific away team (all history)
  home_venue  — home team win rate at this venue (all history)
"""
import numpy as np
import pandas as pd

FORM_N = 5  # number of recent games used for form calculation

FEATURE_COLS = ["home_form", "away_form", "form_diff", "h2h", "home_venue"]


def to_df(games):
    """Convert raw Squiggle game list to a cleaned DataFrame of completed games."""
    df = pd.DataFrame(games)
    if df.empty:
        return df

    df = df[df["complete"] == 100].copy()
    df = df.dropna(subset=["hscore", "ascore"])
    df["hscore"] = pd.to_numeric(df["hscore"], errors="coerce")
    df["ascore"] = pd.to_numeric(df["ascore"], errors="coerce")
    df = df.dropna(subset=["hscore", "ascore"])

    # home_win: 1 if home team won, 0 if away team won. Draws (~0 in modern AFL) → 0.
    df["home_win"] = (df["hscore"] > df["ascore"]).astype(int)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _team_results(df, team):
    """
    All completed games involving team, with a 'won' column:
      1 = team won, 0 = team lost.
    """
    mask = (df["hteam"] == team) | (df["ateam"] == team)
    sub = df[mask].copy()
    sub["won"] = np.where(sub["hteam"] == team, sub["home_win"], 1 - sub["home_win"])
    return sub


def _form(df, team, before_date, n=FORM_N):
    """Win rate of team in their last n games before before_date."""
    results = _team_results(df, team)
    results = results[results["date"] < before_date].tail(n)
    return float(results["won"].mean()) if len(results) > 0 else 0.5


def _h2h(df, home_team, away_team, before_date):
    """Home team's historical win rate against away team before before_date."""
    mask = (
        ((df["hteam"] == home_team) & (df["ateam"] == away_team)) |
        ((df["hteam"] == away_team) & (df["ateam"] == home_team))
    ) & (df["date"] < before_date)
    sub = df[mask]
    if len(sub) == 0:
        return 0.5
    wins = (
        ((sub["hteam"] == home_team) & (sub["home_win"] == 1)).sum() +
        ((sub["ateam"] == home_team) & (sub["home_win"] == 0)).sum()
    )
    return float(wins / len(sub))


def _venue_rate(df, team, venue, before_date):
    """Team's win rate at a specific venue before before_date."""
    results = _team_results(df, team)
    results = results[(results["venue"] == venue) & (results["date"] < before_date)]
    return float(results["won"].mean()) if len(results) > 0 else 0.5


def build_training_features(df):
    """
    Build a feature + label DataFrame from all completed games in df.
    Each row uses only games *prior* to that game's date (no leakage).
    """
    rows = []
    for _, row in df.iterrows():
        home  = row["hteam"]
        away  = row["ateam"]
        date  = row["date"]
        venue = row["venue"]

        hf = _form(df, home, date)
        af = _form(df, away, date)
        rows.append({
            "home_form":  hf,
            "away_form":  af,
            "form_diff":  hf - af,
            "h2h":        _h2h(df, home, away, date),
            "home_venue": _venue_rate(df, home, venue, date),
            "home_win":   int(row["home_win"]),
        })

    return pd.DataFrame(rows)


def build_prediction_features(df, upcoming_games):
    """
    Build a feature DataFrame for a list of upcoming (incomplete) games.
    Uses all available historical data (no date cutoff — the future hasn't happened yet).
    """
    now = pd.Timestamp.now(tz="UTC")
    rows = []

    for g in upcoming_games:
        home  = (g.get("hteam")  or "").strip()
        away  = (g.get("ateam")  or "").strip()
        venue = (g.get("venue")  or "").strip()
        if not home or not away:
            continue

        hf = _form(df, home, now)
        af = _form(df, away, now)
        rows.append({
            "home_team":  home,
            "away_team":  away,
            "venue":      venue,
            "round":      g.get("round"),
            "roundname":  g.get("roundname") or f"Round {g.get('round')}",
            "date":       g.get("date", ""),
            "home_form":  hf,
            "away_form":  af,
            "form_diff":  hf - af,
            "h2h":        _h2h(df, home, away, now),
            "home_venue": _venue_rate(df, home, venue, now),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()
