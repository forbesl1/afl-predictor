"""
features.py — converts raw game data into a feature matrix for modelling.

Features computed for each game:
  home_form        — home team win rate in last 5 games
  away_form        — away team win rate in last 5 games
  form_diff        — home_form - away_form
  h2h              — home team win rate vs this specific away team (all history)
  home_venue       — home team win rate at this venue (all history)
  home_avg_margin  — home team average score margin in last 5 games (signed)
  away_avg_margin  — away team average score margin in last 5 games (signed)
  margin_diff      — home_avg_margin - away_avg_margin
  home_days_rest   — days since home team's last game
  away_days_rest   — days since away team's last game
  rest_diff        — home_days_rest - away_days_rest
  home_ladder_pct  — home team season win % up to this game
  away_ladder_pct  — away team season win % up to this game
  ladder_diff      — home_ladder_pct - away_ladder_pct
  tipster_consensus — fraction of Squiggle tipsters backing home team (0.5 if unavailable)
"""
import numpy as np
import pandas as pd

FORM_N = 5  # number of recent games used for form/margin calculation

# Elo constants (tuned for AFL)
ELO_K        = 40     # update factor per game
ELO_HOME_ADV = 65     # home ground advantage in Elo points
ELO_MEAN     = 1500   # starting / mean rating
ELO_REGRESS  = 0.3    # fraction of gap from mean regressed away each off-season

FEATURE_COLS = [
    "home_form",       "away_form",       "form_diff",
    "h2h",             "home_venue",
    "home_avg_margin", "away_avg_margin", "margin_diff",
    "home_days_rest",  "away_days_rest",  "rest_diff",
    "home_ladder_pct", "away_ladder_pct", "ladder_diff",
    "tipster_consensus",
    "home_elo",        "away_elo",        "elo_diff",        "elo_win_prob",
]


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
    df["home_win"]    = (df["hscore"] > df["ascore"]).astype(int)
    df["home_margin"] = df["hscore"] - df["ascore"]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _team_results(df, team):
    """
    All completed games involving team with derived columns:
      won    — 1 if team won, 0 if lost
      margin — signed score difference from team's perspective (positive = winning)
    """
    mask = (df["hteam"] == team) | (df["ateam"] == team)
    sub = df[mask].copy()
    is_home = sub["hteam"] == team
    sub["won"]    = np.where(is_home, sub["home_win"],             1 - sub["home_win"])
    sub["margin"] = np.where(is_home, sub["hscore"] - sub["ascore"],
                                      sub["ascore"] - sub["hscore"])
    return sub


def _form(df, team, before_date, n=FORM_N):
    """Win rate of team in their last n games before before_date."""
    results = _team_results(df, team)
    results = results[results["date"] < before_date].tail(n)
    return float(results["won"].mean()) if len(results) > 0 else 0.5


def _avg_margin(df, team, before_date, n=FORM_N):
    """Average signed margin in team's last n games before before_date."""
    results = _team_results(df, team)
    results = results[results["date"] < before_date].tail(n)
    return float(results["margin"].mean()) if len(results) > 0 else 0.0


def _days_rest(df, team, before_date):
    """Days since team's last game before before_date. Defaults to 7 if no history."""
    results = _team_results(df, team)
    results = results[results["date"] < before_date]
    if len(results) == 0:
        return 7
    last_game = results["date"].max()
    delta = before_date - last_game
    return int(delta.days)


def _ladder_pct(df, team, before_date):
    """Team's season win percentage up to before_date (all-time, not per season)."""
    results = _team_results(df, team)
    results = results[results["date"] < before_date]
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


def _elo_expected(home_elo, away_elo):
    """Expected win probability for the home team given both Elo ratings."""
    return 1 / (1 + 10 ** ((away_elo - home_elo - ELO_HOME_ADV) / 400))


def compute_elo(df):
    """
    Compute pre-game Elo ratings for every completed game in df.
    df must already be sorted by date ascending (to_df() guarantees this).

    Returns:
        elo_by_idx  — {df_index: (home_elo, away_elo)} — used for training features
        current_elo — {team: elo} after all games — used for prediction features
    """
    ratings    = {}
    elo_by_idx = {}
    prev_year  = None

    for idx, row in df.iterrows():
        home = row["hteam"]
        away = row["ateam"]
        year = row["date"].year

        if home not in ratings:
            ratings[home] = ELO_MEAN
        if away not in ratings:
            ratings[away] = ELO_MEAN

        # Partial regression toward mean at the start of each new season
        if prev_year is not None and year != prev_year:
            for team in list(ratings):
                ratings[team] = ELO_MEAN + (1 - ELO_REGRESS) * (ratings[team] - ELO_MEAN)
        prev_year = year

        home_elo = ratings[home]
        away_elo = ratings[away]
        elo_by_idx[idx] = (home_elo, away_elo)

        # Update ratings from result
        expected = _elo_expected(home_elo, away_elo)
        actual   = float(row["home_win"])
        ratings[home] += ELO_K * (actual - expected)
        ratings[away] += ELO_K * ((1 - actual) - (1 - expected))

    return elo_by_idx, ratings


def build_training_features(df, tips_lookup=None, elo_lookup=None):
    """
    Build a feature + label DataFrame from all completed games in df.
    Each row uses only games *prior* to that game's date (no leakage).

    tips_lookup: {game_id: home_consensus} from Squiggle tipsters.
    elo_lookup:  {df_index: (home_elo, away_elo)} from compute_elo().
    """
    if tips_lookup is None:
        tips_lookup = {}
    if elo_lookup is None:
        elo_lookup = {}

    rows = []
    for idx, row in df.iterrows():
        home    = row["hteam"]
        away    = row["ateam"]
        date    = row["date"]
        venue   = row["venue"]
        game_id = row.get("id")

        hf  = _form(df, home, date)
        af  = _form(df, away, date)
        hm  = _avg_margin(df, home, date)
        am  = _avg_margin(df, away, date)
        hr  = _days_rest(df, home, date)
        ar  = _days_rest(df, away, date)
        hl  = _ladder_pct(df, home, date)
        al  = _ladder_pct(df, away, date)

        home_elo, away_elo = elo_lookup.get(idx, (ELO_MEAN, ELO_MEAN))

        rows.append({
            "home_form":        hf,
            "away_form":        af,
            "form_diff":        hf - af,
            "h2h":              _h2h(df, home, away, date),
            "home_venue":       _venue_rate(df, home, venue, date),
            "home_avg_margin":  hm,
            "away_avg_margin":  am,
            "margin_diff":      hm - am,
            "home_days_rest":   hr,
            "away_days_rest":   ar,
            "rest_diff":        hr - ar,
            "home_ladder_pct":  hl,
            "away_ladder_pct":  al,
            "ladder_diff":      hl - al,
            "tipster_consensus": tips_lookup.get(game_id, 0.5),
            "home_elo":         home_elo,
            "away_elo":         away_elo,
            "elo_diff":         home_elo - away_elo,
            "elo_win_prob":     _elo_expected(home_elo, away_elo),
            "home_win":         int(row["home_win"]),
            "home_margin":      int(row["home_margin"]),
        })

    return pd.DataFrame(rows)


def build_prediction_features(df, upcoming_games, tips_lookup=None, current_elo=None):
    """
    Build a feature DataFrame for upcoming (incomplete) games.
    Uses all available historical data as context.

    current_elo: {team: elo} from compute_elo() — ratings after all training games.
    """
    if tips_lookup is None:
        tips_lookup = {}
    if current_elo is None:
        current_elo = {}

    now  = pd.Timestamp.now(tz="UTC")
    rows = []

    for g in upcoming_games:
        home    = (g.get("hteam")  or "").strip()
        away    = (g.get("ateam")  or "").strip()
        venue   = (g.get("venue")  or "").strip()
        game_id = g.get("id")
        if not home or not away:
            continue

        hf  = _form(df, home, now)
        af  = _form(df, away, now)
        hm  = _avg_margin(df, home, now)
        am  = _avg_margin(df, away, now)
        hr  = _days_rest(df, home, now)
        ar  = _days_rest(df, away, now)
        hl  = _ladder_pct(df, home, now)
        al  = _ladder_pct(df, away, now)

        home_elo = current_elo.get(home, ELO_MEAN)
        away_elo = current_elo.get(away, ELO_MEAN)

        rows.append({
            "home_team":        home,
            "away_team":        away,
            "venue":            venue,
            "round":            g.get("round"),
            "roundname":        g.get("roundname") or f"Round {g.get('round')}",
            "date":             g.get("date", ""),
            "game_id":          game_id,
            "home_form":        hf,
            "away_form":        af,
            "form_diff":        hf - af,
            "h2h":              _h2h(df, home, away, now),
            "home_venue":       _venue_rate(df, home, venue, now),
            "home_avg_margin":  hm,
            "away_avg_margin":  am,
            "margin_diff":      hm - am,
            "home_days_rest":   hr,
            "away_days_rest":   ar,
            "rest_diff":        hr - ar,
            "home_ladder_pct":  hl,
            "away_ladder_pct":  al,
            "ladder_diff":      hl - al,
            "tipster_consensus": tips_lookup.get(game_id, 0.5),
            "home_elo":         home_elo,
            "away_elo":         away_elo,
            "elo_diff":         home_elo - away_elo,
            "elo_win_prob":     _elo_expected(home_elo, away_elo),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()
