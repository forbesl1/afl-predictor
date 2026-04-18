"""
afl_tables.py — downloads per-game team statistics from afltables.com.

Uses the season player stats CSV (https://afltables.com/afl/stats/{YEAR}_stats.txt),
which contains one row per player per game for an entire season. These are
summed by (team, round) to produce team-level totals per game, which are
then used as rolling-average features in the prediction model.

Column mapping (afltables abbreviation → our key):
  IF  → I50   Inside 50s     — attacking pressure
  CL  → CL    Clearances     — midfield dominance
  DI  → D     Disposals      — possession volume
  TK  → T     Tackles        — defensive pressure

Completed seasons are cached to .cache/. Falls back gracefully (returns {})
if the download fails — the pipeline continues without these features and
XGBoost treats missing values as NaN.
"""
import datetime
import io
import json
import os
import time

import pandas as pd
import requests

CACHE_DIR = ".cache"
HEADERS   = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "text/plain,text/html,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.5",
    "Referer":         "https://afltables.com/afl/afl_index.html",
}

# afltables CSV column → feature key
STAT_MAP = {
    "IF": "I50",   # Inside 50s
    "CL": "CL",    # Clearances
    "DI": "D",     # Disposals
    "TK": "T",     # Tackles
}

# Squiggle team name → afltables team name where they differ
# Add entries here if the first run logs "unmatched teams"
TEAM_NAME_FIXES = {
    "Greater Western Sydney": "GWS Giants",
    "Brisbane Lions":         "Brisbane",
}


def _squiggle_to_afltables(name):
    return TEAM_NAME_FIXES.get(name, name)


def _afltables_to_squiggle(name):
    reverse = {v: k for k, v in TEAM_NAME_FIXES.items()}
    return reverse.get(name, name)


def _fetch_season_csv(year):
    """
    Download the afltables season player stats CSV for a given year.
    Returns a pandas DataFrame, or None on failure.
    """
    url = f"https://afltables.com/afl/stats/{year}_stats.txt"
    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        if "Broked" in resp.text[:200]:
            return None
        # Try comma-separated first, then tab-separated
        for sep in (",", "\t"):
            try:
                df = pd.read_csv(io.StringIO(resp.text), sep=sep, low_memory=False)
                if len(df.columns) > 5:
                    return df
            except Exception:
                continue
        return None
    except Exception:
        return None


def _aggregate_to_team_games(df, year):
    """
    Sum player stats by (team, round) to get team totals per game.
    Returns: {(squiggle_team_name, year, round_num): {I50, CL, D, T}}

    Prints a diagnostic on the first call so URL/column issues are visible.
    """
    # Find the team and round columns (column names vary slightly by year)
    team_col  = next((c for c in df.columns if c.strip().lower() in ("team", "tm")), None)
    round_col = next((c for c in df.columns if c.strip().lower() in ("rnd", "round", "rd", "r")), None)

    if team_col is None or round_col is None:
        print(f"  [afltables] Could not find team/round columns in {year} CSV. Columns: {list(df.columns[:15])}")
        return {}

    # Check which stat columns are present
    available = {k: v for k, v in STAT_MAP.items() if k in df.columns}
    if not available:
        print(f"  [afltables] No stat columns found in {year} CSV. Columns: {list(df.columns[:20])}")
        return {}

    # Convert stat columns to numeric, coerce errors to NaN
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to regular-season rounds (integer round numbers only; skip finals)
    df[round_col] = pd.to_numeric(df[round_col], errors="coerce")
    df = df.dropna(subset=[round_col])
    df[round_col] = df[round_col].astype(int)

    agg_dict = {k: "sum" for k in available}
    grouped = df.groupby([team_col, round_col]).agg(agg_dict).reset_index()

    lookup = {}
    for _, row in grouped.iterrows():
        aflt_team = str(row[team_col]).strip()
        squiggle_team = _afltables_to_squiggle(aflt_team)
        rnd = int(row[round_col])
        stats = {available[k]: (row[k] if pd.notna(row[k]) else None) for k in available}
        lookup[(squiggle_team, year, rnd)] = stats

    return lookup


def fetch_season_stats(year, verbose=True):
    """
    Fetch team-game stats for one season.
    Returns: {(squiggle_team_name, year, round_num): {I50, CL, D, T}}
    Caches completed seasons to .cache/.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    current_year = datetime.date.today().year
    cache_path = os.path.join(CACHE_DIR, f"aflt_{year}.json") if year < current_year else None

    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
        # JSON keys are strings; restore tuple keys
        return {(t, y, r): stats for (t, y, r), stats in
                ((json.loads(k), v) for k, v in raw.items())}

    if verbose:
        print(f"  Fetching afltables {year}...", end=" ", flush=True)

    df = _fetch_season_csv(year)
    if df is None:
        if verbose:
            print("FAILED (download error)")
        return {}

    lookup = _aggregate_to_team_games(df, year)
    time.sleep(0.5)

    if verbose:
        print(f"{len(lookup)} team-game entries")

    if cache_path and lookup:
        # Serialize tuple keys as JSON strings
        with open(cache_path, "w") as f:
            json.dump({json.dumps(list(k)): v for k, v in lookup.items()}, f)

    return lookup


def build_stats_lookup(start_year, end_year, verbose=True):
    """
    Fetch team-game stats for a range of seasons.
    Returns: {(squiggle_team_name, year, round_num): {I50, CL, D, T}}
    """
    combined = {}
    for year in range(start_year, end_year + 1):
        season = fetch_season_stats(year, verbose=verbose)
        combined.update(season)
    return combined
