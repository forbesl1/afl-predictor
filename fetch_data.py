"""
fetch_data.py — pulls game data from the Squiggle API.
https://api.squiggle.com.au

Completed seasons are cached to .cache/ to avoid redundant API calls.
The current season is never cached (data is still being added).
"""
import json
import os
import time
import datetime
import requests

BASE_URL  = "https://api.squiggle.com.au/"
CACHE_DIR = ".cache"
HEADERS   = {"User-Agent": "afl-predictor/1.0 (github.com/forbesl1/afl-predictor)"}


def _fetch(params, cache_key=None):
    os.makedirs(CACHE_DIR, exist_ok=True)

    if cache_key:
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

    # Build query string manually — requests would URL-encode semicolons,
    # but the Squiggle API requires them unencoded (e.g. ?q=games;year=2022).
    query = "&".join(f"{k}={v}" for k, v in params.items())
    resp = requests.get(f"{BASE_URL}?{query}", headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    time.sleep(0.5)  # be polite to the API

    if cache_key:
        with open(cache_path, "w") as f:
            json.dump(data, f)

    return data


def fetch_season(year):
    """All games for a given year. Caches completed seasons only."""
    current_year = datetime.date.today().year
    cache_key = f"games_{year}" if year < current_year else None
    data = _fetch({"q": f"games;year={year}"}, cache_key=cache_key)
    return data.get("games", [])


def fetch_upcoming():
    """Incomplete games for the current year (never cached)."""
    year = datetime.date.today().year
    data = _fetch({"q": f"games;year={year};complete=0"})
    return data.get("games", [])


def fetch_training_games(start_year, end_year):
    """Fetch all games across a range of years."""
    games = []
    for year in range(start_year, end_year + 1):
        season = fetch_season(year)
        print(f"  {year}: {len(season)} games")
        games.extend(season)
    return games
