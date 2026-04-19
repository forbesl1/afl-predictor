"""
afl_tables.py — scrapes per-game team statistics from afltables.com.

For each season, fetches seas/{year}.html to get (round, game_url) pairs,
then scrapes each individual match page for team-level stat totals.

Column mapping (afltables abbreviation → feature key):
  IF  → I50   Inside 50s     — attacking pressure
  CL  → CL    Clearances     — midfield dominance
  DI  → D     Disposals      — possession volume
  TK  → T     Tackles        — defensive pressure

Completed seasons are cached to .cache/aflt_{year}.json. Individual game
pages are cached to .cache/game_{filename}.json so interrupted runs resume.
Falls back gracefully — pipeline continues with NaN for missing features.
"""
import datetime
import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup

CACHE_DIR = ".cache"
BASE_URL  = "https://afltables.com"
HEADERS   = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "text/html,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.5",
    "Referer":         "https://afltables.com/afl/afl_index.html",
}

STAT_MAP = {
    "IF": "I50",
    "CL": "CL",
    "DI": "D",
    "TK": "T",
}

# afltables team name → Squiggle team name where they differ
TEAM_NAME_FIXES = {
    "GWS Giants": "Greater Western Sydney",
    "Brisbane":   "Brisbane Lions",
}


def _afltables_to_squiggle(name):
    return TEAM_NAME_FIXES.get(name, name)


def _get(url, retries=2):
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                time.sleep(0.3)
                return resp.text
        except Exception:
            pass
        if attempt < retries:
            time.sleep(1.0)
    return None


def _fetch_season_game_links(year):
    """
    Scrape seas/{year}.html and return [(abs_url, filename), ...] for every
    game stat link. Round numbers are extracted from each game page instead.
    """
    html = _get(f"{BASE_URL}/afl/seas/{year}.html")
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    seen = set()
    games = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if f"stats/games/{year}/" in href:
            filename = href.split("/")[-1]
            if filename not in seen:
                seen.add(filename)
                games.append((f"{BASE_URL}/afl/stats/games/{year}/{filename}", filename))
    return games


def _row_cells_with_colspan(tr):
    """
    Return a flat list of cell texts from a <tr>, expanding colspan so that
    each physical column position maps correctly to the header at that index.
    """
    cells = []
    for td in tr.find_all(["th", "td"]):
        text = td.get_text(strip=True)
        span = int(td.get("colspan", 1))
        cells.append(text)
        for _ in range(span - 1):
            cells.append("")
    return cells


def _parse_game_stats(html):
    """
    Parse a match stats page and return:
      (round_num, {afltables_team_name: {I50, CL, D, T}})
    round_num is None for finals or when the round can't be parsed (caller skips those).
    """
    soup = BeautifulSoup(html, "lxml")

    # Round number: <b>Round: </b>1  — text node immediately after the <b> tag
    round_num = None
    round_b = soup.find("b", string=re.compile(r"Round:\s*$"))
    if round_b and round_b.next_sibling:
        m = re.match(r"\s*(\d+)", str(round_b.next_sibling))
        if m:
            round_num = int(m.group(1))

    result = {}

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 3:
            continue

        # Team name is in the first row: "Melbourne Match Statistics [Season][Game by Game]"
        first_row_text = re.sub(r"\s+", " ", rows[0].get_text()).strip()
        if "Match Statistics" not in first_row_text:
            continue
        before = first_row_text[:first_row_text.index("Match Statistics")].strip()
        team_name = re.sub(r"\b(Season|Game\s*by\s*Game)\b", "", before, flags=re.IGNORECASE).strip()
        if not team_name:
            continue

        # Header row: first row (after the team name row) with stat abbreviations in <th> cells
        header_map = {}
        header_idx = None
        for i, tr in enumerate(rows[1:], 1):
            ths = [th.get_text(strip=True) for th in tr.find_all("th")]
            matched = {col: ths.index(col) for col in STAT_MAP if col in ths}
            if matched:
                header_map = matched
                header_idx = i
                break

        if not header_map:
            continue

        # Totals row
        totals_row = None
        for tr in rows[header_idx + 1:]:
            first = tr.find(["th", "td"])
            if first and re.match(r"Totals?$", first.get_text(strip=True), re.IGNORECASE):
                totals_row = tr
                break

        if totals_row is None:
            continue

        # The Totals label cell is a single <th> with no colspan, while the header has
        # separate # and Player columns — this creates a 1-position offset.
        # If the label has colspan >= 2 it already spans both, so offset is 0.
        label_cell = totals_row.find(["th", "td"])
        offset = 0 if int(label_cell.get("colspan", 1)) >= 2 else 1

        cells = _row_cells_with_colspan(totals_row)
        stats = {}
        for col, feat in STAT_MAP.items():
            if col in header_map:
                idx = header_map[col] - offset
                if 0 <= idx < len(cells):
                    try:
                        stats[feat] = float(cells[idx])
                    except (ValueError, TypeError):
                        pass

        if stats:
            result[team_name] = stats

    return round_num, result


def fetch_season_stats(year, verbose=True):
    """
    Fetch team-game stats for one season.
    Returns: {(squiggle_team_name, year, round_num): {I50, CL, D, T}}
    Caches completed seasons to .cache/aflt_{year}.json.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    current_year = datetime.date.today().year
    cache_path = os.path.join(CACHE_DIR, f"aflt_{year}.json") if year < current_year else None

    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
        return {(t, y, r): stats for (t, y, r), stats in
                ((json.loads(k), v) for k, v in raw.items())}

    if verbose:
        print(f"  Fetching afltables {year}...", end=" ", flush=True)

    game_links = _fetch_season_game_links(year)
    if not game_links:
        if verbose:
            print("FAILED (no game links found)")
        return {}

    lookup = {}
    for url, filename in game_links:
        game_cache = os.path.join(CACHE_DIR, f"game_{filename}.json")
        if os.path.exists(game_cache):
            with open(game_cache) as f:
                cached = json.load(f)
            round_num  = cached.get("round_num")
            game_stats = cached.get("stats", {})
        else:
            html = _get(url)
            if not html:
                continue
            round_num, game_stats = _parse_game_stats(html)
            if game_stats:
                with open(game_cache, "w") as f:
                    json.dump({"round_num": round_num, "stats": game_stats}, f)

        if round_num is None:
            continue  # finals — not used as features
        for aflt_team, stats in game_stats.items():
            squiggle_team = _afltables_to_squiggle(aflt_team)
            lookup[(squiggle_team, year, round_num)] = stats

    if verbose:
        print(f"{len(lookup)} team-game entries")

    if cache_path and lookup:
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
