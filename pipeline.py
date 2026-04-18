"""
pipeline.py — orchestrates the full fetch → train → predict → publish flow.

Run locally:  python pipeline.py
GitHub Actions runs this on a cron schedule each Thursday morning AEST.
"""
import datetime
import os

import pandas as pd

from collections import defaultdict

from fetch_data import fetch_all_tips, fetch_training_games, fetch_upcoming, fetch_upcoming_tips
from features import build_prediction_features, build_training_features, compute_elo, to_df
from predict import predict
from train import train, train_margin
from afl_tables import build_stats_lookup

START_YEAR = 2015
END_YEAR   = 2025   # train on completed seasons; 2026 games are predicted
DOCS_DIR   = "docs"


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_tips_lookup(tips_list):
    """
    Convert a flat list of Squiggle tip objects into a lookup dict:
      {game_id: home_consensus}
    where home_consensus = fraction of tipsters who tipped the home team.
    """
    game_tips = defaultdict(list)
    for tip in tips_list:
        game_id = tip.get("gameid")
        if game_id is None:
            continue
        tipped_home = 1 if tip.get("tip") == tip.get("hteam") else 0
        game_tips[game_id].append(tipped_home)
    return {
        gid: sum(tips) / len(tips)
        for gid, tips in game_tips.items()
        if tips
    }


def _next_round_games(upcoming):
    """Return (games, round_num, roundname) for the earliest incomplete round."""
    rounds = [g["round"] for g in upcoming if g.get("round") is not None]
    if not rounds:
        return [], None, "Unknown Round"
    next_round = min(rounds)
    games = [g for g in upcoming if g.get("round") == next_round]
    roundname = (games[0].get("roundname") or f"Round {next_round}") if games else f"Round {next_round}"
    return games, next_round, roundname


def _confidence_class(conf):
    if conf >= 0.70:
        return "high"
    if conf >= 0.58:
        return "med"
    return "low"


def _format_dt(dt):
    """Cross-platform datetime string (avoids %-d which fails on Windows)."""
    day  = dt.day
    hour = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return dt.strftime(f"%A {day} %B %Y, {hour}:%M {ampm}")


# ── HTML generation ───────────────────────────────────────────────────────────

def _game_rows_html(results):
    if results.empty:
        return '<tr><td colspan="7" style="text-align:center;color:#888;padding:32px 16px">No upcoming games found.</td></tr>'

    has_margin = "predicted_margin" in results.columns
    rows = ""
    for _, r in results.sort_values("confidence", ascending=False).iterrows():
        cc   = _confidence_class(r["confidence"])
        home_bold = r["predicted_winner"] == r["home_team"]
        h_style = "font-weight:700;color:#002B5C;" if home_bold else "color:#444;"
        a_style = "font-weight:700;color:#002B5C;" if not home_bold else "color:#444;"
        margin_td = f'<td class="margin">+{r["predicted_margin"]} pts</td>' if has_margin else ""
        rows += f"""
    <tr>
      <td style="{h_style}">{r['home_team']}</td>
      <td class="vs">vs</td>
      <td style="{a_style}">{r['away_team']}</td>
      <td class="venue">{r['venue']}</td>
      <td class="winner">{r['predicted_winner']}</td>
      {margin_td}
      <td class="conf {cc}">{r['confidence']:.0%}</td>
    </tr>"""
    return rows


def generate_html(results, roundname, generated_at, accuracy):
    conf_counts = {"high": 0, "med": 0, "low": 0}
    if not results.empty:
        for _, r in results.iterrows():
            conf_counts[_confidence_class(r["confidence"])] += 1

    has_margin  = not results.empty and "predicted_margin" in results.columns
    game_rows   = _game_rows_html(results)
    total_games = len(results)
    date_str    = _format_dt(generated_at)
    margin_th   = "<th>Margin</th>" if has_margin else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AFL Predictions — {roundname}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f4f6f9; color: #333; min-height: 100vh; }}

    .header {{
      background: linear-gradient(135deg, #002B5C 0%, #0055A5 100%);
      color: white; padding: 32px 24px 28px; text-align: center;
    }}
    .header h1 {{ font-size: 2em; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 6px; }}
    .header .sub {{ opacity: 0.8; font-size: 0.95em; }}

    .container {{ max-width: 960px; margin: 32px auto; padding: 0 16px 48px; }}

    .badges {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
    .badge {{
      background: white; border-radius: 10px; padding: 14px 20px;
      flex: 1; min-width: 130px; box-shadow: 0 1px 4px rgba(0,0,0,.08); text-align: center;
    }}
    .badge .val {{ font-size: 1.6em; font-weight: 800; color: #002B5C; line-height: 1; }}
    .badge .lbl {{ font-size: 0.72em; color: #999; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }}

    .legend {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; font-size: 0.8em; }}
    .legend span {{ padding: 3px 12px; border-radius: 12px; }}

    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    thead tr {{ background: #002B5C; color: white; }}
    th {{ padding: 12px 16px; font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; text-align: left; }}
    td {{ padding: 14px 16px; font-size: 0.93em; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }}
    tr:last-child td {{ border-bottom: none; }}
    tbody tr:hover td {{ background: #f6f8ff; }}

    .vs     {{ color: #bbb; font-size: 0.82em; width: 28px; text-align: center; padding: 0; }}
    .venue  {{ color: #888; font-size: 0.82em; }}
    .winner {{ font-weight: 600; color: #0055A5; }}
    .margin {{ color: #555; font-size: 0.82em; font-style: italic; }}
    .conf   {{ font-weight: 700; border-radius: 20px; padding: 4px 10px; font-size: 0.82em; text-align: center; width: 70px; }}
    .conf.high {{ background: #e6f4ea; color: #1e7e34; }}
    .conf.med  {{ background: #fff3cd; color: #856404; }}
    .conf.low  {{ background: #fde8e8; color: #c0392b; }}

    .footer {{ text-align: center; color: #bbb; font-size: 0.78em; margin-top: 28px; line-height: 1.8; }}
    .footer a {{ color: #bbb; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>AFL Predictions</h1>
    <div class="sub">{roundname} &nbsp;·&nbsp; Generated {date_str}</div>
  </div>

  <div class="container">
    <div class="badges">
      <div class="badge"><div class="val">{total_games}</div><div class="lbl">Games this round</div></div>
      <div class="badge"><div class="val">{conf_counts['high']}</div><div class="lbl">High confidence</div></div>
      <div class="badge"><div class="val">{conf_counts['med']}</div><div class="lbl">Medium confidence</div></div>
      <div class="badge"><div class="val">{accuracy:.0%}</div><div class="lbl">Model accuracy (CV)</div></div>
    </div>

    <div class="legend">
      <span style="background:#e6f4ea;color:#1e7e34">&#9646; ≥70% High confidence</span>
      <span style="background:#fff3cd;color:#856404">&#9646; 58–69% Medium</span>
      <span style="background:#fde8e8;color:#c0392b">&#9646; &lt;58% Low / coin flip</span>
    </div>

    <table>
      <thead>
        <tr>
          <th>Home</th><th></th><th>Away</th><th>Venue</th>
          <th>Predicted Winner</th>{margin_th}<th>Confidence</th>
        </tr>
      </thead>
      <tbody>
        {game_rows}
      </tbody>
    </table>

    <div class="footer">
      Model: XGBoost · Trained on AFL {START_YEAR}–{END_YEAR} via
      <a href="https://api.squiggle.com.au">Squiggle API</a> ·
      Features: form, margin, days rest, ladder, H2H, venue, Elo rating, tipster consensus ·
      <a href="https://github.com/forbesl1/afl-predictor">Source on GitHub</a>
    </div>
  </div>
</body>
</html>"""


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    print("=== AFL Predictor Pipeline ===\n")

    print(f"[1/5] Fetching training data ({START_YEAR}–{END_YEAR})...")
    raw = fetch_training_games(START_YEAR, END_YEAR)
    print(f"  Total raw games: {len(raw)}")

    print("\n[2/5] Processing completed games...")
    df = to_df(raw)
    print(f"  Completed games: {len(df)}")

    print("\n[3/5] Fetching tipster data...")
    all_tips    = fetch_all_tips(START_YEAR, END_YEAR)
    tips_lookup = _build_tips_lookup(all_tips)
    print(f"  Tips indexed: {len(tips_lookup)} games")

    print("\n[3b] Fetching team stats from afltables...")
    stats_lookup = build_stats_lookup(START_YEAR, END_YEAR)
    print(f"  Team-game stat entries: {len(stats_lookup)}")

    print("\n[4/5] Building training features + training model...")
    elo_lookup, current_elo = compute_elo(df)
    feat_df = build_training_features(df, tips_lookup=tips_lookup, elo_lookup=elo_lookup, stats_lookup=stats_lookup)
    print(f"  Feature rows: {len(feat_df)}")
    model, accuracy = train(feat_df)
    margin_model = train_margin(feat_df)
    print(f"  Margin model trained.")

    print("\n[5/5] Predicting next round (2026)...")
    upcoming = fetch_upcoming()
    games, round_num, roundname = _next_round_games(upcoming)

    if not games:
        print("  No upcoming games found — writing off-season page.")
        roundname = "Off Season"
        results   = pd.DataFrame()
    else:
        print(f"  {roundname}: {len(games)} games")
        current_year    = datetime.date.today().year
        upcoming_tips   = fetch_upcoming_tips(current_year, round_num) if round_num else []
        upcoming_lookup = _build_tips_lookup(upcoming_tips)
        print(f"  Tipster picks available: {len(upcoming_lookup)}/{len(games)} games")
        pred_df = build_prediction_features(df, games, tips_lookup=upcoming_lookup, current_elo=current_elo, stats_lookup=stats_lookup)
        results = predict(pred_df, model, margin_model=margin_model)

    os.makedirs(DOCS_DIR, exist_ok=True)
    out_path = os.path.join(DOCS_DIR, "index.html")
    html = generate_html(results, roundname, datetime.datetime.now(), accuracy)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nDone. Output: {out_path}")
    if not results.empty:
        print("\nPredictions summary:")
        for _, r in results.sort_values("confidence", ascending=False).iterrows():
            margin_str = f"  +{r['predicted_margin']}pts" if "predicted_margin" in r else ""
            print(f"  {r['predicted_winner']:25s} ({r['confidence']:.0%}){margin_str}  —  {r['home_team']} vs {r['away_team']}")


if __name__ == "__main__":
    run()
