# AFL Predictor — Comprehensive Guide

## Overview

AFL Predictor is a fully automated machine learning pipeline that fetches Australian Rules Football data, trains a predictive model, and publishes weekly win/loss predictions to a public webpage. It runs on a GitHub Actions cron schedule every Thursday morning (AEST), publishing predictions before the first game of each round.

**Live predictions:** https://forbesl1.github.io/afl-predictor/

---

## Architecture

```
afl_predictor/
├── fetch_data.py          — pulls game results and tipster data from the Squiggle API
├── afl_tables.py          — scrapes per-game team stats from afltables.com
├── features.py            — computes the feature matrix (all statistics and derived signals)
├── train.py               — trains the XGBoost model and saves it to disk
├── predict.py             — applies the trained model to upcoming games
├── pipeline.py            — orchestrates all steps end-to-end; generates the HTML output
├── analyse.py             — on-demand model performance analysis and plot generation
├── requirements.txt       — Python dependencies
├── docs/
│   └── index.html         — generated predictions page (served via GitHub Pages)
└── .github/
    └── workflows/
        └── predict.yml    — GitHub Actions cron workflow
```

The pipeline is entirely self-contained. It has no database, no server, and no external accounts beyond GitHub. Data comes from the Squiggle API and afltables.com. The model is retrained from scratch on every run. When running locally, completed seasons are cached to `.cache/` so repeated runs don't re-download historical data. On GitHub Actions, the `.cache/` directory is persisted between runs using `actions/cache`.

---

## Data Sources

### Squiggle API (`https://api.squiggle.com.au`)

Free, no authentication required. Three endpoints are used:

| Endpoint | Used for |
|----------|----------|
| `?q=games;year=YEAR` | Historical match results (scores, venues, dates, teams) |
| `?q=games;year=YEAR;complete=0` | Upcoming incomplete games (fixtures for next round) |
| `?q=tips;year=YEAR` | Tipster picks from ~30 prediction models registered on Squiggle |

**Caching:** Completed seasons are cached to `.cache/games_YYYY.json` and `.cache/tips_YYYY.json`. The current season is never cached. On first local run, fetching 2015–2025 takes about 30 seconds; subsequent runs are nearly instant.

**User-Agent:** The API requires a descriptive User-Agent header. The pipeline sends `afl-predictor/1.0 (github.com/forbesl1/afl-predictor)`. Note: semicolons in query parameters must be passed as raw characters — the `requests` library URL-encodes them by default, which the API rejects. The URL is manually constructed to avoid this.

**Training range:** 2015–2025 (2,258 completed games). The current year (2026) is predicted.

### afltables.com

Per-game team statistics scraped from individual match pages. Used to compute rolling team stat averages as additional features.

**Scraping approach:** `afl_tables.py` fetches `seas/{year}.html` to discover all game URLs for a season, then scrapes each match page (`stats/games/{year}/XXXXX.html`) for the team-level Totals row. Only regular-season rounds are included (finals excluded).

**Stats collected per game per team:**

| afltables column | Feature key | Description |
| --------------- | ----------- | ----------- |
| `IF` | `I50` | Inside 50s — attacking pressure |
| `CL` | `CL` | Clearances — midfield dominance |
| `DI` | `D` | Disposals — possession volume |
| `TK` | `T` | Tackles — defensive pressure |

**Caching:** Season-level results cached to `.cache/aflt_{year}.json`. Individual game pages cached to `.cache/game_{filename}.json`, so interrupted scrape runs can resume without re-fetching completed games. On first run, scraping all seasons (~2,000 game pages) takes around 15–20 minutes. GitHub Actions preserves this cache between runs via `actions/cache`.

**Fallback:** If a page fails to download or parse, that game's stats are silently skipped. XGBoost handles the resulting NaN values natively — the pipeline continues without those features for affected games.

---

## Pipeline Steps

Running `python pipeline.py` executes the following steps in order:

### Step 1 — Fetch training data
Downloads all completed games from 2015–2025 via Squiggle. Builds a flat list of ~2,258 game objects, each containing team names, scores, venue, date, and round.

### Step 2 — Process games
Converts the raw game list to a cleaned pandas DataFrame:
- Filters to `complete == 100` (finished games only)
- Converts scores to numeric
- Derives `home_win` (1 if home team won, 0 otherwise; draws treated as home loss — extremely rare in modern AFL)
- Parses dates as timezone-aware UTC timestamps
- Sorts chronologically

### Step 3 — Fetch tipster data
Downloads tipster picks for 2015–2025. Squiggle aggregates picks from ~30 registered prediction models. Tips are indexed by `gameid` into a lookup dict: `{game_id: home_consensus}` where `home_consensus` is the fraction of tipsters who predicted the home team to win (0.0–1.0). Data is available from 2017 onward; earlier games default to 0.5.

### Step 3b — Fetch team stats from afltables

Scrapes per-game team statistics from afltables.com for 2015–2025. Builds a lookup dict keyed by `(team, year, round)`. Completed seasons are cached; individual game pages are also cached so interrupted scrapes resume. See the afltables data source section above for full details.

### Step 4 — Build features + train models
For every completed game in the training set, computes 34 features using only data available *before* that game (no data leakage). Then trains two XGBoost models on the resulting feature matrix: a classifier for win probability and a regressor for predicted point margin. See the Features and Models sections below.

### Step 5 — Predict next round
Fetches incomplete games for 2026, filters to the next incomplete round, computes features using all 2015–2025 history as context, and applies the trained model to generate win probabilities. Writes `docs/index.html`.

---

## Features

All 34 features are computed in `features.py`. They fall into six categories:

### Form (recent results)

| Feature | Description |
|---------|-------------|
| `home_form` | Home team's win rate in their last 5 games |
| `away_form` | Away team's win rate in their last 5 games |
| `form_diff` | `home_form - away_form` — composite advantage signal |

Win rate is computed over the last `FORM_N = 5` games prior to the game date. If a team has fewer than 5 prior games (e.g. early in season 1), the available games are used. If a team has no history at all, 0.5 is used as a neutral prior.

### Scoring margin

| Feature | Description |
|---------|-------------|
| `home_avg_margin` | Home team's average signed score margin in last 5 games |
| `away_avg_margin` | Away team's average signed score margin in last 5 games |
| `margin_diff` | `home_avg_margin - away_avg_margin` |

Margin is computed as `(team score - opponent score)` — positive means winning, negative means losing. This is a richer signal than win/loss alone; a team winning by 60 points is more dominant than one winning by 2.

### Rest and fatigue

| Feature | Description |
|---------|-------------|
| `home_days_rest` | Days since home team's last game |
| `away_days_rest` | Days since away team's last game |
| `rest_diff` | `home_days_rest - away_days_rest` |

AFL scheduling creates meaningful rest asymmetries. Under the Collective Bargaining Agreement (CBA), clubs cannot play on a 4-day break, so the minimum turnaround is 5 days (e.g. Sunday game → Friday night the following week). Five-day breaks are permitted but capped at three per club per season, and when a team plays on a 5-day break their opponent must themselves have had no more than a 6-day break. A team with a bye has 13+ days off. Travel compounds the effect — a Perth-based side playing an away game in Victoria on a 5-day break faces a greater disadvantage than the raw day count suggests. Short turnarounds measurably affect performance, particularly for older squads. Defaults to 7 days if no prior game exists.

### Season ladder position

| Feature | Description |
|---------|-------------|
| `home_ladder_pct` | Home team's season win percentage up to this game |
| `away_ladder_pct` | Away team's season win percentage up to this game |
| `ladder_diff` | `home_ladder_pct - away_ladder_pct` |

Rather than actual ladder rank (which requires knowing all teams' records simultaneously), this uses each team's own win percentage — a clean, leakage-free proxy for season standing.

### Head-to-head and venue

| Feature | Description |
|---------|-------------|
| `h2h` | Home team's historical win rate against this specific opponent (all available history) |
| `home_venue` | Home team's historical win rate at this venue |

Both use all games prior to the game date across the full training window. Defaults to 0.5 if no historical matchup or venue data exists.

### Elo ratings

| Feature | Description |
|---------|-------------|
| `home_elo` | Home team's Elo rating entering the game |
| `away_elo` | Away team's Elo rating entering the game |
| `elo_diff` | `home_elo - away_elo` |
| `elo_win_prob` | Elo's implied win probability for home team |

**How Elo works:**

Each team starts at a rating of 1500. After every game:

```
expected_home = 1 / (1 + 10^((away_elo - home_elo - HOME_ADV) / 400))
home_elo += K * (actual - expected_home)
away_elo += K * ((1 - actual) - (1 - expected_home))
```

Constants used (tuned for AFL):
- `K = 40` — update factor (how much ratings shift per game)
- `HOME_ADV = 65` — home ground advantage in Elo points (~58% baseline win rate for home teams)
- `MEAN = 1500` — initial and mean rating

**Season reset:** At the start of each new year, all ratings are partially regressed toward 1500:
```
new_rating = 1500 + 0.7 * (old_rating - 1500)
```
This reflects that player turnover, coaching changes, and off-season preparation partially reset form. The 30% regression is standard for AFL Elo models.

Elo is computed in `compute_elo(df)` which walks games chronologically, storing the pre-game ratings for training and the final ratings for prediction. It is one of the strongest individual predictors of AFL outcomes.

### Tipster consensus

| Feature | Description |
|---------|-------------|
| `tipster_consensus` | Fraction of Squiggle tipsters who predict the home team to win |

Squiggle aggregates picks from ~30 registered models including statistical models, machine learning systems, and expert tipsters. A high consensus (e.g. 0.85) means the broader prediction community strongly favours one team. This feature is a powerful meta-signal — it partially captures information the other features miss (player availability, weather, travel, etc.), since the individual tipsters incorporate this in their own models.

Defaults to 0.5 when no tips are available (pre-2017 training data, or if the round hasn't been tipped yet).

### Team stats (afltables)

| Feature | Description |
| ------- | ----------- |
| `home_I50_avg` | Home team's average Inside 50s per game, last 5 rounds |
| `away_I50_avg` | Away team's average Inside 50s per game, last 5 rounds |
| `I50_avg_diff` | `home_I50_avg - away_I50_avg` |
| `home_CL_avg` | Home team's average clearances per game, last 5 rounds |
| `away_CL_avg` | Away team's average clearances per game, last 5 rounds |
| `CL_avg_diff` | `home_CL_avg - away_CL_avg` |
| `home_D_avg` | Home team's average disposals per game, last 5 rounds |
| `away_D_avg` | Away team's average disposals per game, last 5 rounds |
| `D_avg_diff` | `home_D_avg - away_D_avg` |
| `home_T_avg` | Home team's average tackles per game, last 5 rounds |
| `away_T_avg` | Away team's average tackles per game, last 5 rounds |
| `T_avg_diff` | `home_T_avg - away_T_avg` |

Rolling averages use the same `FORM_N = 5` window as form features. Stats are looked up from the `(team, year, round)` keyed afltables cache. XGBoost handles NaN natively when data is unavailable for a game, so these features degrade gracefully for early seasons where scraping may have gaps.

---

## Models

The pipeline trains two XGBoost models on every run, both using the same 34-feature matrix.

### Classifier — win probability (`XGBClassifier`)

Predicts the probability that the home team wins. XGBoost was chosen over logistic regression because:

- **Non-linear interactions:** e.g. rest advantage matters more when Elo ratings are close; this can't be captured by a linear model
- **Feature importance:** built-in feature importance scores help interpret what the model is learning
- **Proven performance:** XGBoost consistently outperforms linear models on tabular sports prediction tasks

**Hyperparameters**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `n_estimators` | 300 | Number of trees |
| `max_depth` | 4 | Maximum depth per tree — limits overfitting |
| `learning_rate` | 0.05 | Slow learning rate with many trees — more stable |
| `subsample` | 0.8 | 80% of rows per tree — adds randomness, reduces overfitting |
| `colsample_bytree` | 0.8 | 80% of features per tree |
| `eval_metric` | logloss | Optimises for probability calibration, not just accuracy |

**Evaluation**

5-fold cross-validation is run on the training data. Current OOF accuracy: **~65.4%** (mean across 2015–2025). Season-by-season accuracy ranges from ~59% in unpredictable years to ~72% in more consistent ones. This is consistent with published AFL prediction benchmarks — the best public models achieve 68–72%, and the theoretical maximum (given how much genuine randomness AFL has) is estimated around 72–75%.

**Output**

`model.predict_proba(X)` returns `[P(away wins), P(home wins)]` for each game. The higher probability team is the predicted winner. Confidence is `max(home_prob, away_prob)`.

Confidence thresholds:
- **≥70%** — High confidence (green)
- **58–69%** — Medium confidence (yellow)
- **<58%** — Low / coin flip (red)

### Regressor — predicted margin (`XGBRegressor`)

Predicts the home team's final point margin (`hscore - ascore`). A positive value means the home team wins; negative means the away team wins. The same 34 features and hyperparameters are used (minus `eval_metric`, which is not applicable to regression).

The predicted margin is displayed in the output table from the predicted winner's perspective (always shown as a positive number, e.g. "+18 pts"). This gives a richer picture than win probability alone — a predicted 40-point win is more convincing than a 2-point win at the same confidence level.

Both models are retrained from scratch on every pipeline run, ensuring they incorporate the most recent results automatically.

---

## Output: GitHub Pages

The pipeline writes `docs/index.html` — a self-contained HTML page with inline CSS. It shows:
- Round name and generation timestamp (Melbourne time)
- Summary badges: games count, confidence breakdown, model CV accuracy
- Predictions table sorted by confidence (highest first), with:
  - Predicted winner **bolded** in each matchup row
  - Predicted winning margin (e.g. "+18 pts") from the regressor
  - Colour-coded confidence pills (green / yellow / red)

GitHub Pages is configured to serve from the `docs/` folder on the `master` branch. The GitHub Actions workflow commits the updated `docs/index.html` after each run, which triggers an automatic Pages redeploy.

---

## Automation: GitHub Actions

**File:** `.github/workflows/predict.yml`

**Schedule:** `0 4 * * 4` — Thursday 4am UTC = Thursday 2pm AEST

This timing is chosen to run after Squiggle tipsters have published their picks for the round (typically by Thursday midday) but before the first game (Thursday ~7:30pm AEST).

**Also supports:** Manual trigger via the GitHub Actions tab → "Run workflow"

**What the workflow does:**
1. Checks out the repository
2. Restores `.cache/` from the GitHub Actions cache (key prefix `afltables-`) — avoids re-scraping afltables history on every run
3. Installs Python 3.11 and dependencies from `requirements.txt`
4. Runs `python pipeline.py` (with `TZ=Australia/Melbourne` so timestamps display correctly)
5. Commits and pushes the updated `docs/index.html` to `master`
6. Saves the updated `.cache/` back to the Actions cache for the next run

**Permissions:** `contents: write` is required to allow the workflow to commit back to the repository.

---

## Analysis

`analyse.py` is an on-demand script for evaluating model performance. It is not part of the automated pipeline — run it manually when you want a full diagnostic.

```bash
python analyse.py
```

It fetches the same data as `pipeline.py`, runs 5-fold out-of-fold (OOF) cross-validation, prints a summary, and saves five plots to `analysis/` (git-ignored):

| Plot | Description |
| ---- | ----------- |
| `feature_importance.png` | Top 20 features by XGBoost gain |
| `calibration.png` | Predicted probability vs actual win rate — checks if 70% really means 70% |
| `season_accuracy.png` | OOF accuracy per season (2015–2025) vs overall mean |
| `confidence_tiers.png` | Accuracy broken down by high / medium / low confidence predictions |
| `margin_scatter.png` | Actual vs predicted margins with OOF MAE |

All accuracy metrics use OOF predictions to avoid optimistic bias from evaluating on training data. The final model trained on the full dataset is used only for feature importance.

---

## Running Locally

```bash
# Clone and set up
git clone https://github.com/forbesl1/afl-predictor.git
cd afl-predictor
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt

# Run the full pipeline
python pipeline.py

# Open the output
start docs/index.html       # Windows
```

On first run, the pipeline fetches and caches 2015–2025 game and tips data (~30 seconds). Subsequent runs use the cache for historical seasons and are much faster (~2–3 minutes for feature computation).

**Cache location:** `.cache/` — contains:

- `games_YYYY.json` / `tips_YYYY.json` — Squiggle game and tipster data per season
- `aflt_YYYY.json` — aggregated afltables team stats per season
- `game_XXXXX.json` — individual afltables match page stats (allows interrupted scrapes to resume)

Pre-current-year seasons are cached indefinitely; the current season is never cached.

**Git ignores:** `.cache/`, `model.pkl`, `__pycache__/`, `.venv/`

---

## Future Improvements

### High impact

**Player availability / injury lists**
The single biggest missing signal. If a team's key forward is ruled out Thursday afternoon, the model doesn't know. Possible implementation: scrape AFL.com.au or Footywire for the final selected 22, compare against a "baseline" squad using historical fantasy score averages as a proxy for player value. This requires a headless browser (Playwright) since AFL.com.au is a JavaScript-rendered app, or a Footywire scraper.

**Hyperparameter tuning**
Run `optuna` or `sklearn GridSearchCV` over `n_estimators`, `max_depth`, `learning_rate`, `subsample`. This is currently skipped for simplicity but could improve CV accuracy by 1–2%.

### Medium impact

**Extend Elo back further**
Start Elo computation from 2000 or earlier (Squiggle has data from 1897) to allow ratings to fully converge before the training window. Currently the 2015 ratings start at 1500 and may take 2–3 seasons to reflect true team quality.

**Venue-specific Elo**
Some teams are dramatically stronger at their home ground (MCG teams, GMHBA Stadium). A separate "venue Elo" component — distinct from overall Elo — would capture this more precisely than the `home_venue` win-rate feature.

### Lower impact / longer term

**Email notifications**
Add `notify.py` using Gmail SMTP + App Password to send the HTML output as an email each Thursday. Avoids needing to visit the GitHub Pages URL manually.

**Multi-season Elo reset tuning**
The current season regression factor (30%) is a common default. Empirical tuning against held-out seasons could find a better value for AFL specifically.

**Injury impact score**
Rather than binary availability, weight player absence by their contribution: `impact = sum(fantasy_avg of absent players) / sum(fantasy_avg of full squad)`. A 0.15 absence rate for a star player is very different from a 0.15 rate for a depth player.
