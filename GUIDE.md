# AFL Predictor — Comprehensive Guide

## Overview

AFL Predictor is a fully automated machine learning pipeline that fetches Australian Rules Football data, trains a predictive model, and publishes weekly win/loss predictions to a public webpage. It runs on a GitHub Actions cron schedule every Thursday morning (AEST) during the AFL season (March–October), publishing predictions before the first game of each round. Off-season runs exit early and publish nothing new.

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

**Training range:** 2015 through the most recently completed season — `END_YEAR` is computed as the current year − 1, so the training window grows automatically each season with no maintenance. The current year is predicted. (As of 2026: 2015–2025, 2,258 completed games.)

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

### Step 1 — Check for upcoming games
Fetches incomplete games for the current year and selects the earliest incomplete round. **Off-season early exit:** if there are no upcoming games, the pipeline writes the "Off Season" placeholder page and stops — no data fetching, no training. If the placeholder is already published it writes nothing at all, so repeat off-season runs produce no commit.

### Step 2 — Fetch training data
Downloads all completed games in the training range via Squiggle. Builds a flat list of game objects, each containing team names, scores, venue, date, and round.

### Step 3 — Process games
Converts the raw game list to a cleaned pandas DataFrame:
- Filters to `complete == 100` (finished games only)
- Converts scores to numeric
- Derives `home_win` (1 if home team won, 0 otherwise; draws treated as home loss — extremely rare in modern AFL)
- Parses dates as timezone-aware UTC timestamps
- Sorts chronologically

### Step 4 — Fetch tipster data
Downloads tipster picks for the training range. Squiggle aggregates picks from ~30 registered prediction models. Tips are indexed by `gameid` into a lookup dict: `{game_id: home_consensus}` where `home_consensus` is the fraction of tipsters who predicted the home team to win (0.0–1.0). Data is available from 2017 onward; earlier games default to 0.5.

### Step 4b — Fetch team stats from afltables

Scrapes per-game team statistics from afltables.com for the training range. Builds a lookup dict keyed by `(team, year, round)`. Completed seasons are cached; individual game pages are also cached so interrupted scrapes resume. See the afltables data source section above for full details.

### Step 5 — Build features + train models
For every completed game in the training set, computes 31 features using only data available *before* that game (no data leakage). Then trains the model stack on the resulting feature matrix: a classifier for win probability, a regressor for predicted point margin, and the logistic-regression stacker that blends them. See the Features and Models sections below.

### Step 6 — Predict next round
Using the round found in Step 1, fetches that round's tipster picks, computes features using all training history as context, and applies the model stack to generate win probabilities. Writes `docs/index.html`.

---

## Features

All 31 features are computed in `features.py` (`FEATURE_COLS` is the source of truth). They fall into eight categories:

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
| `home_ladder_pct` | Home team's win percentage across all games before this one (full training window, not season-scoped) |
| `away_ladder_pct` | Away team's win percentage across all games before this one (full training window, not season-scoped) |
| `ladder_diff` | `home_ladder_pct - away_ladder_pct` |

Rather than actual ladder rank (which requires knowing all teams' records simultaneously), this uses each team's own cumulative win percentage — a clean, leakage-free proxy for standing. Note it is computed over the team's *entire* history in the training window (2015 onward), not reset each season — despite the "ladder" name, it measures long-run team quality.

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

The pipeline trains two XGBoost models on every run, both using the same 31-feature matrix, plus a logistic-regression **stacker** that blends them into the final published win probability.

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

5-fold out-of-fold accuracy of the published (stacked) probability: **~64.7%**, vs ~64.0% for the classifier alone (see the Stacker section). Season-by-season accuracy ranges from ~59% in unpredictable years to ~72% in more consistent ones. This is consistent with published AFL prediction benchmarks — the best public models achieve 68–72%, and the theoretical maximum (given how much genuine randomness AFL has) is estimated around 72–75%.

**Output**

The classifier's `P(home wins)` is blended with the margin regressor's prediction by the stacker (below) to produce the final probability. The higher-probability team is the predicted winner. Confidence is `max(home_prob, away_prob)`.

Confidence thresholds:
- **≥70%** — High confidence (green)
- **58–69%** — Medium confidence (yellow)
- **<58%** — Low / coin flip (red)

### Regressor — predicted margin (`XGBRegressor`)

Predicts the home team's final point margin (`hscore - ascore`). A positive value means the home team wins; negative means the away team wins. The same 34 features and hyperparameters are used (minus `eval_metric`, which is not applicable to regression).

The margin is displayed in the output table as "+18 pts" when the margin model agrees with the published predicted winner. When it doesn't (rarer since the stacker was adopted, but still possible), the margin cell instead names the team the margin model favours (e.g. "Carlton +4†") with a footnote — the margin is never silently attributed to the predicted winner. See `docs/decisions/margin_disagreement_display.md`.

### Stacker — blending the two models (`LogisticRegression`)

The raw classifier and regressor disagree on the winner in ~15% of games (OOF 2015–2025), and neither is reliably right in those cases (classifier 48.5%, regressor 51.5%) — each carries signal the other misses. A logistic regression is fitted on the two base models' **out-of-fold** predictions `[P(home win), predicted margin]` and produces the final published probability.

OOF comparison (2,258 games, second-level CV so the stacker is never scored on rows it was fitted on):

| Probability | Accuracy | Logloss | Brier |
| ----------- | -------- | ------- | ----- |
| Classifier alone | 63.99% | 0.6490 | 0.2262 |
| Stacked | 64.66% | 0.6217 | 0.2170 |

Within the disagreement games, stacking lifts accuracy from 48.5% to ~53%. The stacked probabilities are also noticeably better calibrated (lower logloss/Brier) — the raw classifier was overconfident, so published confidence percentages became more conservative when the stacker was adopted. See `docs/decisions/ensemble_stacked_probability.md`.

All three models are retrained from scratch on every pipeline run, ensuring they incorporate the most recent results automatically.

---

## Output: GitHub Pages

The pipeline writes `docs/index.html` — a self-contained HTML page with inline CSS. It shows:
- Round name and generation timestamp (Melbourne time)
- Summary badges: games count, confidence breakdown, model CV accuracy
- Predictions table sorted chronologically by kick-off (earliest first), with:
  - Predicted winner **bolded** in each matchup row
  - Kick-off day and time (e.g. "Thu 7:30 PM", AEST/AEDT from Squiggle's `date` field)
  - Predicted margin from the regressor (e.g. "+18 pts"; when the margin model favours the other team, the cell names that team instead, e.g. "Carlton +4†")
  - Colour-coded confidence pills (green / yellow / red)

GitHub Pages is configured to serve from the `docs/` folder on the `master` branch. The GitHub Actions workflow commits the updated `docs/index.html` after each run, which triggers an automatic Pages redeploy.

---

## Automation: GitHub Actions

**File:** `.github/workflows/predict.yml`

**Schedule:** `0 4 * 3-10 4` — Thursday 4am UTC = Thursday 2pm AEST, **March–October only**

The weekly timing is chosen to run after Squiggle tipsters have published their picks for the round (typically by Thursday midday) but before the first game (Thursday ~7:30pm AEST). The month range pauses the workflow over the off-season; October is included so the run after the Grand Final publishes the off-season placeholder page. Off-season runs exit early (no fetching or training) and commit nothing once the placeholder is up.

**Off-season caveat:** GitHub automatically disables scheduled workflows after 60 days of repository inactivity, and a paused off-season provides no bot commits to keep it alive. GitHub emails a warning first; check the Actions tab each March and re-enable the workflow if needed (any push also resets the inactivity timer).

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

## Ideas and decisions

Decision records live in `docs/decisions/`. Ideas and future improvements are tracked privately outside this repo — see `CLAUDE.md` for filing rules.
