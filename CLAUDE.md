# AFL Predictor — session entry point

Automated AFL match prediction pipeline (Python + XGBoost). Fetches game data, retrains from scratch, and publishes weekly predictions to GitHub Pages via a Thursday GitHub Actions cron.

## Read first, in this order

1. `GUIDE.md` (this folder) — architecture, data sources, features, models, automation. Owns all technical depth.
2. `../notes/projects/afl_predictor.md` — dashboard: status, model summary, dated log. Owns "what exists" and "what's pending".

## Filing rules (deliberately differs from the workspace default)

- **Decision records** → `docs/decisions/` in this repo — public; they document the published model.
- **Ideas / future improvements** → `../notes/ideas/afl_predictor_*.md` — **private** (local-only notes repo). Never file ideas in this repo; see `docs/decisions/ideas_private_decisions_public.md`.

## Commands

```bash
python pipeline.py   # full run: fetch → train → predict → write docs/index.html (~2–3 min with warm .cache/)
python analyse.py    # on-demand diagnostics: OOF CV summary + plots to analysis/ (git-ignored)
```

venv setup: `C:\Users\Lach\miniconda3\python.exe -m venv .venv`, then `.venv\Scripts\activate` and `pip install -r requirements.txt`.

## Definition of done (every change)

1. `python pipeline.py` runs end-to-end without errors if code was touched. This regenerates `docs/index.html` — **discard that change** (`git checkout -- docs/index.html`) rather than committing it; the file is bot-owned.
2. One-line entry in the dashboard Log (`../notes/projects/afl_predictor.md`) — detail belongs in GUIDE.md, not the log.
3. GUIDE.md updated if behaviour, features, data sources, or automation changed (keep the feature count exact).
4. Decision record in `docs/decisions/` if a real either/or choice was made; new ideas go to the private notes repo (see filing rules).
5. Commit this repo. `git pull --rebase` before pushing — the Actions bot commits to `master` every Thursday.

## Facts that prevent mistakes

- **Everything under `docs/` is published** — it is the GitHub Pages root. Never put private material there.
- The Actions bot pushes `Update predictions YYYY-MM-DD` to `master` weekly (Thu 4am UTC / 2pm AEST); pull before pushing or you'll create merge commits.
- **Never commit a locally generated `docs/index.html`.** Git can silently mis-merge the generated HTML with the bot's version — no conflict reported, corrupt table on the live site (happened 2026-07-10; see `docs/decisions/generated_html_bot_owned.md`).
- The model has **31 features** — `FEATURE_COLS` in `features.py` is the source of truth; update GUIDE.md and the dashboard if it changes.
- Squiggle API: semicolons in query params must NOT be URL-encoded — URLs are built manually in `fetch_data.py`. A descriptive User-Agent is required.
- `.cache/` is git-ignored locally but persisted on GitHub Actions via `actions/cache` (key prefix `afltables-`); the current season is never cached.
- `predicted_winner` comes from the classifier but `predicted_margin` from the regressor — they can disagree (see private ideas for the open fix).
- This folder and `../notes/` are separate git repos; the parent `aiprojects/` is not a repo — never `git init` there.
