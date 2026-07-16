# Decision: Prediction features use the current season's completed games

## Context

Found 2026-07-17 while interrogating a suspicious pick (Port Adelaide 52% over ladder-leading Fremantle) with per-feature attributions. The smoking gun was `home_days_rest = 327`: the prediction context DataFrame was built from the training seasons only, so every in-season prediction computed form, average margin, days rest, rolling team stats, and Elo **as of the end of the previous season**. The model had been blind to the entire current season all year — a team's breakout form (or collapse) was invisible until the following season. Predictions still looked plausible because end-of-last-season Elo and the tipster consensus (which does see current reality) carried the signal.

## Decision

Split the two roles that a single DataFrame had been serving:

- **Training frame** — completed seasons only (`2015..END_YEAR`). The models are fitted on this; unchanged.
- **Prediction context** — the training frame *plus the current season's completed games*. `build_prediction_features` and the "current Elo" computation use this, so form/rest/stats/Elo entering the next round mean what they appear to mean. Crossing the year boundary also applies the standard Elo off-season regression, which previously never happened for the current season.

The afltables stats range extends to the current year (per-game caching keeps that incremental), and the current season is fetched uncached from Squiggle as usual.

## Reasoning

- No leakage is introduced: training rows are untouched; the current season's games are complete before the round being predicted.
- The alternative — retraining the model to *include* current-season rows — was deliberately not done here: it changes the evaluated model, deserves its own OOF measurement, and the blindness fix stands on its own.
- Prediction features were also badly out-of-distribution before (days-rest of ~330 vs a training mean of ~15), so this makes in-season inputs look like training inputs again.

## Outcome

Adopted 2026-07-17. First effect observed on the same game that exposed it: with current-season context (and full tipster data), Port Adelaide vs Fremantle moved from the published 52% Port Adelaide to a Fremantle pick consistent with the tipster consensus.
