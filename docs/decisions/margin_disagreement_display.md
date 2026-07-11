# Decision: When the margin model disagrees with the win model, name the team — don't reconcile

## Context

The page shows a predicted winner (from the XGBClassifier) and a predicted margin (from the XGBRegressor). The margin was displayed as the absolute value of the regressor's home-margin prediction, implicitly attributed to the predicted winner. When the two models disagreed on the winner, the page silently showed a margin that actually belonged to the *other* team, with no way to tell.

Out-of-fold analysis (5-fold CV, 2015–2025, 2,258 games) showed this is not rare:

- The models disagree on the winner in **14.7% of games** (332).
- Disagreements are not all coin flips: median regressor margin is 5 points but reaches 30, and 40 of them occur at high (≥70%) classifier confidence.
- Neither model is the authority when they disagree: classifier correct 48.5%, regressor correct 51.5%.

## Decision

Keep both models' outputs and display them honestly. When the margin model agrees with the predicted winner: "+18 pts" as before. When it disagrees: the margin cell names the team the margin model favours (e.g. "Carlton +4†"), styled distinctly, with a footnote under the table explaining the conflict. `predict.py` now emits a `margin_team` column alongside `predicted_margin`.

## Reasoning

- **Clamping to "close game" was rejected** — disagreements reach 30-point margins and 86% classifier confidence; pretending they're all near-zero hides real model conflict.
- **Switching the winner to the regressor (or an ensemble) was rejected for now** — the 48.5/51.5 split gives no statistical basis for preferring either, and changing the prediction itself alters measured accuracy. An ensemble is a possible future model change; this decision is only about honest display.
- **A signed margin ("−4 pts") was rejected** — "predicted to win by −4" reads as nonsense to casual readers; naming the team is self-explanatory without the footnote.

## Outcome

Implemented 2026-07-11. First live example the same day: Carlton vs Hawthorn — win model Hawthorn 66%, margin model Carlton +5.
