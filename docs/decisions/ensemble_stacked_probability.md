# Decision: Publish a stacked probability (classifier + margin regressor)

## Context

The page's win probability came from the XGBClassifier alone; the XGBRegressor's margin was display-only. OOF analysis for the margin-display fix (see `margin_disagreement_display.md`) showed the two models disagree on the winner in 14.7% of games with neither reliably right (48.5% vs 51.5%) — evidence that each carries signal the other misses. An ensemble experiment was run per the pre-agreed bar: adopt only if OOF accuracy, logloss, and calibration all improve over the classifier-only baseline.

## Experiment (5-fold OOF, 2015–2025, 2,258 games)

All fitted blend components were evaluated with a second-level 5-fold CV over the OOF predictions, so nothing was scored on data it was fitted on.

| Approach | Accuracy | Logloss | Brier |
| -------- | -------- | ------- | ----- |
| Classifier only (baseline) | 63.99% | 0.6490 | 0.2262 |
| Margin → logistic alone | 64.35% | 0.6233 | 0.2177 |
| Fixed blends (w · clf + (1−w) · margin-prob, w = 0.3–0.7) | 63.64–64.70% | 0.6211–0.6298 | — |
| **Stacked LR on [clf prob, margin]** | **64.66%** | **0.6217** | **0.2170** |
| Shrink confidence on disagreement | 63.99% | 0.6446 | 0.2242 |

Within the 332 disagreement games: classifier 48.5%, stacked 53.0%. Notably, the margin regressor alone is better calibrated than the classifier.

## Decision

Adopt the stacked logistic regression. `train_ensemble()` fits both base models on all data plus a `LogisticRegression` stacker on their out-of-fold predictions; `predict()` publishes the stacker's probability. The page's accuracy badge shows the stacked OOF accuracy.

The stacker was chosen over the marginally-better fixed 0.3/0.7 blend because it learns the weighting itself — the fixed-blend numbers are best-of-five hand-tried weights scored on the same data, a mild selection bias the stacker's second-level CV avoids.

## Consequences

- Published confidences became more conservative (e.g. a 98% pick became 85%) — the raw classifier was overconfident, and the stacker's calibration is honest. Confidence-tier counts on the page shifted accordingly.
- Winner/margin disagreements on the page are much rarer (the stacker weighs the margin heavily), but still possible; the "name the margin model's team" display from `margin_disagreement_display.md` remains as the safety net.
- `analyse.py` now evaluates and plots the stacked probability (with the classifier as a printed baseline).

## Outcome

Adopted 2026-07-11. First visible effect: Carlton vs Hawthorn flipped from "Hawthorn 66% / margin Carlton +5†" to a consistent "Carlton 51%, +5 pts".
