# Decision: Predictions table sorted by kick-off time, not confidence

## Context

The predictions table originally sorted by confidence (highest first), putting the model's "best bets" at the top. A "When" column was added showing each game's kick-off (e.g. "Thu 7:30 PM"), which raised the question of which column should drive the row order.

## Decision

Sort chronologically by kick-off, earliest game first. The kick-off time comes from Squiggle's `date` field, which is AEST/AEDT (venue-local time is a separate `localtime` field and is not used).

## Reasoning

- The page reads like a fixture: people scan a round in the order the games are played, Thursday night through Sunday.
- No information is lost — confidence remains visible as the colour-coded pill on every row, so high-confidence picks still stand out in any order.
- With ~9 games per round, "best bets first" added little; a "Best bet" badge on the top pick could be added later if that ordering is missed.
- The raw `date` strings ("YYYY-MM-DD HH:MM:SS") sort chronologically as plain text, so no date parsing is needed for the sort itself.

## Outcome

Implemented 2026-07-11 in `pipeline.py` (new "When" column, chronological sort in both the HTML table and the console summary).
