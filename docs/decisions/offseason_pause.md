# Decision: Pause the cron over the off-season (with an early-exit safety net)

## Context

The AFL season runs roughly March–September, but the workflow cron ran every Thursday year-round. Off-season runs executed the full pipeline (fetch, scrape, train all three models) just to write an "Off Season" placeholder — and because the page embeds a generation timestamp, the no-change commit guard never triggered, producing a pointless bot commit every week from October to March. Separately, `END_YEAR` was hardcoded, so each January the pipeline silently stopped short of the most recent completed season until someone bumped the constant.

## Decision

Three changes, adopted together:

1. **Cron month range:** `0 4 * 3-10 4` — Thursdays, March through October. October is included so the first run after the Grand Final publishes the off-season placeholder.
2. **Early exit:** the pipeline now checks for upcoming games *first*. With none found, it skips all fetching and training, writes the placeholder only if it isn't already published, and exits — so repeat off-season runs (manual or October) commit nothing.
3. **Dynamic training window:** `END_YEAR = current year − 1`, removing the annual maintenance trap.

## Reasoning

- The cron range is declarative and zero-cost; the early exit makes any run outside the season cheap and idempotent, so the two guards back each other up.
- Alternatives considered: keeping year-round runs (wasteful, noisy history); disabling the workflow manually each October (relies on memory twice a year); detecting season end in the workflow YAML (duplicates logic the pipeline can decide better).

## Trade-off accepted

GitHub disables scheduled workflows after **60 days of repository inactivity**. The old weekly off-season commits — the very noise being removed — were what kept the workflow alive. With the pause, the off-season will exceed 60 quiet days and GitHub will likely disable the schedule (it emails a warning first). **Each March: check the Actions tab and re-enable the workflow.** Noted in CLAUDE.md and the workflow file itself.

## Outcome

Adopted 2026-07-17. Off-season path verified by test (placeholder written once, repeat run writes nothing); in-season path verified with a full pipeline run (correctly predicted the next round).
