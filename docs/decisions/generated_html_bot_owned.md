# Decision: docs/index.html is bot-owned — never commit a locally generated copy

## Context

On 2026-07-10, a `git pull` produced a merge (`ee2bc9a`) between a local commit that included a stale locally generated `docs/index.html` (Round 6, left over from an April pipeline run) and the Actions bot's fresh Round 18 page (`3985bb6`). Git auto-merged the two versions **without reporting a conflict**: the generated HTML is full of identical repeated lines (`<tr>`, `</tr>`, `<td class="vs">vs</td>`), so git matched false context and silently interleaved fragments of the Round 6 table into the Round 18 table — a phantom extra game row and a stray `<td>` that broke the table layout on the live site.

## Decision

`docs/index.html` is owned exclusively by the GitHub Actions bot. Local pipeline runs regenerate it as a side effect; that change must be discarded before committing:

```bash
git checkout -- docs/index.html
```

## Reasoning

- The file is fully regenerated on every run — there is never a reason to hand-merge or preserve a local copy; the bot's next Thursday run overwrites everything.
- Textual merging of generated HTML is unsafe: repeated identical lines make false-clean merges likely, and the result is silently corrupt (no conflict markers to notice).
- Alternatives considered: a `.gitattributes` merge driver (built-in options don't express "always take upstream"), or git-ignoring the file (impossible — GitHub Pages serves it from the repo). A simple ownership rule is enough.

## Outcome

Adopted 2026-07-11. The corrupted page was repaired by restoring the file from the bot's commit. Rule recorded in `CLAUDE.md` (filing rules + gotchas).
