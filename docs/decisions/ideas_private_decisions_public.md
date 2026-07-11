# Decision: Ideas kept private, decision records public

## Context

The workspace convention (see `notes/README.md` in the workspace) is that project-scoped ideas *and* decisions both live in the project repo under `docs/`, so the repo is self-sufficient. This repo is public on GitHub, and its `docs/` folder is additionally the GitHub Pages root — anything filed there is not just in the repo but served on the public site.

## Decision

Split the two:

- **Decision records stay here** (`docs/decisions/`), public.
- **Ideas and future improvements move to the maintainer's private notes** and are not filed in this repo. The former "Future Improvements" section of GUIDE.md was relocated there.

## Reasoning

- Decision records explain *why the published model is the way it is* — they are documentation of the public artifact and benefit readers of the repo.
- Ideas are speculative: unbuilt features, half-formed plans, and notes on model weaknesses. There is no benefit to publishing them, and keeping them private preserves flexibility.
- A git-ignored ideas folder inside this repo was rejected: no version history and easily lost on a fresh clone. The private notes repo is versioned locally.

## Outcome

Adopted 2026-07-11. GUIDE.md's Future Improvements section removed; contents relocated to private idea notes.
