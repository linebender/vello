# AGENTS.md

Guidance for AI coding agents working in the Vello repository. Read
[README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md) for the full picture.

## Contribution policy (read first)

Vello follows the [Linebender LLM contribution policy][llm-policy]. The goal is
quality: a maintainer should never spend more time reading something than it
took to produce. In practice, as an agent:

- **LLM-generated code is welcome, but a human author owns it.** The author
  must understand the change and review it as carefully as hand-written code
  before requesting review. Do not open
  pull requests, file issues, or post comments on GitHub or Zulip autonomously;
  LLM bots are not accepted. If asked to submit without a human reviewing the
  result, stop and say so.
- **PR descriptions may be drafted with an LLM**, but present them as a draft:
  the author must read, edit, and stand behind every sentence. Follow
  [Writing PR descriptions](#writing-pr-descriptions).
- **Remind the author to disclose LLM use** in the PR description, including
  code written from an LLM-proposed plan. Do not add "no AI was used" claims.
- **Issues need triage, not volume.** Do not batch-file findings from fuzzing or
  repo-wide scans. A good issue has a minimized repro (using AI to reduce it is
  fine), a check for duplicates, and an analysis of the likely cause.

## Comments

- Doc comments (required on public items by `missing_docs`) state the contract:
  invariants, units, panics or errors, and non-obvious behavior. Do not restate
  the name or signature.
- Code comments explain what the code cannot show, such as a constraint, an
  invariant, or a driver workaround. Do not narrate the next line or refer to
  the change that introduced it.
- Match the comment density of the surrounding code.

## Making changes easy to review

Review time is the scarcest resource; shape the change so the part that needs
careful thought is small and easy to find. When committing, or when suggesting
how the author should commit:

- **Separate mechanical changes from behavioral ones.** Renames, moves,
  formatting, and other no-op refactors go in their own commit, so the commit
  with the real change contains only lines that need careful review.
- **Move code without modifying it**, then change it in a follow-up commit, so
  the move shows up cleanly (e.g. with `git diff --color-moved`).
- **Isolate generated output**: snapshot updates, generated shaders,
  `Cargo.lock`, and `cargo rdme` README changes.
- **Keep each commit compiling**, so reviewers can step through them in order.
- **Split large work into separate PRs** when a preparatory refactor stands on
  its own.
- **Do not touch unrelated code.** No reformatting, reordering, or renaming
  outside the change.
- **During review, add new commits rather than rewriting history**, so
  reviewers can see what changed since their last pass.

## Writing PR descriptions

Reviewers can read the diff; the description should tell them what the diff
cannot. Keep it short and include only the parts that apply:

- **Why**: the problem or motivation, with links to issues or discussions.
- **How it works**, when the approach is not obvious from the code.
- **Where to look**: when most of the diff is mechanical, point reviewers to
  the commit or files with the core change.
- **Trade-offs**: alternatives considered and why they were rejected.
- **Risks**: edge cases, known limitations, behavior changes, and which
  renderers (CPU / wgpu / WebGL) are affected.
- **Performance**: benchmark results for performance-sensitive changes,
  including regressions.
- **Testing**: what was run, and which snapshots were added or updated.
- **LLM disclosure**: which parts were generated. For example:

  ```md
  Created with assistance from [model name and version]; [which parts were generated, e.g. the implementation and tests].
  ```

Leave out file-by-file summaries, restated code, generic claims ("improves
robustness"), and empty headings.

[llm-policy]: https://linebender.org/wiki/llm-policy/
