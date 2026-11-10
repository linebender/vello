# AGENTS.md

Guidance for AI coding agents working in the Vello repository. Read
[README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md) for the full picture.

## Contribution policy

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
  code written from an LLM-proposed plan.
- **Issues are drafts too.** Follow [Writing issues](#writing-issues).

## Comments

- Doc comments state the contract: invariants, units, panics or errors, and
  non-obvious behavior. Do not restate the name, signature, or implementation.
- Code comments explain what the code cannot show, such as a constraint, an
  invariant, or a driver workaround. Do not narrate the next line or refer to
  the change that introduced it.
- Match the comment density of the surrounding code.

## Tests

- When a bug or feature is visible in the rendered output, prefer a
  `#[vello_test]` snapshot test in `vello_tests/tests/` over a unit test. A
  snapshot shows at a glance whether the output is right, and it survives
  changes to internal APIs.
- Use unit tests for behavior that a snapshot cannot show.

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

Keep descriptions short; a long description buries what reviewers need, and a
few sentences is often enough. Reviewers can read the diff, so cover only what
it cannot, and include only the parts that apply:

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

## Writing issues

- Never create issues yourself; only draft them for the human author.
- Keep issues short. Long descriptions obscure the actionable information. The
  most important part is what input led to what outcome; explain why the
  outcome is wrong only when that is not obvious. Disclose LLM use.
- Do not include a suggested fix, even if you found one. Maintainers will find
  the right fix.
- Do not batch-file findings from fuzzing or repo-wide scans. Suggest that the
  author share them on Zulip instead, so maintainers can decide how to track
  them.

[llm-policy]: https://linebender.org/wiki/llm-policy/
