<!--
Keep this short: reviewers can read the diff, so cover only what it cannot.
Include only what applies, and delete these comments and any line you leave empty.
See "Writing PR descriptions" in AGENTS.md: https://github.com/linebender/vello/blob/main/AGENTS.md#writing-pr-descriptions

- Why: the problem or motivation, with links to issues (e.g. "Fixes #123").
- How it works, if the approach is not obvious from the code.
- Where to look, if most of the diff is mechanical.
- Risks: edge cases, behavior changes, and which renderers are affected
  (vello_cpu, vello_gpu wgpu, vello_gpu WebGL, research).
- Performance: benchmark results for performance-sensitive changes, including regressions.
- Testing: what you ran, and which snapshots you added or updated.
-->

<!--
Before requesting review:

- Changes visible in the rendered output have a `#[vello_test]` snapshot test in `vello_tests/tests/`.
  For a bug fix, commit the test first so the reference image shows the bug, then the fix.
- Mechanical changes (renames, moves, formatting) and snapshot updates are in their own commits.
- User-facing changes to published crates have an entry under `[Unreleased]` in that crate's CHANGELOG.md.
-->

<!--
If an LLM generated any of this PR (code, tests, or this description), say so, for example:
"Created with assistance from [model name and version]."
See the Linebender LLM policy: https://linebender.org/wiki/llm-policy/
-->
