# Contributing to Vello

Contributions are welcome by pull request. The [Rust code of conduct] applies.
For questions and design discussion, use the [#vello channel on Zulip][zulip].

Reviewer time is the scarcest resource on this project. [AGENTS.md](AGENTS.md)
describes what makes a change quick to review. It is written for coding agents,
but it applies to every contribution:

- [Contribution policy](AGENTS.md#contribution-policy): LLM-generated code is
  welcome, but you must understand it, review it yourself, and disclose LLM use
  in the PR description.
- [Tests](AGENTS.md#tests): prefer a `#[vello_test]` snapshot test for anything
  visible in the rendered output.
- [Making changes easy to review](AGENTS.md#making-changes-easy-to-review):
  keep mechanical changes, snapshot updates and behavior changes in separate
  commits.
- [Writing PR descriptions](AGENTS.md#writing-pr-descriptions) and
  [Writing issues](AGENTS.md#writing-issues).

## Before opening a pull request

The snapshot reference images are stored with [Git LFS][git-lfs], so run
`git lfs pull` after cloning. Then run the checks that CI runs most often:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test -p vello_tests
```

A new `#[vello_test]` creates its reference image on the first run and fails
once so that you look at it. To update a reference after an intended change,
run the test with `REPLACE=1`. A failing test writes an image showing the
reference, the difference and the actual output to `vello_tests/diffs/`; check
it before committing a new reference. See [vello_tests/README.md](vello_tests/README.md) for the WebGL
tests.

Add an entry under `[Unreleased]` in the crate's `CHANGELOG.md` for user-facing
changes to a published crate.

[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
[zulip]: https://xi.zulipchat.com/#narrow/channel/197075-vello
[git-lfs]: https://git-lfs.com/
