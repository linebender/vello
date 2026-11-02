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

## Project overview

Vello is a Rust workspace containing three 2D vector renderers:

- **Vello CPU** (`vello_cpu/`): SIMD and multithreaded CPU renderer. Most mature.
- **Vello GPU** (`vello_gpu/`): CPU preprocessing plus GPU raster/compositing via
  wgpu or native WebGL2. No compute shaders required.
- **Vello research (compute)** (`research/vello_research/`, published as `vello`): the
  original compute-shader renderer. Experimental, with little review bandwidth;
  prefer small fixes for clear failures (panics, obviously wrong rendering).

Vello CPU and Vello GPU share the Sparse Strips pipeline in `vello_common/` and
text/glyph support in `glifo/`. Changes to shared crates affect both renderers.

## Repository layout

| Path | Purpose |
| --- | --- |
| `vello_common/` | Shared geometry, flattening, tiling, strips, paint, atlas |
| `vello_cpu/` | CPU renderer (`coarse/`, `fine/`, `dispatch/`) |
| `vello_gpu/` | GPU renderer (`render/wgpu`, `render/webgl`, `schedule/`) |
| `vello_gpu_shaders/` | WESL shader sources, generated WGSL/GLSL |
| `glifo/` | Glyph runs, glyph atlas cache, COLR |
| `vello_tests/` | Shared CPU/GPU/WebGL snapshot tests (`#[vello_test]`) |
| `vello_bench/` | Sparse Strips benchmarks |
| `research/` | Research renderer, its encoding, shaders, tests, examples |
| `fuzz/` | Fuzz targets (excluded from the workspace) |

## Toolchain

- Edition 2024, MSRV **1.89** (`rust-version` in `Cargo.toml`). Do not use
  newer std APIs in published crates.
- CI uses stable **1.95** for clippy/tests and nightly for docs.
- Snapshot images are stored in Git LFS; run `git lfs pull` before snapshot tests.

## Commands

Run the narrowest command that covers the change (e.g. `-p vello_cpu`) before
workspace-wide checks.

```sh
# Format (Rust + TOML)
cargo fmt --all
taplo fmt

# Lint (mirrors CI; -D warnings)
cargo clippy --workspace --locked -- -D warnings
cargo clippy --workspace --locked --tests --benches --examples -- -D warnings

# Sparse Strips tests (CPU + wgpu)
cargo test -p vello_tests --release

# Research renderer tests
cargo test -p vello_research_tests --release

# Docs (match docs.rs)
RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo +nightly doc --workspace --all-features --no-deps --document-private-items

# Regenerate README sections from lib.rs docs
cargo rdme --workspace-project=<vello_cpu|vello_common|vello_gpu|glifo>
```

The workspace clippy run does not cover `no_std`, both `vello_cpu` pipelines, or
the WebGL backend. Run these when touching the affected crates (needs
`rustup target add x86_64-unknown-none wasm32-unknown-unknown`); CI also checks
each feature separately with `cargo hack`.

```sh
# no_std: vello_common, vello_cpu, glifo
cargo clippy -p vello_common -p vello_cpu -p glifo --locked --target x86_64-unknown-none --no-default-features --features libm,vello_cpu/text,vello_cpu/u8_pipeline,vello_cpu/f32_pipeline -- -D warnings

# vello_cpu pipelines, one at a time
cargo clippy -p vello_cpu --locked --no-default-features --features std,text,png,u8_pipeline -- -D warnings
cargo clippy -p vello_cpu --locked --no-default-features --features std,text,png,f32_pipeline -- -D warnings

# vello_gpu WebGL backend (wasm32 only)
cargo clippy -p vello_gpu --locked --target wasm32-unknown-unknown --features webgl -- -D warnings
```

## Code conventions

- Every `.rs` file starts with:

  ```rust
  // Copyright <year> the Vello Authors
  // SPDX-License-Identifier: Apache-2.0 OR MIT
  ```

  Shaders under `research/vello_shaders/{shader,src/cpu}` use
  `Apache-2.0 OR MIT OR Unlicense`. Checked by `.github/copyright.sh`.
- Workspace lints follow the [Linebender lint set][lints]. Notably:
  `missing_docs`, `unreachable_pub`, `cast_possible_truncation`, and
  `allow_attributes_without_reason`. Prefer `#[expect(lint, reason = "...")]`
  over bare `#[allow]`.
- `vello_common`, `vello_cpu`, and `glifo` must build as `no_std` (use `core` /
  `alloc`; math through the `libm` feature). Gate std-only code behind `std`,
  and do not add dependencies that pull in `std`.
- `vello_cpu` has `u8_pipeline` and `f32_pipeline` features; changes to `fine/`
  must compile under both.
- Do not edit the `README.md` sections generated by `cargo rdme`. Edit the
  crate-level docs in `lib.rs` and regenerate.
- Add new spelling exceptions to `.typos.toml` rather than misspelling words.

### Comments

- Doc comments (required on public items by `missing_docs`) state the contract:
  invariants, units, panics or errors, and non-obvious behavior. Do not restate
  the name or signature.
- Code comments explain what the code cannot show, such as a constraint, an
  invariant, or a driver workaround. Do not narrate the next line or refer to
  the change that introduced it.
- Match the comment density of the surrounding code.

## Testing

- New rendering behavior needs a `#[vello_test]` snapshot in `vello_tests/tests/`.
  The macro generates CPU, wgpu, and WebGL variants; see
  `vello_tests/vello_dev_macros/src/test.rs` for attributes.
- Update Sparse Strips snapshots with `REPLACE=1 cargo test -p vello_tests --release`.
  Diffs are written to `vello_tests/diffs/`.
- Update research renderer snapshots with `VELLO_TEST_UPDATE=all` (or
  `VELLO_TEST_CREATE=all` for new tests).
- Only update snapshots when the visual change is intended.
- WebGL tests: `wasm-pack test --headless --chrome --features webgl --release`
  from `vello_tests/`.
- Fuzz regressions go in `vello_tests/tests/fuzz_regression.rs`.

## Benchmarks

Validate performance-sensitive changes against `vello_bench` instead of
reasoning about speed. See [vello_bench/README.md](vello_bench/README.md).

```sh
# Compare against a baseline
cargo bench --bench main -- --save-baseline control [FILTER]
# ...apply the change, then rerun and compare
cargo bench -- [FILTER]
cargo bench --bench main -- --load-baseline new --baseline control

# Allocation regressions (fails if a recorded limit is exceeded)
cargo bench --bench allocations
```

Only raise an allocation limit when the increase is intentional, and say so.

## Shaders

- Vello GPU shaders are WESL in `vello_gpu_shaders/shaders`; the build script
  relinks them automatically. Inspect generated GLSL with
  `cargo run -p vello_gpu_shaders --features glsl`.
- Research renderer shaders are WGSL in `research/vello_shaders/shader` with a custom
  `import` / `ifdef` preprocessor.

## Changelogs

Each published crate has a `CHANGELOG.md` (Keep a Changelog format). For
user-visible changes, add an entry under `## [Unreleased]` in the right
subsection (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`):

```md
- Breaking change: Renamed `Foo::bar` to `baz`. ([#1234][] by [@handle][])
```

Prefix API breaks with `Breaking change:`.

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

## Before handing work back

- fmt, clippy (including the feature/target checks above when relevant), and
  the relevant tests pass.
- Benchmarks run for performance-sensitive changes.
- Changelog entry added for user-visible changes.
- Change is scoped to one concern, with mechanical and behavioral changes in
  separate commits.
- You have reviewed your own diff for typos, edge cases, missing tests, and
  leftover debug code, and told the author what you generated, which renderers
  (CPU / wgpu / WebGL) were tested, and which snapshots changed.

## Things to avoid

- Bumping `wgpu`/`naga` or `wasm-bindgen`/`web-sys` versions without being asked.
- Committing files under `target/`, `vello_tests/diffs/`, or `*/comparisons/`.
- Using `--all-targets` for clippy; CI splits lib/bin and aux targets on purpose.

[llm-policy]: https://linebender.org/wiki/llm-policy/
[lints]: https://linebender.org/wiki/canonical-lints/
