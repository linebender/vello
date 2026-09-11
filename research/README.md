# Vello compute renderer research

This directory contains Vello's original compute-centric renderer and the code,
examples, tests, and design history dedicated to it. The renderer performs most
of its work in GPU compute shaders and requires a compute-capable GPU.

For the CPU and broadly compatible GPU renderers at the repository root, start
with the [project README](../README.md). See
[ARCHITECTURE.md](../ARCHITECTURE.md) for a comparison of the renderer families.

## Packages

The research packages are organized as follows:

- [`vello_research/`](vello_research) contains the package published as
  [`vello`](https://crates.io/crates/vello).
- [`vello_encoding/`](vello_encoding) contains the package published as
  [`vello_encoding`](https://crates.io/crates/vello_encoding).
- [`vello_shaders/`](vello_shaders) contains the package published as
  [`vello_shaders`](https://crates.io/crates/vello_shaders).

## Contents

- [`vello_research/`](vello_research) — public `vello` rendering API and wgpu
  backend.
- [`vello_encoding/`](vello_encoding) — scene encoding shared with the compute
  pipeline.
- [`vello_shaders/`](vello_shaders) — compute shaders, shader preprocessing,
  and CPU reference kernels.
- [`vello_research_tests/`](vello_research_tests) — integration tests,
  snapshots, and comparison helpers.
- [`examples/`](examples) — standalone examples, including winit and headless
  applications.
- [`xtask/`](xtask) — snapshot and comparison maintenance commands.
- [`doc/`](doc) — historical design documents, roadmaps, and development
  notes.
- [`CHANGELOG.md`](CHANGELOG.md) — release history for the `vello` package.

## Getting started

Run the main windowed example from the repository root:

```shell
cargo run -p with_winit --release
```

Other entry points include the `simple`, `simple_sdl2`, and `headless` example
packages under [`examples/`](examples). The package README in
[`vello_research/`](vello_research) has API setup, features, WebAssembly
instructions, integrations, current limitations, and additional examples.

## Repository migration

The source directories moved as follows:

- `vello/` → `research/vello_research/`
- `vello_encoding/` → `research/vello_encoding/`
- `vello_shaders/` → `research/vello_shaders/`
- the former compute-renderer `vello_tests/` →
  `research/vello_research_tests/`
- `examples/`, `doc/`, and `xtask/` → their corresponding paths under
  `research/`

These are repository path changes. The published `vello`, `vello_encoding`, and
`vello_shaders` package identities remain unchanged.

The current root-level `vello_tests/` is a different, development-only package
for Vello CPU and Vello GPU. It was formerly named `vello_sparse_tests`.
