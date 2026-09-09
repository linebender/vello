<div align="center">

# Vello

**High-performance 2D renderers written in Rust**

[![Linebender Zulip](https://img.shields.io/badge/Linebender-%23vello-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/197075-vello)
[![dependency status](https://deps.rs/repo/github/linebender/vello/status.svg)](https://deps.rs/repo/github/linebender/vello)
[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
[![Build status](https://github.com/linebender/vello/workflows/CI/badge.svg)](https://github.com/linebender/vello/actions)

</div>

Vello is a project for high-performance 2D vector rendering. It contains three renderer implementations with different hardware requirements and maturity profiles.

## Choose a renderer

### Vello CPU

[`vello_cpu`](vello_cpu) is a CPU-only Sparse Strips renderer optimized for multithreading and SIMD. It is the most mature choice when predictable CPU rendering, software rendering, or support for devices without a suitable GPU is required.

### Vello GPU

[`vello_gpu`](vello_gpu) is a Sparse Strips renderer that preprocesses paths on the CPU and uses the GPU for rasterization and compositing. It supports wgpu rendering and a WebGL2 backend without requiring compute shaders. Choose it when GPU acceleration and broad GPU compatibility are important. Vello GPU is intended to become the primary renderer for production GPU use cases as it matures; Vello CPU currently has the more mature Sparse Strips implementation.

### Vello compute renderer

The [`vello`](research/vello_research) crate is the original compute-centric renderer. It performs most rendering work in GPU compute shaders and remains an experimental implementation for compute-capable GPUs. Its source and supporting crates now live under [`research/`](research/README.md).

Vello CPU and Vello GPU share the Sparse Strips architecture and common infrastructure in [`vello_common`](vello_common). See [ARCHITECTURE.md](ARCHITECTURE.md) for how the renderer families differ.

## Quick start

Run the windowed example for the renderer you want to try:

```shell
# Vello CPU
cargo run -p vello_cpu_winit --release

# Vello GPU
cargo run -p vello_gpu_winit --release

# Compute-centric research renderer
cargo run -p with_winit --release
```

Package-specific setup and API examples are in the [`vello_cpu`](vello_cpu), [`vello_gpu`](vello_gpu), and [`vello`](research/vello_research) READMEs.

## Repository layout

- [`vello_cpu/`](vello_cpu) and [`vello_gpu/`](vello_gpu) contain the user-facing Sparse Strips renderers.
- [`vello_common/`](vello_common) and [`vello_gpu_shaders/`](vello_gpu_shaders) support those renderers.
- [`glifo/`](glifo) provides text and glyph-run support shared by both Sparse Strips renderers.
- [`vello_tests/`](vello_tests) contains their shared development tests, snapshots, scenes, and browser tooling.
- [`research/`](research/README.md) contains the compute-centric renderer, its encoding and shader crates, examples, tests, historical design documents, and changelog.
- [`vello_bench/`](vello_bench) contains Sparse Strips benchmarks.

Release histories are maintained in the [`vello` changelog](research/CHANGELOG.md), [`vello_cpu` changelog](vello_cpu/CHANGELOG.md), [`vello_gpu` changelog](vello_gpu/CHANGELOG.md), and [`vello_common` changelog](vello_common/CHANGELOG.md).

## Package names and source folders

Cargo package names do not have to match their source directory names. The research folders were renamed to make their role in this repository explicit without forcing existing crates.io users to migrate:

- `research/vello_research/` still publishes as [`vello`](https://crates.io/crates/vello).
- `research/vello_encoding/` still publishes as [`vello_encoding`](https://crates.io/crates/vello_encoding).
- `research/vello_shaders/` still publishes as [`vello_shaders`](https://crates.io/crates/vello_shaders).

Existing users of these three packages keep the same dependency and Rust import names. Only local or Git-based tooling that refers directly to repository folders needs to use the new `research/` paths.

The repository's Sparse Strips package names did change: `vello_hybrid` became `vello_gpu`, and `vello_sparse_shaders` became `vello_gpu_shaders`. Git and local path dependencies that track this repository must use the new package and Rust import names. Existing crates.io releases under the old names remain available, and publication under the new names may lag behind the repository rename; check crates.io before selecting a released version.

The development-only `vello_sparse_tests` package became the root-level `vello_tests`, while the research test helper package became `vello_research_tests`.

There is no empty root-level `vello/` placeholder. A future shared abstraction can introduce a root directory when it has a concrete API; the current folder layout does not require changing the established `vello` package identity.

## Motivation

Vello is intended to fill the same place in the graphics stack as vector renderers such as [Skia](https://skia.org/), [Cairo](https://www.cairographics.org/), and its predecessor project [Piet](https://github.com/linebender/piet). Its renderers draw shapes, images, gradients, and text using a PostScript-inspired imaging model like those behind SVG and the browser [`<canvas>` element](https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D).

The implementations explore complementary ways to use SIMD, multithreading, and GPUs while sharing the same broader rendering goals.

## Minimum supported Rust Version (MSRV)

This version of Vello has been verified to compile with **Rust 1.89** and later.

Future versions of Vello might increase the Rust version requirement. It will not be treated as a breaking change and as such can even happen with small patch releases.

<details>
<summary>Click here if compiling fails.</summary>

As time has passed, some of Vello's dependencies could have released versions with a higher Rust requirement. If you encounter a compilation issue due to a dependency and don't want to upgrade your Rust toolchain, then you could downgrade the dependency.

```sh
# Use the problematic dependency's name and version
cargo update -p package_name --precise 0.1.1
```

</details>

## Community

Discussion of Vello development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically the [#vello channel](https://xi.zulipchat.com/#narrow/channel/197075-vello). All public content can be read without logging in.

Contributions are welcome by pull request. The [Rust code of conduct] applies.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache 2.0 license, shall be licensed as noted in the [License](#license) section, without any additional terms or conditions.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

In addition, all files in the [`research/vello_shaders/shader`](https://github.com/linebender/vello/tree/main/research/vello_shaders/shader) and [`research/vello_shaders/src/cpu`](https://github.com/linebender/vello/tree/main/research/vello_shaders/src/cpu) directories and subdirectories thereof are alternatively licensed under the Unlicense ([research/vello_shaders/shader/UNLICENSE](https://github.com/linebender/vello/blob/main/research/vello_shaders/shader/UNLICENSE) or <http://unlicense.org/>). For clarity, these files are also licensed under either of the above licenses. The intent is for this research to be used in as broad a context as possible.

The files in subdirectories of the [`assets`](https://github.com/linebender/vello/tree/main/assets) directory are licensed solely under their respective licenses, available in the `LICENSE` file in their directories.

[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
