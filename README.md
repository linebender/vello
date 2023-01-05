<div align="center">

# 🖌 vello

**An experimental GPU compute-centric 2D renderer**

[![Xi Zulip](https://img.shields.io/badge/Xi%20Zulip-%23gpu-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/stream/197075-gpu)
[![dependency status](https://deps.rs/repo/github/linebender/vello/status.svg)](https://deps.rs/repo/github/linebender/vello)
[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](#License)
<!-- [![Crates.io](https://img.shields.io/crates/v/vello.svg)](https://crates.io/crates/vello) -->
<!-- [![Docs](https://docs.rs/vello/badge.svg)](https://docs.rs/vello) -->
<!-- [![Build status](https://github.com/linebender/vello/workflows/CI/badge.svg)](https://github.com/linebender/vello/actions) -->

</div>

Vello is a 2d graphics rendering engine, using [`wgpu`].
It efficiently draws large 2d scenes with interactive or near-interactive performance.

<!-- Impressive picture here -->

It is used as the rendering backend for [xilem], a UI toolkit.

## Examples

Our examples are provided in seperate packages in the [`examples`](examples) folder. 
This allows them to have independent dependencies and faster builds.
Examples must be selected using the `--package` (or `-p`) Cargo flag.

### Winit

Our [winit] example ([examples/with_winit](examples/with_winit)) demonstrates rendering to a [winit] window.
It also includes a collection of test scenes showing the capabilities of vello.
One of these scenes uses an incomplete svg parser/renderer to render the [GhostScript tiger].

```shell
cargo run -p with_winit
```

### Bevy

The [Bevy] example ([examples/with_bevy](examples/with_bevy)) demonstrates using vello within a [Bevy] application.
This currently draws to a [`wgpu`] `Texture` using `vello`, then uses that texture as the faces of a cube.

```shell
cargo run -p with_bevy
```

### Web

Because Vello relies heavily on compute shaders, we rely on the emerging WebGPU standard to run on the web.
Until browser support becomes widespread, it will probably be necessary to use development browser versions (e.g. Chrome Canary) and explicitly enable WebGPU.

The following command builds and runs a web version of the [winit demo](#Winit).

```shell
cargo run --release -p run-wasm -- --package with_winit
```

Additionally, the web is not currently a primary target, so other issues are likely to arise.

## Shader templating

We implement a limited, simple preprocessor for our shaders, as wgsl has insufficient code-sharing for our needs.

This implements only classes of statements.
1. `import`, which imports from `shader/shared`.
2. `ifdef`, `ifndef`, `else` and `endif`, as standard.
  These must be at the start of their lines.  
  Note that there is no support for creating definitions in-shader, these are only specified externally (in `src/shaders.rs`).
  Note also that this definitions cannot currently be used in-code (`import`s may be used instead)

This format is compatible with [`wgsl-analyzer`], and we recommend using.
If you run into any issues, please report them on Zulip ([#gpu > wgsl-analyzer issues](https://xi.zulipchat.com/#narrow/stream/197075-gpu/topic/wgsl-analyzer.20issues)), and/or on the [`wgsl-analyzer`] issue tracker.  
Note that new imports must currently be added to `.vscode/settings.json` for this support to work correctly.`wgsl-analyzer` only supports imports in very few syntactic locations, so we limit their use to these places.

## History

Vello was previously known as `piet-gpu`. This prior incarnation used a custom cross-API hardware abstraction layer, called `piet-gpu-hal`, instead of [`wgpu`].
<!-- Some discussion of this transition can be found in the blog post [A requiem to piet-gpu-hal]() TODO: Once the blog post is published -->

An archive of this version can be found in the branches [`custom-hal-archive-with-shaders`] and [`custom-hal-archive`].
This succeeded the previous prototype, [piet-metal], and included work adapted from [piet-dx12] by Brian Merchant.

<!-- Some mention of `google/forma` here -->

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

In addition, all files in the [`shader`](shader) directory and subdirectories thereof are alternatively 
licensed under the Unlicense ([shader/UNLICENSE](shader/UNLICENSE) or http://unlicense.org/).
For clarity, these files are also licensed under either of the above licenses.
The intent is for this research to be used in as broad a context as possible.

The files in subdirectories of the [`examples/assets`](examples/assets) directory are licensed solely under 
their respective licenses, available in the `LICENSE` file in their directories.
As a consequence of this, the (built) examples may have additional restrictions due to their use of these assets.

## Contribution

Contributions are welcome by pull request. The [Rust code of conduct] applies.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
licensed as above, without any additional terms or conditions.

> **Warning**  
> Previous versions of this README are included below for reference whilst we are still updating the README.

This repo contains the new prototype for a new compute-centric 2D GPU renderer (formerly known as piet-gpu).


<!-- TODO: Are we transitioning to more production? If so, should we rewrite the README a bit? -->

## Goals

The main goal is to answer research questions about the future of 2D rendering:

-   Is a compute-centered approach better than rasterization ([Direct2D])? How much so?

-   To what extent do "advanced" GPU features (subgroups, descriptor arrays) help?

-   Can we improve quality and extend the imaging model in useful ways?

## Blogs and other writing

Much of the research progress on piet-gpu is documented in blog entries. See [doc/blogs.md](doc/blogs.md) for pointers to those.

There is a much larger and detailed [vision](doc/vision.md) that explains the longer-term goals of the project, and how we might get there.

[piet-metal]: https://github.com/linebender/piet-metal
[direct2d]: https://docs.microsoft.com/en-us/windows/win32/direct2d/direct2d-portal
[`wgpu`]: https://wgpu.rs/
[xilem]: https://github.com/linebender/xilem/
[rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
[`custom-hal-archive-with-shaders`]: https://github.com/linebender/piet-gpu/tree/custom-hal-archive-with-shaders
[`custom-hal-archive`]: https://github.com/linebender/piet-gpu/tree/custom-hal-archive
[piet-dx12]: https://github.com/bzm3r/piet-dx12
[GhostScript tiger]: https://commons.wikimedia.org/wiki/File:Ghostscript_Tiger.svg
[`wgsl-analyzer`]: https://marketplace.visualstudio.com/items?itemName=wgsl-analyzer.wgsl-analyzer

# vello

This crate is currently a highly experimental proof-of-concept port of the piet-gpu renderer to the WGSL shader language, so it could be run on WebGPU. Depending on how well it works out, it may become the authoritative source for piet-gpu.

The shaders are actually handlebars templates over WGSL, as it's important to share common data structures; it's likely we'll use the template mechanism to supply various parameters which are not supported by the WGSL language, for example to specify grayscale or RGBA buffer output for fine rasterization.

This crate also uses a very different approach to the GPU abstraction than piet-gpu. That is essentially a HAL that supports an immediate mode approach to creating resources and submitting commands. Here, we generate a `Recording`, which is basically a simple value type, then an `Engine` plays that recording to the actual GPU. The idea is that this can abstract easily over multiple GPU back-ends, without either the render logic needing to be polymorphic or having dynamic dispatch at the GPU abstraction. The goal is to be more agile.

Scene encoding is shared with piet-gpu, and currently uses piet-scene in the same repo with no changes.

This module is still an experimental work in progress. Contributions can be made with the same policy as the root repo, but expect things to change quickly.
