<div align="center">

# Vello

**An experimental GPU compute-centric 2D renderer**

[![Xi Zulip](https://img.shields.io/badge/Xi%20Zulip-%23gpu-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/stream/197075-gpu)
[![dependency status](https://deps.rs/repo/github/linebender/vello/status.svg)](https://deps.rs/repo/github/linebender/vello)
[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](#license)
[![wgpu version](https://img.shields.io/badge/wgpu-v0.15-orange.svg)](https://crates.io/crates/wgpu)
<!-- [![Crates.io](https://img.shields.io/crates/v/vello.svg)](https://crates.io/crates/vello) -->
<!-- [![Docs](https://docs.rs/vello/badge.svg)](https://docs.rs/vello) -->
<!-- [![Build status](https://github.com/linebender/vello/workflows/CI/badge.svg)](https://github.com/linebender/vello/actions) -->

</div>

Vello is a 2d graphics rendering engine, using [`wgpu`].
It efficiently draws large 2d scenes with interactive or near-interactive performance.

<!-- Impressive picture here -->

It is used as the rendering backend for [Xilem], a UI toolkit.

Quickstart to run an example program:
```shell
cargo run -p with_winit
```

## Integrations

### SVG

This repository also includes [`vello_svg`](./integrations/vello_svg/), which supports converting 
a [`usvg`](https://crates.io/crates/usvg) `Tree` into a Vello scene.

This is currently incomplete; see its crate level documentation for more information.

This is used in the [winit](#winit) example for the SVG rendering.

## Examples

Our examples are provided in separate packages in the [`examples`](examples) folder.
This allows them to have independent dependencies and faster builds.
Examples must be selected using the `--package` (or `-p`) Cargo flag.

### Winit

Our [winit] example ([examples/with_winit](examples/with_winit)) demonstrates rendering to a [winit] window.
By default, this renders [GhostScript Tiger] all SVG files in [examples/assets/downloads](examples/assets/downloads) directory (using [`vello_svg`](#svg)).
A custom list of SVG file paths (and directories to render all SVG files from) can be provided as arguments instead.
It also includes a collection of test scenes showing the capabilities of `vello`, which can be shown with `--test-scenes`.

```shell
cargo run -p with_winit 
```

Some default test scenes can be downloaded from Wikimedia Commons using the `download` subcommand.
This also supports downloading from user-provided URLS.

```shell
cargo run -p with_winit -- download
```

<!-- ### Headless -->

### Bevy

The [Bevy] example ([examples/with_bevy](examples/with_bevy)) demonstrates using Vello within a [Bevy] application.
This currently draws to a [`wgpu`] `Texture` using `vello`, then uses that texture as the faces of a cube.

```shell
cargo run -p with_bevy
```

## Platforms

We aim to target all environments which can support WebGPU with the [default limits](https://www.w3.org/TR/webgpu/#limits). 
We defer to [`wgpu`] for this support.
Other platforms are more tricky, and may require special building/running procedures.

### Web

Because Vello relies heavily on compute shaders, we rely on the emerging WebGPU standard to run on the web.
Until browser support becomes widespread, it will probably be necessary to use development browser versions (e.g. Chrome Canary) and explicitly enable WebGPU.

The following command builds and runs a web version of the [winit demo](#winit). 
This uses [`cargo-run-wasm`](https://github.com/rukai/cargo-run-wasm) to build the example for web, and host a local server for it

Other examples use the `-p` shorthand, but `cargo-run-wasm` requires the full `--package` to be specified

```shell
cargo run_wasm --package with_winit
```

> **Warning**  
> The web is not currently a primary target for vello, and WebGPU implementations are incomplete, so you might run into issues running this example.

### Android

The [`with_winit`](#winit) example supports running on Android, using [cargo apk](https://crates.io/crates/cargo-apk).

```
cargo apk run -p with_winit
```

> **Note**  
> cargo apk doesn't support running in release mode without configuration. 
> See [their crates page docs](https://crates.io/crates/cargo-apk) (around `package.metadata.android.signing.<profile>`).
> 
> See also [cargo-apk#16](https://github.com/rust-mobile/cargo-apk/issues/16).
> To run in release mode, you must add the following to `examples/with_winit/Cargo.toml` (changing `$HOME` to your home directory):

```
[package.metadata.android.signing.release]
path = "$HOME/.android/debug.keystore"
keystore_password = "android"
```

## Community

[![Xi Zulip](https://img.shields.io/badge/Xi%20Zulip-%23gpu-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/stream/197075-gpu)

Discussion of Vello development happens in the [Xi Zulip](https://xi.zulipchat.com/), specifically the [#gpu stream](https://xi.zulipchat.com/#narrow/stream/197075-gpu). All public content can be read without logging in

## Shader templating

We implement a limited, simple preprocessor for our shaders, as wgsl has insufficient code-sharing for our needs.

This implements only classes of statements.

1. `import`, which imports from `shader/shared`
2. `ifdef`, `ifndef`, `else` and `endif`, as standard.
  These must be at the start of their lines.  
  Note that there is no support for creating definitions in-shader, these are only specified externally (in `src/shaders.rs`).
  Note also that this definitions cannot currently be used in-code (`import`s may be used instead)

This format is compatible with [`wgsl-analyzer`], which we recommend using.
If you run into any issues, please report them on Zulip ([#gpu > wgsl-analyzer issues](https://xi.zulipchat.com/#narrow/stream/197075-gpu/topic/wgsl-analyzer.20issues)), and/or on the [`wgsl-analyzer`] issue tracker.  
Note that new imports must currently be added to `.vscode/settings.json` for this support to work correctly.
`wgsl-analyzer` only supports imports in very few syntactic locations, so we limit their use to these places.

## GPU abstraction

Our rendering code does not directly interact with `wgpu`.
Instead, we generate a `Recording`, a simple value type, then an `Engine` plays that recording to the actual GPU.
The only currently implemented `Engine` uses `wgpu`.

The idea is that this can abstract easily over multiple GPU back-ends, without either the render logic needing to be polymorphic or having dynamic dispatch at the GPU abstraction.
The goal is to be more agile.

## Goals

The major goal of Vello is to provide a high quality GPU accelerated renderer suitable for a range of 2D graphics applications, including rendering for GUI applications, creative tools, and scientific visualization.
The [roadmap for 2023](doc/roadmap_2023.md) explains the goals and plans for the next few months of development

Vello emerges from being a research project, which attempts to answer these hypotheses:

- To what extent is a compute-centered approach better than rasterization ([Direct2D])?

- To what extent do "advanced" GPU features (subgroups, descriptor arrays, device-scoped barriers) help?

- Can we improve quality and extend the imaging model in useful ways?

Another goal of the overall project is to explain how the renderer is built, and to advance the state of building applications on GPU compute shaders more generally. 
Much of the progress on Vello is documented in blog entries.
See [doc/blogs.md](doc/blogs.md) for pointers to those.

## History

Vello was previously known as `piet-gpu`. 
This prior incarnation used a custom cross-API hardware abstraction layer, called `piet-gpu-hal`, instead of [`wgpu`].

An archive of this version can be found in the branches [`custom-hal-archive-with-shaders`] and [`custom-hal-archive`].
This succeeded the previous prototype, [piet-metal], and included work adapted from [piet-dx12].

The decision to lay down `piet-gpu-hal` in favor of WebGPU is discussed in detail in the blog post [Requiem for piet-gpu-hal].

A [vision](doc/vision.md) document dated December 2020 explained the longer-term goals of the project, and how we might get there.
Many of these items are out-of-date or completed, but it still may provide some useful background.

## Related projects

Vello takes inspiration from many other rendering projects, including:

* [Pathfinder](https://github.com/servo/pathfinder)
* [Spinel](https://fuchsia.googlesource.com/fuchsia/+/refs/heads/master/src/graphics/lib/compute/spinel/)
* [Forma](https://github.com/google/forma)
* [Massively Parallel Vector Graphics](https://w3.impa.br/~diego/projects/GanEtAl14/)
* [Random-access rendering of general vector graphics](https://hhoppe.com/proj/ravg/)

## License

Licensed under either of

- Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license
   ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

In addition, all files in the [`shader`](shader) directory and subdirectories thereof are alternatively
licensed under the Unlicense ([shader/UNLICENSE](shader/UNLICENSE) or <http://unlicense.org/>).
For clarity, these files are also licensed under either of the above licenses.
The intent is for this research to be used in as broad a context as possible.

The files in subdirectories of the [`examples/assets`](examples/assets) directory are licensed solely under
their respective licenses, available in the `LICENSE` file in their directories.

## Contribution

Contributions are welcome by pull request. The [Rust code of conduct] applies.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
licensed as above, without any additional terms or conditions.

[piet-metal]: https://github.com/linebender/piet-metal
[direct2d]: https://docs.microsoft.com/en-us/windows/win32/direct2d/direct2d-portal
[`wgpu`]: https://wgpu.rs/
[Xilem]: https://github.com/linebender/xilem/
[rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
[`custom-hal-archive-with-shaders`]: https://github.com/linebender/piet-gpu/tree/custom-hal-archive-with-shaders
[`custom-hal-archive`]: https://github.com/linebender/piet-gpu/tree/custom-hal-archive
[piet-dx12]: https://github.com/bzm3r/piet-dx12
[GhostScript tiger]: https://commons.wikimedia.org/wiki/File:Ghostscript_Tiger.svg
[winit]: https://github.com/rust-windowing/winit
[Bevy]: https://bevyengine.org/
[`wgsl-analyzer`]: https://marketplace.visualstudio.com/items?itemName=wgsl-analyzer.wgsl-analyzer
[Requiem for piet-gpu-hal]: https://raphlinus.github.io/rust/gpu/2023/01/07/requiem-piet-gpu-hal.html
