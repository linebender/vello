<!--

This repo-level readme needs restructuring, pending some Linebender templating decisions.
https://xi.zulipchat.com/#narrow/stream/419691-linebender/topic/Bikeshedding.20badges

For now, prefer updating the package-level readmes, e.g. vello/README.md.

-->

<div align="center">

# Vello

**A GPU compute-centric 2D renderer**

[![Linebender Zulip](https://img.shields.io/badge/Linebender-%23gpu-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/stream/197075-gpu)
[![dependency status](https://deps.rs/repo/github/linebender/vello/status.svg)](https://deps.rs/repo/github/linebender/vello)
[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](#license)
[![wgpu version](https://img.shields.io/badge/wgpu-v23.0.1-orange.svg)](https://crates.io/crates/wgpu)

[![Crates.io](https://img.shields.io/crates/v/vello.svg)](https://crates.io/crates/vello)
[![Docs](https://docs.rs/vello/badge.svg)](https://docs.rs/vello)
[![Build status](https://github.com/linebender/vello/workflows/CI/badge.svg)](https://github.com/linebender/vello/actions)

</div>

Vello is a 2D graphics rendering engine written in Rust, with a focus on GPU compute.
It can draw large 2D scenes with interactive or near-interactive performance, using [`wgpu`] for GPU access.

Quickstart to run an example program:

```shell
cargo run -p with_winit
```

![image](https://github.com/linebender/vello/assets/8573618/cc2b742e-2135-4b70-8051-c49aeddb5d19)

It is used as the rendering backend for [Xilem], a Rust GUI toolkit.

> [!WARNING]
> Vello can currently be considered in an alpha state. In particular, we're still working on the following:
>
> - [Implementing blur and filter effects](https://github.com/linebender/vello/issues/476).
> - [Conflations artifacts](https://github.com/linebender/vello/issues/49).
> - [GPU memory allocation strategy](https://github.com/linebender/vello/issues/366)
> - [Glyph caching](https://github.com/linebender/vello/issues/204)

Significant changes are documented in [the changelog].

## Motivation

Vello is meant to fill the same place in the graphics stack as other vector graphics renderers like [Skia](https://skia.org/), [Cairo](https://www.cairographics.org/), and its predecessor project [Piet](https://github.com/linebender/piet).
On a basic level, that means it provides tools to render shapes, images, gradients, text, etc, using a PostScript-inspired API, the same that powers SVG files and [the browser `<canvas>` element](https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D).

Vello's selling point is that it gets better performance than other renderers by better leveraging the GPU.
In traditional PostScript-style renderers, some steps of the render process like sorting and clipping either need to be handled in the CPU or done through the use of intermediary textures.
Vello avoids this by using prefix-sum algorithms to parallelize work that usually needs to happen in sequence, so that work can be offloaded to the GPU with minimal use of temporary buffers.

This means that Vello needs a GPU with support for compute shaders to run.

## Getting started

Vello is meant to be integrated deep in UI render stacks.
While drawing in a Vello scene is easy, actually rendering that scene to a surface requires setting up a wgpu context, which is a non-trivial task.

To use Vello as the renderer for your PDF reader / GUI toolkit / etc, your code will have to look roughly like this:

```rust
use vello::{
    kurbo::{Affine, Circle},
    peniko::{Color, Fill},
    *,
};

// Initialize wgpu and get handles
let (width, height) = ...;
let device: wgpu::Device = ...;
let queue: wgpu::Queue = ...;
let surface: wgpu::Surface<'_> = ...;
let texture_format: wgpu::TextureFormat = ...;
let mut renderer = Renderer::new(
   &device,
   RendererOptions {
      surface_format: Some(texture_format),
      use_cpu: false,
      antialiasing_support: AaSupport::all(),
      num_init_threads: NonZeroUsize::new(1),
   },
).expect("Failed to create renderer");

// Create scene and draw stuff in it
let mut scene = Scene::new();
scene.fill(
   Fill::NonZero,
   Affine::IDENTITY,
   Color::from_rgba8(242, 140, 168, 255),
   None,
   &Circle::new((420.0, 200.0), 120.0),
);

// Draw more stuff
scene.push_layer(...);
scene.fill(...);
scene.stroke(...);
scene.pop_layer(...);

// Render to your window/buffer/etc.
let surface_texture = surface.get_current_texture()
   .expect("failed to get surface texture");
renderer
   .render_to_surface(
      &device,
      &queue,
      &scene,
      &surface_texture,
      &RenderParams {
         base_color: palette::css::BLACK, // Background color
         width,
         height,
         antialiasing_method: AaConfig::Msaa16,
      },
   )
   .expect("Failed to render to surface");
surface_texture.present();
```

See the [`examples`](https://github.com/linebender/vello/tree/main/examples) directory for code that integrates with frameworks like winit.

## Performance

We've observed 177 fps for the paris-30k test scene on an M1 Max, at a resolution of 1600 pixels square, which is excellent performance and represents something of a best case for the engine.

More formal benchmarks are on their way.

## Integrations

### SVG

A separate Linebender integration for rendering SVG files is available through [`vello_svg`](https://github.com/linebender/vello_svg).

### Lottie

A separate Linebender integration for playing Lottie animations is available through [`velato`](https://github.com/linebender/velato).

### Bevy

A separate Linebender integration for rendering raw scenes or Lottie and SVG files in [Bevy] through [`bevy_vello`](https://github.com/linebender/bevy_vello).

### Cosmic Text

An example scene demonstrating the integration of COSMIC text for font loading and text layout through [COSMIC Text Scene](https://github.com/linebender/vello/blob/main/examples/scenes/src/cosmic_text_scene.rs).
This scene can be run with:

```shell
cargo run --package with_winit --features cosmic_text -- --test-scenes
````

## Examples

Our examples are provided in separate packages in the [`examples`](https://github.com/linebender/vello/tree/main/examples) directory.
This allows them to have independent dependencies and faster builds.
Examples must be selected using the `--package` (or `-p`) Cargo flag.

### Winit

Our [winit] example ([examples/with_winit](https://github.com/linebender/vello/tree/main/examples/with_winit)) demonstrates rendering to a [winit] window.
By default, this renders the [GhostScript Tiger] as well as all SVG files you add in the [examples/assets/downloads](https://github.com/linebender/vello/tree/main/examples/assets/downloads) directory.
A custom list of SVG file paths (and directories to render all SVG files from) can be provided as arguments instead.
It also includes a collection of test scenes showing the capabilities of `vello`, which can be shown with `--test-scenes`.

```shell
cargo run -p with_winit
```

<!-- ### Headless -->

## Platforms

We aim to target all environments which can support WebGPU with the [default limits](https://www.w3.org/TR/webgpu/#limits).
We defer to [`wgpu`] for this support.
Other platforms are more tricky, and may require special building/running procedures.

### Web

Because Vello relies heavily on compute shaders, we rely on the emerging WebGPU standard to run on the web.
Browser support for WebGPU is still evolving.
Vello has been tested using production versions of Chrome, but WebGPU support in Firefox and Safari is still experimental.
It may be necessary to use development browsers and explicitly enable WebGPU.

The following command builds and runs a web version of the [winit demo](#winit).
This uses [`cargo-run-wasm`](https://github.com/rukai/cargo-run-wasm) to build the example for web, and host a local server for it

```shell
# Make sure the Rust toolchain supports the wasm32 target
rustup target add wasm32-unknown-unknown

# The binary name must also be explicitly provided as it differs from the package name
cargo run_wasm -p with_winit --bin with_winit_bin
```

There is also a web demo [available here](https://linebender.github.io/vello) on supporting web browsers.

> [!WARNING]
> The web is not currently a primary target for Vello, and WebGPU implementations are incomplete, so you might run into issues running this example.

### Android

The [`with_winit`](#winit) example supports running on Android, using [cargo apk](https://crates.io/crates/cargo-apk).

```shell
cargo apk run -p with_winit --lib
```

> [!TIP]
> cargo apk doesn't support running in release mode without configuration.
> See [their crates page docs](https://crates.io/crates/cargo-apk) (around `package.metadata.android.signing.<profile>`).
>
> See also [cargo-apk#16](https://github.com/rust-mobile/cargo-apk/issues/16).
> To run in release mode, you must add the following to `examples/with_winit/Cargo.toml` (changing `$HOME` to your home directory):

```toml
[package.metadata.android.signing.release]
path = "$HOME/.android/debug.keystore"
keystore_password = "android"
```

> [!NOTE]
> As `cargo apk` does not allow passing command line arguments or environment variables to the app when ran, these can be embedded into the
> program at compile time (currently for Android only)
> `with_winit` currently supports the environment variables:
>
> - `VELLO_STATIC_LOG`, which is equivalent to `RUST_LOG`
> - `VELLO_STATIC_ARGS`, which is equivalent to passing in command line arguments

For example (with unix shell environment variable syntax):

```sh
VELLO_STATIC_LOG="vello=trace" VELLO_STATIC_ARGS="--test-scenes" cargo apk run -p with_winit --lib
```

## Minimum supported Rust Version (MSRV)

This version of Vello has been verified to compile with **Rust 1.82** and later.

Future versions of Vello might increase the Rust version requirement.
It will not be treated as a breaking change and as such can even happen with small patch releases.

<details>
<summary>Click here if compiling fails.</summary>

As time has passed, some of Vello's dependencies could have released versions with a higher Rust requirement.
If you encounter a compilation issue due to a dependency and don't want to upgrade your Rust toolchain, then you could downgrade the dependency.

```sh
# Use the problematic dependency's name and version
cargo update -p package_name --precise 0.1.1
```

</details>

## Community

Discussion of Vello development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically the [#gpu stream](https://xi.zulipchat.com/#narrow/stream/197075-gpu).
All public content can be read without logging in.

Contributions are welcome by pull request.
The [Rust code of conduct] applies.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache 2.0 license, shall be licensed as noted in the [License](#license) section, without any additional terms or conditions.

## History

Vello was previously known as `piet-gpu`.
This prior incarnation used a custom cross-API hardware abstraction layer, called `piet-gpu-hal`, instead of [`wgpu`].

An archive of this version can be found in the branches [`custom-hal-archive-with-shaders`] and [`custom-hal-archive`].
This succeeded the previous prototype, [piet-metal], and included work adapted from [piet-dx12].

The decision to lay down `piet-gpu-hal` in favor of WebGPU is discussed in detail in the blog post [Requiem for piet-gpu-hal].

A [vision](https://github.com/linebender/vello/tree/main/doc/vision.md) document dated December 2020 explained the longer-term goals of the project, and how we might get there.
Many of these items are out-of-date or completed, but it still may provide some useful background.

## Related projects

Vello takes inspiration from many other rendering projects, including:

- [Pathfinder](https://github.com/servo/pathfinder)
- [Spinel](https://fuchsia.googlesource.com/fuchsia/+/refs/heads/master/src/graphics/lib/compute/spinel/)
- [Forma](https://github.com/google/forma)
- [Massively Parallel Vector Graphics](https://w3.impa.br/~diego/projects/GanEtAl14/)
- [Random-access rendering of general vector graphics](https://hhoppe.com/proj/ravg/)

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

In addition, all files in the [`vello_shaders/shader`](https://github.com/linebender/vello/tree/main/vello_shaders/shader) and [`vello_shaders/src/cpu`](https://github.com/linebender/vello/tree/main/vello_shaders/src/cpu) directories and subdirectories thereof are alternatively licensed under the Unlicense ([vello_shaders/shader/UNLICENSE](https://github.com/linebender/vello/tree/main/vello_shaders/shader/UNLICENSE) or <http://unlicense.org/>).
For clarity, these files are also licensed under either of the above licenses.
The intent is for this research to be used in as broad a context as possible.

The files in subdirectories of the [`examples/assets`](https://github.com/linebender/vello/tree/main/examples/assets) directory are licensed solely under their respective licenses, available in the `LICENSE` file in their directories.

[piet-metal]: https://github.com/linebender/piet-metal
[`wgpu`]: https://wgpu.rs/
[Xilem]: https://github.com/linebender/xilem/
[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
[`custom-hal-archive-with-shaders`]: https://github.com/linebender/piet-gpu/tree/custom-hal-archive-with-shaders
[`custom-hal-archive`]: https://github.com/linebender/piet-gpu/tree/custom-hal-archive
[piet-dx12]: https://github.com/bzm3r/piet-dx12
[GhostScript tiger]: https://commons.wikimedia.org/wiki/File:Ghostscript_Tiger.svg
[winit]: https://github.com/rust-windowing/winit
[Bevy]: https://bevyengine.org/
[Requiem for piet-gpu-hal]: https://raphlinus.github.io/rust/gpu/2023/01/07/requiem-piet-gpu-hal.html
[the changelog]: https://github.com/linebender/vello/tree/main/CHANGELOG.md
