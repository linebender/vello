<!--

This repo-level readme needs restructuring, pending some Linebender templating decisions.
https://xi.zulipchat.com/#narrow/channel/419691-linebender/topic/Bikeshedding.20badges/with/452312397

For now, prefer updating the package-level readmes, e.g. vello/README.md.

-->

<div align="center">

# Vello

**A GPU compute-centric 2D renderer**

[![Linebender Zulip](https://img.shields.io/badge/Linebender-%23vello-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/197075-vello)
[![dependency status](https://deps.rs/repo/github/linebender/vello/status.svg)](https://deps.rs/repo/github/linebender/vello)
[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
[![wgpu version](https://img.shields.io/badge/wgpu-v29.0.1-orange.svg)](https://crates.io/crates/wgpu)

[![Crates.io](https://img.shields.io/crates/v/vello.svg)](https://crates.io/crates/vello)
[![Docs](https://docs.rs/vello/badge.svg)](https://docs.rs/vello)
[![Build status](https://github.com/linebender/vello/workflows/CI/badge.svg)](https://github.com/linebender/vello/actions)

</div>

The Vello project is a set of high-performance vector renderers written in Rust:

- **Vello** is an experimental renderer with a focus on doing as much work as possible with GPU compute.
- **Vello CPU** is a pure Rust, CPU-only renderer optimized for multithreading and SIMD.
- **Vello Hybrid** is a renderer that does heavy pre-processing on the CPU but still does most of the work on the GPU. It aims to be the main Vello implementation for production use-cases.

Quickstart to run an example program:

```shell
cargo run -p with_winit
```

![image](https://github.com/linebender/vello/assets/8573618/cc2b742e-2135-4b70-8051-c49aeddb5d19)

Significant changes are documented in [the changelog].

## Motivation

Vello is meant to fill the same place in the graphics stack as other vector graphics renderers like [Skia](https://skia.org/), [Cairo](https://www.cairographics.org/), and its predecessor project [Piet](https://github.com/linebender/piet).
On a basic level, that means it provides tools to render shapes, images, gradients, text, etc, using a PostScript-inspired API, the same that powers SVG files and [the browser `<canvas>` element](https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D).

The selling point of Vello's implementations is that they get better performance than other renderers by better leveraging the SIMD, multithreading and the GPU.


## Minimum supported Rust Version (MSRV)

This version of Vello has been verified to compile with **Rust 1.88** and later.

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

Discussion of Vello development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically the [#vello channel](https://xi.zulipchat.com/#narrow/channel/197075-vello).
All public content can be read without logging in.

Contributions are welcome by pull request.
The [Rust code of conduct] applies.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache 2.0 license, shall be licensed as noted in the [License](#license) section, without any additional terms or conditions.

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
