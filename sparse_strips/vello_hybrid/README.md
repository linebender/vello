<div align="center">

# Vello Hybrid

**Hybrid CPU/GPU renderer**

[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![Linebender Zulip chat.](https://img.shields.io/badge/Linebender-%23vello-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/197075-vello)
[![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/linebender/vello/ci.yml?logo=github&label=CI)](https://github.com/linebender/vello/actions)

</div>

<!-- We use cargo-rdme to update the README with the contents of lib.rs.
To edit the following section, update it in lib.rs, then run:
cargo rdme --workspace-project=vello_hybrid
Full documentation at https://github.com/orium/cargo-rdme -->

<!-- Intra-doc links used in lib.rs should be evaluated here.
See https://linebender.org/blog/doc-include/ for related discussion. -->

<!-- cargo-rdme start -->

A hybrid CPU/GPU renderer for 2D vector graphics.

This crate provides a rendering API that combines CPU and GPU operations for efficient
vector graphics processing.
The hybrid approach balances flexibility and performance by:

- Using the CPU for path processing and initial geometry setup
- Leveraging the GPU for fast rendering and compositing
- Minimizing data transfer between CPU and GPU

## Key Features

- Efficient path rendering with CPU-side processing
- GPU-accelerated compositing and blending
- Support for both windowed and headless rendering

## Feature Flags

- `wgpu` (enabled by default): Enables the GPU rendering backend via wgpu and includes the required sparse shaders.
- `wgpu_default` (enabled by default): Enables wgpu with its default hardware backends (such as Vulkan, Metal, and DX12).
- `webgl`: Enables the WebGL rendering backend for browser support, using GLSL shaders for compatibility.

If you need to customize the set of enabled wgpu features, disable this crate's default features then enable its `wgpu` feature.
You can then depend on wgpu directly, setting the specific features you require.
Don't forget to also disable wgpu's default features.

## Architecture

The renderer is split into several key components:

- `Scene`: Manages the render context and path processing on the CPU
- `Renderer` or `WebGlRenderer`: Handles GPU resource management and executes draw operations
- `Scheduler`: Manages and schedules draw operations on the renderer.

See the individual module documentation for more details on usage and implementation.

<!-- cargo-rdme end -->

## Minimum supported Rust Version (MSRV)

This version of Vello Hybrid has been verified to compile with **Rust 1.88** and later.

Future versions of Vello Hybrid might increase the Rust version requirement.
It will not be treated as a breaking change and as such can even happen with small patch releases.

<details>
<summary>Click here if compiling fails.</summary>

As time has passed, some of Vello Hybrid's dependencies could have released versions with a higher Rust requirement.
If you encounter a compilation issue due to a dependency and don't want to upgrade your Rust toolchain, then you could downgrade the dependency.

```sh
# Use the problematic dependency's name and version
cargo update -p package_name --precise 0.1.1
```

</details>

## Community

Discussion of Vello Hybrid development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically the [#vello channel](https://xi.zulipchat.com/#narrow/channel/197075-vello).
All public content can be read without logging in.

Contributions are welcome by pull request.
The [Rust code of conduct] applies.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
