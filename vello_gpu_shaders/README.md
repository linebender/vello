<div align="center">

# Vello GPU Shaders

[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![Linebender Zulip chat.](https://img.shields.io/badge/Linebender-%23vello-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/197075-vello) [![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/linebender/vello/ci.yml?logo=github&label=CI)](https://github.com/linebender/vello/actions)

</div>

This crate contains the WESL programs and linked WGSL output, plus optionally
generated GLSL shader programs, used by the Vello GPU renderer.

## Features

- Single source of truth authored as [WESL](https://wesl-lang.dev/) programs.
- Automated build step that links WESL, then uses
  [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) to minify the resulting
  WGSL.
- Optional generation of minified GLSL and reflection metadata for WebGL.
- The `blurred_rounded_rect`, `full_image_support`, and `gradient_sweep` features are
  enabled by default and include their respective shader paths.
- The `wgpu_external_texture` feature (enabled by default) additionally embeds a
  WGSL variant of the render shader with a `texture_external` binding for
  `wgpu::ExternalTexture`s. The generated GLSL never includes it.

## Usage

This crate provides linked WGSL programs and the build step for GLSL programs
used by Vello GPU.

Whenever the WESL shaders are updated, the build script automatically relinks
and minifies the WGSL. When the `glsl` feature is enabled, it also regenerates
the minified GLSL programs and reflection metadata used by `vello_gpu`.

To inspect the generated WebGL GLSL and the embedded `compiled_shaders.rs`
module used by `vello_gpu`, create local copies by running:

```sh
cargo run -p vello_gpu_shaders --features glsl
```

The generated files will be written into the `generated_glsl` folder.

To retain authored identifiers and Naga's readable formatting for debugging,
enable the diagnostic `unminified` feature as well:

```sh
cargo run -p vello_gpu_shaders --features glsl,unminified
```

## Formatting

The WESL sources are formatted with [`wgslfmt`] using its default options
(4-space indentation). `wgslfmt` is the formatter bundled with the
[wgsl-analyzer] editor extension, so formatting a `.wesl` file from an editor
with that extension produces the same result.

Install the pinned version (building it requires Rust 1.98.1 or later):

```sh
cargo install --git https://github.com/wgsl-analyzer/wgsl-analyzer --tag 2026-09-30 wgslfmt --locked
```

Format the shaders, or check them without modifying any files (CI runs the
check on every pull request):

```sh
wgslfmt vello_gpu_shaders/shaders
wgslfmt --check vello_gpu_shaders/shaders
```

In VS Code-based editors, the repository's workspace settings make wgsl-analyzer
the default formatter for `.wesl` files; enable `editor.formatOnSave` for
`[wesl]` to format on save.

[`wgslfmt`]: https://github.com/wgsl-analyzer/wgsl-analyzer/tree/main/crates/wgslfmt
[wgsl-analyzer]: https://marketplace.visualstudio.com/items?itemName=wgsl-analyzer.wgsl-analyzer

## Minimum supported Rust Version (MSRV)

This version of Vello GPU Shaders has been verified to compile with **Rust
1.89** and later.

Future versions of Vello GPU Shaders might increase the Rust version
requirement. It will not be treated as a breaking change and as such can even
happen with small patch releases.

<details>
<summary>Click here if compiling fails.</summary>

As time has passed, some of Vello GPU's dependencies could have released
versions with a higher Rust requirement. If you encounter a compilation issue
due to a dependency and don't want to upgrade your Rust toolchain, then you
could downgrade the dependency.

```sh
# Use the problematic dependency's name and version
cargo update -p package_name --precise 0.1.1
```

</details>

## Community

Discussion of Vello GPU development happens in the
[Linebender Zulip](https://xi.zulipchat.com/), specifically the
[#vello channel](https://xi.zulipchat.com/#narrow/channel/197075-vello). All
public content can be read without logging in.

Contributions are welcome by pull request. The [Rust code of conduct] applies.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.

[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
