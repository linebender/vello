<div align="center">

# Vello CPU

**CPU-based renderer**

[![Latest published version.](https://img.shields.io/crates/v/vello_cpu.svg)](https://crates.io/crates/vello_cpu)
[![Documentation build status.](https://img.shields.io/docsrs/vello_cpu.svg)](https://docs.rs/vello_cpu)
[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![Linebender Zulip chat.](https://img.shields.io/badge/Linebender-%23vello-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/197075-vello)
[![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/linebender/vello/ci.yml?logo=github&label=CI)](https://github.com/linebender/vello/actions)
[![Dependency staleness status.](https://deps.rs/crate/vello_cpu/latest/status.svg)](https://deps.rs/crate/vello_cpu)

</div>

<!-- We use cargo-rdme to update the README with the contents of lib.rs.
To edit the following section, update it in lib.rs, then run:
cargo rdme --workspace-project=vello_cpu
Full documentation at https://github.com/orium/cargo-rdme -->

<!-- Intra-doc links used in lib.rs should be evaluated here.
See https://linebender.org/blog/doc-include/ for related discussion. -->

[`RenderContext`]: https://docs.rs/vello_cpu/latest/vello_cpu/struct.RenderContext.html
[RenderContext::set_paint]: https://docs.rs/vello_cpu/latest/vello_cpu/struct.RenderContext.html#method.set_paint
[RenderContext::fill_path]: https://docs.rs/vello_cpu/latest/vello_cpu/struct.RenderContext.html#method.fill_path
[RenderContext::stroke_path]: https://docs.rs/vello_cpu/latest/vello_cpu/struct.RenderContext.html#method.stroke_path
[RenderContext::glyph_run]: https://docs.rs/vello_cpu/latest/vello_cpu/struct.RenderContext.html#method.glyph_run
[RenderMode::OptimizeSpeed]: https://docs.rs/vello_cpu/latest/vello_cpu/enum.RenderMode.html#variant.OptimizeSpeed
[RenderMode::OptimizeQuality]: https://docs.rs/vello_cpu/latest/vello_cpu/enum.RenderMode.html#variant.OptimizeQuality
[`RenderContext::render_to_pixmap`]: https://docs.rs/vello_cpu/latest/vello_cpu/struct.RenderContext.html#method.render_to_pixmap
[`Pixmap`]: https://docs.rs/vello_cpu/latest/vello_cpu/struct.Pixmap.html

<!-- cargo-rdme start -->

Vello CPU is a 2D graphics rendering engine written in Rust, for devices with no or underpowered GPUs.

We also develop [Vello](https://crates.io/crates/vello), which makes use of the GPU for 2D rendering and has higher performance than Vello CPU.
Vello CPU is being developed as part of work to address shortcomings in Vello.

## Usage

To use Vello CPU, you need to:

- Create a [`RenderContext`][], a 2D drawing context for a fixed-size target area.
- For each object in your scene:
  - Set how the object will be painted, using [`set_paint`][RenderContext::set_paint].
  - Set the shape to be drawn for that object, using methods like [`fill_path`][RenderContext::fill_path],
    [`stroke_path`][RenderContext::stroke_path], or [`glyph_run`][RenderContext::glyph_run].
- Render it to an image using [`RenderContext::render_to_pixmap`][].

```rust
use vello_cpu::{RenderContext, Pixmap, RenderMode};
use vello_cpu::{color::{palette::css, PremulRgba8}, kurbo::Rect};
let width = 10;
let height = 5;
let mut context = RenderContext::new(width, height);
context.set_paint(css::MAGENTA);
context.fill_rect(&Rect::from_points((3., 1.), (7., 4.)));

let mut target = Pixmap::new(width, height);
// While calling `flush` is only strictly necessary if you are rendering using
// multiple threads, it is recommended to always do this.
context.flush();
context.render_to_pixmap(&mut target);

let expected_render = b"\
    0000000000\
    0001111000\
    0001111000\
    0001111000\
    0000000000";
let magenta = css::MAGENTA.premultiply().to_rgba8();
let transparent = PremulRgba8 {r: 0, g: 0, b: 0, a: 0};
let mut result = Vec::new();
for pixel in target.data() {
    if *pixel == magenta {
        result.push(b'1');
    } else if *pixel == transparent {
        result.push(b'0');
    } else {
         panic!("Got unexpected pixel value {pixel:?}");
    }
}
assert_eq!(&result, expected_render);
```

Feel free to take a look at some further
[examples](https://github.com/linebender/vello/tree/main/sparse_strips/vello_cpu/examples)
to better understand how to interact with Vello CPU's API,

## Features

- `std` (enabled by default): Get floating point functions from the standard library
  (likely using your target's libc).
- `libm`: Use floating point implementations from [libm][].
- `png`(enabled by default): Allow loading [`Pixmap`]s from PNG images.
  Also required for rendering glyphs with an embedded PNG. Implies `std`.
- `multithreading`: Enable multi-threaded rendering. Implies `std`.
- `text` (enabled by default): Enables glyph rendering ([`glyph_run`][RenderContext::glyph_run]).
- `u8_pipeline` (enabled by default): Enable the u8 pipeline, for speed focused rendering using u8 math.
  The `u8` pipeline will be used for [`OptimizeSpeed`][RenderMode::OptimizeSpeed], if both pipelines are enabled.
  If you're using Vello CPU for application rendering, you should prefer this pipeline.
- `f32_pipeline`: Enable the `f32` pipeline, which is slower but has more accurate
  results. This is espectially useful for rendering test snapshots.
  The `f32` pipeline will be used for [`OptimizeQuality`][RenderMode::OptimizeQuality], if both pipelines are enabled.

At least one of `std` and `libm` is required; `std` overrides `libm`.
At least one of `u8_pipeline` and `f32_pipeline` must be enabled.
You might choose to disable one of these pipelines if your application
won't use it, so as to reduce binary size.

## Caveats

Overall, Vello CPU is already very feature-rich and should be ready for
production use cases. The main caveat at the moment is that the API is
still likely to change and not stable yet. For example, we have
known plans to change the API around how image resources are used.

Additionally, there are certain APIs that are still very much experimental,
including for example support for filters. This will be reflected in the
documentation of those APIs.

Another caveat is that multi-threading with large thread counts
(more than 4) might give diminishing returns, especially when
making heavy use of layers and clip paths.

## Performance

Performance benchmarks can be found [here](https://laurenzv.github.io/vello_chart/),
As can be seen, Vello CPU achieves compelling performance on both,
aarch64 and x86 platforms. We also have SIMD optimizations for WASM SIMD,
meaning that you can expect good performance there as well.

## Implementation

If you want to gain a better understanding of Vello CPU and the
sparse strips paradigm, you can take a look at the [accompanying
master's thesis](https://ethz.ch/content/dam/ethz/special-interest/infk/inst-pls/plf-dam/documents/StudentProjects/MasterTheses/2025-Laurenz-Thesis.pdf)
that was written on the topic. Note that parts of the descriptions might
become outdated as the implementation changes, but it should give a good
overview nevertheless.

<!-- We can't directly link to the libm crate built locally, because our feature is only a pass-through  -->
[libm]: https://crates.io/crates/libm

<!-- cargo-rdme end -->

## Minimum supported Rust Version (MSRV)

This version of Vello CPU has been verified to compile with **Rust 1.88** and later.

Future versions of Vello CPU might increase the Rust version requirement.
It will not be treated as a breaking change and as such can even happen with small patch releases.

<details>
<summary>Click here if compiling fails.</summary>

As time has passed, some of Vello CPU's dependencies could have released versions with a higher Rust requirement.
If you encounter a compilation issue due to a dependency and don't want to upgrade your Rust toolchain, then you could downgrade the dependency.

```sh
# Use the problematic dependency's name and version
cargo update -p package_name --precise 0.1.1
```

</details>

## Community

Discussion of Vello CPU development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically the [#vello channel](https://xi.zulipchat.com/#narrow/channel/197075-vello).
All public content can be read without logging in.

Contributions are welcome by pull request.
The [Rust code of conduct] applies.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
