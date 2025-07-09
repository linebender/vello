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
cargo rdme --workspace-project=vello_cpu --heading-base-level=0
Full documentation at https://github.com/orium/cargo-rdme -->

<!-- Intra-doc links used in lib.rs should be evaluated here.
See https://linebender.org/blog/doc-include/ for related discussion. -->

[`RenderContext`]: https://docs.rs/vello_cpu/latest/vello_cpu/struct.RenderContext.html
[RenderContext::set_paint]: https://docs.rs/vello_cpu/latest/vello_cpu/struct.RenderContext.html#method.set_paint
[RenderContext::fill_path]: https://docs.rs/vello_cpu/latest/vello_cpu/struct.RenderContext.html#method.fill_path
[RenderContext::stroke_path]: https://docs.rs/vello_cpu/latest/vello_cpu/struct.RenderContext.html#method.stroke_path
[RenderContext::glyph_run]: https://docs.rs/vello_cpu/latest/vello_cpu/struct.RenderContext.html#method.glyph_run
[`RenderContext::render_to_pixmap`]: https://docs.rs/vello_cpu/latest/vello_cpu/struct.RenderContext.html#method.render_to_pixmap
[`Pixmap`]: https://docs.rs/vello_cpu/latest/vello_cpu/struct.Pixmap.html
[libm]: https://crates.io/crates/libm

<!-- cargo-rdme start -->

Vello CPU is a 2D graphics rendering engine written in Rust, for devices with no or underpowered GPUs.

It is currently available as an alpha.
See the [Caveats](#caveats) section for things you need to be aware of.

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
// This is only necessary if you activated the `multithreading` feature.
context.flush();
context.render_to_pixmap(&mut target, RenderMode::default());

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

## Features

- `std` (enabled by default): Get floating point functions from the standard library
  (likely using your target's libc).
- `libm`: Use floating point implementations from `libm`.
- `png`(enabled by default): Allow loading [`Pixmap`]s from PNG images.
  Also required for rendering glyphs with an embedded PNG.
- `multithreading`: Enable multi-threaded rendering.

At least one of `std` and `libm` is required; `std` overrides `libm`.

## Caveats

Vello CPU is an alpha for several reasons, including the following.

### API stability

This API has been developed for an initial version, and has no stability guarantees.
Whilst we are in the `0.0.x` release series, any release is likely to breaking.
We have known plans to change the API around how image resources are used.

### Documentation

We have not yet put any work into documentation.

### Performance

We do not perform several important optimisations, such as the use of multithreading and SIMD.
Additionally, some algorithms we use aren't final, and will be replaced with higher-performance variants.

## Implementation

TODO: Point to documentation of sparse strips pattern.

<!-- cargo-rdme end -->

## Minimum supported Rust Version (MSRV)

This version of Vello CPU has been verified to compile with **Rust 1.85** and later.

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
