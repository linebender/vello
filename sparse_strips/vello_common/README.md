<div align="center">

# Vello Common

**Shared data structures**

[![Latest published version.](https://img.shields.io/crates/v/vello_common.svg)](https://crates.io/crates/vello_common)
[![Documentation build status.](https://img.shields.io/docsrs/vello_common.svg)](https://docs.rs/vello_common)
[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![Linebender Zulip chat.](https://img.shields.io/badge/Linebender-%23vello-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/197075-vello)
[![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/linebender/vello/ci.yml?logo=github&label=CI)](https://github.com/linebender/vello/actions)
[![Dependency staleness status.](https://deps.rs/crate/vello_common/latest/status.svg)](https://deps.rs/crate/vello_common)

</div>

<!-- We use cargo-rdme to update the README with the contents of lib.rs.
To edit the following section, update it in lib.rs, then run:
cargo rdme --workspace-project=vello_common
Full documentation at https://github.com/orium/cargo-rdme -->

<!-- Intra-doc links used in lib.rs should be evaluated here.
See https://linebender.org/blog/doc-include/ for related discussion. -->

[libm]: https://crates.io/crates/libm
[crate::pixmap::Pixmap]: https://docs.rs/vello_common/latest/vello_common/pixmap/struct.Pixmap.html
[`glyph`]: https://docs.rs/vello_common/latest/vello_common/glyph/index.html

<!-- cargo-rdme start -->

This crate includes common geometry representations, tiling logic, and other fundamental components used by both [Vello CPU][vello_cpu] and Vello Hybrid.

## Usage

This crate should not be used on its own, and you should instead use one of the renderers which use it.
At the moment, only [Vello CPU][vello_cpu] is published, and you probably want to use that.

We also develop [Vello](https://crates.io/crates/vello), which makes use of the GPU for 2D rendering and has higher performance than Vello CPU.
Vello CPU is being developed as part of work to address shortcomings in Vello.
Vello does not use this crate.

## Features

- `std` (enabled by default): Get floating point functions from the standard library
  (likely using your target's libc).
- `libm`: Use floating point implementations from [libm][].
- `png` (enabled by default): Allow loading [`Pixmap`][crate::pixmap::Pixmap]s from PNG images.
  Also required for rendering glyphs with an embedded PNG.
  Implies `std`.
- `text` (enabled by default): Enables glyph rendering (see the [`glyph`][] module).

At least one of `std` and `libm` is required; `std` overrides `libm`.

## Contents

- Shared data structures for paths, tiles, and strips
- Geometry processing utilities
- Common logic for rendering stages

This crate acts as a foundation for `vello_cpu` and `vello_hybrid`, providing essential components to minimize duplication.

[vello_cpu]: https://crates.io/crates/vello_cpu

<!-- cargo-rdme end -->

## Minimum supported Rust Version (MSRV)

This version of Vello Common has been verified to compile with **Rust 1.88** and later.

Future versions of Vello Common might increase the Rust version requirement.
It will not be treated as a breaking change and as such can even happen with small patch releases.

<details>
<summary>Click here if compiling fails.</summary>

As time has passed, some of Vello Common's dependencies could have released versions with a higher Rust requirement.
If you encounter a compilation issue due to a dependency and don't want to upgrade your Rust toolchain, then you could downgrade the dependency.

```sh
# Use the problematic dependency's name and version
cargo update -p package_name --precise 0.1.1
```

</details>

## Community

Discussion of Vello Common development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically the [#vello channel](https://xi.zulipchat.com/#narrow/channel/197075-vello).
All public content can be read without logging in.

Contributions are welcome by pull request.
The [Rust code of conduct] applies.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
