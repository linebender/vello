<div align="center">

# Glifo

**APIs for efficiently rendering text**

[![Latest published version.](https://img.shields.io/crates/v/glifo.svg)](https://crates.io/crates/glifo)
[![Documentation build status.](https://img.shields.io/docsrs/glifo.svg)](https://docs.rs/glifo)
[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![Linebender Zulip chat.](https://img.shields.io/badge/Linebender-%23vello-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/197075-vello)
[![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/linebender/vello/ci.yml?logo=github&label=CI)](https://github.com/linebender/vello/actions)
[![Dependency staleness status.](https://deps.rs/crate/glifo/latest/status.svg)](https://deps.rs/crate/glifo)

</div>

<!-- We use cargo-rdme to update the README with the contents of lib.rs.
To edit the following section, update it in lib.rs, then run:
cargo rdme --workspace-project=glifo
Full documentation at https://github.com/orium/cargo-rdme -->

<!-- Intra-doc links used in lib.rs should be evaluated here.
See https://linebender.org/blog/doc-include/ for related discussion. -->

<!-- cargo-rdme start -->

Glifo provides APIs for efficiently rendering glyphs and paint styles like underline.

## Goals

Glifo is under rapid development. Consider it experimental for now. Its goals are to:

- Provide an API surface that accepts glyphs and their positions and renders them to a surface.
- Cache those glyphs so that repeated renders of a glyph are fast.
- Support rendering paint styles like underline, strikethrough, and brush color.
- Share expensive structs and data between the shaper and renderer like the hinting instance and hinted advance.

## Features

- `std` (enabled by default): Get floating point functions from the standard library
  (likely using your target's libc).
- `libm`: Use floating point implementations from `libm`.
- `png`: Enables PNG support for drawing bitmap glyphs.

At least one of `std` and `libm` is required.

<!-- cargo-rdme end -->

## Minimum supported Rust Version (MSRV)

This version of Glifo has been verified to compile with **Rust 1.88** and later.

Future versions of Glifo might increase the Rust version requirement.
It will not be treated as a breaking change and as such can even happen with small patch releases.

## Community

Discussion of Glifo development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically the [#vello channel](https://xi.zulipchat.com/#narrow/channel/197075-vello).
All public content can be read without logging in.

Contributions are welcome by pull request.
The [Rust code of conduct] applies.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
