<div align="center">

# Vello Hybrid

**Hybrid CPU/GPU renderer**

[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![Linebender Zulip chat.](https://img.shields.io/badge/Linebender-%23gpu-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/stream/197075-gpu)
[![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/linebender/vello/ci.yml?logo=github&label=CI)](https://github.com/linebender/vello/actions)

</div>

This crate implements a hybrid CPU/GPU renderer for Vello. It offloads fine rasterization and other GPU-suited tasks while keeping core path processing on the CPU, making it a balanced solution for a variety of hardware.

## Features
- Hybrid rendering approach with CPU-based tiling and GPU-accelerated fine rasterization
- Efficient path processing with sparse strip representation
- Designed for performance across different hardware capabilities

## Usage
This crate serves as an optimized hybrid rendering engine, leveraging both CPU and GPU where appropriate.

## Minimum supported Rust Version (MSRV)

TODO: Fill in the MSRV when known.

## Community

Discussion of Vello Hybrid development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically the [#gpu stream](https://xi.zulipchat.com/#narrow/stream/197075-gpu).
All public content can be read without logging in.

Contributions are welcome by pull request.
The [Rust code of conduct] applies.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
[Vello]: https://github.com/linebender/vello
[the changelog]: https://github.com/linebender/vello/tree/main/CHANGELOG.md
