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

<!-- cargo-rdme start -->

This crate implements the CPU-based renderer for Vello.
It is optimized for running on lower-end GPUs or CPU-only environments, leveraging parallelism and SIMD acceleration where possible.

This crate implements a CPU-based renderer, optimized for SIMD and multithreaded execution.
It is optimized for CPU-bound workloads and serves as a standalone renderer for systems
without GPU acceleration.

## Features

- Fully CPU-based path rendering pipeline
- SIMD and multithreading optimizations
- Fine rasterization and antialiasing techniques

## Usage

This crate serves as a standalone renderer or as a fallback when GPU resources are constrained.

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
