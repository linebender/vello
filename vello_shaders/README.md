<div align="center">

# Vello Shaders

**Integrate [Vello] shaders into any renderer project**

[![Latest published version.](https://img.shields.io/crates/v/vello_shaders.svg)](https://crates.io/crates/vello_shaders)
[![Documentation build status.](https://img.shields.io/docsrs/vello_shaders.svg)](https://docs.rs/vello_shaders)
[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![Linebender Zulip chat.](https://img.shields.io/badge/Linebender-%23gpu-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/stream/197075-gpu)
[![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/linebender/vello/ci.yml?logo=github&label=CI)](https://github.com/linebender/vello/actions)
[![Dependency staleness status.](https://deps.rs/crate/vello_shaders/latest/status.svg)](https://deps.rs/crate/vello_shaders)

</div>

This is a utility library to help integrate the [Vello] shader modules into any renderer project.
It provides the necessary metadata to construct the individual compute pipelines on any GPU API while leaving the responsibility of all API interactions (such as resource management and command encoding) up to the client.

The shaders can be pre-compiled to any target shading language at build time based on feature flags.
Currently only WGSL and Metal Shading Language are supported.

## Minimum supported Rust Version (MSRV)

This version of Vello Shaders has been verified to compile with **Rust 1.75** and later.

Future versions of Vello Shaders might increase the Rust version requirement.
It will not be treated as a breaking change and as such can even happen with small patch releases.

<details>
<summary>Click here if compiling fails.</summary>

As time has passed, some of Vello Shaders' dependencies could have released versions with a higher Rust requirement.
If you encounter a compilation issue due to a dependency and don't want to upgrade your Rust toolchain, then you could downgrade the dependency.

```sh
# Use the problematic dependency's name and version
cargo update -p package_name --precise 0.1.1
```
</details>

[Vello]: https://github.com/linebender/vello
