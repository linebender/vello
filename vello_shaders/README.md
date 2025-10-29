<div align="center">

# Vello Shaders

**Integrate [Vello] shaders into any renderer project**

[![Latest published version.](https://img.shields.io/crates/v/vello_shaders.svg)](https://crates.io/crates/vello_shaders)
[![Documentation build status.](https://img.shields.io/docsrs/vello_shaders.svg)](https://docs.rs/vello_shaders)
[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![Linebender Zulip chat.](https://img.shields.io/badge/Linebender-%23vello-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/197075-vello)
[![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/linebender/vello/ci.yml?logo=github&label=CI)](https://github.com/linebender/vello/actions)
[![Dependency staleness status.](https://deps.rs/crate/vello_shaders/latest/status.svg)](https://deps.rs/crate/vello_shaders)

</div>

This is a utility library to help integrate the [Vello] shader modules into any renderer project.
It provides the necessary metadata to construct the individual compute pipelines on any GPU API while leaving the responsibility of all API interactions (such as resource management and command encoding) up to the client.

The shaders can be pre-compiled to any target shading language at build time based on feature flags.
Currently only WGSL and Metal Shading Language are supported.

Significant changes are documented in [the changelog].

## Minimum supported Rust Version (MSRV)

This version of Vello Shaders has been verified to compile with **Rust 1.88** and later.

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

## Community

Discussion of Vello Shaders development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically the [#vello channel](https://xi.zulipchat.com/#narrow/channel/197075-vello).
All public content can be read without logging in.

Contributions are welcome by pull request.
The [Rust code of conduct] applies.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

In addition, all files in the [`shader`](https://github.com/linebender/vello/tree/main/vello_shaders/shader) and [`src/cpu`](https://github.com/linebender/vello/tree/main/vello_shaders/src/cpu) directories and subdirectories thereof are alternatively licensed under the Unlicense ([shader/UNLICENSE](https://github.com/linebender/vello/tree/main/vello_shaders/shader/UNLICENSE) or <http://unlicense.org/>).
For clarity, these files are also licensed under either of the above licenses.
The intent is for this research to be used in as broad a context as possible.

[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
[Vello]: https://github.com/linebender/vello
[the changelog]: https://github.com/linebender/vello/tree/main/CHANGELOG.md
