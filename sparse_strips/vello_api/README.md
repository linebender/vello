<div align="center">

# Vello API

**Public API types**

[![Latest published version.](https://img.shields.io/crates/v/vello_api.svg)](https://crates.io/crates/vello_api)
[![Documentation build status.](https://img.shields.io/docsrs/vello_api.svg)](https://docs.rs/vello_api)
[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![Linebender Zulip chat.](https://img.shields.io/badge/Linebender-%23vello-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/197075-vello)
[![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/linebender/vello/ci.yml?logo=github&label=CI)](https://github.com/linebender/vello/actions)
[![Dependency staleness status.](https://deps.rs/crate/vello_api/latest/status.svg)](https://deps.rs/crate/vello_api)

</div>

<!-- We use cargo-rdme to update the README with the contents of lib.rs.
To edit the following section, update it in lib.rs, then run:
cargo rdme --workspace-project=vello_api
Full documentation at https://github.com/orium/cargo-rdme -->

<!-- Intra-doc links used in lib.rs should be evaluated here.
See https://linebender.org/blog/doc-include/ for related discussion. -->

<!-- markdownlint-disable MD053 -->
<!-- cargo-rdme start -->

Vello API is the rendering API of the 2d renderers in the Vello project.

There are currently two [supported Vello renderers](#renderers), each with different tradeoffs.
This crate allows you to write the majority of your application logic to support either of those renderers.
These renderers are [Vello CPU](todo) and [Vello Hybrid](todo).

## Usage

The main entry-point in this crate is the [`Renderer`] trait, which is implemented by [`VelloCPU`](todo) and [`VelloHybrid`](todo).
Once you have a created a renderer, you then create scenes.
These are then scheduled to be run against specific textures.
You can also make textures from CPU content.

TODO: This is a stub just to have an outline to push.

## Renderers

The Vello renderers which support this API are:

- Vello CPU, an extremely portable 2d renderer which does not require a GPU.
  It is one of the fastest CPU-only 2d renderers in Rust.
- Vello Hybrid, which runs the most compute intensive portions of rendering on the GPU, improving performance over Vello CPU.
  It has wide compatibility with most devices, so long as they have a GPU, and it runs well on the web.
<!-- We might also have, to be determined:
- Vello Classic, which performs almost all rendering on the GPU, which gives great performance on devices with decent GPUs.
  However, it cannot run well on devices with weak GPUs, or in contexts without support for compute shaders, such as the web.
  It also has unavoidably high memory usage, and can silently fail to render if the scene gets too big.
-->

As a general guide for consumers, you should prefer Vello Hybrid for applications, and Vello CPU for headless use cases
(e.g. screenshot tests or server-rendered previews).
Note that applications using Vello Hybrid might need to support falling back to Vello CPU for compatibility or performance reasons.

This abstraction is tailored for the Vello renderers, as we believe that these have a sufficiently broad coverage of the trade-off
space to be viable for any consumer.
Vello API guarantees identical rendering between these renderers, barring subpixel differences due to precision/different rounding.
<!-- TODO: Is ^ true? -->

## Abstraction Boundaries

The abstractions in this crate are focused on 2d rendering, and the resources required to perform that.
In particular, this does abstract over strategies for:

- creating the renderer.
- bringing external content into the renderer (for example, already resident GPU textures); nor
- presenting rendered content to an operating system window.

These functionalities are however catered for where applicable by APIs on the specific renderers.
The renderer API supports downcasting to the specific renderer, so that these extensions can be called.
Each supported renderer will/does have examples showing how to achieve this yourself.

## Text

Vello API does not handle text/glyph rendering itself.
This allows for improved resource sharing of intermediate text layout data, for hinting and ink splitting underlines.

Text can be rendered to Vello API scenes using the "Parley Draw" crate.
We also support rendering using using traditional glyph atlases, which may be preferred by some consumers.
This is especially useful to achieve subpixel rendering, such as ClearType, which Vello doesn't currently support directly.

<!-- cargo-rdme end -->
<!-- markdownlint-enable MD053 -->

## Minimum supported Rust Version (MSRV)

This version of Vello API has been verified to compile with **Rust 1.88** and later.

Future versions of Vello API might increase the Rust version requirement.
It will not be treated as a breaking change and as such can even happen with small patch releases.

<details>
<summary>Click here if compiling fails.</summary>

As time has passed, some of Vello API's dependencies could have released versions with a higher Rust requirement.
If you encounter a compilation issue due to a dependency and don't want to upgrade your Rust toolchain, then you could downgrade the dependency.

```sh
# Use the problematic dependency's name and version
cargo update -p package_name --precise 0.1.1
```

</details>

## Community

Discussion of Vello API development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically the [#vello stream](https://xi.zulipchat.com/#narrow/channel/197075-vello).
All public content can be read without logging in.

Contributions are welcome by pull request.
The [Rust code of conduct] applies.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
