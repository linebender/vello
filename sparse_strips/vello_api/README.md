<div align="center">

# Vello API

**Experimental public API for Vello Renderers**

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

<!-- N.B. These links are intentionally incomplete, as the below doc will need to be rewritten -->

<!-- cargo-rdme start -->

Vello API is the **experimental** rendering API of the 2D renderers in the Vello project.

Warning: Vello API is currently only released as a preview, and is not ready for usage beyond short-term experiments.
We know of several design decisions which will change before its use can be recommended.

There are currently two [supported Vello renderers](#renderers), each with different tradeoffs.
This crate allows you to write the majority of your application logic to support either of those renderers.
These renderers are [Vello CPU](todo) and [Vello Hybrid](todo).

## Usage

TODO: Mention Renderer trait when it exists. Otherwise, this code isn't really usable yet.
TODO: This is a stub just to have an outline to push.

## Renderers

The Vello renderers which support this API are:

- Vello CPU, an extremely portable 2D renderer which does not require a GPU.
  It is one of the fastest CPU-only 2D renderers in Rust.
- Vello Hybrid, which runs the most compute intensive portions of rendering on the GPU, improving performance over Vello CPU.
  It will have wide compatibility with most devices, so long as they have a GPU, including running well on the web.
<!-- We might also have, to be determined:
- Vello Classic, which performs almost all rendering on the GPU, which gives great performance on devices with decent GPUs.
  However, it cannot run well on devices with weak GPUs, or in contexts without support for compute shaders, such as the web.
  It also has unavoidably high memory usage, and can silently fail to render if the scene gets too big.
-->

Currently, Vello CPU is much more mature, and so most consumers should currently prefer that implementation.
We hope that in the not-too distant future, application developers will be able to migrate to Vello Hybrid.
We expect headless use cases (such as screenshot tests or server-rendered previews) to prefer Vello CPU, due to
its cross-platform consistency and lower latency.

Note that even once Vello Hybrid is more stable, applications using Vello Hybrid might need to support falling
back to Vello CPU for compatibility or performance reasons.

This abstraction is tailored for the Vello renderers, as we believe that these have a sufficiently broad coverage of the trade-off
space to be viable for any consumer.
Vello API guarantees identical rendering between renderers which implement it, barring subpixel differences due to precision/different rounding.
This doesn't apply to renderer-specific features.
<!-- TODO: Is ^ true? -->

## Abstraction Boundaries

The abstractions in this crate are focused on 2D rendering, and the resources required to perform that.
In particular, this does not abstract over strategies for:

- creating the renderer (which can require more context, such as a wgpu `Device`); nor
- bringing external content into the renderer (for example, already resident GPU textures); nor
- presenting rendered content to an operating system window.

These functionalities are however catered for where applicable by APIs on the specific renderers.
The renderer API supports downcasting to the specific renderer, so that these extensions can be called.
Each supported renderer will/does have examples showing how to achieve this yourself.

## Text

Vello API does not handle text/glyph rendering itself.
This allows for improved resource sharing of intermediate text layout data, for hinting and ink splitting underlines.

Text can be rendered to Vello API scenes using the "Parley Draw" crate.
Note that this crate is not currently implemented; design work is ongoing.
We also support rendering using using traditional glyph atlases, which may be preferred by some consumers.
This is especially useful to achieve subpixel rendering, such as ClearType, which Vello doesn't currently support directly.

## Unimplemented Features

NOTE: This section is not complete; in particular, we have only pushed a half-version of this API to make review more scoped.

The current version of Vello API is a minimal viable product for exploration and later expansion.
As such, there are several features which we expect to be included in this API, but which are not yet exposed in this crate.
These are categorised as follows:

### Out of scope/Renderer specific

<!-- This section can be removed once the other three classes are empty -->
As discussed above, some features are out-of-scope, as they have concerns which need to be handled individually by each renderer.
This includes:

- Rendering directly to a surface.
- Importing "external" textures (e.g. from a `wgpu::Texture`)

### Excluded for expedience

- Renderer specific painting commands (i.e. using downcasting).
  This is intended to be an immediate follow-up to the MVP landing.
- Pushing/popping clip paths (i.e. non-isolated clipping).
  This feature should be easy to restore, although it isn't clear how it will work with "Vello GPU", i.e. Hybrid with GPU sparse strip rendering.
- Downloading rendered textures back to the CPU/host.
  This is currently supported through individual methods on the renderers, but we hope to have a portable API for coordinating this.
- Anti-aliasing threshold.
- Fill rules for clip paths.
- Proper error types; we currently return `()`. This is to ensure that these error types are explicitly interim.
- Texture resizing; this might not be viably possible
- More explicit texture atlas support
- Proper handling of where `TextureHandle` and `TextureId` should be passed.
- "Unsetting" the brush; this is mostly useful for append style operations, which may unexpectedly change the brush.
- Dynamic version of `PaintScene`, allowing `dyn PaintScene` for cases where that would be relevant.

### Not cross-renderer

There are some features which are only implemented in one of our target renderers, so cannot yet be included in the generalised API.
As an interim solution, we intend to expose these features as renderer specific extensions (see the "excluded for expedience section").

For Vello CPU, these are (i.e. Vello Hybrid does not implement these):

- Masks
- Filter effects
- Non-isolated blending (this is "supported" by Vello Hybrid, but currently silently ignored)
- Blurred rounded rectangles (note that currently this is actually included in the abstraction, despite this status)

There are currently no such features the other way around (i.e. which only Vello Hybrid supports).

### Not implemented

- Path caching. This feature is intended to allow re-using paths efficiently, primarily for glyphs.
- Blurred rounded rectangle paints in custom shapes (e.g. to exclude the unblurred parts).
  (TODO: This actually does exist as a method, but no renderer implements it; we should maybe remove that method?)
- Mipmaps

For even more detail on some of these, see the `design.md` file.
Note however that file is very uncurated.

<!-- cargo-rdme end -->

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
