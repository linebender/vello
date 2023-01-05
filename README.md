# vello

This repo contains the new prototype for a new compute-centric 2D GPU renderer (formerly known as piet-gpu).

> **Warning**  
> This README is a work in progress. Previous versions are included below for reference

It succeeds the previous prototype, [piet-metal].

The latest version is a middleware for [`wgpu`]. This is used as the rendering backend for
[xilem], a UI toolkit.

<!-- TODO: Are we transitioning to more production? If so, should we rewrite the README a bit? -->

## Goals

The main goal is to answer research questions about the future of 2D rendering:

-   Is a compute-centered approach better than rasterization ([Direct2D])? How much so?

-   To what extent do "advanced" GPU features (subgroups, descriptor arrays) help?

-   Can we improve quality and extend the imaging model in useful ways?

## Blogs and other writing

Much of the research progress on piet-gpu is documented in blog entries. See [doc/blogs.md](doc/blogs.md) for pointers to those.

There is a much larger and detailed [vision](doc/vision.md) that explains the longer-term goals of the project, and how we might get there.

## History

A prior incarnation used a custom cross-API hal. An archive of this version can be found in the branches [`custom-hal-archive-with-shaders`] and [`custom-hal-archive`].

## License and contributions.

The piet-gpu project is dual-licensed under both [Apache 2.0](LICENSE-APACHE) and [MIT](LICENSE_MIT) licenses.

In addition, the shaders are provided under the terms of the [Unlicense](./shader/UNLICENSE). The intent is for this research to be used in as broad a context as possible.

The dx12 backend was adapted from piet-dx12 by Brian Merchant.

Contributions are welcome by pull request. The [Rust code of conduct] applies.

[piet-metal]: https://github.com/linebender/piet-metal
[direct2d]: https://docs.microsoft.com/en-us/windows/win32/direct2d/direct2d-portal
[`wgpu`]: https://wgpu.rs/
[xilem]: https://github.com/linebender/xilem/
[rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
[`custom-hal-archive-with-shaders`]: https://github.com/linebender/piet-gpu/tree/custom-hal-archive-with-shaders
[`custom-hal-archive`]: https://github.com/linebender/piet-gpu/tree/custom-hal-archive


# vello

This crate is currently a highly experimental proof-of-concept port of the piet-gpu renderer to the WGSL shader language, so it could be run on WebGPU. Depending on how well it works out, it may become the authoritative source for piet-gpu.

The shaders are actually handlebars templates over WGSL, as it's important to share common data structures; it's likely we'll use the template mechanism to supply various parameters which are not supported by the WGSL language, for example to specify grayscale or RGBA buffer output for fine rasterization.

This crate also uses a very different approach to the GPU abstraction than piet-gpu. That is essentially a HAL that supports an immediate mode approach to creating resources and submitting commands. Here, we generate a `Recording`, which is basically a simple value type, then an `Engine` plays that recording to the actual GPU. The idea is that this can abstract easily over multiple GPU back-ends, without either the render logic needing to be polymorphic or having dynamic dispatch at the GPU abstraction. The goal is to be more agile.

Scene encoding is shared with piet-gpu, and currently uses piet-scene in the same repo with no changes.

This module is still an experimental work in progress. Contributions can be made with the same policy as the root repo, but expect things to change quickly.
