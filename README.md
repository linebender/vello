# vello (formerly piet-gpu)

This repo contains the new prototype for a new compute-centric 2D GPU renderer.

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

In addition, the shaders are provided under the terms of the [Unlicense](UNLICENSE). The intent is for this research to be used in as broad a context as possible.

The dx12 backend was adapted from piet-dx12 by Brian Merchant.

Contributions are welcome by pull request. The [Rust code of conduct] applies.

[piet-metal]: https://github.com/linebender/piet-metal
[direct2d]: https://docs.microsoft.com/en-us/windows/win32/direct2d/direct2d-portal
[`wgpu`]: https://wgpu.rs/
[xilem]: https://github.com/linebender/xilem/
[rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
[`custom-hal-archive-with-shaders`]: https://github.com/linebender/piet-gpu/tree/custom-hal-archive-with-shaders
[`custom-hal-archive`]: https://github.com/linebender/piet-gpu/tree/custom-hal-archive
