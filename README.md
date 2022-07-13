# piet-gpu

This repo contains the new prototype for a new compute-centric 2D GPU renderer.

It succeeds the previous prototype, [piet-metal].

## Goals

The main goal is to answer research questions about the future of 2D rendering:

* Is a compute-centered approach better than rasterization ([Direct2D])? How much so?

* To what extent do "advanced" GPU features (subgroups, descriptor arrays) help?

* Can we improve quality and extend the imaging model in useful ways?

Another goal is to explore a standards-based, portable approach to GPU compute.

## Blogs and other writing

Much of the research progress on piet-gpu is documented in blog entries. See [doc/blogs.md](doc/blogs.md) for pointers to those.

There is a much larger and detailed [vision](doc/vision.md) that explains the longer-term goals of the project, and how we might get there.

### Why not gfx-hal?

It makes a lot of sense to use gfx-hal, as it addresses the ability to write kernel and runtime code once and run it portably. But in exploring it I've found some points of friction, especially in using more "advanced" features. To serve the research goals, I'm enjoying using Vulkan directly, through [ash], which I've found does a good job tracking Vulkan releases. One example is experimenting with `VK_EXT_subgroup_size_control`.

The hal layer in this repo is strongly inspired by gfx-hal, but with some differences. One is that we're shooting for a compile-time pipeline to generate GPU IR on DX12 and Metal, while gfx-hal ships [SPIRV-Cross] in the runtime. To access [Shader Model 6], that would also require bundling [DXC] at runtime, which is not yet implemented (though it's certainly possible).

### Why not wgpu?

The case for wgpu is also strong, but it's even less mature. I'd love to see it become a solid foundation, at which point I'd use it as the main integration with [Druid].

In short, the goal is to facilitate the research now, collect the data, and then use that to choose a best path for shipping later.

## License and contributions.

The piet-gpu project is dual-licensed under both [Apache 2.0](LICENSE-APACHE) and [MIT](LICENSE_MIT) licenses.

In addition, the shaders are provided under the terms of the [Unlicense]. The intent is for this research to be used in as broad a context as possible.

The dx12 backend was adapted from piet-dx12 by Brian Merchant.

Contributions are welcome by pull request. The [Rust code of conduct] applies. Pull requests should be against the `dev` branch; see [shader_compilation.md] for explanation and details.

[piet-metal]: https://github.com/linebender/piet-metal
[Direct2D]: https://docs.microsoft.com/en-us/windows/win32/direct2d/direct2d-portal
[ash]: https://github.com/MaikKlein/ash
[SPIRV-Cross]: https://github.com/KhronosGroup/SPIRV-Cross
[Shader Model 6]: https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12
[DXC]: https://github.com/microsoft/DirectXShaderCompiler
[Druid]: https://github.com/linebender/druid
[Unlicense]: https://unlicense.org/
[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
[shader_compilation.md]: ./doc/shader_compilation.md
