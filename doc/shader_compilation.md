# How shader compilation works

We use git branches to support shader compilation in the cloud. The `dev` branch contains only shader source files (in GLSL format), while the `main` branch contains generated shaders. On every push to the `dev` branch, a GitHub action runs which compiles the shaders and pushes those to `main`.

Thus, you can run piet-gpu from the `main` branch without requiring any shader compilation tools. Also, the `dev` branch has a relatively clean history, and PRs can be made against it without having to worry about merge conflicts in the generated shader files.

If you do want to make changes to the shaders, you'll need some tools installed:

* [Ninja]
* [Vulkan SDK] (mostly for glslangValidate, spirv-cross)
* [DirectX Shader Compiler][DXC]

The GitHub action runs on Windows so the DXC signing can succeed (note that [hassle-rs] may provide an alternate solution). We currently only compile to MSL on Metal, not AIR, due to tooling friction. The Metal shader compiler is available on Windows, but a barrier to running in CI is that downloading it appears to require an Apple account. Longer term we will want to figure out a solution to this, because the piet-gpu vision involves ahead-of-time compilation of shaders as much as possible.

Right now the scripts for compiling shaders are done in hand-written ninja files. This is likely to change, as the number of permutations will increase, and we also may want access to metadata from the shader compilation process.

Following a few general rules should hopefully keep things running smoothly:

* Prepare all PRs against the `dev` branch, not `main`.
* Don't commit generated shaders in the PR.
* Don't commit directly to `main`, it will cause divergence.

[Ninja]: https://ninja-build.org/
[Vulkan SDK]: https://www.lunarg.com/vulkan-sdk/
[DXC]: https://github.com/microsoft/DirectXShaderCompiler
