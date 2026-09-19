# Architecture

Vello contains two renderer families:

1. [`vello_cpu`](vello_cpu), a CPU-only renderer, and
   [`vello_gpu`](vello_gpu), a GPU-accelerated renderer, both based on the
   Sparse Strips architecture.
2. The experimental [`vello`](research/vello_research) renderer, which is based
   on compute shaders.

See the [project README](README.md) for renderer selection and package
migration guidance.

## Goals

The project aims to provide high-quality, high-performance 2D rendering for GUI
applications, creative tools, scientific visualization, and similar workloads.
Its implementations explore how CPU SIMD, multithreading, conventional GPU
rasterization, and GPU compute can be applied to vector graphics.

## Sparse Strips renderers

Vello CPU and Vello GPU share a CPU-side pipeline implemented primarily in
`vello_common`. Paths are flattened, binned into tiles, and represented as
sparse horizontal strips. Coverage data is retained where it is needed around
path boundaries, while fully covered regions can be represented compactly. This
limits intermediate work and memory to the parts of the scene that affect the
final image.

The resulting representation can be consumed by different backends:

- `vello_cpu` rasterizes and composites the strips into a CPU pixmap. It uses
  SIMD and can use multiple threads, but requires no GPU.
- `vello_gpu` performs the same broad preprocessing on the CPU, then uploads
  scheduled strips and paint data for GPU rasterization and compositing. Its
  wgpu and WebGL2 backends use vertex and fragment rendering rather than
  requiring compute shaders.

The two renderers share geometry, tiling, paint, image, filter, and strip data
structures through `vello_common`, as well as test scenes and snapshot
infrastructure under `vello_tests`. Text and glyph-run support is shared through
`glifo`.

The design grew from the sparse rendering approach described in
[*Potato: a hybrid CPU/GPU 2D renderer design*][potato]. The
[Vello CPU thesis][vello-cpu-thesis] provides a more detailed explanation of
the pipeline, although parts of the description in the thesis are already
outdated and implementation details continue to evolve.

## Compute-centric research renderer

The package published as `vello` is the original compute-centric renderer. A
scene is encoded into compact path, draw, transform, and resource buffers. Those
buffers are resolved into a `Recording` of GPU operations, and `WgpuEngine`
uploads resources and dispatches the compute pipeline. Prefix-scan algorithms
parallelize work that traditional renderers often perform sequentially.

This approach can perform very well on dynamic, vector-heavy scenes, but it
requires compute shader support and has different compatibility and memory
trade-offs from the Sparse Strips renderers. It remains under `research/`
together with its dedicated shaders, examples, tests, and design history.

CPU implementations of parts of the compute pipeline live in
`research/vello_shaders/src/cpu`. They are used for testing and debugging; they
do not provide a complete standalone CPU renderer. Use `vello_cpu` for that.

## Repository structure

The user-facing Sparse Strips implementation is organized at the root:

- `vello_common/` — shared geometry, tiling, paint, and strip infrastructure.
- `vello_cpu/` — CPU renderer.
- `vello_gpu/` — GPU renderer with CPU-side preprocessing.
- `vello_gpu_shaders/` — WESL sources and generated WGSL/GLSL used by
  `vello_gpu`.
- `glifo/` — text and glyph-run support shared by the Sparse Strips renderers.
- `vello_tests/` — shared integration tests, snapshots, scenes, and browser
  tooling.
- `vello_bench/` — Sparse Strips benchmarks.

Compute-centric packages and support code are grouped under `research/`:

- `research/vello_research/` — source for the published `vello` package.
- `research/vello_encoding/` — compact scene encoding used by `vello`.
- `research/vello_shaders/` — compute shaders, preprocessing, and CPU
  reference kernels.
- `research/vello_research_tests/` — compute-renderer tests and snapshots.
- `research/examples/` — standalone compute-renderer examples.
- `research/xtask/` — snapshot and comparison maintenance tooling.
- `research/doc/` — historical design documents, roadmaps, and blog links.

## Shader source systems

Vello GPU shaders are authored as WESL in `vello_gpu_shaders/shaders`. The crate
links them into WGSL and can generate GLSL plus reflection metadata for the
WebGL2 backend.

The compute renderer uses WGSL sources in `research/vello_shaders/shader`.
Because WGSL has no built-in metaprogramming, these shaders use a small
preprocessor supporting:

1. `import`, which imports shared code from `shader/shared`.
2. `ifdef`, `ifndef`, `else`, and `endif`, controlled by definitions supplied
   outside the shader.

This format is compatible with [`wgsl-analyzer`]. New imports must also be
listed in `.vscode/settings.json` for editor support.

## Research history

Historical documents about the compute renderer are in `research/doc/`.
[blogs.md](research/doc/blogs.md) links to development articles, and the
[2023 roadmap](research/doc/roadmap_2023.md) records earlier project goals. The
[path segment encoding](research/doc/pathseg.md) document describes the compute
renderer's compact path representation.

[`wgsl-analyzer`]: https://marketplace.visualstudio.com/items?itemName=wgsl-analyzer.wgsl-analyzer
[potato]: https://docs.google.com/document/d/1gEqf7ehTzd89Djf_VpkL0B_Fb15e0w5fuv_UzyacAPU/edit
[vello-cpu-thesis]: https://ethz.ch/content/dam/ethz/special-interest/infk/inst-pls/plf-dam/documents/StudentProjects/MasterTheses/2025-Laurenz-Thesis.pdf
