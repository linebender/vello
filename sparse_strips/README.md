# Vello Sparse Strips

We are developing a new implementation for Vello that aims to:

- Be compatible with a wider range of devices (should be able to run on GPUs without compute shader support, using only fragment and vertex shaders).
- Mitigate some performance cliffs.
- Handle a wider range of memory conditions (e.g., when less memory is available).

The renderer crates are grouped in the repository's [`renderers`](../renderers) directory. This directory contains their shared implementation, tests, examples, benchmarks, and development tooling.

This implementation is based on the **sparse rendering approach** outlined by Raph Levien (@raphlinus) in  
[*Potato: a hybrid CPU/GPU 2D renderer design*](https://docs.google.com/document/d/1gEqf7ehTzd89Djf_VpkL0B_Fb15e0w5fuv_UzyacAPU/edit).  
It leverages **efficient tiling, sorting, and sparse strip allocation** to optimize rendering for both CPU and hybrid CPU/GPU workloads.

## Overview

The Sparse Strips architecture has two renderers:

- **[`vello_cpu`](../renderers/vello_cpu)** – Implements a CPU-based renderer optimized for multithreading and SIMD.
- **[`vello_hybrid`](../renderers/vello_hybrid)** – Implements a hybrid CPU/GPU renderer, balancing workload between CPU and GPU.

The principal support crates remaining in this directory are:

- **`vello_common`** – Provides shared data structures and utilities for rendering.
- **`vello_sparse_shaders`** – Provides WGSL-to-GLSL compilation for the WebGL `vello_hybrid` backend.

## Development Status

This structure is **under active development** and subject to changes as the integration progresses. Contributions and feedback are welcome!

## WebAssembly tooling

The [`web`](web) directory contains development scripts for producing
browser-ready WebAssembly builds and checking their sizes. Generated modules
are written to `target/sparse-strips-wasm`; they are not pre-built artifacts
included in this directory. Run the SIMD128 size check from the repository root
with:

```bash
./sparse_strips/web/check_wasm_sizes.sh
```

## Community

Discussion of Vello Hybrid development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically the [#vello channel](https://xi.zulipchat.com/#narrow/channel/197075-vello).
All public content can be read without logging in.

Contributions are welcome by pull request.
The [Rust code of conduct] applies.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
