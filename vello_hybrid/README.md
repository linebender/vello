# Vello Hybrid Crates

This implementation is based on the **sparse rendering approach** outlined by Raph Levien (@raphlinus) in  
[*Potato: a hybrid CPU/GPU 2D renderer design*](https://docs.google.com/document/d/1gEqf7ehTzd89Djf_VpkL0B_Fb15e0w5fuv_UzyacAPU/edit).  
It leverages **efficient tiling, sorting, and sparse strip allocation** to optimize rendering for both CPU and hybrid CPU/GPU workloads.

## Overview

This directory contains the core crates for the Vello rendering. Each crate serves a distinct role in the architecture, allowing modular development and easier maintenance.

### Crates
- **`vello_api`** – Defines the public API types shared across implementations.
- **`vello_common`** – Provides shared data structures and utilities for rendering.
- **`vello_cpu`** – Implements a CPU-based renderer optimized for multithreading and SIMD.
- **`vello_hybrid`** – A hybrid CPU/GPU renderer, balancing workload between CPU and GPU.

## Development Status

This structure is **under active development** and subject to changes as the integration progresses. Contributions and feedback are welcome!

## Community

Discussion of Vello Hybrid development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically the [#gpu stream](https://xi.zulipchat.com/#narrow/stream/197075-gpu).
All public content can be read without logging in.

Contributions are welcome by pull request.
The [Rust code of conduct] applies.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
[Vello]: https://github.com/linebender/vello
[the changelog]: https://github.com/linebender/vello/tree/main/CHANGELOG.md