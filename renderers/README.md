# Vello renderers

This directory contains Vello's user-facing renderer crates.

- [`vello_cpu`](vello_cpu) renders entirely on the CPU and is optimized for SIMD and optional multithreading.
- [`vello_hybrid`](vello_hybrid) processes paths on the CPU and uses the GPU for rendering and compositing.

The shared implementation, tests, benchmarks, and development tooling for
these renderers remain in [`sparse_strips`](../sparse_strips).
