## WebGL Benchmark

A browser-based benchmark tool for Vello Hybrid's WebGL2 renderer. Renders animated rectangles
with tweakable parameters (count, speed, size, paint mode) and a live FPS counter.

## Running

Run with SIMD enabled (recommended):

```
RUSTFLAGS=-Ctarget-feature=+simd128 cargo run_wasm -p webgl_bench --release
```

Scalar (non-SIMD) build:

```
cargo run_wasm -p webgl_bench --release
```

## Controls

The sidebar on the left provides:

- **FPS display** -- rolling average over 60 frames
- **Scene selector** -- switch between benchmark scenes
- **Rectangles** -- number of animated rectangles (1--10,000)
- **Speed** -- animation speed multiplier
- **Paint** -- solid color or linear gradient
- **Rect Size** -- dimensions of each rectangle
