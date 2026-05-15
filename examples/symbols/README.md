# symbols

Headless symbol rendering example built on top of Vello Classic.

This crate demonstrates how to use a rendering pipeline to generate high-quality 2D map-like graphics, including polygons, roads, water layers, and point symbols, and export the result as a PNG image.

It is designed as an example of integrating vector-style rendering with a GPU backend using `wgpu`.

---

## Features

- Headless (offscreen) rendering using GPU
- Map-style layered rendering:
    - Polygon landuse layers
    - Water and river rendering
    - Cased road rendering
    - Grid overlay
    - Point symbols with multi-layer styling
- Gradient fills, strokes, and clipping support
- Export rendered result to PNG
- Configurable antialiasing (MSAA / area)
- Optional CPU rendering mode

---

## Usage
```bash
cargo run -p symbols
cargo run -p symbols -- --width 1600 --height 900 --aa msaa16 --output output.png
```