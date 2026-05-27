# vello_example_scenes

A collection of scenes used for Vello Sparse Strips examples.

This crate provides various scene implementations including:
- Basic shapes and paths
- Text rendering
- Image handling
- Gradients and blending
- SVG rendering
- Clipping operations
- COLR emoji stress testing

## Emoji Grid Font

The Emoji Grid scene uses the bundled Noto Color Emoji COLR subset by default.
Native builds can render a different COLR font by setting `NOTO_COLR_PATH` to
the font file path before running the example. For it to work in WASM, you have to
change two lines in the source file, see there for more information.
