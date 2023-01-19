# Vello SVG viewer

This example program parses SVG files with [usvg](https://crates.io/crates/usvg) and renders them with Vello.

The rendering is extremely simplistic and does not yet support:

- group opacity
- mix-blend-modes
- clipping
- masking
- filter effects
- group background
- path visibility
- path paint order
- path shape-rendering
- embedded images
- text
- gradients
- patterns

## Usage

Running the viewer without any arguments will render a set of default, public-domain SVG images:

```bash
$ cargo run --release
```

Optionally, you can pass in paths to SVG files that you want to render:

```bash
$ cargo run --release -- [SVG FILES]
```

## Controls

- Mouse drag-and-drop will translate the image.
- Mouse scroll wheel will zoom.
- Arrow keys switch between SVG images in the current set.
- Space resets the position and zoom of the image.
- Escape exits the program.
