## Usage

Running the viewer without any arguments will render a built-in set of public-domain SVG images:

```bash
$ cargo run -p with_winit --release
```

Optionally, you can pass in paths to SVG files that you want to render:

```bash
$ cargo run -p with_winit --release -- [SVG FILES]
```

## Controls

- Mouse drag-and-drop will translate the image.
- Mouse scroll wheel will zoom.
- Arrow keys switch between SVG images in the current set.
- Space resets the position and zoom of the image.
- S toggles the frame statistics layer
- C resets the min/max frame time tracked by statistics
- V toggles VSync on/off (default: on)
- Escape exits the program.
