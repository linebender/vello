# Vello Hybrid Winit Example

Run with:

```sh
cargo run -p vello_hybrid_winit --release
```

Optionally, you can pass in paths to SVG files that you want to render:

```bash
cargo run -p vello_hybrid_winit --release -- [SVG FILES]
```

Alternatively, you can pass a scene index to render a specific scene from the built-in set:

```bash
cargo run -p vello_hybrid_winit --release -- [SCENE INDEX]
```

## Controls

- Mouse drag-and-drop will translate the image.
- Mouse scroll wheel will zoom.
- Arrow keys switch between SVG images in the current set.
- Space resets the position and zoom of the image.
- Escape exits the program.
