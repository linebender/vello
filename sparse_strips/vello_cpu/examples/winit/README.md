# Vello CPU Winit Example

This example demonstrates rendering with Vello CPU using the winit windowing library.

## Features

- CPU-based rendering using Vello CPU
- Interactive scene viewer with multiple example scenes
- Pan and zoom controls
- FPS counter and performance metrics
- Multi-threaded rendering support

## Running

```sh
cargo run -p vello_cpu_winit --release
```

## Controls

- **Left/Right Arrow**: Switch between scenes
- **Space**: Reset view transform
- **r**: Single-step rotation clockwise around window center
- **R**: Single-step rotation counter-clockwise around window center
- **Cmd+R**: Toggle continuous rotation around window center
- **s**: Single-step shear in positive X direction around window center
- **S**: Single-step shear in negative X direction around window center
- **Cmd+S**: Toggle continuous shear oscillation (back-and-forth)
- **Left Mouse Button + Drag**: Pan the scene
- **Mouse Wheel**: Zoom in/out (centered at cursor)
- **Pinch Gesture**: Zoom on touchpad
- **Escape**: Exit application

## Performance

This example uses multi-threaded CPU rendering. The number of threads used is determined automatically based on the number of available CPU cores. You can see the rendering performance in the window title, which updates every second with the average FPS and frame time.

For better performance, always run with the `--release` flag.

## Heap Profiling with DHAT

This example supports heap profiling using the [dhat](https://docs.rs/dhat) crate. This is useful for analyzing memory allocation patterns and identifying allocation hotspots.

### Running with Heap Profiling

```sh
cargo run -p vello_cpu_winit --profile profiling --features dhat-heap
```

The program will run slower than normal due to allocation tracking.

When you exit the application, you'll see a summary like:

```
dhat: Total:     1,256 bytes in 6 blocks
dhat: At t-gmax: 1,256 bytes in 6 blocks
dhat: At t-end:  1,256 bytes in 6 blocks
dhat: The data has been saved to dhat-heap.json, and is viewable with dhat/dh_view.html
```

### Viewing Results

Open the generated `dhat-heap.json` file in [DHAT's online viewer](https://nnethercote.github.io/dh_view/dh_view.html) to explore:

- Total allocations and their call sites
- Peak heap usage (t-gmax)
- Memory still allocated at exit (t-end)
- Allocation hotspots with full backtraces

