# Vello Toy

Vello Toy is a small internal crate that contains a number of utility binaries useful for development.

## debug.rs
When running this binary, you can provide an SVG path that you want to render and inspect the results of different stages of the rendering pipeline, in the form of an SVG. 

For example, if you run:

`cargo run --bin debug -- --path "M 5 5 L 40 23 L 7 44 Z"  --stages line_segments,tile_areas`

A new SVG file will be generated that allows you to easily visualize the generated flattened lines and tiles.

## svg.rs
This binary allows you to render SVG files to PNG. Note that support is very primitive, and only very basic filling/stroking as well as clip paths are currently supported. In addition to that, the binary also allows you to define a target time during which it should be running.

For example, if you run:

`cargo run --bin svg --release  -- --path examples/assets/Ghostscript_Tiger.svg --scale 5 --runtime 2000`

The binary will run for two seconds and render the ghostscript tiger in a loop for 2 seconds, until it finally saves the result as a PNG file and prints the average runtime per iteration.