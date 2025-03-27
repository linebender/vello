# Setup

In order to run the integration benchmarks with custom SVGs, you need to add the SVGs you want to run into the `data` folder. For each SVG file in that folder, a corresponding integration test will be generated automatically.

If you don't add any SVGs, the benchmarking harness will only use the ghostscript tiger by default.

In order to run the benches, you can simply run `cargo bench`. However, in most cases you probably don't
want to rerun all benchmarks, in which case you can also provide a filter for the name of the benchmarks
you want to run, like `cargo bench -- fine/fill`