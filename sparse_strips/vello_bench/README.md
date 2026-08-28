# Setup

In order to run the integration benchmarks with custom SVGs, you need to add the SVGs you want to run into the `data` folder. For each SVG file in that folder, a corresponding integration test will be generated automatically.

If you don't add any SVGs, the benchmarking harness will only use the ghostscript tiger by default.

Run the core benchmarks with `cargo bench`. Enable the `extended` feature to include all benchmarks:

```shell
cargo bench --features extended
```

You can also provide a filter for the name of the benchmarks you want to run, like
`cargo bench -- fine/fill`.

## Workflow

Save a control run with:

```shell
cargo bench --bench main -- --save-baseline control [TEST NAME FILTER]
```

Then, apply some changes to the code and compare it to the control with:

```shell
# Rerun bench against new changes
cargo bench -- [TEST NAME FILTER]
# Compare it against control
cargo bench --bench main -- --load-baseline new --baseline control
```
