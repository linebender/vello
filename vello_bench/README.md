# Setup

In order to run the integration benchmarks with custom SVGs, you need to add the SVGs you want to run into the `data` folder. For each SVG file in that folder, a corresponding integration test will be generated automatically.

If you don't add any SVGs, the benchmarking harness will only use the ghostscript tiger by default.

Run the core benchmarks with `cargo bench`. Enable the `extended` feature to include all benchmarks:

```shell
cargo bench --features extended
```

You can also provide a filter for the name of the benchmarks you want to run, like
`cargo bench -- fine/fill`.

Run the deterministic CPU allocation regression benchmarks with:

```shell
cargo bench --bench allocations
```

These benchmarks use fixed workloads, fallback SIMD, and single-threaded rendering. They fail when
an allocation metric exceeds its recorded limit; update a limit only after verifying that the
increase is intentional. One frame is measured by default; pass `--frames` to measure a longer run:

```shell
cargo bench --bench allocations -- --frames 1,10,100
# Space-separated counts are also accepted:
cargo bench --bench allocations -- --frames 1 10 100
```

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

## Paired comparisons with Tango

Benchmarks written against `vello_bench::registry::Registry` (currently only `flatten`) can also
run under [Tango](https://github.com/bazhenov/tango). Tango runs the baseline and the candidate
builds alternately, so both see the same machine noise. This keeps comparisons stable on shared
machines such as CI runners.

Install [`cargo-export`](https://crates.io/crates/cargo-export) via `cargo install cargo-export`,
then run these commands from the workspace root:

```shell
# Build and save the baseline
cargo export target/benchmarks -- bench -p vello_bench --features tango --bench tango
# Apply your changes, then compare against the baseline
cargo bench -p vello_bench --features tango --bench tango -- compare "$PWD/target/benchmarks/tango"
```

The path must be absolute because `cargo bench` runs from the `vello_bench` directory. Changes
marked with `*` are statistically significant, and the comparison exits with an error if any are
found.
