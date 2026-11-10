# Setup

In order to run the integration benchmarks with custom SVGs, you need to add the SVGs you want to run into the `data` folder. For each SVG file in that folder, a corresponding integration test will be generated automatically.

If you don't add any SVGs, the benchmarking harness will only use the ghostscript tiger by default.

The benchmarks run under [Tango](https://github.com/bazhenov/tango). Run the core benchmarks with:

```shell
cargo bench -p vello_bench --bench main -- solo
```

Enable the `extended` feature to include all benchmarks. Pass `-f` with a glob to run only some of
them, and use `list` to see their names:

```shell
cargo bench -p vello_bench --bench main --features extended -- solo -f 'fine/fill/*'
cargo bench -p vello_bench --bench main -- list
```

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

## Comparing against a baseline

Tango runs the baseline and the candidate builds alternately, each in its own process, so both see
the same machine noise. This keeps comparisons stable on shared machines such as CI runners.

Install [`cargo-export`](https://crates.io/crates/cargo-export) via `cargo install cargo-export`,
then run these commands from the workspace root:

```shell
# Build and save the baseline
cargo export target/benchmarks -- bench -p vello_bench --bench main
# Apply your changes, then compare against the baseline
cargo bench -p vello_bench --bench main -- compare "$PWD/target/benchmarks/main"
```

The path must be absolute because `cargo bench` runs from the `vello_bench` directory. `compare`
also takes `-f`. Changes marked with `*` are statistically significant, and the comparison exits
with an error if any are found.
