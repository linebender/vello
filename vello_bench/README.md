# Vello benchmarks

## Native benchmarks

By default, the runner includes SIMD, non-extended, u8 benchmarks:

```shell
bash vello_bench/bench.sh cli
```

Pass a benchmark path substring to select a group or individual case:

```shell
bash vello_bench/bench.sh cli fine/fill
```

Add other variants independently with `--non-simd`, `--extended`, or `--f32`:

```shell
bash vello_bench/bench.sh cli --non-simd
bash vello_bench/bench.sh cli --extended
bash vello_bench/bench.sh cli --f32
```

Use `--warmup-ms`, `--measurement-ms`, and `--samples` to change the default 250-millisecond
warmup, two-second target measurement time, and 30 measured samples:

```shell
bash vello_bench/bench.sh cli fine/fill --warmup-ms 500 --measurement-ms 3000 --samples 40
```

List available cases with:

```shell
cargo run --release -p vello_bench --bin vello-bench -- list
cargo run --release -p vello_bench --bin vello-bench -- list --non-simd --extended --f32
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

## Comparing revisions

Compare matching cases from two revisions:

```shell
bash vello_bench/bench.sh cli --ab REVISION_A REVISION_B fine/fill
```

Use `--non-simd`, `--extended`, and `--f32` to include those variants, and
`--warmup-ms`, `--measurement-ms`, and `--samples` to change the timing.

Comparison mode requires a clean checkout. It builds both revisions as dynamic libraries in a
temporary worktree using the current benchmark definitions, loads both libraries into one process,
then removes the worktree when it exits. The current checkout is not modified.

## Browser benchmarks

Build and serve the browser benchmarks:

```shell
bash vello_bench/bench.sh web
```

Compare two revisions in the browser with:

```shell
bash vello_bench/bench.sh web --ab REVISION_A REVISION_B
```

The browser contains the same timing benchmark groups as the native runner. Use the page's
checkboxes to show extended, non-SIMD, and f32 variants, select cases, and change the timing.

## Measurement

Each case warms up for the configured duration while the harness estimates the iteration count that
will make one sample take approximately the target measurement time divided by the sample count. It
then records exactly that number of samples. Results report the average time per iteration and the
sample standard deviation as a percentage of the average.

Revision comparisons warm up both artifacts independently, use their independently estimated
iteration counts, and alternate their measurement order. They report the average and standard
deviation for each artifact and the average paired change between normalized times per iteration.

## Data-driven benchmarks

The Ghostscript tiger is always included in the pipeline benchmarks. Add SVG files to `data` to
include additional scenes in both native and browser builds. The SVGs are embedded at build time,
so rebuild after adding or changing one.
