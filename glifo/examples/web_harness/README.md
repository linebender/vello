# Glifo web harness

Browser pages for benchmarking Glifo text rendering on Vello CPU (WASM, presented with
`putImageData`) and Vello GPU (native WebGL2). Every page runs the same deterministic
workloads, so results are comparable across renderers.

## Running

```sh
glifo/examples/web_harness/build.sh --serve
# open http://localhost:8000/
```

The script builds the `glifo_web_harness` crate for `wasm32-unknown-unknown` (release with
`simd128` by default), runs `wasm-bindgen`, and copies `web/` into
`$CARGO_TARGET_DIR/glifo_web_harness` (default `target/glifo_web_harness`). It uses a
`wasm-bindgen` CLI matching `Cargo.lock`, installing one under the target directory if the CLI on
`PATH` has a different version. Options: `--profile dev|release|wasm-size`, `--no-simd`,
`--port <port>`; set `WASM_BINDGEN` to use a specific CLI. `--serve` sends cross-origin isolation
headers, which raise the `performance.now()` resolution from 100 µs to 5 µs.

Keep the tab visible while measuring: browsers throttle timers and `requestAnimationFrame` in
background tabs.

## Pages

- **Live** (`live.html`): one renderer with per-frame timings (encode, render, present, GPU sync,
  GPU timer queries) as a chart and as rolling averages, glyph throughput, and start-up costs.
  Drag to pan and scroll to zoom. Settings are mirrored in the URL, so views can be shared.
- **Benchmark** (`bench.html`): any selection of workload × renderer × cache on/off scenarios,
  picked in a grid. Every scenario runs on a fresh renderer, with warmup and measured frames at a fixed 1/60 s animation step, repeated for
  several interleaved rounds. Optionally, every GPU frame waits for completion. Results export as
  JSON or Markdown, and can be saved as baselines.

## Comparing against a baseline

1. Run the benchmark on the base build and use **Save as baseline…** (for example `main`).
2. Rebuild with the change, run the same matrix, and pick the baseline under **Compare with**.

When the run finishes, each compared metric shows its change relative to the baseline: green for
better, red for worse, and gray when the change is not significant. A summary above the table
counts faster and slower scenarios by mean frame time. A change counts when Welch's t-test finds
it significant at the 5% level and it exceeds the noise threshold (5% by default). With 3 or more
rounds on both sides, the test uses per-round means, which include run-to-run noise; frames within
one run are correlated, so testing them individually overstates significance. Even so, a session
can drift by a few percent, so run the same build twice to see how noisy your machine is before
trusting small changes. GPU sync and GPU timer columns are noisier than the CPU frame-time columns.

Independently of baselines, each cache-on row shows how many times faster or slower it is than
cache off on the same renderer in the same run. Every round runs all scenarios, so this uses a
paired test over rounds, which stays reliable even when a whole round runs slower.

Baselines live in the browser's IndexedDB for the page's origin, so keep the same port. The list
shows what each baseline contains and lets you view, overwrite, rename, export, import, or delete
it.

## Workloads

| Name        | What it stresses                                                                                         |
| ----------- | -------------------------------------------------------------------------------------------------------- |
| `paragraph` | Static prose and code at several sizes; steady state is all hits.                                        |
| `scroll`    | Long document scrolling at fractional speeds with horizontal drift (subpixel buckets, app-side culling). |
| `sizes`     | One line per size from 6 to 96 px (atlas area).                                                          |
| `zoom`      | Continuously animated zoom: new effective sizes every frame, so misses and atlas growth.                 |
| `storm`     | ~20k glyphs per frame (× `density`) in drifting runs of mixed sizes and colors (throughput).             |
| `charset`   | Every mapped glyph of Roboto and Inconsolata at three sizes (many unique entries, multiple pages).       |
| `emoji`     | COLR and bitmap (CBDT) emoji at several sizes (color entries, bitmap uploads).                           |
| `blank`     | Background only; renderer overhead baseline.                                                             |

## Automation

- `bench.html?autorun=1&workloads=paragraph,storm&renderers=gpu&cache=on&frames=300&warmup=60`
  runs on load; the three filters combine. `scenarios=paragraph/gpu/on,storm/cpu/off` picks exact
  scenarios instead. Other parameters are `rounds`, `width`, `height`, `scale`, `density`, `atlas`,
  `sync` and `hint`. `baseline=<name>` compares against a saved baseline, `save=<name>` saves the
  results as a baseline after the run, and `noise=<percent>` sets the noise threshold.
- `await window.glifoBench.run({ frames: 120 })` returns the results,
  `window.glifoBench.markdown()` formats them as a Markdown table (with changes when comparing),
  and `glifoBench.saveBaseline(name)` and `glifoBench.compareWith(name)` script baselines.
- On the live page, `window.glifo.harness` exposes the wasm `Harness` for console use.

## Notes on the numbers

- Frame times are CPU wall-clock times measured inside wasm. On the GPU backend, `render` only
  issues WebGL commands. Enable GPU sync (a 1×1 `readPixels`) to include GPU execution, or read
  `GPU time` from `EXT_disjoint_timer_query_webgl2`, which arrives a few frames late.
- Vello GPU cannot draw bitmap glyphs without the atlas cache, so the benchmark skips `emoji` on
  the GPU with the cache off.
