# RGBA8 premultiplication browser benchmark

This benchmark compares the previous scalar premultiplication loop with the new
64-byte interleaved implementation in a browser. It runs the same 800 × 500,
1920 × 1080, and 3840 × 2160 inputs as the Criterion benchmark.

Build the SIMD128 module from the repository root:

```bash
./sparse_strips/premultiply_web_bench/build.sh simd
```

Then serve the directory:

```bash
python3 -m http.server 8000 --directory sparse_strips/premultiply_web_bench
```

Open <http://localhost:8000>, run the benchmark, and use **Copy results** to
copy the report.

To inspect the scalar fearless-simd fallback instead, rebuild with:

```bash
./sparse_strips/premultiply_web_bench/build.sh non-simd
```

The harness checks that the output difference between exact `/ 255` and the
`div_255` helper is at most one, then reports how many channel bytes differ.
Each benchmark runs for three seconds by default and reports the average from
the total active execution time and iteration count. Warm-up and measurement
are split into short chunks so the browser remains responsive; buffer resets
and browser rendering time are excluded from the measured duration.
