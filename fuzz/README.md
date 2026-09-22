# Vello fuzz targets

This package contains [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) targets and is excluded
from the main Cargo workspace so normal builds do not compile libFuzzer.

## CPU/GPU differential rendering

`cpu_gpu_differential` decodes bounded, structured scenes with `arbitrary`, renders each scene with
Vello CPU and Vello GPU, and compares their premultiplied RGBA8 output. The target reuses both
renderers between iterations. A panic, render failure, or image difference beyond the configured
tolerance is saved by libFuzzer as a reproducible artifact.

The scene model covers:

- Shapes: rectangles and paths made of lines, quadratics, and cubics, under affine transforms.
- Fills with both fill rules; strokes with joins, caps, miter limits, and dash patterns.
- Solid paints.
- Gradient paints: linear, radial, or sweep; pad/repeat/reflect; sRGB, linear sRGB, or Oklab
  interpolation.
- Image paints: the snapshot suite's assets in `vello_tests/tests/assets` (RGB, RGBA, luma, and
  luma-alpha images from 2x2 to 16x16) with per-axis extend modes and a sampler alpha. Sampling
  quality is `Low` (nearest) unless `--image-quality=all` is passed, because `Medium` and `High`
  are known to differ between the backends and would drown other findings.
- A paint transform (`set_paint_transform`) for gradient and image paints.
- Clip layers, opacity layers, and supported `SrcOver` blend modes.
- Non-isolated clip paths (`push_clip_rect`/`push_clip_path`), freely interleaved with layers.

Add new features incrementally so known differences do not hide unrelated failures. libFuzzer grows
input length slowly; pass `-len_control=0` to reach scenes that use gradients, images, and dashes
right away.

Install cargo-fuzz and run the target with a bounded input size:

```shell
cargo install cargo-fuzz
cargo +nightly fuzz run cpu_gpu_differential -- -max_len=4096
```

Run a short smoke session:

```shell
cargo +nightly fuzz run cpu_gpu_differential -- -max_len=4096 -max_total_time=60
```

Everything after `--` goes to the fuzz binary. Single-dash flags such as `-max_len` belong to
libFuzzer (`-help=1` lists them); libFuzzer ignores double-dash flags, which is how the target
receives its own options below. Target flags work the same with `run` and `tmin`.

By default a pixel may differ by up to 2 per channel, and up to 4 pixels per image may exceed
that. Both thresholds can be set on the command line, and a generated snapshot test carries the
tolerances it was found with:

```shell
cargo +nightly fuzz run cpu_gpu_differential -- -max_len=4096 --channel-tolerance=0 --max-outlier-pixels=0
```

libFuzzer stops at the first failure. To keep fuzzing after image mismatches and collect them all,
enable continue mode:

```shell
cargo +nightly fuzz run cpu_gpu_differential -- -max_len=4096 --continue
```

Each new mismatching input is saved as `artifacts/cpu_gpu_differential/mismatch-<hash>` together
with its `.png` diff image, `.json` report, and `.rs` snapshot test (see below), and a
`MISMATCH #n` line with the reproduce command is printed. A panic inside one of the renderers is
saved the same way as `crash-<hash>` with the panic message in `crash-<hash>.txt`, printed as a
`CRASH #n` line, and the renderer that panicked is rebuilt before the next input. Panics outside
the renderers (or on another thread) still abort the run. Expect many findings per minute if the
tolerances are very strict: every anti-aliased edge can differ by one.

Mutations often change bytes without changing the symptom, so one divergence can fill the artifact
directory with near-identical findings. Add `--dedup` to save only the first finding per signature
(the set of differing pixels for a mismatch, the backend and panic location for a crash) and print
`MISMATCH: duplicate of #14, not saved` for the rest:

```shell
cargo +nightly fuzz run cpu_gpu_differential -- -max_len=4096 --continue --dedup
```

Reproduce or minimize a saved artifact:

```shell
cargo +nightly fuzz run cpu_gpu_differential fuzz/artifacts/cpu_gpu_differential/<artifact>
cargo +nightly fuzz tmin cpu_gpu_differential fuzz/artifacts/cpu_gpu_differential/<artifact>
```

Render a saved input on both backends and write a `cpu | diff | gpu` image plus a per-pixel JSON
report, in the same format the snapshot suite writes to `vello_tests/diffs/`:

```shell
cargo +nightly fuzz run cpu_gpu_differential fuzz/artifacts/cpu_gpu_differential/<artifact> -- --diff
```

This writes `<artifact>.png` and `<artifact>.json` beside the input and never panics, so it also
works on inputs that pass the comparison. Red pixels in the middle panel exceed the channel tolerance.

Decode a saved input into a snapshot test without rendering it:

```shell
cargo +nightly fuzz run cpu_gpu_differential fuzz/artifacts/cpu_gpu_differential/<artifact> -- --decode-to-rs
```

This writes `<artifact>.rs` beside the input. The file is a complete `vello_test` function that
replays the scene through the same `Renderer` calls the fuzz target used (`set_transform`,
`set_paint`, `fill_path`, `push_clip_layer`, ...), with the fuzz target's tolerances encoded in the
attribute. Image paints load their asset with the suite's `load_image!` macro. Like the tolerances,
the sampling quality depends on the target flags, so pass the same `--image-quality` when decoding
an artifact found with it.

To check a finding against the whole snapshot suite in one step, which also covers the SIMD and
multithreaded CPU variants the fuzz target does not run:

```shell
fuzz/validate.sh fuzz/artifacts/cpu_gpu_differential/<artifact> --channel-tolerance=0 --max-outlier-pixels=0
```

The script adds the test function (`fuzz_regression_mismatch_<hash>`,
`fuzz_regression_crash_<hash>`, ... after the artifact) to the scratch module `vello_tests/tests/fuzz_regression.rs`, creates the reference image from the
scalar CPU f32 pipeline, and compares every variant against it. An existing `.rs` (continue mode
writes one for every finding) is reused as is, since it carries the tolerances the input was found
with; passing target flags regenerates it with those tolerances instead. An `.rs` path can also be
given directly.

The scratch module is committed empty and is never meant to be committed with tests in it. Added
tests and their reference images stay in place, so several findings can be collected and rerun
directly while investigating; the diff images and per-pixel JSON reports in `vello_tests/diffs/`
are refreshed each time:

```shell
cargo test -p vello_tests --test tests fuzz_regression::<test_name>
cargo test -p vello_tests --test tests fuzz_regression::
```

Running the script again for the same artifact replaces that function and its reference. Once a
finding is understood, move its test into the module matching its topic (or `issues.rs`) under a
descriptive name, rename the reference image in `vello_tests/snapshots/` to match, and reset the
scratch module with `git checkout vello_tests/tests/fuzz_regression.rs`. The script exits non-zero
if any variant failed. Note
that the suite's GPU threshold is never below 1, so a finding whose largest difference is exactly 1
cannot be reproduced there.

libFuzzer stores every input that reached new coverage in `corpus/<target>/`. The corpus is a
local cache that speeds up later sessions and is not committed; a session started from an empty
corpus discovers the structured scene format within seconds. Failing inputs are written to
`artifacts/<target>/` instead.

The differential target requires a working native `wgpu` adapter.
