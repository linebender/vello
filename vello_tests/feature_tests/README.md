# GPU shader feature combinations

Compares the supported elements of the common probe scene against PNG references on WGPU
and WebGL, with depth buffering enabled.

`run.sh` tests five combinations: no optional shader features, each feature alone, and all
features together.

```sh
bash vello_tests/feature_tests/run.sh --features wgpu
bash vello_tests/feature_tests/run.sh --headless --chrome --features webgl
```

References live in `vello_tests/snapshots`; native diffs go to `vello_tests/diffs`.
Generate references on a native target before compiling browser tests:

```sh
cargo test -p vello_gpu_feature_tests --no-default-features \
    --features gradient_sweep --test features feature_tests_probe_reference
```

Use `REPLACE=1` to update references. Creating or replacing a reference intentionally fails
that run for review.
