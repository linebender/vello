<div align="center">

# Vello Sparse Tests

[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![Linebender Zulip chat.](https://img.shields.io/badge/Linebender-%23vello-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/197075-vello)
[![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/linebender/vello/ci.yml?logo=github&label=CI)](https://github.com/linebender/vello/actions)

</div>

This is a development-only crate for testing the sparse_strip renderers across a corpus of reference
images:
- CPU
- WGPU
- WASM32 WebGL

The `vello_test` proc macro will create a snapshot test for each supported renderer target. See the
below example usage.

```rs
// Draws a filled triangle into a 125x125 scene.
#[vello_test(width = 125, height = 125)]
fn filled_triangle(ctx: &mut impl Renderer) {
    let path = {
        let mut path = BezPath::new();
        path.move_to((5.0, 5.0));
        path.line_to((95.0, 50.0));
        path.line_to((5.0, 95.0));
        path.close_path();

        path
    };

    ctx.set_paint(LIME);
    ctx.fill_path(&path);
}
```

See all the attributes that can be passed to `vello_test` in `vello_dev_macros/test.rs`.

## Testing WebGL on the Browser

Requirements:
 - on MacOS, a minimum Clang major version of 20 is required.

To run the `vello_sparse_tests` suite including the WebGL tests:

```sh
wasm-pack test --headless --chrome --features webgl --release
```

To debug the output images in webgl, run the same command without `--headless`. Any tests that fail
will have their diff image appended to the bottom of the page.
