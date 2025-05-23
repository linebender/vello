<div align="center">

# Vello Sparse Tests

[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![Linebender Zulip chat.](https://img.shields.io/badge/Linebender-%23vello-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/197075-vello)
[![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/linebender/vello/ci.yml?logo=github&label=CI)](https://github.com/linebender/vello/actions)

</div>

This is a development only crate for testing the sparse_strip renderers across a corpus of reference
images:
 - cpu
 - wgpu
 - wasm32 WebGL

## Testing WebGL on the Browser

To run the `vello_sparse_tests` suite on WebGL headless:

```
wasm-pack test --headless --chrome --features webgl
```

To debug the output images in webgl, run the same command without `--headless`. Any tests that fail
will have their diff image appended to the bottom of the page.
