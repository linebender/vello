# WebGL Demo

This folder contains a minimal WebGL2 repro for the bug where a linked program
reports `_group_0_binding_2_fs` as missing.

The four fragment variants are:

- `Minimal Pass`
- `Minimal Fail`
- `Render-Like Fail`
- `Render-Like Fixed`

The only preserved Vello detail is the sampler name. Everything else is reduced
to the minimum needed for the working vs failing comparisons.

## Run

Serve from inside this folder:

```sh
cd webgl_demo
python3 -m http.server
```

Then open:

```text
http://localhost:8000/
```
