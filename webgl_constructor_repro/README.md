# WebGL Constructor Repro

This folder contains a tiny static page that compiles several WebGL2 shaders
matching the constructor-heavy GLSL patterns emitted by `vello_sparse_shaders`.

It is intended to answer one question:

Does the device fail on these constructor forms by themselves?

## Run

From the repo root:

```sh
python3 -m http.server
```

Then open:

```text
http://localhost:8000/webgl_constructor_repro/
```

## What it tests

- `float[3](...)` array constructor
- struct constructor with array arguments
- struct constructor with `uint[12](...)`
- large struct constructor with many arguments
- a conservative field-assignment rewrite

If the device fails one or more constructor-heavy shaders while the safe rewrite
passes, that strongly suggests the Vello issue is GLSL compiler portability.
