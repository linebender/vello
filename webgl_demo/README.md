# WebGL Demo

This folder contains a tiny static page that links a few WebGL2 programs and
then inspects uniform locations.

It is intended to answer one question:

Does the device return `null` for Vello's clip-input sampler uniform even though
the shader links successfully?

## Run

From the repo root:

```sh
python3 -m http.server
```

Then open:

```text
http://localhost:8000/webgl_demo/
```

## What it tests

- a tiny control shader with an unconditionally used sampler
- a reduced branchy sampler-use shader
- the real generated `render_strips.vert.glsl` + `render_strips.frag.glsl`

For each linked program it reports:

- the link log
- the active uniform list
- whether `getUniformLocation("_group_0_binding_2_fs")` is `present` or `null`
