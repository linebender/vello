# Architecture

This document should be updated semi-regularly. Feel free to open an issue if it hasn't been updated in more than a year.

## Goals

The major goal of Vello is to provide a high quality GPU accelerated renderer suitable for a range of 2D graphics applications, including rendering for GUI applications, creative tools, and scientific visualization.

Vello emerges from being a research project, which attempts to answer these hypotheses:

- To what extent is a compute-centered approach better than rasterization ([Direct2D])?
- To what extent do advanced GPU features (subgroups, descriptor arrays, device-scoped barriers) help?
- Can we improve quality and extend the imaging model in useful ways?

Another goal of the overall project is to explain how the renderer is built, and to advance the state of building applications on GPU compute shaders more generally.
Much of the progress on Vello is documented in blog entries.
See [blogs.md](blogs.md) for pointers to those.

Ideally, we'd like our documentation to be more structured; we may refactor it in the future (see [#488]).


## Roadmap

The [roadmap for 2023](roadmap_2023.md) is still largely applicable.
The "Semi-stable encoding format" section and most of the "CPU fallback" section can be considered implemented.

Our current priority is to fill in missing features and to fix rendering artifacts, so that Vello can reach feature parity with other 2D graphics engines.


## File structure

The repository is structured as such:

- `doc/` - Various documents detailing the vision for Vello as it was developed. This directory should probably be refactored away; adding to it not recommended.
- `examples/` - Example projects using Vello. Each example is its own crate, with its own dependencies. The simplest example is called `simple`.
- `vello/` - Code for the main `vello` crate.
- `vello_encoding/` - Types that represent the data that needs to be rendered.
- `vello_shaders/` - Infrastructure to preprocess and cross-compile shaders at compile time; see "Shader templating".
  - `shader/` - This is where the magic happens. WGSL shaders that define the compute operations (often variations of prefix sum) that Vello does to render a scene.
    - `shared/` - Shared types, functions and constants included in other shaders through non-standard `#import` preprocessor directives (see "Shader templating").
  - `cpu/` - Functions that perform the same work as their equivalently-named WGSL shaders for the CPU fallbacks. The name is a bit loose; they're "shaders" in the sense that they work on resource bindings with the exact same layout as actual GPU shaders.
- `vello_tests/` - Helper code for writing tests; current has a single smoke test and not much else.


## Shader templating

WGSL has no meta-programming support, which limits code-sharing.
We use a strategy common to many projects (eg Bevy) which is to implement a limited, simple preprocessor for our shaders.

This preprocessor implements the following directives:

1. `import`, which imports from `shader/shared`
2. `ifdef`, `ifndef`, `else` and `endif`, as standard.
  These must be at the start of their lines.  
  Note that there is no support for creating definitions in-shader, these are only specified externally (in `src/shaders.rs`).
  Note also that this definitions cannot currently be used in-code (`import`s may be used instead)

This format is compatible with [`wgsl-analyzer`], which we recommend using.
If you run into any issues, please report them on Zulip ([#vello > wgsl-analyzer issues](https://xi.zulipchat.com/#narrow/channel/197075-vello/topic/wgsl-analyzer.20issues/with/429480999)), and/or on the [`wgsl-analyzer`] issue tracker.  
Note that new imports must currently be added to `.vscode/settings.json` for this support to work correctly.
`wgsl-analyzer` only supports imports in very few syntactic locations, so we limit their use to these places.


## Path encoding

See [Path segment encoding](./pathseg.md) document.


## Intermediary layers

There are multiple layers of separation between "draw shape in Scene" and "commands are sent to wgpu":

- First, everything you do in `Scene` appends data to an `Encoding`.
The encoding owns multiple buffers representing compressed path commands, draw commands, transforms, etc. It's a linearized representation of the things you asked the `Scene` to draw.
- From that encoding, we generate a `Recording`, which is an array of commands; each `Command` represents an operation interacting with the GPU (think "upload buffer", "dispatch", "download buffer", etc).
- We then use `WgpuEngine` to send these commands to the actual GPU.

In principle, other backends could consume a `Recording`, but for now the only implemented wgpu backend is `WgpuEngine`.


### CPU rendering

The code in `vello_shaders/src/cpu/*.rs` and `vello_shaders/src/cpu.rs` provides *some* support for CPU-side rendering. It's in an awkward place right now:

- It's called through WgpuEngine, so the dependency on wgpu is still there.
- Fine rasterization (the part at the end that puts pixels on screen) doesn't work in CPU yet (see [#386]).
- Every single WGSL shader needs a CPU equivalent, which is pretty cumbersome.

Still, it's useful for testing and debugging.


[`wgsl-analyzer`]: https://marketplace.visualstudio.com/items?itemName=wgsl-analyzer.wgsl-analyzer
[direct2d]: https://docs.microsoft.com/en-us/windows/win32/direct2d/direct2d-portal
[#488]: https://github.com/linebender/vello/issues/488
[#467]: https://github.com/linebender/vello/issues/467
[#386]: https://github.com/linebender/vello/issues/386
