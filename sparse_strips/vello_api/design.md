# Vello API design thoughts

> [!WARNING]
>
> These design notes have not been curated carefully, and instead were written to focus thoughts.
> They might be useful as a reference to understand decisions made, but the ground-truth is the code

Considerations for Vello API; use cases, required features, etc.

## Scope

### Still TO-DO

- Draw images/gradients; images must be pre-uploaded, maybe gradients too.
  - What does it mean to pre-upload gradients?
  - Error handling for missing/dead image resources. ✓, impossible to be missing, by using handles.
- Filter effects, a core required set and support for externally registered effects.
  Externally registered effects *should* be able to:
  - Have a `static` identifier (this allows a third-party filter effect library)
  - Support detecting whether or not they're available, somehow.
  - Have runtime configuration options, ideally type-checked.
  - All of this suggests that using [`TypeId`](core::any::TypeId)s and a trait for validation is the way.
- Image atlasing, i.e. drawing from a subregion of a registered image.
  - "Needed" for atlas based glyph rendering.
  - Quite hard because of mipmaps, might need to be pre-registered.
- Layers
  - Being non-isolated.
  - Filters.
- Downcasting, for renderer-specific options (e.g. [`set_aliasing_threshold`]). (Maybe works, but untested)
- Prepared Paths:
  - Translation
- Clip optimisation (which is equivalent to non-isolated `Normal` blending), ✓-ish

### "Completed"

- Fill/Stroke paths. ✓
- Transforms applied to fills/strokes and their contents. ✓
- Blurred rounded rectangles (we should maybe test it against the actual blurring?) ✓
- Split the renderer and the "scene". ✓
- Layers. These need:
  - A "clip path" (this being optional, i.e. falling back to the viewport, is useful). ✓
  - A blend mode. ✓
  - An opacity. ✓
- Cached intermediates for subpaths (i.e. for glyph caching if outlines are drawn "inline"). ✓
  - Support for applying a translation to this cache, including recalculation if needed. ✓, current design makes recalculation never needed
  - Need for care when:
    - Scaling, so that the right caches are kept. ✓, responsibility of consumer, i.e. Parley Draw
    - Subpixel translations (e.g. for unhinted glyphs, probably want 4 horizontal and zero vertical subpixels for each glyph). ✓, responsibility of consumer, i.e. Parley Draw
    - Translating previously off-screen items to be on-screen. ✓, API leaves this unsupported/unrepresentable
  - Is it sound to support changing the paint? ✓, has no paint information
- Some sort of reset mechanism? ✓-ish, re-use handled by the "renderer".

- No per-operation allocations. ✓

## Medium Scope

- "Upload" CPU images "immediately", i.e. I have a CPU texture, and want to display it on-screen.
  - This needs to support cleanup.
  - Is there any sense in which it makes sense to support uploading images asynchronously?
    - Answer: Yes, if we ever get a "remote" backend.
      This would be atomically async, i.e. the "content generator" would be uploading a complete image, but with lower priority.
      GPUs are already effectively a remote backend, but we can't make any specific task "lower priority".
      Imagined use cases:
        - Farming off massive render to server? Seems more likely that you'd also prepare the scene on that server.
        - Local machine not powerful enough to do even simple render - would it be powerful enough to play a livestream?
        - Remote machine generating commands to be rendered by a thin client; e.g. a remote (sandboxed?) browser.
    - Answer: Progressive enhancement also demands this, i.e. for slowly downloading images.
      That's a different kind of async, though, not atomically async.
- There are arguments for supporting "global"/`static` image ids, like filters.
- Mipmaps, both explicit and automatic.
- Output image from one render "pass" and use it in another.
  This should work "immediately".
  Again, this needs support for cleanup.
- Download render results back to the CPU. This must be "async".
  - We should think about how this interacts with tests.
- Masks: How do they work?
  - Do we want to support rendering *something* in 1 channel?
  - What about RGB-only?
- Portability between renderers, especially of (image) resources.
- Interaction with multi-threading.
  - Should be possible to make scenes on multiple threads
  - Anything which is ambiguous should require an explicit happens-before ordering
    (e.g. thread 1: render to A; thread 2: render to A; thread 3: use A in render)
  - Dispatch of all "fine" rasterisation maybe happens on one thread (e.g. so we can have a single wgpu `submit` call)?
  - Reasoning: Multi-threaded dispatch/splitout must be explicit.
- Minimal per-frame allocations

## As yet undetermined

- Fully generic/backend-interchangable recording
  - What are the use cases for this?
- Precision, in intermediate calculations and similar.
- Rendering color space/HDR/Tonemapping.
- Rendering "over" a previous image. ✓
  Support for such invalidation might be valuable for external textures.
- Per-path blendmodes.
- Interaction with compositor layers; this could *feasibly* be explicitly out-of-scope.
- No per-operation dynamic dispatch. (Impossible?)

Out of scope:

- Creating the actual renderer.
- Use image already on GPU (i.e. `wgpu::Texture`); this a renderer-specific extension.
- Glyph rendering, outlining and caching (Scoped to "Parley Draw").
- Registering custom filter effects.
- Drive a GPU surface (i.e. output to a buffer).
  Note that we *do* need to not *block* this.

## Image Resources

The classes of image resources are:

- Render outputs.
- Input (CPU) images.
- External textures.

Do we want to allow rendering output to a subset of an input image?
Servo wants to be able to render more content to a render output, which ideally we'd support in a memory-efficient way.
Is that backend specific? The question becomes how is multithreading handled.
Explicit usages for the textures?

[`set_aliasing_threshold`]: https://docs.rs/vello_cpu/latest/vello_cpu/struct.RenderContext.html#method.set_aliasing_threshold
