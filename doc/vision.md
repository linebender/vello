## The piet-gpu vision

Raph Levien, 2020-12-10

Note: `vello` was previously called `piet-gpu`.

I’ve done several [blog posts](./blogs.md) about piet-gpu already, and more generally GPU compute, but this document is a little different in scope. Rather than showing off a prototype and presenting a research result, it will set forth a bold and ambitious plan for where this might go. I find this vision compelling, and it’s motivated me to spend a lot of energy mastering some difficult material. The grand vision is much more than one person can do, so I’ll do some of it myself and maybe inspire collaboration for the rest of it.

The full vision for piet-gpu is a 2D rendering engine that is considerably faster, higher quality, and more flexible than the current state of the art, and runs on a wide variety of hardware. I’ll go into some detail about why I think this goal is possible and what kind of work is needed to get there.

The current state of the piet-gpu codebase is an early stage prototype, largely to test whether the ideas are viable and to gather empirical performance data on some of the more intensive parts of the rendering problem, so far mainly antialiased vector filling and stroking.

## Compute-based 2D rendering

The central theme of piet-gpu is to do most or all of the rendering steps in compute shaders. This is quite a different philosophy to the traditional rasterization-based approach to 2D rendering, which breaks the scene (on the CPU side) into a series of draw calls, which are then sent to the GPU. This works extremely well when the mapping to draw calls is simple (which is the case for imgui-style UI made up of text and simple graphic elements), but otherwise much less so. In using GPU compute extensively, piet-gpu draws much inspiration from [Spinel].

Using compute shaders has profound effects at two particular stages in the pipeline. First, in early stages, it lets the GPU ingest a scene description that is, as much as possible, a straightforward binary encoding of the scene. That, in turn, makes the CPU-side part of the job simple and efficient, allowing higher frame rates on complex scenes without jank.

Second, in the last stage (“fine rasterization”), compositing takes place within the compute shader, using vector registers rather than texture buffers in global memory for intermediate RGBA values.

Note that the benefits depend on the scene. For a static (or mostly static) scene, the CPU-side encoding cost might not matter much because it can be done ahead of time. Similarly, if the scene doesn’t require sophisticated compositing, but is just a series of alpha-blended draws, existing rasterization pipelines can handle those very efficiently. But piet-gpu should fly with dynamic scenes with lots of masking and blending, where existing 2D engines would struggle.

The intermediate stages benefit too. The coarse rasterization step can employ sophisticated logic to enable optimizations on a per-tile granularity that would otherwise rely on brute force.

## Retained scene graph fragments

Applications vary in their degree of dynamism. At one extreme, the scene is mostly static, with perhaps a few variable elements and perhaps some animation done at compositing time (I think of this as the iPhone style of UI, as it’s so well adapted to mechanisms like Core Animation). At the other extreme, every rendered frame is completely different from the one before, so encoding needs to be done entirely from scratch every time; these applications are well adapted to an “immediate mode” approach.

I’m most interested in cases in the middle. I believe the best approach is to split the encoding process so that the static parts of the scene graph can be encoded once into a retained scene graph fragment, then these fragments can be stitched together, along with the dynamically encoded parts of the scene, with a minimum of CPU effort.

Much of the piet-gpu architecture is geared towards supporting this goal. Notably, the global affine transformation is not baked into the encoding of vector paths, so the same binary encoding of a vector path can be instanced (perhaps multiple times within a scene) with different transforms. Applying the transform is done GPU-side, early in the [pipeline][sort-middle architecture]. Thus, animating the transform should be very efficient, and the vector paths will be re-rendered at full resolution with vector crispness.

Even so, fully realizing retained scene graph fragments will be one of the more difficult parts of the vision. It requires a good API to represent retained fragments, as well as incrementally update parameters such as transformation and opacity. It also requires a sophisticated approach to resource management so that resources backing the retained fragments can be efficiently cached GPU-side without hogging relatively scarce GPU memory. As such, I will focus on immediate mode first, as that is also an important case. But make no mistake, the goal of retaining scene fragments is motivating a number of design decisions, in particular leading me away from shortcuts such as applying affine transforms CPU-side during encoding.

## Portable compute runtime

One challenge facing piet-gpu is the lack of adequate infrastructure for portable GPU compute. Most research is done on CUDA, as that is the only truly viable platform for GPU compute today, but that would make it essentially impossible to deploy the work on any hardware other than Nvidia.

I strongly believe that Vulkan is emerging as a viable low-level platform for utilizing GPU compute resources. I’m also not the only one thinking along these lines. The [VkFFT] project is an impressive demonstration that a Vulkan deployment of one math-intensive algorithm can be just as performant as the CUDA version. In addition, there are early steps toward running machine learning workloads on Vulkan, particularly TensorFlow Lite.

Of course, while it’s possible to run Vulkan on a pretty wide range of hardware, it doesn’t solve all portability problems. “Runs Vulkan” is not a binary, but rather a portal to a vast matrix of optional features and limits from the various combinations of hardware, drivers, and compatibility shims ([vulkan.gpuinfo.org] is an excellent resource). In particular, Apple forces the use of Metal. In theory, MoltenVk — or, more generally, the [Vulkan Portability Extension] — lets you run Vulkan code on Apple hardware, but in practice it doesn’t quite work (see [#42]), and there are compatibility and integration advantages to DX12 over Vulkan on Windows; older CPU generations such as Haswell and Broadwell don’t support Vulkan at all. To this end, I’ve started a portability layer (piet-gpu-hal) which should be able to run natively on these other API’s.

### Why not wgpu?

The compatibility layer has overlapping goals as [wgpu], and WebGPU more broadly. Why not just use that, as much of the Rust ecosystem has done?

It’s *very* tempting, but there is also some divergence of goals. The main one is that to keep the piet-gpu runtime light and startup time quick, I really want to do ahead-of-time compilation of shaders, so that the binary embeds intermediate representation for the target platform (DXIL for Windows 10, etc). Further, by using Vulkan directly, we can experiment with advanced features such as subgroups, the memory model, etc., which are not yet well supported in wgpu, though it certainly would be possible to add these features. I don’t know how much these advanced features contribute, but that’s one of the research questions to be addressed. If the gain is modest, then implementing them is a low priority. If the gain is significant, then that should increase motivation for runtimes such as wgpu to include them.

Also see the section on incremental present, below, which is another feature that is not yet well supported in wgpu, so working with lower level APIs should reduce the friction.

At the same time, wgpu continues to improve, including focus on making the runtime leaner (using the new [naga] shader compilation engine rather than spirv-cross is one such advance). My sense is this: a primary reason for piet-gpu to have its own compatibility layer is so that we can really clarify and sharpen the requirements for a more general GPU compute runtime.

### Compatibility fallback

One challenge of a compute-centric approach is that there is not (yet) an ironclad guarantee that the GPU and drivers will actually be able to handle the compute shaders and resource management patterns (the latter may actually be more of a challenge, as piet-gpu relies on [descriptor indexing] to address multiple images during fine rasterization).

There are a number of approaches to this problem, including building hybrid pipelines and otherwise doing lots of compatibility engineering, to target platforms well on their way to becoming obsolete. But I worry quite a bit about the complexity burden, as well as pressure away from the absolute best solution to a problem if it poses compatibility challenges.

I’m more inclined to fall back to CPU rendering. Projects such as [Blend2D] show that CPU rendering can be performant, though nowhere nearly as much as a GPU. Of course, that means coming up with CPU implementations of the algorithms.

One intriguing possibility is to automatically translate the Vulkan compute shaders to CPU runnable code. This approach has the advantage of maintaining one codebase for the pipeline, reducing friction for adding new features, and guaranteeing pixel-perfect consistency. The biggest question is whether such an approach would be adequately performant. A very good way to get preliminary answers is to use [SwiftShader] or Mesa’s [Lavapipe], which do JIT generation of CPU side code. Obviously, for reasons of startup time and binary size it would be better to ship ahead-of-time translated shaders, but that’s a practical rather than conceptual problem.

There are examples of compile time translation of shaders to CPU code. An intriguing possibility is the [spirv to ispc translator], which doesn’t seem to be actively developed, but would seem to be a path to reasonably good CPU performance from shaders. Another, actually used in production in WebRender, is [glsl-to-cxx].

A truly universal compute infrastructure with unified shader source would have implications far beyond 2D rendering. The domain most likely to invest in this area is AI (deployment to consumer hardware; for server side and in-house deployment, they’ll obviously just use CUDA and neural accelerators). I’ll also note that this problem is ostensibly within scope of OpenCL, but they have so far failed to deliver, largely because they’ve historically been entirely dependent on driver support from the GPU manufacturer. I expect *something* to happen.

There is another perfectly viable path this could take, less dependent on shader compilation infrastructure: a software renderer developed in parallel with the GPU one. Possible existing Rust code bases to draw on include [raqote] and [tiny-skia]. These make more sense as community sub-projects (see below).

## Text

An essential part of any 2D library is text rendering. This really breaks down into text layout and painting of glyphs. Both are important to get right.

The Piet of today is primarily an abstraction layer over platform 2D graphics libraries, and that’s equally true of text. We’ve lately made some really good progress in a common [rich text API] and implementations over DirectWrite and Core Text. However, it is currently lacking a Linux backend. (As a placeholder, we use the Cairo “toy text API,” but that is unsatisfying for a number of reasons.)

I think we want to move away from abstracting over platform capabilities, for several reasons. One is that it’s harder to ensure consistent results. Another is that it’s hard to add new features, such as hz-style justification (see below). Thus, we follow a similar trajectory as Web browsers.

As a project related to piet-gpu, I’d love to build (or mentor someone to build) a text layout engine, in Rust, suitable for most UI work. This wouldn’t be my first time; I wrote the original version of [Minikin], the text layout engine first shipped in Android Lollipop.

### Painting

Ultimately, I’d like piet-gpu to support 3 sources of glyph data for painting.

The first is bitmaps produced by the platform. These have the advantage of matching native UI, and also take maximum advantage of hinting and subpixel RGB rendering, thus improving contrast and clarity. These bitmaps would be rendered mostly CPU-side, and uploaded into a texture atlas. The actual rasterization is just texture lookups, and should be super efficient.

The second is dynamic vector rendering from glyph outlines. This source is best optimized for large text, animation (including supporting pinch-to-zoom style gestures), and possible extension into 3D, including VR and AR. The lack of hinting and RGB subpixel rendering is not a serious issue on high-dpi screens, and is not an expectation on mobile. Early measurements from piet-gpu suggest that it should be possible to maintain 60fps of text-heavy scenes on most GPUs, but power usage might not be ideal.

Thus, the third source is vector rendering through a glyph cache, something of a hybrid of the first two sources. Originally, management of the cache will be CPU-side, and managed during encoding (likely using [Guillotière], [Étagère], or something similar), but in the future we might explore GPU-side algorithms to manage the cache in parallel, reducing CPU requirements further.

### GPU-side variable fonts

A very intriguing possibility is to offload most of the work of rendering variable fonts to GPU. There are reasons to believe this would work well: [variable font technology] is fundamentally based on multiplying vectors of “deltas” with basis functions and adding those up, a task ideally suited to GPU.

A challenge is representing the coordinate data and deltas in a GPU-friendly format; the [glyf] and [gvar] table formats are designed for compact data representation and (reasonably) simple decoding by scalar CPUs, but are challenging for massively parallel algorithms. Decoding to fixed-size numbers is straightforward but might use a lot of GPU memory and bandwidth to represent the font data (especially a problem for CJK fonts). One intriguing approach is to re-encode the underlying data using a self-synchronizing variable integer encoding, which would reduce the memory requirements but preserve the ability to do processing in parallel.

The major advantages of GPU-side variable font rendering are to allow efficient animation of variable font axes, and also to open up the possibility of adjusting the axes to improve text layout, for example to improve the quality of paragraph justification as pioneered by the [hz] prototype and recently demonstrated with [amstelvar], or to support calligraphic styles and complex scripts better, for example to make more beautiful [kashida] for Arabic, all without significantly reducing performance.

## Improving rendering quality

The question of quality in GPU 2D rendering has long been complex. Many rasterization based approaches are dependent on [MSAA] in the GPU’s fixed-function pipeline, which may not always be available or perhaps only practical at lower settings (especially on mobile). Thus, GPU accelerated 2D rendering quality has gotten something of a bad name.

A compute-centric approach changes the story. All actual pixels are generated by code; the quality of the rendering is entirely up to the author of that code. The current piet-gpu codebase uses an exact-area approach to antialiasing (in the [tradition of libart]), and thus does not exhibit stepping or graininess characteristic of MSAA at low or medium settings. The quality should be the same as a good software renderer, because it *is* a software renderer, just one that happens to be running on hardware with orders of magnitude more parallelism than any reasonable CPU.

Even so, I believe it’s possible to do even better. A CPU-bound renderer has barely enough performance to get pixels to the screen, so takes whatever shortcuts are needed to get the job done in that performance budget. A GPU typically has an order of magnitude more raw compute bandwidth, so there is headroom that can be used to improve quality.

The details of what I have in mind could be a blog post in and of itself, but I’ll sketch out the highlights.

Perhaps the most important quality problem is that of so-called “conflation artifacts,” the seams that happen when compositing antialiased elements (see [#49]). Most of the academic literature on 2D rendering on GPU addresses this question. I think it’s practical to do in the piet-gpu architecture, basically by swapping out soft-alpha compositing in the fine rasterizer with one based on supersampling. Some of the academic literature also takes the opportunity at that stage in the pipeline to apply a reconstruction filter more sophisticated than a box filter, but I am not yet convinced that the improvement is worth it, especially as physical display resolution increases.

The next major area of potential quality improvement is getting gamma right. This is a surprisingly tricky area, as a theoretically “correct” approach to gamma often yields text and hairline strokes that appear weak and spindly. Another concern is document compatibility; simply changing the gamma of the colorspace in which alpha blending happens will change the color of the result. Likely, a perfect solution to this problem will require cooperation with the application driving the renderer; if it is designed with gamma-perfect rendering in mind, there is no real problem, but otherwise it’s likely that various heuristics will need to be applied to get good results. (Note that [stem darkening] is one approach used specifically for text rendering, and among other things is a source of considerable variation between platforms.)

When driving low-dpi displays (which still exist), one opportunity to improve quality is more sophisticated RGB [subpixel rendering]. Currently, that’s basically text-only, but could be applied to vector rendering as well, and often doesn’t survive sophisticated compositing, as an RGBA texture with a transparent background cannot represent RGB subpixel text. One solution is to do compositing with per-channel alpha, which can be done very efficiently when compositing in a compute shader, but would be a serious performance problem if intermediate texture buffers needed to be written out to global memory.

These potential quality improvements may well provide the answer to the question, “why move to a new rendering architecture instead of incrementally improving what we’ve got now?”

## Enriching the imaging model

There is consensus on “the modern 2D imaging model,” roughly encompassing PDF, SVG, HTML Canvas, and Direct2D, but it is not set in stone and with considerable variation in advanced features within those systems (for example, gradient meshes are more or less unique to PDF — the feature was proposed for SVG 2 but [then removed](https://librearts.org/2018/05/gradient-meshes-and-hatching-to-be-removed-from-svg-2-0/)).

I like this consensus 2D imaging model because I feel it is extremely well suited for UI and documents of considerable richness and complexity, and is quite designer-friendly. There is also tension pulling away from it, I think for two reasons. One is that it is not always implemented efficiently on GPU, especially with deeply nested soft clipping and other nontrivial compositing requirements. The other is that it’s possible to do things on GPU (especially using custom shaders) that are not easily possible with the standard 2D api. Shadertoy shows *many* things that are possible in shaders. One idea I’d like to explore is watercolor brush strokes (see [Computer-Generated Watercolor](https://grail.cs.washington.edu/projects/watercolor/paper_small.pdf) for inspiration). I think it would be possible to get pretty far with distance fields and procedural noise, and a simple function to go from those to paint values for paint-like compositing.

Another direction the imaging model should go is support for [HDR] (strong overlap with the gamma issue above). This will require color transformations for tone mapping in the compositing pipeline, which again can be written as shaders.

One interesting existing 2D engine with extension points is Direct2D, which lets users provide [Custom effects](https://docs.microsoft.com/en-us/windows/win32/direct2d/custom-effects) by linking in compute shaders. Of course, it is a major challenge to make such a thing portable, but I’m encouraged about building on existing GPU infrastructure efforts. In particular, over time, I think WebGPU could become a standard way to provide such an extension point portably.

Blurs are a specific case that should probably be done early, as they’re very widely used in UI. In the general case, it will require allocating temporary buffers for the contents being blurred, which is not exactly in the spirit of piet-gpu compositing, largely because it requires a lot of resource management and pipeline building CPU-side, but is possible. I’ve already done research on a special case, a [blurred rounded rectangle], which can be computed extremely efficiently as a fairly simple shader. The encoder would apply a peephole-like optimization during encoding time, pattern matching the blurred contents and swapping in the more efficient shader when possible.

## Incremental present

In the old days, UI tracked “dirty rectangles,” and only redrew what actually changed, as computers just weren’t fast enough to redraw the entire screen contents in a single refresh period. Games, on the other hand, need to redraw every pixel every frame, so the GPU pipeline became optimized for those, and many rendering engines got more relaxed about avoiding redrawing, as the GPU was plenty fast for that.

Today, the GPU is still plenty fast, but there are still gains to be had from incremental present, primarily power consumption. Blinking a cursor in a text editor should not run the battery down. Also, on low resource devices, incremental present can reduce latency and increase the chance of smooth running without dropped frames.

The tile-based architecture of piet-gpu is extremely well suited to incremental present, as the various pipeline stages are optimized to only do work within the viewport (render region). This is especially true for fine rasterization, which doesn’t touch any work outside that region.

A small challenge is support by the GPU infrastructure, which tends to be more optimized for games than UI. DirectX has long had [good support](https://docs.microsoft.com/en-us/windows/win32/api/dxgi1_2/nf-dxgi1_2-idxgiswapchain1-present1). The Vulkan world is spottier, as it’s available as an extension. That extension tends to be available on Linux (largely because [Gnome can make good use of it](https://feaneron.com/2019/10/05/incremental-present-in-gtk4/)), and some on Android, but in my experiments less so on desktop. And of course Metal can’t do it at all.

## Roadmap and community

This vision is *very* ambitious. There’s no way one person could do it all in a reasonable amount of time. It’s a multi-year project at best, and that’s not counting the year and a half since the first piet-metal prototype.

There are a few ways I plan to deal with this. First is to be explicit that it is a research project. That means that certain elements, especially dealing with compatibility, are a lower priority. Other projects in a similar space have sunk a lot of time and energy into working around driver bugs and dealing with the complex landscape of GPU capability diversity (especially on older devices and mobile). The initial goal is to prove that the concepts work on a reasonably modern GPU platform.

Another strategy is to split up the work so that at least some parts can be taken up by the community. There are a number of interesting subprojects. Also, it would be wonderful for the runtime work to be taken up by another project, as most of it is not specific to the needs of 2D rendering.

I’d really like to build a good open-source community around piet-gpu, and that’s already starting to happen. The #gpu stream on [xi.zulipchat.com] hosts some really interesting discussions. In addition, the [gio] project is exploring adopting the compute shaders of piet-gpu (with the CPU runtime in Go) and has made substantive contributions to the code base. There’s a lot of research potential in piet-gpu, and knowledge about GPU compute programming in general, that I think is valuable to share, so it’s my intent to keep creating blog posts and other materials to spread that knowledge. Academic papers are also within scope, and I’m open to collaboration on those.

I'm really excited to see where this goes. I think there's the potential to build something truly great, and I look forward to working with others to realize that vision.

There's been some great discussion on [/r/rust](https://www.reddit.com/r/rust/comments/kal8ac/the_pietgpu_vision/).

[hz]: https://en.wikipedia.org/wiki/Hz-program
[spirv to ispc translator]: https://software.intel.com/content/www/us/en/develop/articles/spir-v-to-ispc-convert-gpu-compute-to-the-cpu.html
[tiny-skia]: https://github.com/RazrFalcon/tiny-skia
[raqote]: https://github.com/jrmuizel/raqote
[Blend2D]: https://blend2d.com/
[amstelvar]: https://variablefonts.typenetwork.com/topics/spacing/justification
[kashida]: https://andreasmhallberg.github.io/stretchable-kashida/
[SwiftShader]: https://swiftshader.googlesource.com/SwiftShader
[Lavapipe]: https://www.phoronix.com/scan.php?page=news_item&px=Mesa-Vulkan-Lavapipe
[glsl-to-cxx]: https://github.com/servo/webrender/tree/master/glsl-to-cxx
[sort-middle architecture]: https://raphlinus.github.io/rust/graphics/gpu/2020/06/12/sort-middle.html
[vulkan.gpuinfo.org]: https://vulkan.gpuinfo.org
[Vulkan Portability Extension]: https://www.khronos.org/blog/fighting-fragmentation-vulkan-portability-extension-released-implementations-shipping
[xi.zulipchat.com]: https://xi.zulipchat.com
[glyf]: https://docs.microsoft.com/en-us/typography/opentype/spec/glyf
[gvar]: https://docs.microsoft.com/en-us/typography/opentype/spec/gvar
[VkFFT]: https://github.com/DTolm/VkFFT
[Spinel]: https://fuchsia.googlesource.com/fuchsia/+/refs/heads/master/src/graphics/lib/compute/spinel/
[wgpu]: https://github.com/gfx-rs/wgpu
[naga]: https://github.com/gfx-rs/naga
[descriptor indexing]: http://chunkstories.xyz/blog/a-note-on-descriptor-indexing/
[rich text API]: https://www.cmyr.net/blog/piet-text-work.html
[Guillotière]: https://github.com/nical/guillotiere
[Étagère]: https://crates.io/crates/etagere
[variable font technology]: https://docs.microsoft.com/en-us/typography/opentype/spec/otvaroverview
[MSAA]: https://en.wikipedia.org/wiki/Multisample_anti-aliasing
[tradition of libart]: https://people.gnome.org/~mathieu/libart/internals.html
[stem darkening]: https://freetype.org/freetype2/docs/hinting/text-rendering-general.html
[subpixel rendering]: https://en.wikipedia.org/wiki/Subpixel_rendering
[HDR]: https://en.wikipedia.org/wiki/High-dynamic-range_imaging
[blurred rounded rectangle]: https://raphlinus.github.io/graphics/2020/04/21/blurred-rounded-rects.html
[gio]: https://gioui.org/
[Minikin]: https://android.googlesource.com/platform/frameworks/minikin/
[#42]: https://github.com/linebender/piet-gpu/issues/42
[#49]: https://github.com/linebender/piet-gpu/issues/49
