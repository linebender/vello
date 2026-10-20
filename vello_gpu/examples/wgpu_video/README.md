# Vello GPU wgpu video example

A wgpu-native example that renders decoded NV12 video frames through Vello
GPU's external-texture API (`ImageSource::external_texture` +
`TextureBindings::insert_wgpu_external_texture`) backed by `wgpu::ExternalTexture`.

## What it does

macOS only. Scans `~/Downloads/videos/` for `.mp4` / `.mov` files, opens each
through a small AVFoundation / VideoToolbox pipeline (see
[`src/video/`](src/video)), decodes to NV12, and wraps the resulting Y/UV plane
`wgpu::Texture`s in a `wgpu::ExternalTexture` that Vello samples directly. wgpu
performs the YCbCr → RGB conversion when Vello's shader loads from the external
texture; there is no CPU-side or GPU-side intermediate RGBA copy. The
plane textures are zero-copy — backed by the same `IOSurface`s VT decoded into.
All videos play simultaneously in a grid layout. A still image
(`assets/splash-flower.jpg`, uploaded once with `Renderer::upload_image`) sits in the
free space between the first two videos of the top row. A large yellow "Vello GPU" text
overlay is rendered on top of the grid using `Scene::glyph_run` with the bundled
Roboto font, and inherits the user's pan / zoom / rotation.

On other platforms the binary still builds, but `main` immediately exits with a
diagnostic — there is no software-decode fallback.

## Architecture

The macOS pipeline is a minimal AVFoundation / VideoToolbox decoder:

```
Demuxer (AVAssetReader, compressed CMSampleBuffers)
  -> VideoDecoder (VTDecompressionSession, NV12 CVPixelBuffers in decode order)
  -> MetalImporter (CVMetalTextureCache, one wgpu::Texture per plane)
  -> VideoFrame { external_texture, pts_ns, color_space, ... }
  -> VideoFileSource (reorders to display order, paces by timestamp)
  -> Scene::set_paint (ImageSource::external_texture) + TextureBindings::insert_wgpu_external_texture
```

Each frame's `wgpu::ExternalTexture` is created from two plane views — a
full-resolution `R8Unorm` Y plane and a half-resolution `Rg8Unorm` interleaved
Cb/Cr plane — with `ExternalTextureFormat::Nv12` and a YCbCr → RGB matrix built
from the source's color-space metadata (matrix coefficients and range). Vello's
`render.wesl` sees a WGSL `texture_external` and receives RGB from `textureLoad`.

`wgpu::Features::EXTERNAL_TEXTURE` is required, which wgpu currently implements
only on Metal, DX12 and WebGPU.

A frame's planes live in an IOSurface that VideoToolbox recycles once the frame is
dropped. `VideoFileSource` therefore keeps each frame it takes off screen until the
GPU reports (through `Queue::on_submitted_work_done`) that every submission that
may have sampled it has finished.

Limitations:
* Decoding is synchronous, on the render thread. A real player would decode on a
  background thread and hand frames over through a bounded queue.
* Small fixed reorder buffer (4 frames). Handles typical H.264 / HEVC B-frame
  patterns; sources with extreme reorder distance may still display out of
  order.
* No seeking or format-change handling; restarting reopens the file.
* Track rotation (`preferredTransform`), pixel aspect ratio and clean aperture are
  ignored, so portrait phone videos show sideways.
* Only the YCbCr matrix and range are applied. HDR (PQ / HLG) and BT.2020 content
  is not color managed and looks washed out; a warning is logged when opening it.
* No IOSurface-keyed plane cache — every frame imports its planes fresh.
* No audio.

## Getting sample videos

The example reads every `.mp4` / `.mov` in `~/Downloads/videos/` and needs at
least one. Any H.264 / HEVC file that AVFoundation can open will do; the
Blender Foundation's open movies are a good default since they are freely
redistributable (Creative Commons Attribution) and are the de facto standard
video test content:

```sh
mkdir -p ~/Downloads/videos
cd ~/Downloads/videos

# Big Buck Bunny (2008, CC BY 3.0) — 1280x720 H.264, ~400 MB
curl -LO https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_720p_h264.mov.zip
unzip -j big_buck_bunny_720p_h264.mov.zip '*.mov' && rm big_buck_bunny_720p_h264.mov.zip

# Tears of Steel (2012, CC BY 3.0) — 1280x534 H.264, ~370 MB
curl -LO https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov

# Elephants Dream (2006, CC BY 2.5) — 720x405 H.264, ~150 MB
curl -LO https://download.blender.org/ED/elephantsdream-720-h264-st-aac.mov

# Sintel trailer (2010, CC BY 3.0) — 1920x1080 H.264, ~15 MB
curl -LO https://download.blender.org/durian/trailer/sintel_trailer-1080p.mp4
```

For a high-frame-rate stress test of the PTS pacing, the 2013 re-render at
<https://download.blender.org/demo/movies/BBB/> has 1080p / 2160p MP4s at 30
and 60 fps under the same license.

Only `.mp4` and `.mov` extensions are picked up — rename `.m4v` files or add the
extension to `build_video_slots` in `src/main.rs`. MKV / WebM are not supported
by `AVAssetReader` and will be skipped with a warning.

If you reuse these clips, credit `(c) Blender Foundation` with the respective
project URL (`www.bigbuckbunny.org`, `mango.blender.org`, `orange.blender.org`,
`www.sintel.org`).

## Run

```sh
cargo run -p vello_gpu_wgpu_video --release
```

## Controls

* `Esc` — exit
* `R` — rewind / restart all videos from the first frame (also resets pan/zoom/rotation)
* `L` — toggle auto-replay (loop videos when they reach end-of-stream; on by default)
* `Space` — reset pan / zoom / rotation to identity
* `[` / `]` — rotate scene 15° counter-clockwise / clockwise around cursor (or window center if no cursor)
* **Mouse drag** (left button) — pan
* **Scroll wheel** — zoom at cursor
* **Trackpad pinch** — zoom at cursor
