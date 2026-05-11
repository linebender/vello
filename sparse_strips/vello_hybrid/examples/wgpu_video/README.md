# Vello Hybrid wgpu video example

A wgpu-native example that renders decoded NV12 video frames through Vello
Hybrid's native YCbCr external-texture API
(`Scene::draw_texture_rects` with `ExternalTextureFormat::YCbCrNv12` +
`TextureBindings::insert_ycbcr_nv12`).

## What it does

macOS only. Scans `~/Downloads/videos/` for `.mp4` / `.mov` files, opens each
through a small AVFoundation / VideoToolbox pipeline (see
[`src/video/`](src/video)), decodes to NV12, and binds the resulting Y/UV plane
`wgpu::Texture`s straight into Vello. The YCbCr → RGB conversion happens inside
Vello's shader; there is no CPU-side or GPU-side intermediate RGBA copy. The
plane textures are zero-copy — backed by the same `IOSurface`s VT decoded into.
All videos play simultaneously in a grid layout. A large yellow "Mr. Bean" text
overlay is rendered on top of the grid using `Scene::glyph_run` with the bundled
Roboto font, and inherits the user's pan / zoom / rotation.

On other platforms the binary still builds, but `main` immediately exits with a
diagnostic — there is no software-decode fallback.

## Architecture

The macOS pipeline is a trimmed port of the video half of `lightspeed_av`:

```
Demuxer (AVAssetReader)
  -> VideoDecoder (VTDecompressionSession, NV12 output)
  -> DecodedVideoFrame { y_plane: wgpu::Texture, uv_plane: wgpu::Texture, color_space, ... }
  -> Scene::draw_texture_rects (ExternalTextureFormat::YCbCrNv12) + TextureBindings::insert_ycbcr_nv12
```

The renderer hands Vello two `wgpu::TextureView`s per frame — full-resolution
`R8Unorm` Y plane and half-resolution `Rg8Unorm` interleaved Cb/Cr plane — plus
the source's color-space metadata (`YCbCrInfo` = `{ matrix, range }`). Vello's
`render_strips.wgsl` does the YCbCr → RGB conversion at sample time using
those parameters.

Trimmed relative to `lightspeed_av`:
* Small fixed reorder buffer (4 frames). Handles typical H.264 / HEVC B-frame
  patterns; sources with extreme reorder distance may still display out of
  order.
* No flush / seek / format-change handling.
* No IOSurface-keyed plane cache — every frame imports its planes fresh.
* No audio.

## Run

```sh
cargo run -p vello_hybrid_wgpu_video --release
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
