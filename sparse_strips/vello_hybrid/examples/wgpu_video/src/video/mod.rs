// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Video pipeline: AVAsset demuxer -> VideoToolbox decoder -> NV12 plane import.
//!
//! Trimmed port of the macOS half of `lightspeed_av`. Frames are exposed to the
//! renderer as their *native* two-plane NV12 (`R8` Y plane + `Rg8` interleaved
//! Cb/Cr plane), backed zero-copy by the `IOSurface` `VideoToolbox` decoded into.
//! The renderer feeds those planes straight into Vello's
//! [`Scene::draw_texture_rects`](vello_hybrid::Scene::draw_texture_rects) API
//! with [`ExternalTextureFormat::YCbCrNv12`](vello_hybrid::ExternalTextureFormat::YCbCrNv12)
//! — there is no CPU-side or GPU-side `NV12 -> BGRA` conversion in this
//! example any more.

// Apple FFI-port noise relaxations applied module-wide:
// - `doc_markdown`: backticking every `CVPixelBuffer` / `IOSurface` / `MTLTexture` /
//   `CVMetalTexture` / `VTDecompressionSession` / `AVAssetReader` / `CMSampleBuffer`
//   mention in every doc comment is more line-noise than signal in an FFI port.
// - `cast_possible_truncation`: video frame widths fit in u32 in any practical
//   sense, and PTS values are derived from i64-valued `CMTime` and fit back in
//   i64 by construction.
#![allow(
    clippy::doc_markdown,
    reason = "Apple type names appear in nearly every doc line of this module"
)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "frame dimensions and CMTime-derived PTS fit in their target types by construction"
)]

mod decoder;
mod demuxer;
mod error;
mod frame;
mod metal_import;
mod packet;

use std::path::Path;

pub(crate) use error::AvError;
pub(crate) use frame::{ColorMatrix, ColorRange, ColorSpace, DecodedVideoFrame};

use decoder::VideoDecoder;
use demuxer::Demuxer;

/// Owning bundle of (demuxer + decoder).
///
/// `&mut`-driven: each call to [`Self::next_frame`] advances the pipeline by one
/// frame and returns an owned [`PlayerFrame`] the caller is expected to hold for
/// the duration of any sampling.
pub(crate) struct VideoPlayer {
    demuxer: Demuxer,
    decoder: VideoDecoder,
    eos_sent: bool,
}

/// One NV12 frame ready for binding into Vello's external-texture API.
///
/// Owns the underlying [`DecodedVideoFrame`] (which keeps the `IOSurface`-backed
/// `wgpu::Texture` Y/UV planes alive) plus pre-built `wgpu::TextureView`s for each
/// plane.
pub(crate) struct PlayerFrame {
    pub(crate) y_view: wgpu::TextureView,
    pub(crate) uv_view: wgpu::TextureView,
    /// Width of the Y plane in texels (= the displayed video width).
    pub(crate) width: u16,
    /// Height of the Y plane in texels (= the displayed video height).
    pub(crate) height: u16,
    /// Color-space metadata to feed into the YCbCr → RGB shader conversion.
    pub(crate) color_space: ColorSpace,
    /// Presentation timestamp of this frame in nanoseconds, propagated from
    /// `DecodedVideoFrame::pts_ns`. The renderer uses this to pace playback to
    /// the source's native frame rate.
    pub(crate) pts_ns: i64,
    /// Holds the underlying NV12 `wgpu::Texture`s and the `CVMetalTexture` /
    /// `CVPixelBuffer` retains alive while the renderer samples through the views.
    #[expect(dead_code, reason = "kept alive to back the plane views")]
    decoded: DecodedVideoFrame,
}

impl VideoPlayer {
    /// Open `path` and prepare a decoder against the given wgpu device.
    ///
    /// Reads the first video track only.
    pub(crate) fn open(
        path: &Path,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
    ) -> Result<Self, AvError> {
        let demuxer = Demuxer::open(path)?;
        let track = demuxer.track_info().clone();
        let decoder = VideoDecoder::new(adapter, device, &track)?;

        Ok(Self {
            demuxer,
            decoder,
            eos_sent: false,
        })
    }

    /// Pump the pipeline until a new NV12 frame is ready, returning it owned. Returns
    /// `None` once the demuxer reports end-of-stream and the decoder has nothing left
    /// buffered.
    pub(crate) fn next_frame(&mut self) -> Option<PlayerFrame> {
        loop {
            if let Some(decoded) = self.decoder.recv_frame() {
                let width = clamp_u16(decoded.width);
                let height = clamp_u16(decoded.height);
                let color_space = decoded.color_space;
                let pts_ns = decoded.pts_ns;
                let y_view = decoded
                    .y_plane()
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let uv_view = decoded
                    .uv_plane()
                    .create_view(&wgpu::TextureViewDescriptor::default());
                return Some(PlayerFrame {
                    y_view,
                    uv_view,
                    width,
                    height,
                    color_space,
                    pts_ns,
                    decoded,
                });
            }

            if self.eos_sent {
                return None;
            }

            match self.demuxer.read_packet() {
                Ok(Some(packet)) => {
                    if let Err(err) = self.decoder.send_packet(&packet) {
                        log::warn!("VideoDecoder::send_packet failed: {err}");
                    }
                }
                Ok(None) => {
                    self.eos_sent = true;
                    return None;
                }
                Err(err) => {
                    log::warn!("Demuxer::read_packet failed: {err}");
                    self.eos_sent = true;
                    return None;
                }
            }
        }
    }
}

fn clamp_u16(v: u32) -> u16 {
    u16::try_from(v).unwrap_or(u16::MAX)
}
