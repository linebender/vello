// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! macOS video pipeline: AVAsset demuxer, then VideoToolbox decoder, then NV12 plane import.
//!
//! Frames stay in their native two-plane NV12 layout (`R8` Y plane, `Rg8` interleaved
//! Cb/Cr plane) in the IOSurface VideoToolbox decoded into. Vello samples the planes
//! directly through
//! [`TextureBindings::insert_ycbcr_nv12`](vello_gpu::TextureBindings::insert_ycbcr_nv12),
//! so there is no separate conversion pass.

#![allow(
    clippy::doc_markdown,
    reason = "Apple type names appear in nearly every doc line of this module"
)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "plane dimensions fit in u32 by construction"
)]

mod decoder;
mod demuxer;
mod error;
mod frame;
mod metal_import;

use std::path::{Path, PathBuf};

pub(crate) use error::AvError;
pub(crate) use frame::VideoFrame;

use decoder::VideoDecoder;
use demuxer::Demuxer;
use metal_import::MetalImporter;

/// Demuxer, decoder and plane importer for one video file.
pub(crate) struct VideoPlayer {
    path: PathBuf,
    importer: MetalImporter,
    demuxer: Demuxer,
    decoder: VideoDecoder,
    /// The demuxer has run out of packets and the decoder has been flushed.
    flushed: bool,
}

impl VideoPlayer {
    /// Opens the first video track of `path`. Fails if the file, codec or wgpu backend
    /// is unsupported.
    pub(crate) fn open(
        path: &Path,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
    ) -> Result<Self, AvError> {
        let importer = MetalImporter::new(adapter, device)?;
        let (demuxer, track) = Demuxer::open(path)?;
        let decoder = VideoDecoder::new(&track)?;
        Ok(Self {
            path: path.to_path_buf(),
            importer,
            demuxer,
            decoder,
            flushed: false,
        })
    }

    /// Restarts from the first frame. AVAssetReader can't seek, so the file is
    /// reopened; on failure the player is left as it was.
    pub(crate) fn rewind(&mut self) -> Result<(), AvError> {
        let (demuxer, track) = Demuxer::open(&self.path)?;
        let decoder = VideoDecoder::new(&track)?;
        self.demuxer = demuxer;
        self.decoder = decoder;
        self.flushed = false;
        Ok(())
    }

    /// Feeds packets to the decoder until a frame comes out, in decode order. Returns
    /// `Ok(None)` once every frame has been returned.
    pub(crate) fn next_frame(&mut self) -> Result<Option<VideoFrame>, AvError> {
        loop {
            if let Some(decoded) = self.decoder.recv_frame() {
                let planes = self.importer.import_nv12(decoded.pixel_buffer)?;
                return Ok(Some(VideoFrame::new(
                    planes,
                    decoded.pts_ns,
                    decoded.color_space,
                )));
            }
            if self.flushed {
                return Ok(None);
            }
            match self.demuxer.read_packet() {
                // A bad packet only costs frames until the next keyframe, so keep going.
                Some(packet) => {
                    if let Err(err) = self.decoder.send_packet(&packet) {
                        log::warn!("{}: {err}", self.path.display());
                    }
                }
                None => {
                    self.decoder.flush();
                    self.flushed = true;
                }
            }
        }
    }
}
