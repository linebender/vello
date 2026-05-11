// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Compressed packet wrapper handed from the demuxer to the decoder.
//!
//! Trimmed from `lightspeed_av/packet.rs` — only the fields the VT decoder actually
//! reads. The originating `CMSampleBuffer` is retained so the decoder can feed it
//! straight into `VTDecompressionSessionDecodeFrame` (no re-wrapping).

use objc2_core_foundation::CFRetained;
use objc2_core_media::CMSampleBuffer;

use super::frame::FourCC;

/// A compressed packet pulled from the demuxer.
#[derive(Debug)]
pub(crate) struct Packet {
    /// Track this packet belongs to. Only one video track is ever read in the
    /// example, but the field is kept for parity with the source pipeline and so
    /// that a future multi-track expansion has somewhere to record routing.
    #[expect(dead_code, reason = "carried for future multi-track use")]
    pub(crate) track_id: u32,
    /// Presentation timestamp in nanoseconds.
    #[expect(dead_code, reason = "kept for future PTS-paced playback / debugging")]
    pub(crate) pts_ns: i64,
    pub(crate) sample_buffer: CFRetained<CMSampleBuffer>,
}

// SAFETY: `CMSampleBuffer` is a CFType (atomic retain/release); the buffer's bytes plus
// format description are read-only after the demuxer hands the packet out. Transferring
// the packet across threads is just an atomic refcount bump.
unsafe impl Send for Packet {}

/// Per-track metadata returned by the demuxer.
#[derive(Debug, Clone)]
pub(crate) struct TrackInfo {
    pub(crate) track_id: u32,
    pub(crate) codec: FourCC,
}
