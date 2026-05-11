// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public-shape video frame and codec descriptor types.
//!
//! Mirrors `lightspeed_av/frame.rs` — single decoded NV12 frame whose Y/UV plane
//! `wgpu::Texture`s are imported lazily on first access. Both planes feed straight
//! into Vello's [`Scene::draw_texture_rects`](vello_hybrid::Scene::draw_texture_rects)
//! API with [`ExternalTextureFormat::YCbCrNv12`](vello_hybrid::ExternalTextureFormat::YCbCrNv12);
//! there is no intermediate RGBA conversion in this pipeline.

use super::metal_import::PlatformVideoFrameRetain;

/// A decoded video frame in its native NV12 form.
#[derive(Debug)]
pub(crate) struct DecodedVideoFrame {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Presentation timestamp in nanoseconds. Used by the renderer to pace
    /// playback so videos run at their native frame rate regardless of how
    /// fast the host is rendering.
    pub(crate) pts_ns: i64,
    /// Carried so the renderer can pick the right matrix / range when invoking
    /// [`Scene::draw_texture_rects`](vello_hybrid::Scene::draw_texture_rects)
    /// with [`ExternalTextureFormat::YCbCrNv12`](vello_hybrid::ExternalTextureFormat::YCbCrNv12).
    pub(crate) color_space: ColorSpace,
    /// Backend-private retain handle keeping the underlying `CVPixelBuffer` alive and
    /// providing lazy `y_plane()` / `uv_plane()` accessors.
    pub(crate) retain: PlatformVideoFrameRetain,
}

impl DecodedVideoFrame {
    /// Full-resolution Y plane wgpu texture (`R8Unorm`), backed by the frame's
    /// underlying IOSurface. Resolved lazily on first access.
    pub(crate) fn y_plane(&self) -> &wgpu::Texture {
        self.retain.y_plane()
    }

    /// Half-resolution UV plane wgpu texture (`Rg8Unorm`, interleaved Cb/Cr).
    pub(crate) fn uv_plane(&self) -> &wgpu::Texture {
        self.retain.uv_plane()
    }
}

/// Four-character codec identifier.
///
/// Modelled after the macOS `kCMVideoCodecType_*` FourCCs, packed as `u32` in
/// big-endian byte order so "avc1" / "hvc1" / "av01" survive byte-level equality.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FourCC(pub u32);

impl FourCC {
    pub(crate) const fn from_bytes(bytes: [u8; 4]) -> Self {
        Self(u32::from_be_bytes(bytes))
    }

    pub(crate) const fn as_bytes(self) -> [u8; 4] {
        self.0.to_be_bytes()
    }
}

impl core::fmt::Debug for FourCC {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let bytes = self.as_bytes();
        if bytes.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
            write!(
                f,
                "FourCC(\"{}{}{}{}\")",
                bytes[0] as char, bytes[1] as char, bytes[2] as char, bytes[3] as char,
            )
        } else {
            write!(f, "FourCC(0x{:08x})", self.0)
        }
    }
}

impl core::fmt::Display for FourCC {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let bytes = self.as_bytes();
        if bytes.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
            write!(
                f,
                "{}{}{}{}",
                bytes[0] as char, bytes[1] as char, bytes[2] as char, bytes[3] as char,
            )
        } else {
            write!(f, "0x{:08x}", self.0)
        }
    }
}

/// Common video codec FourCCs.
pub(crate) mod video_codec {
    use super::FourCC;

    pub(crate) const H264: FourCC = FourCC::from_bytes(*b"avc1");
    pub(crate) const HEVC: FourCC = FourCC::from_bytes(*b"hvc1");
    pub(crate) const HEVC_HEV1: FourCC = FourCC::from_bytes(*b"hev1");
    pub(crate) const PRORES_4444: FourCC = FourCC::from_bytes(*b"ap4h");
    pub(crate) const PRORES_422: FourCC = FourCC::from_bytes(*b"apcn");
    pub(crate) const AV1: FourCC = FourCC::from_bytes(*b"av01");
    pub(crate) const MJPEG: FourCC = FourCC::from_bytes(*b"jpeg");
}

/// Colour-space metadata that travels with a video frame.
///
/// We surface matrix coefficients, transfer function, primaries and pixel range so the
/// future Vello-side YCbCr sampler has everything it needs to apply the right conversion.
/// The current `RgbaConverter` path lets `VTPixelTransferSession` consume them on Apple's
/// side, so we only have to *carry* the metadata for now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ColorSpace {
    pub(crate) matrix: ColorMatrix,
    pub(crate) transfer: TransferFunction,
    pub(crate) primaries: ColorPrimaries,
    pub(crate) range: ColorRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColorMatrix {
    Bt601,
    Bt709,
    Bt2020Ncl,
    /// Source did not declare a matrix; treat as Bt709.
    Unspecified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransferFunction {
    /// Rec.709 / Rec.601 / sRGB-ish gamma (~2.2).
    Bt709,
    /// SMPTE ST 2084 (PQ) — HDR. Out of scope.
    SmpteSt2084,
    /// Hybrid Log-Gamma — HDR. Out of scope.
    Hlg,
    /// Source did not declare; treat as Bt709.
    Unspecified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColorPrimaries {
    Bt601_525,
    Bt601_625,
    Bt709,
    Bt2020,
    /// Source did not declare; treat as Bt709.
    Unspecified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColorRange {
    /// Y in 16..=235, CbCr in 16..=240 (8-bit).
    Limited,
    /// Full 0..=255 range ("PC" / "JPEG" range).
    Full,
}

impl Default for ColorSpace {
    fn default() -> Self {
        Self {
            matrix: ColorMatrix::Unspecified,
            transfer: TransferFunction::Unspecified,
            primaries: ColorPrimaries::Unspecified,
            range: ColorRange::Limited,
        }
    }
}
