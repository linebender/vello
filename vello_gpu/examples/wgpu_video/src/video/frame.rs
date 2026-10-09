// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Decoded-frame, codec and color-space types for the video pipeline.

use vello_gpu::{YCbCrInfo, YCbCrMatrix, YCbCrRange};

use super::metal_import::Nv12Planes;

/// A decoded NV12 frame, with a texture view for each plane.
///
/// Both views point at the IOSurface VideoToolbox decoded into. Dropping the frame
/// returns that IOSurface to the decoder's pool, where the next decode may overwrite
/// it, so a frame must outlive every GPU submission that samples its views.
pub(crate) struct VideoFrame {
    /// Full-resolution Y plane (`R8Unorm`).
    pub(crate) y_view: wgpu::TextureView,
    /// Half-resolution interleaved Cb/Cr plane (`Rg8Unorm`).
    pub(crate) uv_view: wgpu::TextureView,
    pub(crate) width: u16,
    pub(crate) height: u16,
    /// Presentation timestamp in nanoseconds, used to pace playback.
    pub(crate) pts_ns: i64,
    pub(crate) color_space: ColorSpace,
    _planes: Nv12Planes,
}

impl VideoFrame {
    pub(super) fn new(planes: Nv12Planes, pts_ns: i64, color_space: ColorSpace) -> Self {
        let clamp = |v: u32| u16::try_from(v).unwrap_or(u16::MAX);
        Self {
            y_view: planes
                .y
                .create_view(&wgpu::TextureViewDescriptor::default()),
            uv_view: planes
                .uv
                .create_view(&wgpu::TextureViewDescriptor::default()),
            width: clamp(planes.y.width()),
            height: clamp(planes.y.height()),
            pts_ns,
            color_space,
            _planes: planes,
        }
    }
}

/// Color metadata of a frame. Vello's YCbCr-to-RGB conversion uses the matrix and
/// range; the transfer function and primaries are only used to warn about content
/// that will display with wrong colors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ColorSpace {
    pub(crate) matrix: ColorMatrix,
    pub(crate) transfer: TransferFunction,
    pub(crate) primaries: ColorPrimaries,
    pub(crate) range: ColorRange,
}

impl ColorSpace {
    /// The conversion parameters for Vello's shader. An unspecified matrix falls back
    /// to BT.709, the usual choice for HD video.
    pub(crate) fn ycbcr_info(self) -> YCbCrInfo {
        let matrix = match self.matrix {
            ColorMatrix::Bt601 => YCbCrMatrix::Bt601,
            ColorMatrix::Bt709 | ColorMatrix::Unspecified => YCbCrMatrix::Bt709,
            ColorMatrix::Bt2020Ncl => YCbCrMatrix::Bt2020Ncl,
        };
        let range = match self.range {
            ColorRange::Limited => YCbCrRange::Limited,
            ColorRange::Full => YCbCrRange::Full,
        };
        YCbCrInfo { matrix, range }
    }

    /// HDR or wide-gamut content, which shows washed-out or desaturated on an SDR
    /// surface because only the YCbCr matrix is applied.
    pub(crate) fn needs_color_management(self) -> bool {
        matches!(
            self.transfer,
            TransferFunction::SmpteSt2084 | TransferFunction::Hlg
        ) || self.primaries == ColorPrimaries::Bt2020
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColorMatrix {
    Bt601,
    Bt709,
    Bt2020Ncl,
    /// Not declared by the source; treated as BT.709.
    Unspecified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransferFunction {
    /// SDR gamma shared by Rec. 709 and Rec. 601.
    Bt709,
    /// HDR perceptual quantizer (PQ); not supported.
    SmpteSt2084,
    /// HDR hybrid log-gamma; not supported.
    Hlg,
    /// Not declared by the source; treated as BT.709.
    Unspecified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColorPrimaries {
    Bt601_525,
    Bt601_625,
    Bt709,
    Bt2020,
    /// Not declared by the source; treated as BT.709.
    Unspecified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColorRange {
    /// Y in 16..=235 and Cb/Cr in 16..=240 (8-bit).
    Limited,
    /// The full 0..=255 range, also called PC or JPEG range.
    Full,
}

/// Four-character codec code such as `avc1`, as used by Core Media. Stored big-endian,
/// so the bytes read in the same order as the characters.
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

/// Codecs the decoder accepts.
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
