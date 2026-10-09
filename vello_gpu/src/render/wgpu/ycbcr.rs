// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Color-space metadata for YCbCr external textures.

/// Color-space metadata describing how to interpret a YCbCr external texture.
///
/// Transfer function and color primaries are not included: the destination color space is
/// assumed to match the source, the same simplification the RGBA external-texture path makes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct YCbCrInfo {
    /// Matrix coefficients for the YCbCr → RGB conversion.
    pub matrix: YCbCrMatrix,
    /// Numeric range of the source samples.
    pub range: YCbCrRange,
}

impl YCbCrInfo {
    /// ITU-R BT.709 with limited (TV) range, the typical output of `H.264` / `HEVC` HD video
    /// decoders.
    pub const BT_709_LIMITED: Self = Self {
        matrix: YCbCrMatrix::Bt709,
        range: YCbCrRange::Limited,
    };
}

/// YCbCr conversion-matrix coefficients.
// Discriminants are packed into encoded paints and must match `helpers/ycbcr.wesl`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum YCbCrMatrix {
    /// ITU-R BT.601 (SD video).
    Bt601 = 0,
    /// ITU-R BT.709 (HD video). The most common matrix for AVC/HEVC content.
    Bt709 = 1,
    /// ITU-R BT.2020 non-constant luminance (UHD/HDR content).
    Bt2020Ncl = 2,
}

/// Numeric range of the YCbCr samples in the underlying texture.
// Discriminants are packed into encoded paints and must match `helpers/ycbcr.wesl`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum YCbCrRange {
    /// "TV" / "video" range: Y in `[16, 235]`, Cb/Cr in `[16, 240]` (8-bit).
    Limited = 0,
    /// "PC" / "JPEG" range: full `[0, 255]` for all channels.
    Full = 1,
}
