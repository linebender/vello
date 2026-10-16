// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Decoded-frame, codec and color-space types for the video pipeline.

use super::metal_import::Nv12Planes;

/// A decoded NV12 frame, wrapped in a [`wgpu::ExternalTexture`] that converts it to RGB
/// when sampled.
///
/// Both planes live in the IOSurface VideoToolbox decoded into. Dropping the frame
/// returns that IOSurface to the decoder's pool, where the next decode may overwrite
/// it, so a frame must outlive every GPU submission that samples it.
pub(crate) struct VideoFrame {
    pub(crate) external_texture: wgpu::ExternalTexture,
    pub(crate) width: u16,
    pub(crate) height: u16,
    /// Presentation timestamp in nanoseconds, used to pace playback.
    pub(crate) pts_ns: i64,
    pub(crate) color_space: ColorSpace,
    _planes: Nv12Planes,
}

impl VideoFrame {
    pub(super) fn new(
        device: &wgpu::Device,
        planes: Nv12Planes,
        pts_ns: i64,
        color_space: ColorSpace,
    ) -> Self {
        let y_view = planes
            .y
            .create_view(&wgpu::TextureViewDescriptor::default());
        let uv_view = planes
            .uv
            .create_view(&wgpu::TextureViewDescriptor::default());
        #[rustfmt::skip]
        let external_texture = device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: Some("NV12 video frame"),
                width: planes.y.width(),
                height: planes.y.height(),
                format: wgpu::ExternalTextureFormat::Nv12,
                yuv_conversion_matrix: color_space.yuv_conversion_matrix(),
                // Identity gamut and transfer functions: the source is assumed to already be in
                // the surface's color space.
                gamut_conversion_matrix: [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
                src_transfer_function: wgpu::ExternalTextureTransferFunction::default(),
                dst_transfer_function: wgpu::ExternalTextureTransferFunction::default(),
                sample_transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                load_transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            },
            &[&y_view, &uv_view],
        );
        let clamp = |v: u32| u16::try_from(v).unwrap_or(u16::MAX);
        Self {
            external_texture,
            width: clamp(planes.y.width()),
            height: clamp(planes.y.height()),
            pts_ns,
            color_space,
            _planes: planes,
        }
    }
}

/// Color metadata of a frame. The YCbCr-to-RGB conversion uses the matrix and range;
/// the transfer function and primaries are only used to warn about content that will
/// display with wrong colors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ColorSpace {
    pub(crate) matrix: ColorMatrix,
    pub(crate) transfer: TransferFunction,
    pub(crate) primaries: ColorPrimaries,
    pub(crate) range: ColorRange,
}

impl ColorSpace {
    /// Column-major matrix taking normalized `[Y, Cb, Cr, 1]` samples to non-linear RGB, as
    /// expected by [`wgpu::ExternalTextureDescriptor::yuv_conversion_matrix`]. An unspecified
    /// matrix falls back to BT.709, the usual choice for HD video.
    pub(crate) fn yuv_conversion_matrix(self) -> [f32; 16] {
        // Luma weights Kr, Kb defined by ITU-R BT.601, BT.709 and BT.2020 (collected in
        // ITU-T H.273).
        let (kr, kb): (f32, f32) = match self.matrix {
            ColorMatrix::Bt601 => (0.299, 0.114),
            ColorMatrix::Bt709 | ColorMatrix::Unspecified => (0.2126, 0.0722),
            ColorMatrix::Bt2020Ncl => (0.2627, 0.0593),
        };
        let kg = 1.0 - kr - kb;
        // Limited range stores 8-bit Y in 16..=235 (219 steps) and Cb/Cr in 16..=240
        // (224 steps); stretch both back to the full range.
        let (y_scale, y_offset, c_scale) = match self.range {
            ColorRange::Limited => (255.0 / 219.0, 16.0 / 255.0, 255.0 / 224.0),
            ColorRange::Full => (1.0, 0.0, 1.0),
        };
        let c_offset = 128.0 / 255.0;

        // The standards give the forward transform; its inverse is
        //   R = Y + 2(1 - Kr) Cr
        //   G = Y - 2Kb(1 - Kb)/Kg Cb - 2Kr(1 - Kr)/Kg Cr
        //   B = Y + 2(1 - Kb) Cb
        let cr_r = c_scale * 2.0 * (1.0 - kr);
        let cb_g = -c_scale * 2.0 * kb * (1.0 - kb) / kg;
        let cr_g = -c_scale * 2.0 * kr * (1.0 - kr) / kg;
        let cb_b = c_scale * 2.0 * (1.0 - kb);
        let y_bias = -y_scale * y_offset;
        #[rustfmt::skip]
        let matrix = [
            // Y column.
            y_scale, y_scale, y_scale, 0.0,
            // Cb column.
            0.0, cb_g, cb_b, 0.0,
            // Cr column.
            cr_r, cr_g, 0.0, 0.0,
            // Constant column, removing the Y and Cb/Cr offsets.
            y_bias - cr_r * c_offset,
            y_bias - (cb_g + cr_g) * c_offset,
            y_bias - cb_b * c_offset,
            1.0,
        ];
        matrix
    }

    /// HDR or wide-gamut content, which shows washed-out or desaturated on an SDR
    /// surface because only the YCbCr matrix is applied. The external texture's gamut
    /// matrix and transfer functions could convert BT.2020 primaries, which this example
    /// doesn't set up, but its gamma-style transfer functions can't express PQ or HLG.
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
