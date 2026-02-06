// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU filter types and conversion utilities.

#![allow(missing_docs)]

use bytemuck::{Pod, Zeroable};

const BYTES_PER_TEXEL: usize = 16;
use vello_common::color::{AlphaColor, Srgb};
use vello_common::filter::drop_shadow::DropShadow;
use vello_common::filter::flood::Flood;
use vello_common::filter::gaussian_blur::{GaussianBlur, MAX_KERNEL_SIZE};
use vello_common::filter::offset::Offset;
use vello_common::filter::InstantiatedFilter;
use vello_common::filter_effects::EdgeMode;

pub(crate) mod filter_type {
    pub const OFFSET: u32 = 0;
    pub const FLOOD: u32 = 1;
    pub const GAUSSIAN_BLUR: u32 = 2;
    pub const DROP_SHADOW: u32 = 3;
}

pub(crate) mod edge_mode {
    pub const DUPLICATE: u32 = 0;
    pub const WRAP: u32 = 1;
    pub const MIRROR: u32 = 2;
    pub const NONE: u32 = 3;
}

#[inline]
pub(crate) fn edge_mode_to_gpu(mode: EdgeMode) -> u32 {
    match mode {
        EdgeMode::Duplicate => edge_mode::DUPLICATE,
        EdgeMode::Wrap => edge_mode::WRAP,
        EdgeMode::Mirror => edge_mode::MIRROR,
        EdgeMode::None => edge_mode::NONE,
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct GpuOffset {
    pub dx: f32,
    pub dy: f32,
    pub _padding: [u32; 2],
}

impl GpuOffset {
    pub const SIZE_TEXELS: u32 = size_of::<Self>().div_ceil(BYTES_PER_TEXEL) as u32;
}

impl From<&Offset> for GpuOffset {
    fn from(offset: &Offset) -> Self {
        Self {
            dx: offset.dx,
            dy: offset.dy,
            _padding: [0; 2],
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct GpuFlood {
    pub color: u32,
    pub _padding: [u32; 3],
}

impl GpuFlood {
    pub const SIZE_TEXELS: u32 = size_of::<Self>().div_ceil(BYTES_PER_TEXEL) as u32;
}

impl From<&Flood> for GpuFlood {
    fn from(flood: &Flood) -> Self {
        Self {
            color: pack_color_srgb(&flood.color),
            _padding: [0; 3],
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct GpuGaussianBlur {
    pub std_deviation: f32,
    pub n_decimations: u32,
    pub kernel_size: u32,
    pub edge_mode: u32,
    pub kernel: [f32; MAX_KERNEL_SIZE],
    pub _padding: [u32; 3],
}

impl GpuGaussianBlur {
    pub const SIZE_TEXELS: u32 = size_of::<Self>().div_ceil(BYTES_PER_TEXEL) as u32;
}

impl From<&GaussianBlur> for GpuGaussianBlur {
    fn from(blur: &GaussianBlur) -> Self {
        Self {
            std_deviation: blur.std_deviation,
            n_decimations: blur.n_decimations as u32,
            kernel_size: blur.kernel_size as u32,
            edge_mode: edge_mode_to_gpu(blur.edge_mode),
            kernel: blur.kernel,
            _padding: [0; 3],
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct GpuDropShadow {
    pub dx: f32,
    pub dy: f32,
    pub color: u32,
    pub edge_mode: u32,
    pub std_deviation: f32,
    pub n_decimations: u32,
    pub kernel_size: u32,
    pub _padding0: u32,
    pub kernel: [f32; MAX_KERNEL_SIZE],
    pub _padding1: [u32; 3],
}

impl GpuDropShadow {
    pub const SIZE_TEXELS: u32 = size_of::<Self>().div_ceil(BYTES_PER_TEXEL) as u32;
}

impl From<&DropShadow> for GpuDropShadow {
    fn from(shadow: &DropShadow) -> Self {
        Self {
            dx: shadow.dx,
            dy: shadow.dy,
            color: pack_color_srgb(&shadow.color),
            edge_mode: edge_mode_to_gpu(shadow.edge_mode),
            std_deviation: shadow.std_deviation,
            n_decimations: shadow.n_decimations as u32,
            kernel_size: shadow.kernel_size as u32,
            _padding0: 0,
            kernel: shadow.kernel,
            _padding1: [0; 3],
        }
    }
}

#[derive(Debug)]
pub(crate) enum GpuFilter {
    Offset(GpuOffset),
    Flood(GpuFlood),
    GaussianBlur(GpuGaussianBlur),
    DropShadow(GpuDropShadow),
}

impl GpuFilter {
    #[inline]
    pub fn filter_type(&self) -> u32 {
        match self {
            Self::Offset(_) => filter_type::OFFSET,
            Self::Flood(_) => filter_type::FLOOD,
            Self::GaussianBlur(_) => filter_type::GAUSSIAN_BLUR,
            Self::DropShadow(_) => filter_type::DROP_SHADOW,
        }
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Offset(f) => bytemuck::bytes_of(f),
            Self::Flood(f) => bytemuck::bytes_of(f),
            Self::GaussianBlur(f) => bytemuck::bytes_of(f),
            Self::DropShadow(f) => bytemuck::bytes_of(f),
        }
    }

    #[inline]
    pub fn size_texels(&self) -> u32 {
        match self {
            Self::Offset(_) => GpuOffset::SIZE_TEXELS,
            Self::Flood(_) => GpuFlood::SIZE_TEXELS,
            Self::GaussianBlur(_) => GpuGaussianBlur::SIZE_TEXELS,
            Self::DropShadow(_) => GpuDropShadow::SIZE_TEXELS,
        }
    }
}

impl From<&InstantiatedFilter> for GpuFilter {
    fn from(filter: &InstantiatedFilter) -> Self {
        match filter {
            InstantiatedFilter::Offset(f) => Self::Offset(GpuOffset::from(f)),
            InstantiatedFilter::Flood(f) => Self::Flood(GpuFlood::from(f)),
            InstantiatedFilter::GaussianBlur(f) => Self::GaussianBlur(GpuGaussianBlur::from(f)),
            InstantiatedFilter::DropShadow(f) => Self::DropShadow(GpuDropShadow::from(f)),
        }
    }
}

#[inline]
fn pack_color_srgb(color: &AlphaColor<Srgb>) -> u32 {
    let premul = color.premultiply().to_rgba8();
    u32::from_le_bytes([premul.r, premul.g, premul.b, premul.a])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        assert_eq!(size_of::<GpuOffset>(), 16);
        assert_eq!(size_of::<GpuFlood>(), 16);
        assert_eq!(size_of::<GpuGaussianBlur>(), 80);
        assert_eq!(size_of::<GpuDropShadow>(), 96);
    }

    #[test]
    fn test_size_texels() {
        assert_eq!(GpuOffset::SIZE_TEXELS, 1);
        assert_eq!(GpuFlood::SIZE_TEXELS, 1);
        assert_eq!(GpuGaussianBlur::SIZE_TEXELS, 5);
        assert_eq!(GpuDropShadow::SIZE_TEXELS, 6);
    }

    #[test]
    fn test_offset_conversion() {
        let offset = Offset::new(10.5, -20.3);
        let gpu_offset = GpuOffset::from(&offset);
        assert_eq!(gpu_offset.dx, 10.5);
        assert_eq!(gpu_offset.dy, -20.3);
    }
}
