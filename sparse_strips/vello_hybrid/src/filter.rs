// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU filter types and conversion utilities.

#![allow(missing_docs)]

use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};
use hashbrown::HashMap;
use vello_common::filter::drop_shadow::DropShadow;
use vello_common::filter::flood::Flood;
use vello_common::filter::gaussian_blur::{GaussianBlur, MAX_KERNEL_SIZE};
use vello_common::filter::offset::Offset;
use vello_common::filter::InstantiatedFilter;
use vello_common::filter_effects::EdgeMode;
use vello_common::render_graph::{LayerId, RenderGraph, RenderNodeKind};

const BYTES_PER_TEXEL: usize = 16;
const FILTER_SIZE_BYTES: usize = 96;
// Keep in sync with FILTER_SIZE_U32 in filters.wgsl
const FILTER_SIZE_U32: usize = FILTER_SIZE_BYTES / 4;

// Keep in sync with FILTER_TYPE_* in filters.wgsl
pub(crate) mod filter_type {
    pub(crate) const OFFSET: u32 = 0;
    pub(crate) const FLOOD: u32 = 1;
    pub(crate) const GAUSSIAN_BLUR: u32 = 2;
    pub(crate) const DROP_SHADOW: u32 = 3;
}

// Keep in sync with EdgeMode in vello_common/src/filter_effects.rs
// and EDGE_MODE_* in filters.wgsl
pub(crate) mod edge_mode {
    pub(crate) const DUPLICATE: u32 = 0;
    pub(crate) const WRAP: u32 = 1;
    pub(crate) const MIRROR: u32 = 2;
    pub(crate) const NONE: u32 = 3;
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

// Keep struct layout in sync with unpack_offset_filter in filters.wgsl
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct GpuOffset {
    pub filter_type: u32,
    pub dx: f32,
    pub dy: f32,
    pub _padding: [u32; 21],
}

impl GpuOffset {
    const SIZE_TEXELS: u32 = size_of::<Self>().div_ceil(BYTES_PER_TEXEL) as u32;
}

impl From<&Offset> for GpuOffset {
    fn from(offset: &Offset) -> Self {
        Self {
            filter_type: filter_type::OFFSET,
            dx: offset.dx,
            dy: offset.dy,
            _padding: [0; 21],
        }
    }
}

// Keep struct layout in sync with unpack_flood_filter in filters.wgsl
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct GpuFlood {
    pub filter_type: u32,
    pub color: u32,
    pub _padding: [u32; 22],
}

impl GpuFlood {
    const SIZE_TEXELS: u32 = size_of::<Self>().div_ceil(BYTES_PER_TEXEL) as u32;
}

impl From<&Flood> for GpuFlood {
    fn from(flood: &Flood) -> Self {
        Self {
            filter_type: filter_type::FLOOD,
            color: flood.color.premultiply().to_rgba8().to_u32(),
            _padding: [0; 22],
        }
    }
}

// Keep struct layout in sync with unpack_gaussian_blur_filter in filters.wgsl
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct GpuGaussianBlur {
    pub filter_type: u32,
    pub std_deviation: f32,
    pub n_decimations: u32,
    pub kernel_size: u32,
    pub edge_mode: u32,
    pub kernel: [f32; MAX_KERNEL_SIZE],
    pub _padding: [u32; 6],
}

impl GpuGaussianBlur {
    const SIZE_TEXELS: u32 = size_of::<Self>().div_ceil(BYTES_PER_TEXEL) as u32;
}

impl From<&GaussianBlur> for GpuGaussianBlur {
    fn from(blur: &GaussianBlur) -> Self {
        Self {
            filter_type: filter_type::GAUSSIAN_BLUR,
            std_deviation: blur.std_deviation,
            n_decimations: blur.n_decimations as u32,
            kernel_size: blur.kernel_size as u32,
            edge_mode: edge_mode_to_gpu(blur.edge_mode),
            kernel: blur.kernel,
            _padding: [0; 6],
        }
    }
}

// Keep struct layout in sync with unpack_drop_shadow_filter in filters.wgsl
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct GpuDropShadow {
    pub filter_type: u32,
    pub dx: f32,
    pub dy: f32,
    pub color: u32,
    pub edge_mode: u32,
    pub std_deviation: f32,
    pub n_decimations: u32,
    pub kernel_size: u32,
    pub kernel: [f32; MAX_KERNEL_SIZE],
    pub _padding: [u32; 3],
}

impl GpuDropShadow {
    const SIZE_TEXELS: u32 = size_of::<Self>().div_ceil(BYTES_PER_TEXEL) as u32;
}

impl From<&DropShadow> for GpuDropShadow {
    fn from(shadow: &DropShadow) -> Self {
        Self {
            filter_type: filter_type::DROP_SHADOW,
            dx: shadow.dx,
            dy: shadow.dy,
            color: shadow.color.premultiply().to_rgba8().to_u32(),
            edge_mode: edge_mode_to_gpu(shadow.edge_mode),
            std_deviation: shadow.std_deviation,
            n_decimations: shadow.n_decimations as u32,
            kernel_size: shadow.kernel_size as u32,
            kernel: shadow.kernel,
            _padding: [0; 3],
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct GpuFilterData {
    data: [u32; FILTER_SIZE_U32],
}

impl GpuFilterData {
    const SIZE_TEXELS: u32 = size_of::<Self>().div_ceil(BYTES_PER_TEXEL) as u32;

    fn filter_type(&self) -> u32 {
        self.data[0]
    }
}

impl From<GpuOffset> for GpuFilterData {
    fn from(filter: GpuOffset) -> Self {
        bytemuck::cast(filter)
    }
}

impl From<GpuFlood> for GpuFilterData {
    fn from(filter: GpuFlood) -> Self {
        bytemuck::cast(filter)
    }
}

impl From<GpuGaussianBlur> for GpuFilterData {
    fn from(filter: GpuGaussianBlur) -> Self {
        bytemuck::cast(filter)
    }
}

impl From<GpuDropShadow> for GpuFilterData {
    fn from(filter: GpuDropShadow) -> Self {
        bytemuck::cast(filter)
    }
}

impl From<&InstantiatedFilter> for GpuFilterData {
    fn from(filter: &InstantiatedFilter) -> Self {
        match filter {
            InstantiatedFilter::Offset(f) => GpuOffset::from(f).into(),
            InstantiatedFilter::Flood(f) => GpuFlood::from(f).into(),
            InstantiatedFilter::GaussianBlur(f) => GpuGaussianBlur::from(f).into(),
            InstantiatedFilter::DropShadow(f) => GpuDropShadow::from(f).into(),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct FilterData {
    filters: Vec<GpuFilterData>,
    offsets: HashMap<LayerId, u32>,
    buffer: Vec<u8>,
}

impl FilterData {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn clear(&mut self) {
        self.filters.clear();
        self.offsets.clear();
        self.buffer.clear();
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    pub(crate) fn filters(&self) -> &[GpuFilterData] {
        &self.filters
    }

    pub(crate) fn offsets(&self) -> &HashMap<LayerId, u32> {
        &self.offsets
    }

    pub(crate) fn total_texels(&self) -> u32 {
        self.filters.len() as u32 * GpuFilterData::SIZE_TEXELS
    }

    pub(crate) fn prepare(&mut self, render_graph: &RenderGraph) {
        self.clear();

        if !render_graph.has_filters() {
            return;
        }

        let mut current_offset = 0u32;
        for node in &render_graph.nodes {
            if let RenderNodeKind::FilterLayer {
                layer_id,
                filter,
                transform,
                ..
            } = &node.kind
            {
                let instantiated = InstantiatedFilter::new(filter, transform);
                let gpu_filter = GpuFilterData::from(&instantiated);
                self.filters.push(gpu_filter);
                self.offsets.insert(*layer_id, current_offset);
                current_offset += GpuFilterData::SIZE_TEXELS;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_filters_same_size() {
        assert_eq!(size_of::<GpuOffset>(), FILTER_SIZE_BYTES);
        assert_eq!(size_of::<GpuFlood>(), FILTER_SIZE_BYTES);
        assert_eq!(size_of::<GpuGaussianBlur>(), FILTER_SIZE_BYTES);
        assert_eq!(size_of::<GpuDropShadow>(), FILTER_SIZE_BYTES);
        assert_eq!(size_of::<GpuFilterData>(), FILTER_SIZE_BYTES);
    }

    #[test]
    fn test_size_texels() {
        const EXPECTED_TEXELS: u32 = (FILTER_SIZE_BYTES / BYTES_PER_TEXEL) as u32;
        assert_eq!(GpuOffset::SIZE_TEXELS, EXPECTED_TEXELS);
        assert_eq!(GpuFlood::SIZE_TEXELS, EXPECTED_TEXELS);
        assert_eq!(GpuGaussianBlur::SIZE_TEXELS, EXPECTED_TEXELS);
        assert_eq!(GpuDropShadow::SIZE_TEXELS, EXPECTED_TEXELS);
        assert_eq!(GpuFilterData::SIZE_TEXELS, EXPECTED_TEXELS);
    }

    #[test]
    fn test_offset_conversion() {
        let offset = Offset::new(10.5, -20.3);
        let gpu_offset = GpuOffset::from(&offset);
        assert_eq!(gpu_offset.filter_type, filter_type::OFFSET);
        assert_eq!(gpu_offset.dx, 10.5);
        assert_eq!(gpu_offset.dy, -20.3);
    }

    #[test]
    fn test_type_erased_cast() {
        let offset = Offset::new(1.0, 2.0);
        let gpu_offset = GpuOffset::from(&offset);
        let erased: GpuFilterData = gpu_offset.into();
        assert_eq!(erased.filter_type(), filter_type::OFFSET);

        let back: GpuOffset = bytemuck::cast(erased);
        assert_eq!(back.dx, 1.0);
        assert_eq!(back.dy, 2.0);
    }
}
