// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU filter types and conversion utilities.

#![allow(missing_docs)]

use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};
use hashbrown::HashMap;
use vello_common::coarse::WideTilesBbox;
use vello_common::filter::InstantiatedFilter;
use vello_common::filter::drop_shadow::DropShadow;
use vello_common::filter::flood::Flood;
use vello_common::filter::gaussian_blur::{GaussianBlur, MAX_KERNEL_SIZE};
use vello_common::filter::offset::Offset;
use vello_common::filter_effects::EdgeMode;
use vello_common::paint::ImageId;
use vello_common::render_graph::LayerId;

use crate::AtlasConfig;
use crate::image_cache::ImageCache;

// Note: Keep these variables and struct layouts in sync with `filters.wgsl`!

const BYTES_PER_TEXEL: usize = 16;
const FILTER_SIZE_BYTES: usize = 48;
const FILTER_SIZE_U32: usize = FILTER_SIZE_BYTES / 4;

pub(crate) mod filter_type {
    pub(crate) const OFFSET: u32 = 0;
    pub(crate) const FLOOD: u32 = 1;
    pub(crate) const GAUSSIAN_BLUR: u32 = 2;
    pub(crate) const DROP_SHADOW: u32 = 3;
}

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

fn pack_header(filter_type: u32) -> u32 {
    debug_assert!(filter_type <= 31);
    filter_type
}

fn pack_with_gaussian_params(
    filter_type: u32,
    edge_mode: u32,
    n_decimations: u32,
    n_linear_taps: u32,
) -> u32 {
    debug_assert!(filter_type <= 31);
    debug_assert!(edge_mode <= 3);
    debug_assert!(n_decimations <= 15);
    debug_assert!(n_linear_taps <= 3);
    filter_type | (edge_mode << 5) | (n_decimations << 7) | (n_linear_taps << 11)
}

// To a large degree, the vello_hybrid implementation of gaussian blur follows the one in vello_cpu.
// However, we apply a specific optimization, where instead of averaging and weighting each sample
// one after the other, we use linear sampling to sample two pixels at once, and adjust the
// weights accordingly so the gaussian blur filter is still valid. See
// https://www.rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/
// for more information.

/// Maximum number of linear-sampling tap pairs per side.
const MAX_TAPS_PER_SIDE: usize = (MAX_KERNEL_SIZE / 2 + 1) / 2;

/// A linear-sampling kernel derived from a discrete Gaussian kernel.
struct LinearKernel {
    /// Weight of the center tap.
    center_weight: f32,
    // Note that we only need to store one side since they are symmetrical.
    /// Merged weights for each tap pair. Only the first `n_taps` entries are valid.
    weights: [f32; MAX_TAPS_PER_SIDE],
    /// The fractional offsets for each tap pair for linear sampling.
    offsets: [f32; MAX_TAPS_PER_SIDE],
    /// The actual number of taps per side.
    n_taps: u8,
}

impl LinearKernel {
    fn new(kernel: &[f32; MAX_KERNEL_SIZE], kernel_size: u8) -> Self {
        let kernel_size = kernel_size as usize;
        let radius = kernel_size / 2;
        let center_weight = kernel[radius];

        let mut weights = [0.0f32; MAX_TAPS_PER_SIDE];
        let mut offsets = [0.0f32; MAX_TAPS_PER_SIDE];
        let mut n_taps = 0u8;

        // The kernel is symmetric, so we can only process the positive side.
        let positive_side = &kernel[radius + 1..kernel_size];
        let (pairs, remainder) = positive_side.as_chunks::<2>();

        // Merge each consecutive pair into a single bilinear tap. See the
        // formulas on the website.
        for (k, &[w1, w2]) in pairs.iter().enumerate() {
            let merged_weight = w1 + w2;
            let offset1 = (2 * k + 1) as f32;
            let merged_offset = if merged_weight > 0.0 {
                (w1 * offset1 + w2 * (offset1 + 1.0)) / merged_weight
            } else {
                offset1
            };
            weights[n_taps as usize] = merged_weight;
            offsets[n_taps as usize] = merged_offset;
            n_taps += 1;
        }

        // If there is a leftover tap, we sample with no fractional offset so that just
        // that single pixel is sampled.
        if let [leftover] = remainder {
            weights[n_taps as usize] = *leftover;
            offsets[n_taps as usize] = radius as f32;
            n_taps += 1;
        }

        Self {
            center_weight,
            weights,
            offsets,
            n_taps,
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct GpuOffset {
    pub header: u32,
    pub dx: f32,
    pub dy: f32,
    pub _padding: [u32; 9],
}

impl From<&Offset> for GpuOffset {
    fn from(offset: &Offset) -> Self {
        Self {
            header: pack_header(filter_type::OFFSET),
            dx: offset.dx,
            dy: offset.dy,
            _padding: [0; 9],
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct GpuFlood {
    pub header: u32,
    pub color: u32,
    pub _padding: [u32; 10],
}

impl From<&Flood> for GpuFlood {
    fn from(flood: &Flood) -> Self {
        Self {
            header: pack_header(filter_type::FLOOD),
            color: flood.color.premultiply().to_rgba8().to_u32(),
            _padding: [0; 10],
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct GpuGaussianBlur {
    pub header: u32,
    pub std_deviation: f32,
    pub center_weight: f32,
    pub linear_weights: [f32; MAX_TAPS_PER_SIDE],
    pub linear_offsets: [f32; MAX_TAPS_PER_SIDE],
    pub _padding: [u32; 3],
}

impl From<&GaussianBlur> for GpuGaussianBlur {
    fn from(blur: &GaussianBlur) -> Self {
        let lk = LinearKernel::new(&blur.kernel, blur.kernel_size);

        Self {
            header: pack_with_gaussian_params(
                filter_type::GAUSSIAN_BLUR,
                edge_mode_to_gpu(blur.edge_mode),
                blur.n_decimations as u32,
                lk.n_taps as u32,
            ),
            std_deviation: blur.std_deviation,
            center_weight: lk.center_weight,
            linear_weights: lk.weights,
            linear_offsets: lk.offsets,
            _padding: [0; 3],
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct GpuDropShadow {
    pub header: u32,
    pub dx: f32,
    pub dy: f32,
    pub color: u32,
    pub std_deviation: f32,
    pub center_weight: f32,
    pub linear_weights: [f32; MAX_TAPS_PER_SIDE],
    pub linear_offsets: [f32; MAX_TAPS_PER_SIDE],
}

impl From<&DropShadow> for GpuDropShadow {
    fn from(shadow: &DropShadow) -> Self {
        let lk = LinearKernel::new(&shadow.kernel, shadow.kernel_size);
        Self {
            header: pack_with_gaussian_params(
                filter_type::DROP_SHADOW,
                edge_mode_to_gpu(shadow.edge_mode),
                shadow.n_decimations as u32,
                lk.n_taps as u32,
            ),
            dx: shadow.dx,
            dy: shadow.dy,
            color: shadow.color.premultiply().to_rgba8().to_u32(),
            std_deviation: shadow.std_deviation,
            center_weight: lk.center_weight,
            linear_weights: lk.weights,
            linear_offsets: lk.offsets,
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct GpuFilterData {
    data: [u32; FILTER_SIZE_U32],
}

impl GpuFilterData {
    pub(crate) const SIZE_TEXELS: u32 = size_of::<Self>().div_ceil(BYTES_PER_TEXEL) as u32;

    fn filter_type(&self) -> u32 {
        self.data[0] & 0x1F
    }

    /// Returns whether this filter requires a scratch buffer for multi-pass rendering.
    pub(crate) fn needs_scratch_buffer(&self) -> bool {
        matches!(
            self.filter_type(),
            filter_type::GAUSSIAN_BLUR | filter_type::DROP_SHADOW
        )
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

/// Context used for keeping track of state necessary for filter rendering.
#[derive(Debug)]
pub(crate) struct FilterContext {
    /// The encoded data for each filter used in the current scene that will be uploaded to the
    /// filter texture.
    pub(crate) filters: Vec<GpuFilterData>,
    /// At what texel offset the filter data for the given layer ID is stored in the texture.
    pub(crate) offsets: HashMap<LayerId, u32>,
    /// Allocated filter textures (as ImageIds in the atlas) for each layer.
    pub(crate) filter_textures: HashMap<LayerId, FilterLayerData>,
    /// Image cache for storing filter intermediate textures.
    pub(crate) filter_texture_cache: ImageCache,
}

impl FilterContext {
    pub(crate) fn new(atlas_config: AtlasConfig) -> Self {
        Self {
            filters: Vec::new(),
            offsets: HashMap::new(),
            filter_textures: HashMap::new(),
            filter_texture_cache: ImageCache::new_with_config(atlas_config),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.filters.clear();
        self.offsets.clear();
        self.filter_textures.clear();
        // TODO: Clear cache?
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    pub(crate) fn offsets(&self) -> &HashMap<LayerId, u32> {
        &self.offsets
    }

    pub(crate) fn filter_textures(&self) -> &HashMap<LayerId, FilterLayerData> {
        &self.filter_textures
    }

    pub(crate) fn total_texels(&self) -> u32 {
        self.filters.len() as u32 * GpuFilterData::SIZE_TEXELS
    }

    /// Returns the filter data for the given layer ID.
    pub(crate) fn get_filter_data(&self, layer_id: &LayerId) -> Option<&GpuFilterData> {
        let offset = self.offsets.get(layer_id)?;
        let index = (*offset / GpuFilterData::SIZE_TEXELS) as usize;
        self.filters.get(index)
    }

    pub(crate) fn serialize_to_buffer(&self, buffer: &mut [u8]) {
        let src = bytemuck::cast_slice::<GpuFilterData, u8>(&self.filters);
        buffer[..src.len()].copy_from_slice(src);
    }
}

/// Filter texture allocation for a single layer.
#[derive(Debug)]
pub(crate) struct FilterLayerData {
    /// Image ID for the main texture holding the raw painted version of the layer.
    pub main_image_id: ImageId,
    /// Image ID for the destination texture holding the final filtered version.
    pub dest_image_id: ImageId,
    /// Optional image ID for scratch texture used in multi-pass filter operations.
    pub scratch_image_id: Option<ImageId>,
    /// The paint index that points to the location in `encoded_paints` where
    /// the final filtered version of the image will be stored.
    pub paint_idx: u32,
    /// The bounding box of the filter layer.
    pub bbox: WideTilesBbox,
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::filter::gaussian_blur::{compute_gaussian_kernel, plan_decimated_blur};

    #[test]
    fn test_all_filters_same_size() {
        assert_eq!(size_of::<GpuOffset>(), FILTER_SIZE_BYTES);
        assert_eq!(size_of::<GpuFlood>(), FILTER_SIZE_BYTES);
        assert_eq!(size_of::<GpuGaussianBlur>(), FILTER_SIZE_BYTES);
        assert_eq!(size_of::<GpuDropShadow>(), FILTER_SIZE_BYTES);
        assert_eq!(size_of::<GpuFilterData>(), FILTER_SIZE_BYTES);
    }

    #[test]
    fn test_offset_conversion() {
        let offset = Offset::new(10.5, -20.3);
        let gpu_offset = GpuOffset::from(&offset);
        assert_eq!(gpu_offset.header & 0x1F, filter_type::OFFSET);
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

    fn check_linear_kernel(kernel: &[f32; MAX_KERNEL_SIZE], size: u8, expected_taps: u8) {
        let lk = LinearKernel::new(kernel, size);
        assert_eq!(lk.n_taps, expected_taps);

        let sum = lk.center_weight + 2.0 * lk.weights.iter().take(lk.n_taps as usize).sum::<f32>();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn linear_kernel_size_1() {
        let (_n_dec, kernel, size) = plan_decimated_blur(0.0);
        assert_eq!(size, 1);
        check_linear_kernel(&kernel, size, 0);
    }

    #[test]
    fn linear_kernel_size_3() {
        let (kernel, size) = compute_gaussian_kernel(0.1);
        assert_eq!(size, 3);
        check_linear_kernel(&kernel, size, 1);
    }

    #[test]
    fn linear_kernel_size_7() {
        let (kernel, size) = compute_gaussian_kernel(1.0);
        assert_eq!(size, 7);
        check_linear_kernel(&kernel, size, 2);
    }

    #[test]
    fn linear_kernel_size_13() {
        let (kernel, size) = compute_gaussian_kernel(2.0);
        assert_eq!(size, 13);
        check_linear_kernel(&kernel, size, 3);
    }
}
