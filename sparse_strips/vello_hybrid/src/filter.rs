// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU filter types and conversion utilities.

use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};
use hashbrown::HashMap;
use vello_common::coarse::{WideTile, WideTilesBbox};
use vello_common::encode::{EncodedImage, EncodedPaint};
use vello_common::filter::InstantiatedFilter;
use vello_common::filter::drop_shadow::DropShadow;
use vello_common::filter::flood::Flood;
use vello_common::filter::gaussian_blur::{GaussianBlur, MAX_KERNEL_SIZE};
use vello_common::filter::offset::Offset;
use vello_common::filter_effects::EdgeMode;
use vello_common::kurbo::{Affine, Vec2};
use vello_common::paint::{ImageId, ImageSource};
use vello_common::peniko::{ImageQuality, ImageSampler};
use vello_common::render_graph::{LayerId, RenderGraph, RenderNodeKind};
use vello_common::tile::Tile;

use crate::AtlasConfig;
use crate::image_cache::ImageCache;
use crate::multi_atlas::{AtlasError, AtlasId};

// Note: Keep these variables and struct layouts in sync with `filters.wgsl`!

// Since we store in RGBA32 texture.
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

pub(crate) fn edge_mode_to_gpu(mode: EdgeMode) -> u32 {
    match mode {
        EdgeMode::Duplicate => edge_mode::DUPLICATE,
        EdgeMode::Wrap => edge_mode::WRAP,
        EdgeMode::Mirror => edge_mode::MIRROR,
        EdgeMode::None => edge_mode::NONE,
    }
}

fn pack_header(filter_type: u32) -> u32 {
    debug_assert!(filter_type <= 31, "filter_type must fit in 5 bits");

    filter_type
}

fn pack_with_gaussian_params(
    filter_type: u32,
    edge_mode: u32,
    n_decimations: u32,
    n_linear_taps: u32,
) -> u32 {
    debug_assert!(filter_type <= 31, "filter_type must fit in 5 bits");
    debug_assert!(edge_mode <= 3, "edge_mode must fit in 2 bits");
    debug_assert!(n_decimations <= 15, "n_decimations must fit in 4 bits");
    debug_assert!(n_linear_taps <= 3, "n_linear_taps must fit in 2 bits");

    filter_type | (edge_mode << 5) | (n_decimations << 7) | (n_linear_taps << 11)
}

// To a large degree, the vello_hybrid implementation of gaussian blur follows the one in vello_cpu.
// However, we apply a specific optimization, where instead of averaging and weighting each sample
// one after the other, we use linear sampling to sample two pixels at once, and adjust the
// weights accordingly so the gaussian blur filter is still valid. See
// https://www.rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/
// for more information.

/// Maximum number of linear-sampling tap pairs per side.
const MAX_TAPS_PER_SIDE: usize = (MAX_KERNEL_SIZE / 2).div_ceil(2);

/// A linear-sampling kernel derived from a discrete Gaussian kernel.
struct LinearKernel {
    /// Weight of the center tap.
    center_weight: f32,
    // Note that we only need to store one side since they are symmetrical.
    /// Merged weights for each tap pair. Only the first `n_taps` entries are valid.
    weights: [f32; MAX_TAPS_PER_SIDE],
    /// The fractional offsets for each tap pair for linear sampling. Only the first `n_taps` entries are valid.
    offsets: [f32; MAX_TAPS_PER_SIDE],
    /// The actual number of taps per side.
    n_taps: u8,
}

impl LinearKernel {
    fn new(kernel: &[f32; MAX_KERNEL_SIZE], kernel_size: u8) -> Self {
        let kernel_size = kernel_size as usize;
        let radius = kernel_size / 2;
        let center_weight = kernel[radius];

        let mut weights = [0.0_f32; MAX_TAPS_PER_SIDE];
        let mut offsets = [0.0_f32; MAX_TAPS_PER_SIDE];
        let mut n_taps = 0_u8;

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
        // that single pixel is sampled fully.
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

// Currently, we assume that each filter struct has the same size so we can cast them into
// the type-erased type and assume uniform offsets. It might be worth exploring variable offsets
// (as is done for encoded paints) in the future, but it doesn't seem to be worth it for filters
// specifically.

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
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
#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
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
#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
pub(crate) struct GpuGaussianBlur {
    pub header: u32,
    pub std_deviation: f32,
    pub center_weight: f32,
    pub linear_weights: [f32; MAX_TAPS_PER_SIDE],
    pub linear_offsets: [f32; MAX_TAPS_PER_SIDE],
    pub _padding: [u32; 3],
}

impl From<&GaussianBlur> for GpuGaussianBlur {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "n_decimations fits in 4 bits"
    )]
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
#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "n_decimations fits in 4 bits"
    )]
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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "filter size is a small constant"
    )]
    pub(crate) const SIZE_TEXELS: u32 = size_of::<Self>().div_ceil(BYTES_PER_TEXEL) as u32;

    fn filter_type(&self) -> u32 {
        self.data[0] & 0x1F
    }

    /// Returns whether this filter requires a scratch buffer.
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

/// Per-instance vertex data for filter rendering.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct FilterInstanceData {
    /// Top-left pixel offset of the source region in `in_tex` of the atlas texture it lives in.
    pub src_offset: [u32; 2],
    /// Width and height (in pixels) of the source region in `in_tex`.
    pub src_size: [u32; 2],
    /// Top-left pixel offset of the destination region in the output atlas.
    pub dest_offset: [u32; 2],
    /// Width and height (in pixels) of the destination region.
    pub dest_size: [u32; 2],
    /// Full dimensions of the destination atlas texture.
    pub dest_atlas_size: [u32; 2],
    /// Texel offset into `filter_data` where this filter's parameters begin.
    pub filter_data_offset: u32,
    /// Top-left pixel offset of the original texture in `original_tex` of the atlas texture it lives in.
    pub original_offset: [u32; 2],
    /// Width and height (in pixels) of the original content region.
    pub original_size: [u32; 2],
    /// Padding for 16-byte alignment.
    pub _padding: u32,
}

/// Describes the resources and instance data needed to execute the second filter pass.
pub(crate) struct FilterPass2Data {
    /// Instance data for the second filter pass.
    pub instance: FilterInstanceData,
    /// Atlas index of the scratch texture to read from (in `filter_context.image_cache`).
    pub scratch_atlas_idx: u32,
    /// Atlas index of the original input texture (in `filter_context.image_cache`).
    pub original_atlas_idx: u32,
    /// Atlas index of the final destination (in the main `image_cache`).
    pub dest_atlas_idx: u32,
}

/// Describes the resources and instance data needed to execute filter passes.
/// Built by `FilterContext`, consumed by backend-specific renderers.
pub(crate) struct FilterPassData {
    /// Instance data for the first filter pass.
    pub pass_1: FilterInstanceData,
    /// Atlas index of the input texture (in `filter_context.image_cache`).
    pub input_atlas_idx: u32,
    /// Atlas index of the first pass output texture.
    pub pass_1_output_atlas_idx: u32,
    /// Whether the first pass output goes to a filter atlas texture (true)
    /// or the main atlas texture array (false).
    pub pass_1_output_is_filter_atlas: bool,
    /// Instance data for the second filter pass, if it exists.
    pub pass_2: Option<FilterPass2Data>,
}

/// Context used for keeping track of state necessary for filter rendering.
#[derive(Debug)]
pub(crate) struct FilterContext {
    /// The encoded data for each filter used in the current scene that will be uploaded to the
    /// filter data texture.
    pub(crate) filters: Vec<GpuFilterData>,
    /// At what texel offset the filter data for the given layer ID is stored in the texture.
    pub(crate) offsets: HashMap<LayerId, u32>,
    /// Data for each filter layer.
    pub(crate) filter_textures: HashMap<LayerId, FilterLayerData>,
    /// Image cache for storing filter intermediate textures.
    pub(crate) image_cache: ImageCache,
}

impl FilterContext {
    pub(crate) fn new(atlas_config: AtlasConfig) -> Self {
        Self {
            filters: Vec::new(),
            offsets: HashMap::new(),
            filter_textures: HashMap::new(),
            image_cache: ImageCache::new_with_config(atlas_config),
        }
    }

    /// Deallocates all filter textures from both the filter image cache and the image atlas cache,
    /// then clears the filter context.
    ///
    /// Note that the client is responsible for clearing (with a transparent color) the existing
    /// images in the atlas, if desired.
    pub(crate) fn deallocate_all_and_clear(&mut self, image_atlas_cache: &mut ImageCache) {
        for filter_textures in self.filter_textures.values() {
            self.image_cache
                .deallocate(filter_textures.initial_image_id);
            image_atlas_cache.deallocate(filter_textures.dest_image_id);
            if let Some(scratch_id) = filter_textures.scratch_image_id {
                self.image_cache.deallocate(scratch_id);
            }
        }

        // Now clear everything.
        self.filters.clear();
        self.offsets.clear();
        self.filter_textures.clear();
    }

    /// Prepares the context for rendering the filter layers that exist in this scene.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "filter dimensions and paint indices won't exceed u32"
    )]
    pub(crate) fn prepare(
        &mut self,
        render_graph: &RenderGraph,
        dest_cache: &mut ImageCache,
        encoded_paints: &mut Vec<EncodedPaint>,
    ) -> Result<(), AtlasError> {
        if !render_graph.has_filters() {
            return Ok(());
        }

        let mut current_offset = 0_u32;
        for node in &render_graph.nodes {
            // During coarse rasterization it can happen that filter layers with a zero-sized
            // bounding box are allocated. Trying to allocate such a texture in our atlas manager
            // would give an error, so we skip those nodes.
            if node.is_empty() {
                continue;
            }

            if let RenderNodeKind::FilterLayer {
                layer_id,
                filter,
                transform,
                wtile_bbox,
            } = &node.kind
            {
                let width = wtile_bbox.width_px() as u32;
                let height = wtile_bbox.height_px() as u32;

                let instantiated = InstantiatedFilter::new(filter, transform);
                let gpu_filter = GpuFilterData::from(&instantiated);
                let needs_scratch = gpu_filter.needs_scratch_buffer();

                // The tricky part! Why do we have two image caches and don't just use the main
                // atlas that is used by renderers to store images? Fundamentally, the problem is
                // that the destination texture where we render the initial contents of the layer
                // with the filter cannot live in that texture array, because during the `render_strips`
                // pass we already bind that texture array as an input bind group. Therefore, it needs
                // to live somewhere else. So we need to create a second image cache, which lives
                // in the filter context.
                let initial_image_id = self.image_cache.allocate(width, height)?;
                let initial_atlas_id = self.image_cache.get(initial_image_id).unwrap().atlas_id;
                // This represents the destination where the final _filtered_ version lives. We store this
                // in the same image atlas where normal images live, allowing us to treat them like normal
                // image fills.
                let dest_image_id = dest_cache.allocate(width, height)?;
                // For multi-pass filters we need an intermediate scratch buffer. Therefore, it
                // is important that inside of the atlas, the image lives on a different texture
                // than the initial texture. Otherwise, we would have to read and write from the
                // same texture (even though the locations are distinct), which is prohibited.
                let scratch_image_id = if needs_scratch {
                    Some(self.image_cache.allocate_excluding(
                        width,
                        height,
                        Some(AtlasId(initial_atlas_id.as_u32())),
                    )?)
                } else {
                    None
                };

                let encoded_paint = EncodedPaint::Image(EncodedImage {
                    source: ImageSource::OpaqueId(dest_image_id),
                    sampler: ImageSampler::new().with_quality(ImageQuality::Low),
                    may_have_opacities: true,
                    // Since filter layers are always shifted to start at (0, 0), we need
                    // to "unshift" them when sampling.
                    transform: Affine::translate((
                        -(wtile_bbox.x0() as f64) * WideTile::WIDTH as f64,
                        -(wtile_bbox.y0() as f64) * Tile::HEIGHT as f64,
                    ))
                        // TODO: Come up with a more unified way of dealing with this shift.
                        // The problem is that currently, in vello_common, we apply a (0.5, 0.5) shift
                        // to images so that vello_cpu will sample from the pixel center. Since the fragment shader
                        // already applies this shift, we undo it in our render backends: https://github.com/linebender/vello/blob/57a77091fa91bdd652fdacb73aff3d0d1c86285e/sparse_strips/vello_hybrid/src/render/wgpu.rs#L352
                        // Therefore, we need to add the shift here as well, so that it cancels out.
                        * Affine::translate((0.5, 0.5)),
                    x_advance: Vec2::new(1.0, 0.0),
                    y_advance: Vec2::new(0.0, 1.0),
                });

                let idx = encoded_paints.len();
                encoded_paints.push(encoded_paint);

                self.filter_textures.insert(
                    *layer_id,
                    FilterLayerData {
                        initial_image_id,
                        dest_image_id,
                        scratch_image_id,
                        paint_idx: idx as u32,
                        bbox: *wtile_bbox,
                    },
                );

                self.filters.push(gpu_filter);
                self.offsets.insert(*layer_id, current_offset);
                current_offset += GpuFilterData::SIZE_TEXELS;
            }
        }

        Ok(())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "filter count won't exceed u32"
    )]
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

    /// Calculate the required height for the filter data texture.
    /// Returns `None` if no filters are present.
    pub(crate) fn required_filter_data_height(&self, max_texture_dimension_2d: u32) -> Option<u32> {
        let required_texels = self.total_texels();

        if required_texels == 0 {
            return None;
        }
        let height = required_texels.div_ceil(max_texture_dimension_2d);
        assert!(
            height <= max_texture_dimension_2d,
            "Filter texture height exceeds max texture dimensions"
        );

        Some(height)
    }

    /// Build backend-agnostic filter pass data for a given layer.
    ///
    /// `dest_image_cache` is the main renderer's image cache (for `dest_image` lookups).
    /// `get_filter_atlas_size` returns the (width, height) of a filter atlas texture.
    /// `get_main_atlas_size` returns the (width, height) of the main atlas texture array.
    pub(crate) fn build_filter_pass_data(
        &self,
        layer_id: &LayerId,
        dest_image_cache: &ImageCache,
        get_filter_atlas_size: impl Fn(u32) -> [u32; 2],
        get_main_atlas_size: impl Fn() -> [u32; 2],
    ) -> FilterPassData {
        let filter_data_offset = self.offsets.get(layer_id).copied().unwrap();
        let uses_scratch = self
            .get_filter_data(layer_id)
            .is_some_and(|f| f.needs_scratch_buffer());
        let filter_textures = self.filter_textures.get(layer_id).unwrap();

        let initial_image = self
            .image_cache
            .get(filter_textures.initial_image_id)
            .expect("Main image must exist");
        let dest_image = dest_image_cache
            .get(filter_textures.dest_image_id)
            .expect("Dest image must exist");

        let input_atlas_idx = initial_image.atlas_id.as_u32();

        let (first_pass_dest_offset, first_pass_dest_size, first_pass_dest_atlas_idx) =
            if uses_scratch {
                let scratch_id = filter_textures.scratch_image_id.unwrap();
                let scratch = self.image_cache.get(scratch_id).unwrap();
                (scratch.offsets(), scratch.size(), scratch.atlas_id.as_u32())
            } else {
                (
                    dest_image.offsets(),
                    dest_image.size(),
                    dest_image.atlas_id.as_u32(),
                )
            };

        let dest_atlas_size = if uses_scratch {
            get_filter_atlas_size(first_pass_dest_atlas_idx)
        } else {
            get_main_atlas_size()
        };

        let pass_1 = FilterInstanceData {
            src_offset: initial_image.offsets(),
            src_size: initial_image.size(),
            dest_offset: first_pass_dest_offset,
            dest_size: first_pass_dest_size,
            dest_atlas_size,
            filter_data_offset,
            _padding: 0,
            // Unused in the first filter pass.
            original_offset: [0, 0],
            original_size: [0, 0],
        };

        let pass_2 = if uses_scratch {
            let scratch_id = filter_textures.scratch_image_id.unwrap();
            let scratch_resource = self.image_cache.get(scratch_id).unwrap();
            let main_atlas_size = get_main_atlas_size();

            let instance = FilterInstanceData {
                src_offset: scratch_resource.offsets(),
                src_size: scratch_resource.size(),
                dest_offset: dest_image.offsets(),
                dest_size: dest_image.size(),
                dest_atlas_size: main_atlas_size,
                filter_data_offset,
                _padding: 0,
                original_offset: initial_image.offsets(),
                original_size: initial_image.size(),
            };

            Some(FilterPass2Data {
                instance,
                scratch_atlas_idx: scratch_resource.atlas_id.as_u32(),
                original_atlas_idx: input_atlas_idx,
                dest_atlas_idx: dest_image.atlas_id.as_u32(),
            })
        } else {
            None
        };

        FilterPassData {
            pass_1,
            input_atlas_idx,
            pass_1_output_atlas_idx: first_pass_dest_atlas_idx,
            pass_1_output_is_filter_atlas: uses_scratch,
            pass_2,
        }
    }
}

/// Data associated with a single filter layer.
#[derive(Debug)]
pub(crate) struct FilterLayerData {
    /// Image ID for the main texture holding the raw initially painted version of the layer.
    pub initial_image_id: ImageId,
    /// Image ID for the destination texture holding the final filtered version. This lives in
    /// the same image atlas as normal images, allowing us to treat filtered layers the same
    /// way as normal images that we can sample from.
    pub dest_image_id: ImageId,
    /// Some filters require intermediate buffers. This field optionally holds the ID of
    /// the intermediate texture we can use for that.
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
    use vello_common::color::AlphaColor;
    use vello_common::filter::gaussian_blur::{compute_gaussian_kernel, plan_decimated_blur};

    #[test]
    fn test_offset_conversion() {
        let offset = Offset::new(10.5, -20.3);
        let gpu_offset = GpuOffset::from(&offset);

        assert_eq!(gpu_offset.header & 0x1F, filter_type::OFFSET);
        assert_eq!(gpu_offset.dx, 10.5);
        assert_eq!(gpu_offset.dy, -20.3);
    }

    fn check_round_trip<T>(gpu: T, expected_type: u32)
    where
        T: Into<GpuFilterData> + Copy + PartialEq + core::fmt::Debug + Pod,
    {
        let erased: GpuFilterData = gpu.into();
        assert_eq!(erased.filter_type(), expected_type);
        assert_eq!(bytemuck::cast::<_, T>(erased), gpu);
    }

    #[test]
    fn test_offset_round_trip() {
        check_round_trip(GpuOffset::from(&Offset::new(1.0, 2.0)), filter_type::OFFSET);
    }

    #[test]
    fn test_flood_round_trip() {
        check_round_trip(
            GpuFlood::from(&Flood::new(AlphaColor::new([0.2, 0.4, 0.6, 0.8]))),
            filter_type::FLOOD,
        );
    }

    #[test]
    fn test_gaussian_blur_round_trip() {
        check_round_trip(
            GpuGaussianBlur::from(&GaussianBlur::new(2.0, EdgeMode::None)),
            filter_type::GAUSSIAN_BLUR,
        );
    }

    #[test]
    fn test_drop_shadow_round_trip() {
        check_round_trip(
            GpuDropShadow::from(&DropShadow::new(
                3.0,
                -4.0,
                1.5,
                EdgeMode::Duplicate,
                AlphaColor::new([0.0, 0.0, 0.0, 1.0]),
            )),
            filter_type::DROP_SHADOW,
        );
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
