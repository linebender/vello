// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// A brief high-level explanation of how filters work in the hybrid renderer:
//
// Filter layers are recorded as normal layers with an attached filter description. During schedule
// building, the layer is allocated in one of the two layer atlas textures with enough space for the
// expanded filter bounds rather than only the raw layer contents. Multi-pass filters also reserve
// regions in the two scratch textures used for ping-ponging.
//
// While executing the schedule, each layer texture pass expands its filters into a sequence of GPU
// passes such as offset, downscale, blur, upscale, and drop-shadow composite. Passes at the same
// step with matching input/output textures are batched together. The round first renders the layer
// contents into the allocated layer texture region, then the backend executes the filter plan
// before the layer is blended or sampled by its parent.
//
// The final filtered pixels are normalized into scratch texture 0 and then copied back into the
// layer texture allocation. From that point on, the rest of the scheduler can treat the layer like
// any other sampled layer: it can be blended into its parent, used with opacity, clipped, or sampled
// by the normal strip rendering path.

//! GPU filter types and conversion utilities.

use crate::copy::GpuCopyInstance;
use crate::schedule::{TextureRegion, TextureTarget, round::FilterOp};
use crate::util::{IntRect, IntSize, pack_u16_pair};
use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};
use vello_common::filter::drop_shadow::DropShadow;
use vello_common::filter::flood::Flood;
use vello_common::filter::gaussian_blur::{DecimationSizer, GaussianBlur, MAX_KERNEL_SIZE};
use vello_common::filter::offset::Offset;
use vello_common::filter::{FilterData, PreparedFilter};
use vello_common::filter_effects::EdgeMode;
use vello_common::util::RetainVec;

/// How much transparent padding to reserve for filter layers within the image. Needed so
/// that the various shader programs can assume transparent pixels on the outside, making
/// the code significantly easier since we don't need to special-case border pixels. Since we
/// do use checked accesses for the offset filter, the bottleneck is formed by the gaussian blur
/// convolution.
#[expect(clippy::cast_possible_truncation, reason = "safe in this case")]
pub(crate) const FILTER_ATLAS_PADDING: u16 = MAX_KERNEL_SIZE as u16 / 2;

// Note: Keep these variables and struct layouts in sync with `filter.wgsl`!

// Since we store in RGBA32 texture.
const BYTES_PER_TEXEL: usize = 16;
const FILTER_SIZE_BYTES: usize = 48;
const FILTER_SIZE_U32: usize = FILTER_SIZE_BYTES / 4;

const _: () = assert!(
    size_of::<GpuFilterData>() == FILTER_SIZE_BYTES,
    "memory size of filters need to match"
);
const _: () = assert!(
    size_of::<GpuOffset>() == FILTER_SIZE_BYTES,
    "memory size of filters need to match"
);
const _: () = assert!(
    size_of::<GpuFlood>() == FILTER_SIZE_BYTES,
    "memory size of filters need to match"
);
const _: () = assert!(
    size_of::<GpuDropShadow>() == FILTER_SIZE_BYTES,
    "memory size of filters need to match"
);
const _: () = assert!(
    size_of::<GpuGaussianBlur>() == FILTER_SIZE_BYTES,
    "memory size of filters need to match"
);

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

pub(crate) mod pass_kind {
    pub(crate) const COPY: u32 = 0;
    pub(crate) const FLOOD: u32 = 1;
    pub(crate) const OFFSET: u32 = 2;
    pub(crate) const DOWNSCALE: u32 = 3;
    pub(crate) const BLUR_H: u32 = 4;
    pub(crate) const BLUR_V: u32 = 5;
    pub(crate) const UPSCALE: u32 = 6;
    pub(crate) const COMPOSITE_DROP_SHADOW: u32 = 7;
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

fn pack_header_with_gaussian_params(
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
        // formulas on the website linked above.
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
// specifically since it's uncommon to have more than a few dozen filters in a single scene.

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
    pub center_weight: f32,
    pub linear_weights: [f32; MAX_TAPS_PER_SIDE],
    pub linear_offsets: [f32; MAX_TAPS_PER_SIDE],
    // Needed since drop shadow has a bigger footprint.
    pub _padding: [u32; 4],
}

impl From<&GaussianBlur> for GpuGaussianBlur {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "n_decimations fits in 4 bits"
    )]
    fn from(blur: &GaussianBlur) -> Self {
        let lk = LinearKernel::new(&blur.kernel, blur.kernel_size);

        Self {
            header: pack_header_with_gaussian_params(
                filter_type::GAUSSIAN_BLUR,
                edge_mode_to_gpu(blur.edge_mode),
                // Note that this could be exceeded in theory, but it would have to be a huge
                // standard deviation! If it turns out to be a problem we can reserve additional
                // bits for it in the future.
                blur.n_decimations as u32,
                lk.n_taps as u32,
            ),
            center_weight: lk.center_weight,
            linear_weights: lk.weights,
            linear_offsets: lk.offsets,
            _padding: [0; 4],
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
pub(crate) struct GpuDropShadow {
    pub header: u32,
    pub center_weight: f32,
    pub linear_weights: [f32; MAX_TAPS_PER_SIDE],
    pub linear_offsets: [f32; MAX_TAPS_PER_SIDE],
    pub dx: f32,
    pub dy: f32,
    pub color: u32,
    pub _padding: [u32; 1],
}

impl From<&DropShadow> for GpuDropShadow {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "n_decimations fits in 4 bits"
    )]
    fn from(shadow: &DropShadow) -> Self {
        let lk = LinearKernel::new(&shadow.kernel, shadow.kernel_size);
        Self {
            header: pack_header_with_gaussian_params(
                filter_type::DROP_SHADOW,
                edge_mode_to_gpu(shadow.edge_mode),
                shadow.n_decimations as u32,
                lk.n_taps as u32,
            ),
            center_weight: lk.center_weight,
            linear_weights: lk.weights,
            linear_offsets: lk.offsets,
            dx: shadow.dx,
            dy: shadow.dy,
            color: shadow.color.premultiply().to_rgba8().to_u32(),
            _padding: [0; 1],
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

    pub(crate) fn filter_type(&self) -> u32 {
        self.data[0] & 0x1F
    }

    /// Returns the number of decimation levels encoded in the header.
    pub(crate) fn n_decimations(&self) -> usize {
        ((self.data[0] >> 7) & 0xF) as usize
    }

    /// Whether the filter is a multi-pass filter, requiring intermediate scratch textures.
    pub(crate) fn is_multi_pass(&self) -> bool {
        matches!(
            self.filter_type(),
            filter_type::GAUSSIAN_BLUR | filter_type::DROP_SHADOW
        )
    }
}

trait CastToFilterData: Pod {}

impl CastToFilterData for GpuOffset {}
impl CastToFilterData for GpuFlood {}
impl CastToFilterData for GpuGaussianBlur {}
impl CastToFilterData for GpuDropShadow {}

impl<T: CastToFilterData> From<T> for GpuFilterData {
    fn from(filter: T) -> Self {
        bytemuck::cast(filter)
    }
}

impl From<&PreparedFilter> for GpuFilterData {
    fn from(filter: &PreparedFilter) -> Self {
        match filter {
            PreparedFilter::Offset(f) => GpuOffset::from(f).into(),
            PreparedFilter::Flood(f) => GpuFlood::from(f).into(),
            PreparedFilter::GaussianBlur(f) => GpuGaussianBlur::from(f).into(),
            PreparedFilter::DropShadow(f) => GpuDropShadow::from(f).into(),
        }
    }
}

/// Per-instance vertex data for filter rendering.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct FilterInstanceData {
    /// Source region in the input atlas.
    pub src: IntRect,
    /// Destination region in the output atlas.
    pub dest: IntRect,
    /// Full pixel dimensions of the destination atlas texture.
    pub dest_atlas_size: IntSize,
    /// Texel offset into `filter_data` where this filter's data is stored.
    pub filter_data_offset: u32,
    /// The region of the original (unfiltered) content.
    pub original: IntRect,
    /// Additional per-pass data.
    ///
    /// Layout:
    /// - bits 0..30: filter pass kind
    /// - bit 31: layer texture index containing the unfiltered layer content
    pub other_data: u32,
}

/// Context used for keeping track of state necessary for filter rendering.
#[derive(Debug)]
pub(crate) struct FilterContext {
    /// The encoded data for each filter used in the current scene that will be uploaded to the
    /// filter data texture.
    filters: Vec<GpuFilterData>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PreparedGpuFilter {
    pub(crate) data_offset: u32,
    pub(crate) data: GpuFilterData,
}

impl PreparedGpuFilter {
    pub(crate) fn scratch_count(self) -> usize {
        if self.data.is_multi_pass() {
            return 2;
        }

        1
    }
}

#[derive(Debug, Default)]
pub(crate) struct FilterPassPlan {
    steps: RetainVec<Vec<FilterInstanceData>>,
    copy_back: Vec<GpuCopyInstance>,
}

impl FilterPassPlan {
    pub(crate) fn is_empty(&self) -> bool {
        self.copy_back.is_empty()
    }

    pub(crate) fn steps(&self) -> impl Iterator<Item = &[FilterInstanceData]> {
        self.steps.as_slice().iter().map(Vec::as_slice)
    }

    pub(crate) fn copy_back(&self) -> &[GpuCopyInstance] {
        &self.copy_back
    }

    fn clear(&mut self) {
        self.steps.clear();
        self.copy_back.clear();
    }

    fn step_mut(&mut self, step: usize) -> &mut Vec<FilterInstanceData> {
        if self.steps.len() <= step {
            self.steps.resize_with(step + 1, Vec::new);
        }

        &mut self.steps[step]
    }
}

pub(crate) fn plan(
    filters: impl IntoIterator<Item = FilterOp>,
    target_texture_size: (u16, u16),
    passes: &mut FilterPassPlan,
) {
    passes.clear();

    for filter in filters {
        let mut builder = FilterPassBuilder::new(filter, target_texture_size, passes);
        match filter.gpu_filter.filter_type() {
            filter_type::OFFSET => {
                builder.emit_to_scratch(pass_kind::OFFSET);
            }
            filter_type::FLOOD => {
                builder.emit_to_scratch(pass_kind::FLOOD);
            }
            filter_type::GAUSSIAN_BLUR => {
                builder.emit_blur_sequence(filter.gpu_filter.n_decimations());
            }
            filter_type::DROP_SHADOW => {
                builder.emit_to_scratch(pass_kind::OFFSET);
                builder.emit_blur_sequence(filter.gpu_filter.n_decimations());
                builder.emit_to_scratch(pass_kind::COMPOSITE_DROP_SHADOW);
            }
            _ => unreachable!("unsupported filter type was encoded"),
        }

        builder.ensure_result_in_scratch0();
        builder.copy_back();
    }
}

#[derive(Debug)]
struct FilterPassBuilder<'a> {
    op: FilterOp,
    target_texture_size: (u16, u16),
    passes: &'a mut FilterPassPlan,
    sizer: DecimationSizer,
    original: TextureTarget,
    current_scratch: Option<usize>,
    step: usize,
}

impl<'a> FilterPassBuilder<'a> {
    fn new(op: FilterOp, target_texture_size: (u16, u16), passes: &'a mut FilterPassPlan) -> Self {
        let mut sizer = DecimationSizer::default();
        sizer.reset(
            op.layer_region.texture.rect.width(),
            op.layer_region.texture.rect.height(),
        );
        let original = TextureTarget::layer(op.layer_region.texture.texture_index);
        Self {
            op,
            target_texture_size,
            passes,
            sizer,
            original,
            current_scratch: None,
            step: 0,
        }
    }

    fn scratch_region(&self, index: usize) -> TextureRegion {
        self.op.scratches[index].expect("filter pass requires allocated scratch region")
    }

    fn texture_offset(&self, texture: TextureTarget) -> [u32; 2] {
        match texture {
            TextureTarget::Layer0 | TextureTarget::Layer1 => [
                u32::from(self.op.layer_region.texture.rect.x0),
                u32::from(self.op.layer_region.texture.rect.y0),
            ],
            TextureTarget::Scratch0 | TextureTarget::Scratch1 => {
                let scratch = self.scratch_region(texture.index());
                [u32::from(scratch.rect.x0), u32::from(scratch.rect.y0)]
            }
        }
    }

    fn current_texture(&self) -> TextureTarget {
        self.current_scratch
            .map_or_else(|| self.original, TextureTarget::scratch)
    }

    fn next_scratch(&self) -> usize {
        self.current_scratch.map_or(0, |scratch| 1 - scratch)
    }

    fn apply_pass_dimensions(&mut self, kind: u32) -> (IntSize, IntSize) {
        match kind {
            pass_kind::DOWNSCALE => {
                let (sw, sh) = self.sizer.current();
                let (dw, dh) = self.sizer.downscale();
                (
                    IntSize([u32::from(sw), u32::from(sh)]),
                    IntSize([u32::from(dw), u32::from(dh)]),
                )
            }
            pass_kind::UPSCALE => {
                let (sw, sh) = self.sizer.current();
                let (dw, dh) = self.sizer.upscale();
                (
                    IntSize([u32::from(sw), u32::from(sh)]),
                    IntSize([u32::from(dw), u32::from(dh)]),
                )
            }
            _ => {
                let (w, h) = self.sizer.current();
                let size = IntSize([u32::from(w), u32::from(h)]);
                (size, size)
            }
        }
    }

    fn emit(&mut self, kind: u32, output: TextureTarget) {
        let (src_size, dest_size) = self.apply_pass_dimensions(kind);
        let input = self.current_texture();
        let src_offset = self.texture_offset(input);
        let dest_offset = self.texture_offset(output);
        let original_offset = self.texture_offset(self.original);
        let other_data = self.other_data(kind);
        self.passes.step_mut(self.step).push(FilterInstanceData {
            src: IntRect::new(src_offset, src_size),
            dest: IntRect::new(dest_offset, dest_size),
            dest_atlas_size: IntSize([
                u32::from(self.target_texture_size.0),
                u32::from(self.target_texture_size.1),
            ]),
            filter_data_offset: self.op.filter_data_offset,
            original: IntRect::new(
                original_offset,
                [
                    u32::from(self.op.layer_region.texture.rect.width()),
                    u32::from(self.op.layer_region.texture.rect.height()),
                ],
            ),
            other_data,
        });
        self.step += 1;
    }

    fn emit_to_scratch(&mut self, kind: u32) {
        let scratch = self.next_scratch();
        self.emit(kind, TextureTarget::scratch(scratch));
        self.current_scratch = Some(scratch);
    }

    fn ensure_result_in_scratch0(&mut self) {
        if self.current_texture() == TextureTarget::Scratch0 {
            return;
        }

        self.emit(pass_kind::COPY, TextureTarget::Scratch0);
        self.current_scratch = Some(0);
    }

    fn emit_blur_sequence(&mut self, n_decimations: usize) {
        for _ in 0..n_decimations {
            self.emit_to_scratch(pass_kind::DOWNSCALE);
        }
        self.emit_to_scratch(pass_kind::BLUR_H);

        let mut final_pass = pass_kind::BLUR_V;
        if n_decimations > 0 {
            self.emit_to_scratch(pass_kind::BLUR_V);
            for _ in 0..n_decimations - 1 {
                self.emit_to_scratch(pass_kind::UPSCALE);
            }
            final_pass = pass_kind::UPSCALE;
        }
        self.emit_to_scratch(final_pass);
    }

    fn other_data(&self, kind: u32) -> u32 {
        const OTHER_DATA_LAYER_TEXTURE_INDEX_SHIFT: u32 = 31;

        let texture_index = u32::try_from(self.op.layer_region.texture.texture_index)
            .expect("layer texture index must fit into u32");
        debug_assert!(texture_index <= 1, "layer texture index must fit in 1 bit");
        kind | (texture_index << OTHER_DATA_LAYER_TEXTURE_INDEX_SHIFT)
    }

    fn copy_back(&mut self) {
        let scratch = self.op.scratches[0].expect("filter copy requires scratch texture 0");

        let copy_instance = GpuCopyInstance {
            target_texture_origin: pack_u16_pair(
                self.op.layer_region.texture.rect.x0,
                self.op.layer_region.texture.rect.y0,
            ),
            source_texture_origin: pack_u16_pair(scratch.rect.x0, scratch.rect.y0),
            copy_rect_size: pack_u16_pair(
                self.op.layer_region.texture.rect.width(),
                self.op.layer_region.texture.rect.height(),
            ),
            target_texture_size: pack_u16_pair(
                self.target_texture_size.0,
                self.target_texture_size.1,
            ),
        };

        self.passes.copy_back.push(copy_instance);
    }
}

impl FilterContext {
    pub(crate) fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.filters.clear();
    }

    pub(crate) fn push(&mut self, filter_data: &FilterData) -> PreparedGpuFilter {
        let data_offset = self.total_texels();
        let prepared = PreparedFilter::new(&filter_data.filter, &filter_data.transform);
        let data = GpuFilterData::from(&prepared);
        self.filters.push(data);
        PreparedGpuFilter { data_offset, data }
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

    pub(crate) fn serialize_to_buffer(&self, buffer: &mut [u8]) {
        let src = bytemuck::cast_slice::<GpuFilterData, u8>(&self.filters);
        debug_assert!(
            buffer.len() >= src.len(),
            "filter data buffer too small: {} < {}",
            buffer.len(),
            src.len()
        );
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
        debug_assert!(
            height <= max_texture_dimension_2d,
            "Filter texture height exceeds max texture dimensions"
        );

        Some(height)
    }
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
