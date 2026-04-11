// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// A very brief explanation of how filters work on a high-level follows here:
//
// During coarse rasterization, filter layers become nodes in the `RenderGraph`. Once we hit
// vello_hybrid, we render those nodes in an order such that all children nodes are done rendering
// before they are needed by their parent nodes. In terms of allocation, we reserve
// 1) A position in a new atlas array (distinct from the atlas array used for normal images), which
// stores the raw, unfiltered version of the layer.
// 2) In case the filter is multi-pass, we also allocate 2 scratch buffers in that new atlas array,
// such that we can do ping-ponging between them.
// 3) A position in the image atlas array, which stores the final filtered version of the layer,
// such that it can be consumed like a normal image in parent layers.
//
// When rendering a filtered layer, as mentioned we first render the normal raw content of the layer
// using the normal existing mechanism for rendering, except for the fact that we render into an
// intermediate texture instead of the final output. Then, in the end, we either apply a single-pass
// filter that just copies the contents from intermediate storage into the image atlas (while applying
// the filter), or we apply a multi-pass filter (currently only used for Gaussian blurring), which
// splits the filter into more atomic filter passes and applies them repeatedly using the scratch
// textures.
//
// Finally, when the parent layer needs to render the filtered child layer, it can simply treat it
// like a normal image.

//! GPU filter types and conversion utilities.

use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};
use hashbrown::HashMap;
use vello_common::coarse::{WideTile, WideTilesBbox};
use vello_common::encode::{EncodedImage, EncodedPaint};
use vello_common::filter::PreparedFilter;
use vello_common::filter::drop_shadow::DropShadow;
use vello_common::filter::flood::Flood;
use vello_common::filter::gaussian_blur::{DecimationSizer, GaussianBlur, MAX_KERNEL_SIZE};
use vello_common::filter::offset::Offset;
use vello_common::filter_effects::EdgeMode;
use vello_common::kurbo::{Affine, Vec2};
use vello_common::paint::{ImageId, ImageSource};
use vello_common::peniko::{ImageQuality, ImageSampler};
use vello_common::render_graph::{LayerId, RenderGraph, RenderNodeKind};
use vello_common::tile::Tile;

use crate::render::common::IMAGE_PADDING;
use crate::util::{IntOffset, IntRect, IntSize};
use vello_common::image_cache::ImageCache;
use vello_common::multi_atlas::AtlasConfig;
use vello_common::multi_atlas::{AtlasError, AtlasId};

/// How much transparent padding to reserve for filter layers within the image. Needed so
/// that the various shader programs can assume transparent pixels on the outside, making
/// the code significantly easier since we don't need to special-case border pixels. Since we
/// do use checked accesses for the offset filter, the bottleneck is formed by the gaussian blur
/// convolution.
#[expect(clippy::cast_possible_truncation, reason = "safe in this case")]
const FILTER_ATLAS_PADDING: u16 = MAX_KERNEL_SIZE as u16 / 2;

// Note: Keep these variables and struct layouts in sync with `filters.wgsl`!

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
    #[expect(dead_code, reason = "Useful in the future")]
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
    /// The filter pass that should be executed.
    pub pass_kind: u32,
}

impl FilterInstanceData {
    /// The scissor rect that should be applied when applying this filter
    /// pass.
    pub(crate) fn scissor_rect(&self, target_size: [u32; 2]) -> [u32; 4] {
        // See the comment in `filters.wgsl`. In general, when applying a filter it's not enough
        // to just cover the destination region of the (possibly downsized) filter area. We also
        // need to include some padding such that stale border pixels are set back to a fully
        // transparent color. In the shader, we always use the original size (e.g. if the original
        // filter layer was 700x700 pixels but the downsized area only 15x15, we would still have fragment
        // shader invocations for the whole 700x700 area, even if the vast majority just shortcut to
        // yielding a transparent color).
        //
        // However, we can actually further reduce this area: In the bottom/right, we only need to
        // cover as many additional pixels as are necessary for padding. So in the above case,
        // we only need to cover (15 + FILTER_ATLAS_PADDING) in each direction. Therefore, when
        // rendering filters we apply a scissor rect to further reduce the area to only the part
        // that really needs to be cleared out. Experiments have shown that especially on low-tier
        // devices, doing this leads to very huge speedups.
        //
        // Note that we never need to clear the top/left area, since the origin of a filter
        // between each pass always stays the same; only the width/height can vary.
        //
        // TODO: Explore whether we can not use scissor rects and instead adjust the vertex shader
        // to cover the reduced area. Unfortunately, this seemed to cause other non-obvious test
        // failures, hence why we just use this approach for now.
        let x = self.dest.offset.0[0];
        let y = self.dest.offset.0[1];
        let width = self.dest.size.0[0];
        let height = self.dest.size.0[1];
        let padding = u32::from(FILTER_ATLAS_PADDING);
        let x1 = x
            .saturating_add(width)
            .saturating_add(padding)
            .min(target_size[0]);
        let y1 = y
            .saturating_add(height)
            .saturating_add(padding)
            .min(target_size[1]);
        [x, y, x1 - x, y1 - y]
    }
}

/// Where a filter pass writes its output.
#[derive(Debug)]
pub(crate) enum FilterPassTarget {
    /// Output to a filter atlas texture.
    FilterAtlas(u32),
    /// Output to the main atlas texture.
    MainAtlas(u32),
}

/// Describes a single filter pass with its resources.
#[derive(Debug)]
pub(crate) struct FilterPass {
    /// Atlas index of the input texture that will be used as the basis for the operation.
    pub(crate) input_atlas_idx: u32,
    /// Where this pass writes its output.
    pub(crate) output: FilterPassTarget,
    /// The index of the atlas that contains the original content. This is only set for
    /// filter passes that actually need it.
    pub(crate) original_atlas_idx: Option<u32>,
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
    // Note that this image cache is separate from the image cache used for used-uploaded images.
    // This means that intermediate content for filters is _not_ stored in the main image texture atlas,
    // but instead in a separate atlas array.
    /// Image cache for storing filter intermediate textures.
    pub(crate) image_cache: ImageCache,
}

#[derive(Default, Debug)]
pub(crate) struct FilterPassState {
    /// Store the most recently generated filter passes.
    filter_passes: Vec<FilterPass>,
    /// The instance data for each filter pass.
    instances: Vec<FilterInstanceData>,
    sizer: DecimationSizer,
}

impl FilterPassState {
    fn clear(&mut self) {
        self.filter_passes.clear();
        self.instances.clear();
    }

    fn push(&mut self, instance: FilterInstanceData, pass: FilterPass) {
        self.instances.push(instance);
        self.filter_passes.push(pass);
    }

    pub(crate) fn filter_passes(&self) -> &[FilterPass] {
        &self.filter_passes
    }

    pub(crate) fn instances(&self) -> &[FilterInstanceData] {
        &self.instances
    }
}

/// An image location within an atlas texture.
struct AtlasLocation {
    /// Index of the atlas texture.
    atlas_idx: u32,
    /// Texel offset of the image within the atlas.
    offset: IntOffset,
    /// Full pixel dimensions of the atlas texture.
    atlas_size: IntSize,
}

/// A helper struct making it easier to schedule blur filters.
struct BlurPassScheduler<'a> {
    state: &'a mut FilterPassState,
    /// Atlas index and offset of the initial (unfiltered) image.
    ///
    /// Unlike `dest` and `scratch`, we dont need to store the size of the atlas
    /// itself since we never write into the initial image when applying filters.
    initial: (u32, IntOffset),
    /// Location of the final destination in its atlas.
    dest: AtlasLocation,
    /// Location of each scratch buffer in its atlas.
    scratch: [AtlasLocation; 2],
    /// Texel offset into the filter data texture.
    filter_data_offset: u32,
    /// Full size of the original content region.
    original_size: IntSize,
    /// Which scratch buffer to write to next.
    toggle: usize,
    /// Whether the next pass is the first (reads from initial instead of scratch).
    first: bool,
}

impl BlurPassScheduler<'_> {
    /// Compute and update source and destination sizes based on the pass kind,
    fn apply_pass_dimensions(&mut self, kind: u32) -> (IntSize, IntSize) {
        match kind {
            pass_kind::DOWNSCALE => {
                let (sw, sh) = self.state.sizer.current();
                let (dw, dh) = self.state.sizer.downscale();
                (
                    IntSize([u32::from(sw), u32::from(sh)]),
                    IntSize([u32::from(dw), u32::from(dh)]),
                )
            }
            pass_kind::UPSCALE => {
                let (sw, sh) = self.state.sizer.current();
                let (dw, dh) = self.state.sizer.upscale();
                (
                    IntSize([u32::from(sw), u32::from(sh)]),
                    IntSize([u32::from(dw), u32::from(dh)]),
                )
            }
            _ => {
                let (w, h) = self.state.sizer.current();
                let size = IntSize([u32::from(w), u32::from(h)]);
                (size, size)
            }
        }
    }

    /// Resolve the input atlas index and offset for the next pass.
    fn input(&mut self) -> (u32, IntOffset) {
        if self.first {
            // Atlas containing the original, unfiltered layer.
            self.first = false;
            (self.initial.0, self.initial.1)
        } else {
            // Atlas containing the layers inside the previous scratch buffer.
            let prev = (self.toggle + 1) % 2;
            (self.scratch[prev].atlas_idx, self.scratch[prev].offset)
        }
    }

    /// Emit a pass to the next scratch buffer.
    fn emit_to_scratch(&mut self, kind: u32) {
        let (src_size, dst_size) = self.apply_pass_dimensions(kind);
        let (input_idx, src_offset) = self.input();
        let s = self.toggle;
        self.toggle = (self.toggle + 1) % 2;

        self.state.push(
            FilterInstanceData {
                src: IntRect::new(src_offset, src_size),
                dest: IntRect::new(self.scratch[s].offset, dst_size),
                dest_atlas_size: self.scratch[s].atlas_size,
                filter_data_offset: self.filter_data_offset,
                original: IntRect::new([0, 0], self.original_size),
                pass_kind: kind,
            },
            FilterPass {
                input_atlas_idx: input_idx,
                output: FilterPassTarget::FilterAtlas(self.scratch[s].atlas_idx),
                // Note: We must not bind the original texture here! See the comment in
                // `emit_composite_to_dest`.
                original_atlas_idx: None,
            },
        );
    }

    /// Emit a pass to the final destination.
    fn emit_to_dest(&mut self, kind: u32) {
        let (src_size, dst_size) = self.apply_pass_dimensions(kind);
        let (input_idx, src_offset) = self.input();

        self.state.push(
            FilterInstanceData {
                src: IntRect::new(src_offset, src_size),
                dest: IntRect::new(self.dest.offset, dst_size),
                dest_atlas_size: self.dest.atlas_size,
                filter_data_offset: self.filter_data_offset,
                original: IntRect::new([0, 0], self.original_size),
                pass_kind: kind,
            },
            FilterPass {
                input_atlas_idx: input_idx,
                output: FilterPassTarget::MainAtlas(self.dest.atlas_idx),
                // Note: We must not bind the original texture here! See the comment in
                // `emit_composite_to_dest`.
                original_atlas_idx: None,
            },
        );
    }

    /// Emit a composite pass that reads from the previous scratch and the original,
    /// and writes to the final destination.
    fn emit_composite_to_dest(&mut self, kind: u32) {
        let (src_size, dst_size) = self.apply_pass_dimensions(kind);
        let (input_idx, src_offset) = self.input();

        self.state.push(
            FilterInstanceData {
                src: IntRect::new(src_offset, src_size),
                dest: IntRect::new(self.dest.offset, dst_size),
                dest_atlas_size: self.dest.atlas_size,
                filter_data_offset: self.filter_data_offset,
                original: IntRect::new(self.initial.1, self.original_size),
                pass_kind: kind,
            },
            FilterPass {
                input_atlas_idx: input_idx,
                output: FilterPassTarget::MainAtlas(self.dest.atlas_idx),
                // There is some important subtlety going on. We have three different "regions":
                // The initial image region (storing the unfiltered representation of the layer)
                // as well as two scratch buffers. Scratch buffer 0 always lives on a different atlas
                // than the initial image, but scratch buffer 1 could live on the same.
                // This is intentional: Right now, we only ever need to access the initial image
                // during drop shadow, where the original image needs to be composited on top of
                // the filtered version. However, since this is the very last step, we can just bind
                // the two textures as input (even if they point to the same atlas) since we only
                // read from them, and the final destination we write to is guaranteed to live somewhere
                // else. However, care needs to be exercised in case we add more filters in the future
                // that also need to sample from the unfiltered texture, but where the write target
                // is still a scratch buffer.
                original_atlas_idx: Some(self.initial.0),
            },
        );
    }

    /// Apply the sequences of passes that is needed to create a full Gaussian blur with
    /// the given number of decimations.
    fn emit_blur_sequence(&mut self, n_decimations: usize, final_to_dest: bool) {
        // TODO: From my experiments, it would very much be worth it to add a
        // UPSCALE_4x and DOWNSCALE_4x pass, since unlike the CPU we can use bilinear
        // filtering for sampling and therefore don't need as many samples, and can reduce
        // the number of render passes for large standard deviations. However, this unfortunately
        // causes higher pixel differences for some tests compared to vello_cpu, since edge
        // pixels will inevitably exhibit different behavior. Therefore, for now we stick to
        // this more straight-forward approach.

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

        if final_to_dest {
            self.emit_to_dest(final_pass);
        } else {
            self.emit_to_scratch(final_pass);
        }
    }
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
    pub(crate) fn deallocate_all_and_clear_context(&mut self, image_atlas_cache: &mut ImageCache) {
        for filter_textures in self.filter_textures.values() {
            self.image_cache
                .deallocate(filter_textures.initial_image_id);
            image_atlas_cache.deallocate(filter_textures.dest_image_id);
            if let Some(scratch_ids) = filter_textures.scratch_image_ids {
                self.image_cache.deallocate(scratch_ids[0]);
                self.image_cache.deallocate(scratch_ids[1]);
            }
        }

        // Now clear everything (except for `image_cache`, where there is nothing to clear).
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
            // would give an error, so we skip those nodes. Later on, we skip nodes with no
            // associated filter layer gracefully.
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

                let instantiated = PreparedFilter::new(filter, transform);
                let gpu_filter = GpuFilterData::from(&instantiated);
                let is_multi_pass = gpu_filter.is_multi_pass();

                // The tricky part! Why do we have two distinct image caches and don't just use the main
                // atlas that is used by renderers to store images? Fundamentally, the problem is
                // that the destination texture where we render the initial contents of the layer
                // with the filter cannot live in that texture array, because during the `render_strips`
                // pass we already bind that texture array as an input bind group so that we can render
                // normal images. Since filter layers can also have normal images, they can't live
                // in the same location. Therefore, it needs to live somewhere else.
                // So we need to create a second image cache, and the initial rendering of the
                // image needs to be stored there.
                let initial_image_id =
                    self.image_cache
                        .allocate(width, height, FILTER_ATLAS_PADDING)?;
                let initial_atlas_id = self.image_cache.get(initial_image_id).unwrap().atlas_id;
                // This represents the destination where the final _filtered_ version lives. We store this
                // in the same image atlas where normal images live, allowing us to treat them like normal
                // image fills.
                let dest_image_id = dest_cache.allocate(width, height, IMAGE_PADDING)?;
                // For multi-pass filters we need two intermediate scratch buffers for ping-pong
                // rendering. Each scratch must live on a different atlas texture than the other
                // and then the initial texture, because we cannot read and write the same texture.
                let scratch_image_ids = if is_multi_pass {
                    // First scratch buffer needs to live on a different texture than the initial image.
                    let scratch_1 = self.image_cache.allocate_excluding(
                        width,
                        height,
                        FILTER_ATLAS_PADDING,
                        Some(AtlasId(initial_atlas_id.as_u32())),
                    )?;
                    let scratch_1_atlas_id = self.image_cache.get(scratch_1).unwrap().atlas_id;

                    // Second scratch buffer needs to live on a different texture than first scratch buffer.
                    // Note: The second scratch buffer is allowed to live on the same atlas
                    // as the initial image texture. See the comment in `emit_composite_to_dest`.
                    let scratch_2 = self.image_cache.allocate_excluding(
                        width,
                        height,
                        FILTER_ATLAS_PADDING,
                        Some(AtlasId(scratch_1_atlas_id.as_u32())),
                    )?;
                    Some([scratch_1, scratch_2])
                } else {
                    None
                };

                let encoded_paint = EncodedPaint::Image(EncodedImage {
                    source: ImageSource::OpaqueId {
                        id: dest_image_id,
                        may_have_opacities: true,
                    },
                    sampler: ImageSampler::new().with_quality(ImageQuality::Low),
                    may_have_opacities: true,
                    // Since filter layers are always shifted to start at (0, 0) relative to
                    // their bounding box, we need to "unshift" them when sampling.
                    transform: Affine::translate((
                        -(wtile_bbox.x0() as f64) * WideTile::WIDTH as f64,
                        -(wtile_bbox.y0() as f64) * Tile::HEIGHT as f64,
                    )),
                    x_advance: Vec2::new(1.0, 0.0),
                    y_advance: Vec2::new(0.0, 1.0),
                    tint: None,
                });

                let idx = encoded_paints.len();
                encoded_paints.push(encoded_paint);

                self.filter_textures.insert(
                    *layer_id,
                    FilterLayerData {
                        initial_image_id,
                        dest_image_id,
                        scratch_image_ids,
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

    /// Build the sequence of filter passes for a given layer and store it in the internal
    /// buffer.
    pub(crate) fn build_filter_passes(
        &self,
        state: &mut FilterPassState,
        layer_id: &LayerId,
        dest_image_cache: &ImageCache,
        get_filter_atlas_size: impl Fn(u32) -> [u32; 2],
        get_image_atlas_size: impl Fn() -> [u32; 2],
    ) {
        state.clear();

        // These unwraps can only panic for filter layers without a bbox, but those are skipped
        // anyway before even getting here, so unwrap should be safe here.
        // See also:
        // In `prepare`, nodes with an empty filter bbox are skipped and thus never inserted
        // here. For all other filter layers we do insert everything.
        // However, in `do_scene`, we skip empty nodes as well, and thus `build_filer_passes` is never
        // called on empty nodes.
        let filter_data_offset = self.offsets.get(layer_id).copied().unwrap();
        let gpu_filter = self.get_filter_data(layer_id).unwrap();
        let filter_type = gpu_filter.filter_type();
        let filter_textures = self.filter_textures.get(layer_id).unwrap();

        let initial_image = self
            .image_cache
            .get(filter_textures.initial_image_id)
            .unwrap();
        let dest_image = dest_image_cache.get(filter_textures.dest_image_id).unwrap();

        let initial_atlas_idx = initial_image.atlas_id.as_u32();
        let dest_atlas_idx = dest_image.atlas_id.as_u32();
        let main_atlas_size = get_image_atlas_size();

        // Short-circuit single-pass filters.
        if !gpu_filter.is_multi_pass() {
            let pass = match filter_type {
                filter_type::OFFSET => pass_kind::OFFSET,
                filter_type::FLOOD => pass_kind::FLOOD,
                // The above are the only single-pass filters currently implemented.
                _ => unimplemented!(),
            };

            state.push(
                FilterInstanceData {
                    src: IntRect::new(initial_image.offsets(), initial_image.size()),
                    dest: IntRect::new(dest_image.offsets(), dest_image.size()),
                    dest_atlas_size: IntSize(main_atlas_size),
                    filter_data_offset,
                    // Note that these two passes don't sample the original atlas, so we
                    // can pass anything here.
                    original: IntRect::new([0, 0], dest_image.size()),
                    pass_kind: pass,
                },
                FilterPass {
                    input_atlas_idx: initial_atlas_idx,
                    output: FilterPassTarget::MainAtlas(dest_atlas_idx),
                    original_atlas_idx: None,
                },
            );

            return;
        }

        // Otherwise, schedule the multi-pass filters.

        let scratch_ids = filter_textures.scratch_image_ids.unwrap();
        let scratch_resources = [
            self.image_cache.get(scratch_ids[0]).unwrap(),
            self.image_cache.get(scratch_ids[1]).unwrap(),
        ];

        state.sizer.reset(
            filter_textures.bbox.width_px(),
            filter_textures.bbox.height_px(),
        );

        let mut builder = BlurPassScheduler {
            state,
            initial: (initial_atlas_idx, IntOffset(initial_image.offsets())),
            dest: AtlasLocation {
                atlas_idx: dest_atlas_idx,
                offset: IntOffset(dest_image.offsets()),
                atlas_size: IntSize(main_atlas_size),
            },
            scratch: [
                AtlasLocation {
                    atlas_idx: scratch_resources[0].atlas_id.as_u32(),
                    offset: IntOffset(scratch_resources[0].offsets()),
                    atlas_size: IntSize(get_filter_atlas_size(
                        scratch_resources[0].atlas_id.as_u32(),
                    )),
                },
                AtlasLocation {
                    atlas_idx: scratch_resources[1].atlas_id.as_u32(),
                    offset: IntOffset(scratch_resources[1].offsets()),
                    atlas_size: IntSize(get_filter_atlas_size(
                        scratch_resources[1].atlas_id.as_u32(),
                    )),
                },
            ],
            filter_data_offset,
            original_size: IntSize([
                filter_textures.bbox.width_px() as u32,
                filter_textures.bbox.height_px() as u32,
            ]),
            toggle: 0,
            first: true,
        };

        match filter_type {
            filter_type::GAUSSIAN_BLUR => {
                let n_decimations = gpu_filter.n_decimations();

                builder.emit_blur_sequence(n_decimations, true);
            }
            filter_type::DROP_SHADOW => {
                let n_decimations = gpu_filter.n_decimations();

                builder.emit_to_scratch(pass_kind::OFFSET);
                builder.emit_blur_sequence(n_decimations, false);
                builder.emit_composite_to_dest(pass_kind::COMPOSITE_DROP_SHADOW);
            }
            // The above are the only supported multi-pass filters for now.
            _ => unimplemented!(),
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
    /// Some filters require intermediate buffers. This field optionally holds the IDs of
    /// two intermediate textures we can use for ping-pong rendering.
    pub scratch_image_ids: Option<[ImageId; 2]>,
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
