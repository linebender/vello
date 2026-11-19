// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::SegmentCount;

use super::{
    BinHeader, Clip, ClipBbox, ClipBic, ClipElement, DrawBbox, DrawMonoid, Layout, LineSoup, Path,
    PathBbox, PathMonoid, PathSegment, Tile,
};
use bytemuck::{Pod, Zeroable};

const TILE_WIDTH: u32 = 16;
const TILE_HEIGHT: u32 = 16;

// TODO: Obtain these from the vello_shaders crate
pub(crate) const PATH_REDUCE_WG: u32 = 256;
const PATH_BBOX_WG: u32 = 256;
const FLATTEN_WG: u32 = 256;
const CLIP_REDUCE_WG: u32 = 256;

/// Counters for tracking dynamic allocation on the GPU.
///
/// This must be kept in sync with the struct in `shader/shared/bump.wgsl`
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct BumpAllocators {
    pub failed: u32,
    // Final needed dynamic size of the buffers. If any of these are larger
    // than the corresponding `_size` element reallocation needs to occur.
    pub binning: u32,
    pub ptcl: u32,
    pub tile: u32,
    pub seg_counts: u32,
    pub segments: u32,
    pub blend: u32,
    pub lines: u32,
}

#[derive(Default)]
pub struct BumpAllocatorMemory {
    pub total: u32,
    pub binning: BufferSize<u32>,
    pub ptcl: BufferSize<u32>,
    pub tile: BufferSize<Tile>,
    pub seg_counts: BufferSize<SegmentCount>,
    pub segments: BufferSize<PathSegment>,
    pub lines: BufferSize<LineSoup>,
}

impl BumpAllocators {
    pub fn memory(&self) -> BumpAllocatorMemory {
        let binning = BufferSize::new(self.binning);
        let ptcl = BufferSize::new(self.ptcl);
        let tile = BufferSize::new(self.tile);
        let seg_counts = BufferSize::new(self.seg_counts);
        let segments = BufferSize::new(self.segments);
        let lines = BufferSize::new(self.lines);
        BumpAllocatorMemory {
            total: binning.size_in_bytes()
                + ptcl.size_in_bytes()
                + tile.size_in_bytes()
                + seg_counts.size_in_bytes()
                + segments.size_in_bytes()
                + lines.size_in_bytes(),
            binning,
            ptcl,
            tile,
            seg_counts,
            segments,
            lines,
        }
    }
}

impl std::fmt::Display for BumpAllocatorMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\n \
                 \tTotal:\t\t\t{} bytes ({:.2} KB | {:.2} MB)\n\
                 \tBinning\t\t\t{} elements ({} bytes)\n\
                 \tPTCL\t\t\t{} elements ({} bytes)\n\
                 \tTile:\t\t\t{} elements ({} bytes)\n\
                 \tSegment Counts:\t\t{} elements ({} bytes)\n\
                 \tSegments:\t\t{} elements ({} bytes)\n\
                 \tLines:\t\t\t{} elements ({} bytes)",
            self.total,
            self.total as f32 / (1 << 10) as f32,
            self.total as f32 / (1 << 20) as f32,
            self.binning.len(),
            self.binning.size_in_bytes(),
            self.ptcl.len(),
            self.ptcl.size_in_bytes(),
            self.tile.len(),
            self.tile.size_in_bytes(),
            self.seg_counts.len(),
            self.seg_counts.size_in_bytes(),
            self.segments.len(),
            self.segments.size_in_bytes(),
            self.lines.len(),
            self.lines.size_in_bytes()
        )
    }
}

/// Storage of indirect dispatch size values.
///
/// The original plan was to reuse [`BumpAllocators`], but the WebGPU compatible
/// usage list rules forbid that being used as indirect counts while also
/// bound as writable.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct IndirectCount {
    pub count_x: u32,
    pub count_y: u32,
    pub count_z: u32,
    pub pad0: u32,
}

/// Uniform render configuration data used by all GPU stages.
///
/// This data structure must be kept in sync with the definition in
/// `shaders/shared/config.wgsl`.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct ConfigUniform {
    /// Width of the scene in tiles.
    pub width_in_tiles: u32,
    /// Height of the scene in tiles.
    pub height_in_tiles: u32,
    /// Width of the target in pixels.
    pub target_width: u32,
    /// Height of the target in pixels.
    pub target_height: u32,
    /// The base background color applied to the target before any blends.
    pub base_color: u32,
    /// Layout of packed scene data.
    pub layout: Layout,
    /// Size of line soup buffer allocation (in [`LineSoup`]s)
    pub lines_size: u32,
    /// Size of binning buffer allocation (in `u32`s).
    pub binning_size: u32,
    /// Size of tile buffer allocation (in [`Tile`]s).
    pub tiles_size: u32,
    /// Size of segment count buffer allocation (in [`SegmentCount`]s).
    pub seg_counts_size: u32,
    /// Size of segment buffer allocation (in [`PathSegment`]s).
    pub segments_size: u32,
    /// Size of blend spill buffer (in `u32` pixels).
    // TODO: Maybe store in TILE_WIDTH * TILE_HEIGHT blocks of pixels instead?
    pub blend_size: u32,
    /// Size of per-tile command list buffer allocation (in `u32`s).
    pub ptcl_size: u32,
}

/// CPU side setup and configuration.
#[derive(Default)]
pub struct RenderConfig {
    /// GPU side configuration.
    pub gpu: ConfigUniform,
    /// Workgroup counts for all compute pipelines.
    pub workgroup_counts: WorkgroupCounts,
    /// Sizes of all buffer resources.
    pub buffer_sizes: BufferSizes,
}

impl RenderConfig {
    pub fn new(layout: &Layout, width: u32, height: u32, base_color: &peniko::Color) -> Self {
        let new_width = width.next_multiple_of(TILE_WIDTH);
        let new_height = height.next_multiple_of(TILE_HEIGHT);
        let width_in_tiles = new_width / TILE_WIDTH;
        let height_in_tiles = new_height / TILE_HEIGHT;
        let n_path_tags = layout.path_tags_size();
        let workgroup_counts =
            WorkgroupCounts::new(layout, width_in_tiles, height_in_tiles, n_path_tags);
        let buffer_sizes = BufferSizes::new(layout, &workgroup_counts);
        Self {
            gpu: ConfigUniform {
                width_in_tiles,
                height_in_tiles,
                target_width: width,
                target_height: height,
                base_color: base_color.premultiply().to_rgba8().to_u32(),
                lines_size: buffer_sizes.lines.len(),
                binning_size: buffer_sizes.bin_data.len() - layout.bin_data_start,
                tiles_size: buffer_sizes.tiles.len(),
                seg_counts_size: buffer_sizes.seg_counts.len(),
                segments_size: buffer_sizes.segments.len(),
                blend_size: buffer_sizes.blend_spill.len(),
                ptcl_size: buffer_sizes.ptcl.len(),
                layout: *layout,
            },
            workgroup_counts,
            buffer_sizes,
        }
    }
}

/// Type alias for a workgroup size.
pub type WorkgroupSize = (u32, u32, u32);

/// Computed sizes for all dispatches.
#[derive(Copy, Clone, Debug, Default)]
pub struct WorkgroupCounts {
    pub use_large_path_scan: bool,
    pub path_reduce: WorkgroupSize,
    pub path_reduce2: WorkgroupSize,
    pub path_scan1: WorkgroupSize,
    pub path_scan: WorkgroupSize,
    pub bbox_clear: WorkgroupSize,
    pub flatten: WorkgroupSize,
    pub draw_reduce: WorkgroupSize,
    pub draw_leaf: WorkgroupSize,
    pub clip_reduce: WorkgroupSize,
    pub clip_leaf: WorkgroupSize,
    pub binning: WorkgroupSize,
    pub tile_alloc: WorkgroupSize,
    pub path_count_setup: WorkgroupSize,
    // Note: `path_count` must use an indirect dispatch
    pub backdrop: WorkgroupSize,
    pub coarse: WorkgroupSize,
    pub path_tiling_setup: WorkgroupSize,
    // Note: `path_tiling` must use an indirect dispatch
    pub fine: WorkgroupSize,
}

impl WorkgroupCounts {
    pub fn new(
        layout: &Layout,
        width_in_tiles: u32,
        height_in_tiles: u32,
        n_path_tags: u32,
    ) -> Self {
        let n_paths = layout.n_paths;
        let n_draw_objects = layout.n_draw_objects;
        let n_clips = layout.n_clips;
        let path_tag_padded = align_up(n_path_tags, 4 * PATH_REDUCE_WG);
        let path_tag_wgs = path_tag_padded / (4 * PATH_REDUCE_WG);
        let use_large_path_scan = path_tag_wgs > PATH_REDUCE_WG;
        let reduced_size = if use_large_path_scan {
            align_up(path_tag_wgs, PATH_REDUCE_WG)
        } else {
            path_tag_wgs
        };
        let draw_object_wgs = n_draw_objects.div_ceil(PATH_BBOX_WG);
        let draw_monoid_wgs = draw_object_wgs.min(PATH_BBOX_WG);
        let flatten_wgs = n_path_tags.div_ceil(FLATTEN_WG);
        let clip_reduce_wgs = n_clips.saturating_sub(1) / CLIP_REDUCE_WG;
        let clip_wgs = n_clips.div_ceil(CLIP_REDUCE_WG);
        let path_wgs = n_paths.div_ceil(PATH_BBOX_WG);
        let width_in_bins = width_in_tiles.div_ceil(16);
        let height_in_bins = height_in_tiles.div_ceil(16);
        Self {
            use_large_path_scan,
            path_reduce: (path_tag_wgs, 1, 1),
            path_reduce2: (PATH_REDUCE_WG, 1, 1),
            path_scan1: (reduced_size / PATH_REDUCE_WG, 1, 1),
            path_scan: (path_tag_wgs, 1, 1),
            bbox_clear: (draw_object_wgs, 1, 1),
            flatten: (flatten_wgs, 1, 1),
            draw_reduce: (draw_monoid_wgs, 1, 1),
            draw_leaf: (draw_monoid_wgs, 1, 1),
            clip_reduce: (clip_reduce_wgs, 1, 1),
            clip_leaf: (clip_wgs, 1, 1),
            binning: (draw_object_wgs, 1, 1),
            tile_alloc: (path_wgs, 1, 1),
            path_count_setup: (1, 1, 1),
            backdrop: (path_wgs, 1, 1),
            coarse: (width_in_bins, height_in_bins, 1),
            path_tiling_setup: (1, 1, 1),
            fine: (width_in_tiles, height_in_tiles, 1),
        }
    }
}

/// Typed buffer size primitive.
#[derive(Copy, Clone, Eq, Default, Debug)]
pub struct BufferSize<T: Sized> {
    len: u32,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Sized> BufferSize<T> {
    /// Creates a new buffer size from number of elements.
    pub const fn new(len: u32) -> Self {
        Self {
            // Each buffer binding must be large enough to hold at least one element to avoid
            // triggering validation errors.
            //
            // Note: not using `Ord::max` here because it doesn't support const eval yet (except
            // in nightly)
            len: if len > 0 { len } else { 1 },
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new buffer size from size in bytes.
    pub const fn from_size_in_bytes(size: u32) -> Self {
        Self::new(size / size_of::<T>() as u32)
    }

    /// Returns the number of elements.
    #[expect(clippy::len_without_is_empty, reason = "The buffer can never be empty")]
    pub const fn len(self) -> u32 {
        self.len
    }

    /// Returns the size in bytes.
    pub const fn size_in_bytes(self) -> u32 {
        size_of::<T>() as u32 * self.len
    }

    /// Returns the size in bytes aligned up to the given value.
    pub const fn aligned_in_bytes(self, alignment: u32) -> u32 {
        align_up(self.size_in_bytes(), alignment)
    }
}

impl<T: Sized> PartialEq for BufferSize<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
    }
}

impl<T: Sized> PartialOrd for BufferSize<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.len.partial_cmp(&other.len)
    }
}

/// Computed sizes for all buffers.
#[derive(Copy, Clone, Debug, Default)]
pub struct BufferSizes {
    // Known size buffers
    pub path_reduced: BufferSize<PathMonoid>,
    pub path_reduced2: BufferSize<PathMonoid>,
    pub path_reduced_scan: BufferSize<PathMonoid>,
    pub path_monoids: BufferSize<PathMonoid>,
    pub path_bboxes: BufferSize<PathBbox>,
    pub draw_reduced: BufferSize<DrawMonoid>,
    pub draw_monoids: BufferSize<DrawMonoid>,
    pub info: BufferSize<u32>,
    pub clip_inps: BufferSize<Clip>,
    pub clip_els: BufferSize<ClipElement>,
    pub clip_bics: BufferSize<ClipBic>,
    pub clip_bboxes: BufferSize<ClipBbox>,
    pub draw_bboxes: BufferSize<DrawBbox>,
    pub bump_alloc: BufferSize<BumpAllocators>,
    pub indirect_count: BufferSize<IndirectCount>,
    pub bin_headers: BufferSize<BinHeader>,
    pub paths: BufferSize<Path>,
    // Bump allocated buffers
    pub lines: BufferSize<LineSoup>,
    pub bin_data: BufferSize<u32>,
    pub tiles: BufferSize<Tile>,
    pub seg_counts: BufferSize<SegmentCount>,
    pub segments: BufferSize<PathSegment>,
    pub blend_spill: BufferSize<u32>,
    pub ptcl: BufferSize<u32>,
}

impl BufferSizes {
    pub fn new(layout: &Layout, workgroups: &WorkgroupCounts) -> Self {
        let n_paths = layout.n_paths;
        let n_draw_objects = layout.n_draw_objects;
        let n_clips = layout.n_clips;
        let path_tag_wgs = workgroups.path_reduce.0;
        let reduced_size = if workgroups.use_large_path_scan {
            align_up(path_tag_wgs, PATH_REDUCE_WG)
        } else {
            path_tag_wgs
        };
        let path_reduced = BufferSize::new(reduced_size);
        let path_reduced2 = BufferSize::new(PATH_REDUCE_WG);
        let path_reduced_scan = BufferSize::new(reduced_size);
        let path_monoids = BufferSize::new(path_tag_wgs * PATH_REDUCE_WG);
        let path_bboxes = BufferSize::new(n_paths);
        let binning_wgs = workgroups.binning.0;
        let draw_monoid_wgs = workgroups.draw_reduce.0;
        let draw_reduced = BufferSize::new(draw_monoid_wgs);
        let draw_monoids = BufferSize::new(n_draw_objects);
        let info = BufferSize::new(layout.bin_data_start);
        let clip_inps = BufferSize::new(n_clips);
        let clip_els = BufferSize::new(n_clips);
        let clip_bics = BufferSize::new(n_clips / CLIP_REDUCE_WG);
        let clip_bboxes = BufferSize::new(n_clips);
        let draw_bboxes = BufferSize::new(n_paths);
        let bump_alloc = BufferSize::new(1);
        let indirect_count = BufferSize::new(1);
        let n_paths_aligned = align_up(n_paths, 256);
        let paths = BufferSize::new(n_paths_aligned);
        let width_in_bins = workgroups.coarse.0;
        let height_in_bins = workgroups.coarse.1;
        let n_bins = width_in_bins * height_in_bins;
        let aligned_n_bins = align_up(n_bins, 256);
        let bin_headers = BufferSize::new(binning_wgs * aligned_n_bins);

        // The bump-allocated buffers below are sized from the frame rather than
        // from fixed constants.
        //
        // Two things drive them, and a buffer needs whichever is larger. Tiling
        // work grows with the tile grid, because a path only costs the tiles it
        // covers. Flattening work grows with the scene, because `flatten` runs
        // before any tiling and turns each path segment into lines whatever the
        // target looks like. A dense drawing on a small canvas is bounded by the
        // second and a plain one on a large canvas by the first, so taking the
        // greater of the two covers both.
        //
        // `ptcl` is the exception, and the clearest case: its static part is not
        // an estimate at all, because `coarse` addresses every tile's command
        // list at `tile_ix * PTCL_INITIAL_ALLOC`, so that region has to be
        // present exactly and only the spill past it is dynamic.
        //
        // The multipliers are headroom. The floors matter as much: a small frame
        // saves little by being sized tightly and has the most room to be denser
        // than its size suggests, so nothing here falls below one.
        let n_tiles = workgroups.fine.0 * workgroups.fine.1;
        // What `path_reduce` was given, which is the padded path tag count and
        // so an upper bound on the segments `flatten` will see.
        let n_path_tags = path_tag_wgs * 4 * PATH_REDUCE_WG;
        // Every tile owns a command list of this many words before `coarse` has
        // to spill. Must be kept in sync with `PTCL_INITIAL_ALLOC` in
        // `shader/shared/ptcl.wgsl`.
        const PTCL_INITIAL_ALLOC: u32 = 64;
        /// Lines a path segment may flatten into before the buffer is the limit.
        /// Four covers the densest scene the old constants were picked for.
        const LINES_PER_TAG: u32 = 4;
        const TILE_HEADROOM: u32 = 16;
        const BIN_HEADROOM: u32 = 4;
        let from_scene = n_path_tags.saturating_mul(LINES_PER_TAG);
        let from_grid = n_tiles.saturating_mul(TILE_HEADROOM);
        let flattened = from_scene.max(from_grid);
        // Each buffer is clamped to the constant it used to be given, so that
        // this can only ever hand out less than before and never more. It is
        // what keeps a frame with a very large path count inside the device's
        // own binding limit, which the old constants were already sized under.
        fn sized<T: Sized>(derived: u32, floor: u32, was: u32) -> BufferSize<T> {
            BufferSize::new(derived.clamp(floor, was))
        }
        let bin_data = BufferSize::new(
            // `binning_size` is this buffer past `bin_data_start`, so the draw
            // data at its head is not headroom and has to be added on top.
            (layout.bin_data_start + n_tiles.saturating_mul(BIN_HEADROOM)).clamp(1 << 14, 1 << 18),
        );
        let tiles = sized(from_grid.max(n_paths.saturating_mul(16)), 1 << 16, 1 << 21);
        let lines = sized(flattened, 1 << 17, 1 << 21);
        let seg_counts = sized(flattened, 1 << 17, 1 << 21);
        let segments = sized(flattened, 1 << 17, 1 << 21);
        // 16 * 16 (1 << 8) is one blend spill, so this allows for one spill per
        // sixty-four tiles, and never fewer than sixty-four spills.
        let blend_spill =
            BufferSize::new(n_tiles.saturating_mul(BIN_HEADROOM).clamp(1 << 14, 1 << 20));
        // The per-tile command lists, which must be present in full, plus as much
        // again for what `coarse` spills past them.
        let ptcl = BufferSize::new(
            n_tiles
                .saturating_mul(PTCL_INITIAL_ALLOC * 2)
                .clamp(1 << 16, 1 << 23),
        );
        Self {
            path_reduced,
            path_reduced2,
            path_reduced_scan,
            path_monoids,
            path_bboxes,
            draw_reduced,
            draw_monoids,
            info,
            clip_inps,
            clip_els,
            clip_bics,
            clip_bboxes,
            draw_bboxes,
            bump_alloc,
            indirect_count,
            lines,
            bin_headers,
            paths,
            bin_data,
            tiles,
            seg_counts,
            segments,
            blend_spill,
            ptcl,
        }
    }
}

const fn align_up(len: u32, alignment: u32) -> u32 {
    len + (len.wrapping_neg() & (alignment - 1))
}
