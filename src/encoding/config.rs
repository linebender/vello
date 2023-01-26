use super::{
    BinHeader, ClipBbox, ClipBic, ClipElement, Clip, Cubic, DrawBbox, DrawMonoid, Layout,
    PackedEncoding, Path, PathBbox, PathMonoid, PathSegment, Tile,
};
use bytemuck::{Pod, Zeroable};
use std::mem;

/// Counters for tracking dynamic allocation on the GPU.
///
/// This must be kept in sync with the struct in shader/shared/bump.wgsl
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct BumpAllocators {
    pub failed: u32,
    // Final needed dynamic size of the buffers. If any of these are larger
    // than the corresponding `_size` element reallocation needs to occur.
    pub binning: u32,
    pub ptcl: u32,
    pub tile: u32,
    pub segments: u32,
    pub blend: u32,
}

/// GPU side configuration.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct GpuConfig {
    /// Width of the scene in tiles.
    pub width_in_tiles: u32,
    /// Height of the scene in tiles.
    pub height_in_tiles: u32,
    /// Width of the target in pixels.
    pub target_width: u32,
    /// Height of the target in pixels.
    pub target_height: u32,
    /// Layout of packed scene data.
    pub layout: Layout,
    /// Size of binning buffer allocation (in u32s).
    pub binning_size: u32,
    /// Size of tile buffer allocation (in Tiles).
    pub tiles_size: u32,
    /// Size of segment buffer allocation (in PathSegments).
    pub segments_size: u32,
    /// Size of per-tile command list buffer allocation (in u32s).
    pub ptcl_size: u32,
}

/// CPU side setup and configuration.
#[derive(Default)]
pub struct CpuConfig {
    /// GPU side configuration.
    pub gpu: GpuConfig,
    /// Workgroup sizes for all compute pipelines.
    pub workgroup_sizes: WorkgroupSizes,
    /// Sizes of all buffer resources.
    pub buffer_sizes: BufferSizes,
}

impl CpuConfig {
    pub fn new(encoding: &PackedEncoding, width: u32, height: u32) -> Self {
        let new_width = next_multiple_of(width, 16);
        let new_height = next_multiple_of(height, 16);
        let mut config = GpuConfig {
            width_in_tiles: new_width / 16,
            height_in_tiles: new_height / 16,
            target_width: width,
            target_height: height,
            binning_size: 0,
            tiles_size: 0,
            segments_size: 0,
            ptcl_size: 0,
            layout: encoding.layout,
        };
        let n_path_tags = encoding.path_tags().len() as u32;
        let workgroup_sizes = WorkgroupSizes::new(&config, n_path_tags);
        let buffer_sizes = BufferSizes::new(&config, &workgroup_sizes, n_path_tags);
        config.binning_size = buffer_sizes.bin_data.len();
        config.tiles_size = buffer_sizes.tiles.len();
        config.segments_size = buffer_sizes.tiles.len();
        config.ptcl_size = buffer_sizes.tiles.len();
        Self {
            gpu: config,
            workgroup_sizes,
            buffer_sizes,
        }
    }
}

const PATH_REDUCE_WG: u32 = 256;
const PATH_BBOX_WG: u32 = 256;
const PATH_COARSE_WG: u32 = 256;
const CLIP_REDUCE_WG: u32 = 256;

/// Type alias for a workgroup size.
pub type WorkgroupSize = (u32, u32, u32);

/// Computed sizes for all dispatches.
#[derive(Copy, Clone, Debug, Default)]
pub struct WorkgroupSizes {
    pub use_large_path_scan: bool,
    pub path_reduce: WorkgroupSize,
    pub path_reduce2: WorkgroupSize,
    pub path_scan1: WorkgroupSize,
    pub path_scan: WorkgroupSize,
    pub bbox_clear: WorkgroupSize,
    pub path_seg: WorkgroupSize,
    pub draw_reduce: WorkgroupSize,
    pub draw_leaf: WorkgroupSize,
    pub clip_reduce: WorkgroupSize,
    pub clip_leaf: WorkgroupSize,
    pub binning: WorkgroupSize,
    pub tile_alloc: WorkgroupSize,
    pub path_coarse: WorkgroupSize,
    pub backdrop: WorkgroupSize,
    pub coarse: WorkgroupSize,
    pub fine: WorkgroupSize,
}

impl WorkgroupSizes {
    pub fn new(config: &GpuConfig, n_path_tags: u32) -> Self {
        let n_paths = config.layout.n_paths;
        let n_draw_objects = config.layout.n_draw_objects;
        let n_clips = config.layout.n_clips;
        let path_tag_padded = align_up(n_path_tags, 4 * PATH_REDUCE_WG);
        let path_tag_wgs = path_tag_padded / (4 * PATH_REDUCE_WG);
        let use_large_path_scan = path_tag_wgs > PATH_REDUCE_WG;
        let path_reduce_wgs = if use_large_path_scan {
            align_up(path_tag_wgs, PATH_REDUCE_WG)
        } else {
            path_tag_wgs
        };
        let draw_object_wgs = (n_draw_objects + PATH_BBOX_WG - 1) / PATH_BBOX_WG;
        let path_coarse_wgs = (n_path_tags + PATH_COARSE_WG - 1) / PATH_COARSE_WG;
        let clip_reduce_wgs = n_clips.saturating_sub(1) / CLIP_REDUCE_WG;
        let clip_wgs = (n_clips + CLIP_REDUCE_WG - 1) / CLIP_REDUCE_WG;
        let path_wgs = (n_paths + PATH_BBOX_WG - 1) / PATH_BBOX_WG;
        let width_in_bins = (config.width_in_tiles + 15) / 16;
        let height_in_bins = (config.height_in_tiles + 15) / 16;
        Self {
            use_large_path_scan,
            path_reduce: (path_reduce_wgs, 1, 1),
            path_reduce2: (PATH_REDUCE_WG, 1, 1),
            path_scan1: (path_reduce_wgs / PATH_REDUCE_WG, 1, 1),
            path_scan: (path_tag_wgs, 1, 1),
            bbox_clear: (draw_object_wgs, 1, 1),
            path_seg: (path_coarse_wgs, 1, 1),
            draw_reduce: (draw_object_wgs, 1, 1),
            draw_leaf: (draw_object_wgs, 1, 1),
            clip_reduce: (clip_reduce_wgs, 1, 1),
            clip_leaf: (clip_wgs, 1, 1),
            binning: (draw_object_wgs, 1, 1),
            tile_alloc: (path_wgs, 1, 1),
            path_coarse: (path_coarse_wgs, 1, 1),
            backdrop: (path_wgs, 1, 1),
            coarse: (width_in_bins, height_in_bins, 1),
            fine: (config.width_in_tiles, config.height_in_tiles, 1),
        }
    }
}

/// Typed buffer size primitive.
#[derive(Copy, Clone, Eq, Ord, Default, Debug)]
pub struct BufferSize<T: Sized> {
    len: u32,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Sized> BufferSize<T> {
    /// Creates a new buffer size from number of elements.
    pub const fn new(len: u32) -> Self {
        Self {
            len,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new buffer size from size in bytes.
    pub const fn from_size_in_bytes(size: u32) -> Self {
        Self::new(size / mem::size_of::<T>() as u32)
    }

    /// Returns the number of elements.
    pub const fn len(self) -> u32 {
        self.len
    }

    /// Returns the size in bytes.
    pub const fn size_in_bytes(self) -> u32 {
        mem::size_of::<T>() as u32 * self.len
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
    pub cubics: BufferSize<Cubic>,
    pub draw_reduced: BufferSize<DrawMonoid>,
    pub draw_monoids: BufferSize<DrawMonoid>,
    pub info: BufferSize<u32>,
    pub clip_inps: BufferSize<Clip>,
    pub clip_els: BufferSize<ClipElement>,
    pub clip_bics: BufferSize<ClipBic>,
    pub clip_bboxes: BufferSize<ClipBbox>,
    pub draw_bboxes: BufferSize<DrawBbox>,
    pub bin_headers: BufferSize<BinHeader>,
    pub paths: BufferSize<Path>,
    // Bump allocated buffers
    pub bin_data: BufferSize<u32>,
    pub tiles: BufferSize<Tile>,
    pub segments: BufferSize<PathSegment>,
    pub ptcl: BufferSize<u32>,
}

impl BufferSizes {
    pub fn new(config: &GpuConfig, workgroups: &WorkgroupSizes, n_path_tags: u32) -> Self {
        let n_paths = config.layout.n_paths;
        let n_draw_objects = config.layout.n_draw_objects;
        let n_clips = config.layout.n_clips;
        let path_tag_wgs = workgroups.path_reduce.0;
        let path_reduced = BufferSize::new(path_tag_wgs);
        let path_reduced2 = BufferSize::new(PATH_REDUCE_WG);
        let path_reduced_scan = BufferSize::new(path_tag_wgs);
        let path_monoids = BufferSize::new(path_tag_wgs * PATH_REDUCE_WG);
        let path_bboxes = BufferSize::new(n_paths);
        let cubics = BufferSize::new(n_path_tags);
        let draw_object_wgs = workgroups.draw_reduce.0;
        let draw_reduced = BufferSize::new(draw_object_wgs);
        let draw_monoids = BufferSize::new(n_draw_objects);
        let info = BufferSize::new(config.layout.bin_data_start);
        let clip_inps = BufferSize::new(n_clips);
        let clip_els = BufferSize::new(n_clips);
        let clip_bics = BufferSize::new(n_clips / CLIP_REDUCE_WG);
        let clip_bboxes = BufferSize::new(n_clips);
        let draw_bboxes = BufferSize::new(n_paths);
        let bin_headers = BufferSize::new(draw_object_wgs * 256);
        let n_paths_aligned = align_up(n_paths, 256);
        let paths = BufferSize::new(n_paths_aligned);
        // TODO: better heuristics. Just use 128k for now
        let initial_bump_size = 128 * 1024;
        let bin_data = BufferSize::from_size_in_bytes(initial_bump_size);
        let tiles = BufferSize::from_size_in_bytes(initial_bump_size);
        let segments = BufferSize::from_size_in_bytes(initial_bump_size);
        let ptcl = BufferSize::from_size_in_bytes(initial_bump_size);
        Self {
            path_reduced,
            path_reduced2,
            path_reduced_scan,
            path_monoids,
            path_bboxes,
            cubics,
            draw_reduced,
            draw_monoids,
            info,
            clip_inps,
            clip_els,
            clip_bics,
            clip_bboxes,
            draw_bboxes,
            bin_headers,
            paths,
            bin_data,
            tiles,
            segments,
            ptcl,
        }
    }
}

const fn align_up(len: u32, alignment: u32) -> u32 {
    len + (len.wrapping_neg() & (alignment - 1))
}

const fn next_multiple_of(val: u32, rhs: u32) -> u32 {
    match val % rhs {
        0 => val,
        r => val + (rhs - r),
    }
}
