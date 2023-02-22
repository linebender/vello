//! Take an encoded scene and create a graph to render it

use bytemuck::{Pod, Zeroable};

use crate::{
    encoding::Encoding,
    engine::{BufProxy, ImageFormat, ImageProxy, Recording, ResourceProxy},
    mask::make_mask_lut,
    shaders::{self, FullShaders, Shaders},
    Scene,
};

/// State for a render in progress.
pub struct Render {
    /// Size of binning and info combined buffer in u32 units
    binning_info_size: u32,
    /// Size of tiles buf in tiles
    tiles_size: u32,
    /// Size of segments buf in segments
    segments_size: u32,
    /// Size of per-tile command list in u32 units
    ptcl_size: u32,
    width_in_tiles: u32,
    height_in_tiles: u32,
    fine: Option<FineResources>,
    mask_buf: Option<ResourceProxy>,
}

/// Resources produced by pipeline, needed for fine rasterization.
struct FineResources {
    config_buf: ResourceProxy,
    bump_buf: ResourceProxy,
    tile_buf: ResourceProxy,
    segments_buf: ResourceProxy,
    ptcl_buf: ResourceProxy,
    gradient_image: ResourceProxy,
    info_bin_data_buf: ResourceProxy,

    out_image: ImageProxy,
}

const TAG_MONOID_SIZE: u64 = 12;
const TAG_MONOID_FULL_SIZE: u64 = 20;
const PATH_BBOX_SIZE: u64 = 24;
const CUBIC_SIZE: u64 = 48;
const DRAWMONOID_SIZE: u64 = 16;
const MAX_DRAWINFO_SIZE: u64 = 44;
const CLIP_BIC_SIZE: u64 = 8;
const CLIP_EL_SIZE: u64 = 32;
const CLIP_INP_SIZE: u64 = 8;
const CLIP_BBOX_SIZE: u64 = 16;
const PATH_SIZE: u64 = 32;
const DRAW_BBOX_SIZE: u64 = 16;
const BUMP_SIZE: u64 = std::mem::size_of::<BumpAllocators>() as u64;
const BIN_HEADER_SIZE: u64 = 8;
const TILE_SIZE: u64 = 8;
const SEGMENT_SIZE: u64 = 24;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct Config {
    width_in_tiles: u32,
    height_in_tiles: u32,
    target_width: u32,
    target_height: u32,
    n_drawobj: u32,
    n_path: u32,
    n_clip: u32,
    bin_data_start: u32,
    pathtag_base: u32,
    pathdata_base: u32,
    drawtag_base: u32,
    drawdata_base: u32,
    transform_base: u32,
    linewidth_base: u32,
}

fn size_to_words(byte_size: usize) -> u32 {
    (byte_size / std::mem::size_of::<u32>()) as u32
}

pub const fn next_multiple_of(val: u32, rhs: u32) -> u32 {
    match val % rhs {
        0 => val,
        r => val + (rhs - r),
    }
}

// This must be kept in sync with the struct in shader/shared/bump.wgsl
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct BumpAllocators {
    failed: u32,
    // Final needed dynamic size of the buffers. If any of these are larger than the corresponding `_size` element
    // reallocation needs to occur
    binning: u32,
    ptcl: u32,
    tile: u32,
    segments: u32,
    blend: u32,
}

#[allow(unused)]
fn render(scene: &Scene, shaders: &Shaders) -> (Recording, BufProxy) {
    let mut recording = Recording::default();
    let data = scene.data();
    let n_pathtag = data.path_tags.len();
    let pathtag_padded = align_up(n_pathtag, 4 * shaders::PATHTAG_REDUCE_WG);
    let pathtag_wgs = pathtag_padded / (4 * shaders::PATHTAG_REDUCE_WG as usize);
    let mut scene: Vec<u8> = Vec::with_capacity(pathtag_padded);
    let pathtag_base = size_to_words(scene.len());
    scene.extend(bytemuck::cast_slice(&data.path_tags));
    scene.resize(pathtag_padded, 0);
    let pathdata_base = size_to_words(scene.len());
    scene.extend(&data.path_data);

    let config = Config {
        width_in_tiles: 64,
        height_in_tiles: 64,
        target_width: 64 * 16,
        target_height: 64 * 16,
        pathtag_base,
        pathdata_base,
        ..Default::default()
    };
    let scene_buf = recording.upload("scene", scene);
    let config_buf = recording.upload_uniform("config", bytemuck::bytes_of(&config));

    let reduced_buf = BufProxy::new(pathtag_wgs as u64 * TAG_MONOID_SIZE, "reduced_buf");
    // TODO: really only need pathtag_wgs - 1
    recording.dispatch(
        shaders.pathtag_reduce,
        (pathtag_wgs as u32, 1, 1),
        [config_buf, scene_buf, reduced_buf],
    );

    let tagmonoid_buf = BufProxy::new(
        pathtag_wgs as u64 * shaders::PATHTAG_REDUCE_WG as u64 * TAG_MONOID_SIZE,
        "tagmonoid_buf",
    );
    recording.dispatch(
        shaders.pathtag_scan,
        (pathtag_wgs as u32, 1, 1),
        [config_buf, scene_buf, reduced_buf, tagmonoid_buf],
    );

    let path_coarse_wgs =
        (n_pathtag as u32 + shaders::PATH_COARSE_WG - 1) / shaders::PATH_COARSE_WG;
    // TODO: more principled size calc
    let tiles_buf = BufProxy::new(4097 * 8, "tiles_buf");
    let segments_buf = BufProxy::new(256 * 24, "segments_buf");
    recording.clear_all(tiles_buf);
    recording.dispatch(
        shaders.path_coarse,
        (path_coarse_wgs, 1, 1),
        [
            config_buf,
            scene_buf,
            tagmonoid_buf,
            tiles_buf,
            segments_buf,
        ],
    );
    recording.dispatch(
        shaders.backdrop,
        (config.height_in_tiles, 1, 1),
        [config_buf, tiles_buf],
    );
    let out_buf_size = config.width_in_tiles * config.height_in_tiles * 256;
    let out_buf = BufProxy::new(out_buf_size as u64, "out_buf");
    recording.dispatch(
        shaders.fine,
        (config.width_in_tiles, config.height_in_tiles, 1),
        [config_buf, tiles_buf, segments_buf, out_buf],
    );

    recording.download(out_buf);
    (recording, out_buf)
}

pub fn render_full(
    scene: &Scene,
    shaders: &FullShaders,
    width: u32,
    height: u32,
) -> (Recording, ResourceProxy) {
    render_encoding_full(scene.data(), shaders, width, height)
}

/// Create a single recording with both coarse and fine render stages.
///
/// This function is not recommended when the scene can be complex, as it does not
/// implement robust dynamic memory.
pub fn render_encoding_full(
    encoding: &Encoding,
    shaders: &FullShaders,
    width: u32,
    height: u32,
) -> (Recording, ResourceProxy) {
    let mut render = Render::new();
    let mut recording = render.render_encoding_coarse(encoding, shaders, width, height, false);
    let out_image = render.out_image();
    render.record_fine(shaders, &mut recording);
    (recording, out_image.into())
}

pub fn align_up(len: usize, alignment: u32) -> usize {
    len + (len.wrapping_neg() & (alignment as usize - 1))
}

impl Render {
    pub fn new() -> Self {
        // These sizes are adequate for paris-30k but should probably be dialed down.
        Render {
            binning_info_size: (1 << 20) / 4,
            tiles_size: (1 << 24) / TILE_SIZE as u32,
            segments_size: (1 << 26) / SEGMENT_SIZE as u32,
            ptcl_size: (1 << 26) / 4 as u32,
            width_in_tiles: 0,
            height_in_tiles: 0,
            fine: None,
            mask_buf: None,
        }
    }

    /// Prepare a recording for the coarse rasterization phase.
    ///
    /// The `robust` parameter controls whether we're preparing for readback
    /// of the atomic bump buffer, for robust dynamic memory.
    pub fn render_encoding_coarse(
        &mut self,
        encoding: &Encoding,
        shaders: &FullShaders,
        width: u32,
        height: u32,
        robust: bool,
    ) -> Recording {
        use crate::encoding::{resource::ResourceCache, PackedEncoding};
        let mut recording = Recording::default();
        let mut resources = ResourceCache::new();
        let mut packed = PackedEncoding::default();
        packed.pack(encoding, &mut resources);
        let (ramp_data, ramps_width, ramps_height) = resources.ramps(packed.resources).unwrap();
        let gradient_image = if encoding.patches.is_empty() {
            ResourceProxy::new_image(1, 1, ImageFormat::Rgba8)
        } else {
            let data: &[u8] = bytemuck::cast_slice(ramp_data);
            ResourceProxy::Image(recording.upload_image(
                ramps_width,
                ramps_height,
                ImageFormat::Rgba8,
                data,
            ))
        };
        // TODO: calculate for real when we do rectangles
        let n_pathtag = encoding.path_tags.len();
        let pathtag_padded = align_up(encoding.path_tags.len(), 4 * shaders::PATHTAG_REDUCE_WG);
        let n_paths = encoding.n_paths;
        let n_drawobj = n_paths;
        let n_clip = encoding.n_clips;

        let new_width = next_multiple_of(width, 16);
        let new_height = next_multiple_of(height, 16);

        let info_size = packed.layout.bin_data_start;
        let config = crate::encoding::Config {
            width_in_tiles: new_width / 16,
            height_in_tiles: new_height / 16,
            target_width: width,
            target_height: height,
            binning_size: self.binning_info_size - info_size,
            tiles_size: self.tiles_size,
            segments_size: self.segments_size,
            ptcl_size: self.ptcl_size,
            layout: packed.layout,
        };
        // println!("{:?}", config);
        let scene_buf = ResourceProxy::Buf(recording.upload("scene", packed.data));
        let config_buf =
            ResourceProxy::Buf(recording.upload_uniform("config", bytemuck::bytes_of(&config)));
        let info_bin_data_buf = ResourceProxy::new_buf(
            (info_size + config.binning_size) as u64 * 4,
            "info_bin_data_buf",
        );
        let tile_buf = ResourceProxy::new_buf(config.tiles_size as u64 * TILE_SIZE, "tile_buf");
        let segments_buf =
            ResourceProxy::new_buf(config.segments_size as u64 * SEGMENT_SIZE, "segments_buf");
        let ptcl_buf = ResourceProxy::new_buf(config.ptcl_size as u64 * 4, "ptcl_buf");

        let pathtag_wgs = pathtag_padded / (4 * shaders::PATHTAG_REDUCE_WG as usize);
        let pathtag_large = pathtag_wgs > shaders::PATHTAG_REDUCE_WG as usize;
        let reduced_size = if pathtag_large {
            align_up(pathtag_wgs, shaders::PATHTAG_REDUCE_WG)
        } else {
            pathtag_wgs
        };
        let reduced_buf =
            ResourceProxy::new_buf(reduced_size as u64 * TAG_MONOID_FULL_SIZE, "reduced_buf");
        // TODO: really only need pathtag_wgs - 1
        recording.dispatch(
            shaders.pathtag_reduce,
            (pathtag_wgs as u32, 1, 1),
            [config_buf, scene_buf, reduced_buf],
        );
        let mut pathtag_parent = reduced_buf;
        let mut large_pathtag_bufs = None;
        if pathtag_large {
            let reduced2_size = shaders::PATHTAG_REDUCE_WG as usize;
            let reduced2_buf =
                ResourceProxy::new_buf(reduced2_size as u64 * TAG_MONOID_FULL_SIZE, "reduced2_buf");
            recording.dispatch(
                shaders.pathtag_reduce2,
                (reduced2_size as u32, 1, 1),
                [reduced_buf, reduced2_buf],
            );
            let reduced_scan_buf = ResourceProxy::new_buf(
                pathtag_wgs as u64 * TAG_MONOID_FULL_SIZE,
                "reduced_scan_buf",
            );
            recording.dispatch(
                shaders.pathtag_scan1,
                (reduced_size as u32 / shaders::PATHTAG_REDUCE_WG, 1, 1),
                [reduced_buf, reduced2_buf, reduced_scan_buf],
            );
            pathtag_parent = reduced_scan_buf;
            large_pathtag_bufs = Some((reduced2_buf, reduced_scan_buf));
        }

        let tagmonoid_buf = ResourceProxy::new_buf(
            pathtag_wgs as u64 * shaders::PATHTAG_REDUCE_WG as u64 * TAG_MONOID_FULL_SIZE,
            "tagmonoid_buf",
        );
        let pathtag_scan = if pathtag_large {
            shaders.pathtag_scan_large
        } else {
            shaders.pathtag_scan
        };
        recording.dispatch(
            pathtag_scan,
            (pathtag_wgs as u32, 1, 1),
            [config_buf, scene_buf, pathtag_parent, tagmonoid_buf],
        );
        recording.free_resource(reduced_buf);
        if let Some((reduced2, reduced_scan)) = large_pathtag_bufs {
            recording.free_resource(reduced2);
            recording.free_resource(reduced_scan);
        }
        let drawobj_wgs = (n_drawobj + shaders::PATH_BBOX_WG - 1) / shaders::PATH_BBOX_WG;
        let path_bbox_buf =
            ResourceProxy::new_buf(n_paths as u64 * PATH_BBOX_SIZE, "path_bbox_buf");
        recording.dispatch(
            shaders.bbox_clear,
            (drawobj_wgs, 1, 1),
            [config_buf, path_bbox_buf],
        );
        let cubic_buf = ResourceProxy::new_buf(n_pathtag as u64 * CUBIC_SIZE, "cubic_buf");
        let path_coarse_wgs =
            (n_pathtag as u32 + shaders::PATH_COARSE_WG - 1) / shaders::PATH_COARSE_WG;
        recording.dispatch(
            shaders.pathseg,
            (path_coarse_wgs, 1, 1),
            [
                config_buf,
                scene_buf,
                tagmonoid_buf,
                path_bbox_buf,
                cubic_buf,
            ],
        );
        let draw_reduced_buf =
            ResourceProxy::new_buf(drawobj_wgs as u64 * DRAWMONOID_SIZE, "draw_reduced_buf");
        recording.dispatch(
            shaders.draw_reduce,
            (drawobj_wgs, 1, 1),
            [config_buf, scene_buf, draw_reduced_buf],
        );
        let draw_monoid_buf =
            ResourceProxy::new_buf(n_drawobj as u64 * DRAWMONOID_SIZE, "draw_monoid_buf");
        let clip_inp_buf =
            ResourceProxy::new_buf(encoding.n_clips as u64 * CLIP_INP_SIZE, "clip_inp_buf");
        recording.dispatch(
            shaders.draw_leaf,
            (drawobj_wgs, 1, 1),
            [
                config_buf,
                scene_buf,
                draw_reduced_buf,
                path_bbox_buf,
                draw_monoid_buf,
                info_bin_data_buf,
                clip_inp_buf,
            ],
        );
        recording.free_resource(draw_reduced_buf);
        let clip_el_buf =
            ResourceProxy::new_buf(encoding.n_clips as u64 * CLIP_EL_SIZE, "clip_el_buf");
        let clip_bic_buf = ResourceProxy::new_buf(
            (n_clip / shaders::CLIP_REDUCE_WG) as u64 * CLIP_BIC_SIZE,
            "clip_bic_buf",
        );
        let clip_wg_reduce = n_clip.saturating_sub(1) / shaders::CLIP_REDUCE_WG;
        if clip_wg_reduce > 0 {
            recording.dispatch(
                shaders.clip_reduce,
                (clip_wg_reduce, 1, 1),
                [
                    config_buf,
                    clip_inp_buf,
                    path_bbox_buf,
                    clip_bic_buf,
                    clip_el_buf,
                ],
            );
        }
        let clip_wg = (n_clip + shaders::CLIP_REDUCE_WG - 1) / shaders::CLIP_REDUCE_WG;
        let clip_bbox_buf = ResourceProxy::new_buf(n_clip as u64 * CLIP_BBOX_SIZE, "clip_bbox_buf");
        if clip_wg > 0 {
            recording.dispatch(
                shaders.clip_leaf,
                (clip_wg, 1, 1),
                [
                    config_buf,
                    clip_inp_buf,
                    path_bbox_buf,
                    clip_bic_buf,
                    clip_el_buf,
                    draw_monoid_buf,
                    clip_bbox_buf,
                ],
            );
        }
        recording.free_resource(clip_inp_buf);
        recording.free_resource(clip_bic_buf);
        recording.free_resource(clip_el_buf);
        let draw_bbox_buf =
            ResourceProxy::new_buf(n_paths as u64 * DRAW_BBOX_SIZE, "draw_bbox_buf");
        let bump_buf = BufProxy::new(BUMP_SIZE, "bump_buf");
        let width_in_bins = (config.width_in_tiles + 15) / 16;
        let height_in_bins = (config.height_in_tiles + 15) / 16;
        let bin_header_buf = ResourceProxy::new_buf(
            (256 * drawobj_wgs) as u64 * BIN_HEADER_SIZE,
            "bin_header_buf",
        );
        recording.clear_all(bump_buf);
        let bump_buf = ResourceProxy::Buf(bump_buf);
        recording.dispatch(
            shaders.binning,
            (drawobj_wgs, 1, 1),
            [
                config_buf,
                draw_monoid_buf,
                path_bbox_buf,
                clip_bbox_buf,
                draw_bbox_buf,
                bump_buf,
                info_bin_data_buf,
                bin_header_buf,
            ],
        );
        recording.free_resource(draw_monoid_buf);
        recording.free_resource(path_bbox_buf);
        recording.free_resource(clip_bbox_buf);
        // Note: this only needs to be rounded up because of the workaround to store the tile_offset
        // in storage rather than workgroup memory.
        let n_path_aligned = align_up(n_paths as usize, 256);
        let path_buf = ResourceProxy::new_buf(n_path_aligned as u64 * PATH_SIZE, "path_buf");
        let path_wgs = (n_paths + shaders::PATH_BBOX_WG - 1) / shaders::PATH_BBOX_WG;
        recording.dispatch(
            shaders.tile_alloc,
            (path_wgs, 1, 1),
            [
                config_buf,
                scene_buf,
                draw_bbox_buf,
                bump_buf,
                path_buf,
                tile_buf,
            ],
        );
        recording.free_resource(draw_bbox_buf);
        recording.dispatch(
            shaders.path_coarse,
            (path_coarse_wgs, 1, 1),
            [
                config_buf,
                scene_buf,
                tagmonoid_buf,
                cubic_buf,
                path_buf,
                bump_buf,
                tile_buf,
                segments_buf,
            ],
        );
        recording.free_resource(tagmonoid_buf);
        recording.free_resource(cubic_buf);
        recording.dispatch(
            shaders.backdrop,
            (path_wgs, 1, 1),
            [config_buf, path_buf, tile_buf],
        );
        recording.dispatch(
            shaders.coarse,
            (width_in_bins, height_in_bins, 1),
            [
                config_buf,
                scene_buf,
                draw_monoid_buf,
                bin_header_buf,
                info_bin_data_buf,
                path_buf,
                tile_buf,
                bump_buf,
                ptcl_buf,
                segments_buf,
            ],
        );
        recording.free_resource(scene_buf);
        recording.free_resource(draw_monoid_buf);
        recording.free_resource(bin_header_buf);
        recording.free_resource(path_buf);
        let out_image = ImageProxy::new(width, height, ImageFormat::Rgba8);
        self.width_in_tiles = config.width_in_tiles;
        self.height_in_tiles = config.height_in_tiles;
        self.fine = Some(FineResources {
            config_buf,
            bump_buf,
            tile_buf,
            segments_buf,
            ptcl_buf,
            gradient_image,
            info_bin_data_buf,
            out_image,
        });
        if robust {
            recording.download(*bump_buf.as_buf().unwrap());
        }
        recording.free_resource(bump_buf);
        recording
    }

    /// Run fine rasterization assuming the coarse phase succeeded.
    pub fn record_fine(&mut self, shaders: &FullShaders, recording: &mut Recording) {
        let fine = self.fine.take().unwrap();
        if self.mask_buf.is_none() {
            let mask_lut = make_mask_lut();
            let buf = recording.upload("mask lut", mask_lut);
            // TODO: the `Render` struct is not persistent, so this reuploads
            // and leaks.
            self.mask_buf = Some(buf.into());
        }
        recording.dispatch(
            shaders.fine,
            (self.width_in_tiles, self.height_in_tiles, 1),
            [
                fine.config_buf,
                fine.tile_buf,
                fine.segments_buf,
                ResourceProxy::Image(fine.out_image),
                fine.ptcl_buf,
                fine.gradient_image,
                fine.info_bin_data_buf,
                self.mask_buf.unwrap(),
            ],
        );
        recording.free_resource(fine.config_buf);
        recording.free_resource(fine.tile_buf);
        recording.free_resource(fine.segments_buf);
        recording.free_resource(fine.ptcl_buf);
        recording.free_resource(fine.gradient_image);
        recording.free_resource(fine.info_bin_data_buf);
    }

    /// Get the output image.
    ///
    /// This is going away, as the caller will add the output image to the bind
    /// map.
    pub fn out_image(&self) -> ImageProxy {
        self.fine.as_ref().unwrap().out_image
    }

    pub fn bump_buf(&self) -> BufProxy {
        *self.fine.as_ref().unwrap().bump_buf.as_buf().unwrap()
    }
}
