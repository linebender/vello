//! Take an encoded scene and create a graph to render it

use bytemuck::{Pod, Zeroable};
use piet_scene::Scene;

use crate::{
    engine::{BufProxy, ImageFormat, ImageProxy, Recording, ResourceProxy},
    shaders::{self, FullShaders, Shaders},
};

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
const BUMP_SIZE: u64 = 16;
const BIN_HEADER_SIZE: u64 = 8;

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
    pathtag_base: u32,
    pathdata_base: u32,
    drawtag_base: u32,
    drawdata_base: u32,
    transform_base: u32,
    linewidth_base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct PathSegment {
    origin: [f32; 2],
    delta: [f32; 2],
    y_edge: f32,
    next: u32,
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

#[allow(unused)]
fn render(scene: &Scene, shaders: &Shaders) -> (Recording, BufProxy) {
    let mut recording = Recording::default();
    let data = scene.data();
    let n_pathtag = data.tag_stream.len();
    let pathtag_padded = align_up(n_pathtag, 4 * shaders::PATHTAG_REDUCE_WG);
    let pathtag_wgs = pathtag_padded / (4 * shaders::PATHTAG_REDUCE_WG as usize);
    let mut scene: Vec<u8> = Vec::with_capacity(pathtag_padded);
    let pathtag_base = size_to_words(scene.len());
    scene.extend(&data.tag_stream);
    scene.resize(pathtag_padded, 0);
    let pathdata_base = size_to_words(scene.len());
    scene.extend(&data.pathseg_stream);

    let config = Config {
        width_in_tiles: 64,
        height_in_tiles: 64,
        target_width: 64 * 16,
        target_height: 64 * 16,
        pathtag_base,
        pathdata_base,
        ..Default::default()
    };
    let scene_buf = recording.upload(scene);
    let config_buf = recording.upload(bytemuck::bytes_of(&config).to_owned());

    let reduced_buf = BufProxy::new(pathtag_wgs as u64 * TAG_MONOID_SIZE);
    // TODO: really only need pathtag_wgs - 1
    recording.dispatch(
        shaders.pathtag_reduce,
        (pathtag_wgs as u32, 1, 1),
        [config_buf, scene_buf, reduced_buf],
    );

    let tagmonoid_buf =
        BufProxy::new(pathtag_wgs as u64 * shaders::PATHTAG_REDUCE_WG as u64 * TAG_MONOID_SIZE);
    recording.dispatch(
        shaders.pathtag_scan,
        (pathtag_wgs as u32, 1, 1),
        [config_buf, scene_buf, reduced_buf, tagmonoid_buf],
    );

    let path_coarse_wgs =
        (n_pathtag as u32 + shaders::PATH_COARSE_WG - 1) / shaders::PATH_COARSE_WG;
    // TODO: more principled size calc
    let tiles_buf = BufProxy::new(4097 * 8);
    let segments_buf = BufProxy::new(256 * 24);
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
    let out_buf = BufProxy::new(out_buf_size as u64);
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
    let mut recording = Recording::default();
    let mut ramps = crate::ramp::RampCache::default();
    let mut drawdata_patches: Vec<(usize, u32)> = vec![];
    let data = scene.data();
    let stop_data = &data.resources.stops;
    for patch in &data.resources.patches {
        use piet_scene::ResourcePatch;
        match patch {
            ResourcePatch::Ramp { offset, stops } => {
                let ramp_id = ramps.add(&stop_data[stops.clone()]);
                drawdata_patches.push((*offset, ramp_id));
            }
        }
    }
    let gradient_image = if drawdata_patches.is_empty() {
        ResourceProxy::new_image(1, 1, ImageFormat::Rgba8)
    } else {
        let data = ramps.data();
        let width = ramps.width();
        let height = ramps.height();
        let data: &[u8] = bytemuck::cast_slice(data);
        // println!(
        //     "gradient image: {}x{} ({} bytes)",
        //     width,
        //     height,
        //     data.len()
        // );
        ResourceProxy::Image(recording.upload_image(width, height, ImageFormat::Rgba8, data))
    };
    let n_pathtag = data.tag_stream.len();
    let pathtag_padded = align_up(n_pathtag, 4 * shaders::PATHTAG_REDUCE_WG);
    // TODO: can compute size accurately, avoid reallocation
    let mut scene: Vec<u8> = Vec::with_capacity(pathtag_padded);
    let pathtag_base = size_to_words(scene.len());
    scene.extend(&data.tag_stream);
    scene.resize(pathtag_padded, 0);
    let pathdata_base = size_to_words(scene.len());
    scene.extend(&data.pathseg_stream);
    let drawtag_base = size_to_words(scene.len());
    scene.extend(bytemuck::cast_slice(&data.drawtag_stream));
    let drawdata_base = size_to_words(scene.len());
    if !drawdata_patches.is_empty() {
        let mut pos = 0;
        for patch in drawdata_patches {
            let offset = patch.0;
            let value = patch.1;
            if pos < offset {
                scene.extend_from_slice(&data.drawdata_stream[pos..offset]);
            }
            scene.extend_from_slice(bytemuck::bytes_of(&value));
            pos = offset + 4;
        }
        if pos < data.drawdata_stream.len() {
            scene.extend_from_slice(&data.drawdata_stream[pos..])
        }
    } else {
        scene.extend(&data.drawdata_stream);
    }
    let transform_base = size_to_words(scene.len());
    scene.extend(bytemuck::cast_slice(&data.transform_stream));
    let linewidth_base = size_to_words(scene.len());
    scene.extend(bytemuck::cast_slice(&data.linewidth_stream));
    let n_path = data.n_path;
    // TODO: calculate for real when we do rectangles
    let n_drawobj = n_path;
    let n_clip = data.n_clip;

    let new_width = next_multiple_of(width, 16);
    let new_height = next_multiple_of(height, 16);

    let config = Config {
        // TODO: Replace with div_ceil once stable
        width_in_tiles: new_width / 16,
        height_in_tiles: new_height / 16,
        target_width: width,
        target_height: height,
        n_drawobj,
        n_path,
        n_clip,
        pathtag_base,
        pathdata_base,
        drawtag_base,
        drawdata_base,
        transform_base,
        linewidth_base,
    };
    // println!("{:?}", config);
    let scene_buf = ResourceProxy::Buf(recording.upload(scene));
    let config_buf = ResourceProxy::Buf(recording.upload(bytemuck::bytes_of(&config).to_owned()));

    let pathtag_wgs = pathtag_padded / (4 * shaders::PATHTAG_REDUCE_WG as usize);
    let reduced_buf = ResourceProxy::new_buf(pathtag_wgs as u64 * TAG_MONOID_FULL_SIZE);
    // TODO: really only need pathtag_wgs - 1
    recording.dispatch(
        shaders.pathtag_reduce,
        (pathtag_wgs as u32, 1, 1),
        [config_buf, scene_buf, reduced_buf],
    );

    let tagmonoid_buf = ResourceProxy::new_buf(
        pathtag_wgs as u64 * shaders::PATHTAG_REDUCE_WG as u64 * TAG_MONOID_FULL_SIZE,
    );
    recording.dispatch(
        shaders.pathtag_scan,
        (pathtag_wgs as u32, 1, 1),
        [config_buf, scene_buf, reduced_buf, tagmonoid_buf],
    );
    let drawobj_wgs = (n_drawobj + shaders::PATH_BBOX_WG - 1) / shaders::PATH_BBOX_WG;
    let path_bbox_buf = ResourceProxy::new_buf(n_path as u64 * PATH_BBOX_SIZE);
    recording.dispatch(
        shaders.bbox_clear,
        (drawobj_wgs, 1, 1),
        [config_buf, path_bbox_buf],
    );
    let cubic_buf = ResourceProxy::new_buf(n_pathtag as u64 * CUBIC_SIZE);
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
    let draw_reduced_buf = ResourceProxy::new_buf(drawobj_wgs as u64 * DRAWMONOID_SIZE);
    recording.dispatch(
        shaders.draw_reduce,
        (drawobj_wgs, 1, 1),
        [config_buf, scene_buf, draw_reduced_buf],
    );
    let draw_monoid_buf = ResourceProxy::new_buf(n_drawobj as u64 * DRAWMONOID_SIZE);
    let info_buf = ResourceProxy::new_buf(n_drawobj as u64 * MAX_DRAWINFO_SIZE);
    let clip_inp_buf = ResourceProxy::new_buf(data.n_clip as u64 * CLIP_INP_SIZE);
    recording.dispatch(
        shaders.draw_leaf,
        (drawobj_wgs, 1, 1),
        [
            config_buf,
            scene_buf,
            draw_reduced_buf,
            path_bbox_buf,
            draw_monoid_buf,
            info_buf,
            clip_inp_buf,
        ],
    );
    let clip_el_buf = ResourceProxy::new_buf(data.n_clip as u64 * CLIP_EL_SIZE);
    let clip_bic_buf =
        ResourceProxy::new_buf((n_clip / shaders::CLIP_REDUCE_WG) as u64 * CLIP_BIC_SIZE);
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
    let clip_bbox_buf = ResourceProxy::new_buf(n_clip as u64 * CLIP_BBOX_SIZE);
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
                info_buf,
            ],
        );
    }
    let draw_bbox_buf = ResourceProxy::new_buf(n_path as u64 * DRAW_BBOX_SIZE);
    let bump_buf = BufProxy::new(BUMP_SIZE);
    let bin_data_buf = ResourceProxy::new_buf(1 << 20);
    let width_in_bins = (config.width_in_tiles + 15) / 16;
    let height_in_bins = (config.height_in_tiles + 15) / 16;
    let n_bins = width_in_bins * height_in_bins;
    let bin_header_buf = ResourceProxy::new_buf((n_bins * drawobj_wgs) as u64 * BIN_HEADER_SIZE);
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
            bin_data_buf,
            bin_header_buf,
        ],
    );
    let path_buf = ResourceProxy::new_buf(n_path as u64 * PATH_SIZE);
    let tile_buf = ResourceProxy::new_buf(1 << 20);
    let path_wgs = (n_path + shaders::PATH_BBOX_WG - 1) / shaders::PATH_BBOX_WG;
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

    let segments_buf = ResourceProxy::new_buf(1 << 24);
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
    recording.dispatch(
        shaders.backdrop,
        (path_wgs, 1, 1),
        [config_buf, path_buf, tile_buf],
    );
    let ptcl_buf = ResourceProxy::new_buf(1 << 24);
    recording.dispatch(
        shaders.coarse,
        (width_in_bins, height_in_bins, 1),
        [
            config_buf,
            scene_buf,
            draw_monoid_buf,
            bin_header_buf,
            bin_data_buf,
            path_buf,
            tile_buf,
            info_buf,
            bump_buf,
            ptcl_buf,
        ],
    );
    let out_image = ImageProxy::new(width, height, ImageFormat::Rgba8);
    recording.dispatch(
        shaders.fine,
        (config.width_in_tiles, config.height_in_tiles, 1),
        [
            config_buf,
            tile_buf,
            segments_buf,
            ResourceProxy::Image(out_image),
            ptcl_buf,
            gradient_image,
            info_buf,
        ],
    );
    (recording, ResourceProxy::Image(out_image))
}

pub fn align_up(len: usize, alignment: u32) -> usize {
    len + (len.wrapping_neg() & alignment as usize - 1)
}
