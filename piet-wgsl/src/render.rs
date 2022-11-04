//! Take an encoded scene and create a graph to render it

use bytemuck::{Pod, Zeroable};
use piet_scene::Scene;

use crate::{
    engine::{BufProxy, Recording},
    shaders::{self, FullShaders, Shaders},
};

const TAG_MONOID_SIZE: u64 = 12;
const TAG_MONOID_FULL_SIZE: u64 = 20;
const PATH_BBOX_SIZE: u64 = 24;
const CUBIC_SIZE: u64 = 40;
const DRAWMONOID_SIZE: u64 = 16;
const MAX_DRAWINFO_SIZE: u64 = 44;
const PATH_SIZE: u64 = 8;
const DRAW_BBOX_SIZE: u64 = 16;
const BUMP_SIZE: u64 = 16;

#[repr(C)]
#[derive(Clone, Copy, Default, Zeroable, Pod)]
struct Config {
    width_in_tiles: u32,
    height_in_tiles: u32,
    n_drawobj: u32,
    n_path: u32,
    pathtag_base: u32,
    pathdata_base: u32,
    drawtag_base: u32,
    drawdata_base: u32,
    transform_base: u32,
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

pub fn render(scene: &Scene, shaders: &Shaders) -> (Recording, BufProxy) {
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

    let path_coarse_wgs = (data.n_pathseg + shaders::PATH_COARSE_WG - 1) / shaders::PATH_COARSE_WG;
    //let cubics_buf = BufProxy::new(data.n_pathseg as u64 * 32);
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

pub fn render_full(scene: &Scene, shaders: &FullShaders) -> (Recording, BufProxy) {
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
    let drawtag_base = size_to_words(scene.len());
    scene.extend(bytemuck::cast_slice(&data.drawtag_stream));
    let drawdata_base = size_to_words(scene.len());
    scene.extend(&data.drawdata_stream);
    let transform_base = size_to_words(scene.len());
    scene.extend(bytemuck::cast_slice(&data.transform_stream));

    let n_path = data.n_path;
    // TODO: calculate for real when we do rectangles
    let n_drawobj = n_path;
    let config = Config {
        width_in_tiles: 64,
        height_in_tiles: 64,
        n_drawobj,
        n_path,
        pathtag_base,
        pathdata_base,
        drawtag_base,
        drawdata_base,
        transform_base,
    };
    let scene_buf = recording.upload(scene);
    let config_buf = recording.upload(bytemuck::bytes_of(&config).to_owned());

    let reduced_buf = BufProxy::new(pathtag_wgs as u64 * TAG_MONOID_FULL_SIZE);
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
    let drawobj_wgs = (n_drawobj + shaders::PATH_BBOX_WG - 1) / shaders::PATH_BBOX_WG;
    let path_bbox_buf = BufProxy::new(n_path as u64 * PATH_BBOX_SIZE);
    recording.dispatch(
        shaders.bbox_clear,
        (drawobj_wgs, 1, 1),
        [config_buf, path_bbox_buf],
    );
    let cubic_buf = BufProxy::new(n_path as u64 * CUBIC_SIZE);
    let path_coarse_wgs = (data.n_pathseg + shaders::PATH_COARSE_WG - 1) / shaders::PATH_COARSE_WG;
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
    let draw_reduced_buf = BufProxy::new(drawobj_wgs as u64 * DRAWMONOID_SIZE);
    recording.dispatch(
        shaders.draw_reduce,
        (drawobj_wgs, 1, 1),
        [config_buf, scene_buf, draw_reduced_buf],
    );
    let draw_monoid_buf = BufProxy::new(n_drawobj as u64 * DRAWMONOID_SIZE);
    let info_buf = BufProxy::new(n_drawobj as u64 * MAX_DRAWINFO_SIZE);
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
        ],
    );
    let draw_bbox_buf = BufProxy::new(n_path as u64 * DRAW_BBOX_SIZE);
    let bump_buf = BufProxy::new(BUMP_SIZE);
    // Not actually used yet.
    let clip_bbox_buf = BufProxy::new(1024);
    let bin_data_buf = BufProxy::new(1 << 16);
    recording.clear_all(bump_buf);
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
        ],
    );
    let path_buf = BufProxy::new(n_path as u64 * PATH_SIZE);
    let tile_buf = BufProxy::new(1 << 20);
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

    //let cubics_buf = BufProxy::new(data.n_pathseg as u64 * 32);
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
            cubic_buf,
            path_buf,
            bump_buf,
            tiles_buf,
            segments_buf,
        ],
    );
    recording.dispatch(
        shaders.backdrop,
        (path_wgs, 1, 1),
        [config_buf, path_buf, tiles_buf],
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

pub fn align_up(len: usize, alignment: u32) -> usize {
    len + (len.wrapping_neg() & alignment as usize - 1)
}
