//! Take an encoded scene and create a graph to render it

use bytemuck::{Pod, Zeroable};
use piet_scene::Scene;

use crate::{
    engine::{BufProxy, DownloadBufUsage, Recording},
    shaders::{self, Shaders},
};

const TAG_MONOID_SIZE: u64 = 12;

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct Config {
    width_in_tiles: u32,
    height_in_tiles: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct PathSegment {
    origin: [f32; 2],
    delta: [f32; 2],
    y_edge: f32,
    next: u32,
}

pub fn render(
    scene: &Scene,
    shaders: &Shaders,
    output_usage: DownloadBufUsage,
) -> (Recording, BufProxy) {
    let mut recording = Recording::default();
    let data = scene.data();
    let n_pathtag = data.tag_stream.len();
    let pathtag_padded = align_up(n_pathtag, 4 * shaders::PATHTAG_REDUCE_WG);
    let pathtag_wgs = pathtag_padded / (4 * shaders::PATHTAG_REDUCE_WG as usize);
    let mut tag_data: Vec<u8> = Vec::with_capacity(pathtag_padded);
    tag_data.extend(&data.tag_stream);
    tag_data.resize(pathtag_padded, 0);
    let pathtag_buf = recording.upload(tag_data);
    let reduced_buf = BufProxy::new(pathtag_wgs as u64 * TAG_MONOID_SIZE);
    // TODO: really only need pathtag_wgs - 1
    recording.dispatch(
        shaders.pathtag_reduce,
        (pathtag_wgs as u32, 1, 1),
        [pathtag_buf, reduced_buf],
    );

    let tagmonoid_buf =
        BufProxy::new(pathtag_wgs as u64 * shaders::PATHTAG_REDUCE_WG as u64 * TAG_MONOID_SIZE);
    recording.dispatch(
        shaders.pathtag_scan,
        (pathtag_wgs as u32, 1, 1),
        [pathtag_buf, reduced_buf, tagmonoid_buf],
    );

    let path_coarse_wgs = (data.n_pathseg + shaders::PATH_COARSE_WG - 1) / shaders::PATH_COARSE_WG;
    // The clone here is kinda BS, think about reducing copies
    // Of course, we'll probably end up concatenating into a single scene binding.
    let pathdata_buf = recording.upload(data.pathseg_stream.clone());
    //let cubics_buf = BufProxy::new(data.n_pathseg as u64 * 32);
    let config = Config {
        width_in_tiles: 64,
        height_in_tiles: 64,
    };
    let config_buf = recording.upload(bytemuck::bytes_of(&config).to_owned());
    // TODO: more principled size calc
    let tiles_buf = BufProxy::new(4097 * 8);
    let segments_buf = BufProxy::new(256 * 24);
    recording.clear_all(tiles_buf);
    recording.dispatch(
        shaders.path_coarse,
        (path_coarse_wgs, 1, 1),
        [
            pathtag_buf,
            tagmonoid_buf,
            pathdata_buf,
            config_buf,
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

    recording.download(out_buf, output_usage);
    (recording, out_buf)
}

pub fn align_up(len: usize, alignment: u32) -> usize {
    len + (len.wrapping_neg() & alignment as usize - 1)
}
