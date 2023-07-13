// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{Clip, ConfigUniform, DrawMonoid, DrawTag, Monoid, PathBbox, RenderConfig};

const WG_SIZE: usize = 256;

fn draw_leaf_main(
    n_wg: u32,
    config: &ConfigUniform,
    scene: &[u32],
    reduced: &[DrawMonoid],
    path_bbox: &[PathBbox],
    draw_monoid: &mut [DrawMonoid],
    info: &mut [u32],
    clip_inp: &mut [Clip],
) {
    let drawtag_base = config.layout.draw_tag_base;
    let mut prefix = DrawMonoid::default();
    for i in 0..n_wg {
        let mut m = prefix;
        for j in 0..WG_SIZE {
            let ix = (i * WG_SIZE as u32) as usize + j;
            let tag_word = DrawTag(scene[(drawtag_base + i * WG_SIZE as u32) as usize + j]);
            // store exclusive prefix sum
            draw_monoid[ix] = m;
            let dd = config.layout.draw_data_base + m.scene_offset;
            let di = m.info_offset as usize;
            if tag_word == DrawTag::COLOR
                || tag_word == DrawTag::LINEAR_GRADIENT
                || tag_word == DrawTag::RADIAL_GRADIENT
                || tag_word == DrawTag::IMAGE
                || tag_word == DrawTag::BEGIN_CLIP
            {
                let bbox = path_bbox[m.path_ix as usize];
                let fill_mode = (bbox.linewidth >= 0.0) as u32;
                let mut linewidth = bbox.linewidth;
                match tag_word {
                    DrawTag::COLOR => {
                        info[di] = f32::to_bits(linewidth);
                    }
                    _ => todo!(),
                }
            }
            // TODO: clips
            m = m.combine(&DrawMonoid::new(tag_word));
        }
        prefix = prefix.combine(&reduced[i as usize]);
    }
}
