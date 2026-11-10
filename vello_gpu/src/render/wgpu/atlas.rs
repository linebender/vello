// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Writing image data into atlas textures.

use vello_common::pixmap::Pixmap;
use wgpu::{Extent3d, Queue, Texture};

pub(crate) fn write_pixmap_to_atlas(
    queue: &Queue,
    pixmap: &Pixmap,
    atlas_texture: &Texture,
    offset: [u16; 2],
) {
    let width = u32::from(pixmap.width());
    let height = u32::from(pixmap.height());
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: atlas_texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: u32::from(offset[0]),
                y: u32::from(offset[1]),
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        pixmap.data_as_u8_slice(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}
