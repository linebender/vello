// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Writing image data into atlas textures.

use crate::render::webgl::{
    WebGlDataTransferOperation, WebGlError, WebGlOperation, WebGlResultExt,
};
use vello_common::pixmap::Pixmap;
use web_sys::{WebGl2RenderingContext, WebGlTexture};

pub(crate) fn write_pixmap_to_atlas(
    gl: &WebGl2RenderingContext,
    pixmap: &Pixmap,
    atlas_texture: &WebGlTexture,
    offset: [u16; 2],
) -> Result<(), WebGlError> {
    gl.active_texture(WebGl2RenderingContext::TEXTURE0);
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(atlas_texture));

    gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
        WebGl2RenderingContext::TEXTURE_2D,
        0,
        offset[0] as i32,
        offset[1] as i32,
        i32::from(pixmap.width()),
        i32::from(pixmap.height()),
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        Some(pixmap.data_as_u8_slice()),
    )
    .map_js_error(WebGlOperation::DataTransfer(
        WebGlDataTransferOperation::TextureUpload,
    ))?;

    Ok(())
}
