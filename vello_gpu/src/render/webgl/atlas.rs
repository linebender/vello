// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Writing image data into atlas textures.

use crate::render::webgl::bindings::WebGlTextureWithDimensions;
use crate::render::webgl::gl::resource::Framebuffer;
use crate::render::webgl::{
    WebGlDataTransferOperation, WebGlError, WebGlOperation, WebGlResultExt,
};
use alloc::sync::Arc;
use vello_common::pixmap::Pixmap;
use web_sys::{WebGl2RenderingContext, WebGlTexture};

/// Trait for types that can write image data directly to the atlas texture in WebGL.
///
/// This allows efficient uploading from different sources:
/// - `Pixmap`: Direct upload using raw pixel data
/// - `ImageData`: Browser `ImageData` objects
/// - Custom implementations for other image sources
pub trait WebGlAtlasWriter {
    /// Get the width of the image.
    fn width(&self) -> u16;
    /// Get the height of the image.
    fn height(&self) -> u16;

    /// Write image data to an atlas texture at the specified offset.
    fn write_to_atlas(
        &self,
        gl: &WebGl2RenderingContext,
        atlas_texture: &WebGlTexture,
        offset: [u16; 2],
        width: u16,
        height: u16,
    ) -> Result<(), WebGlError>;
}

/// Implementation for `Pixmap` - direct upload using raw pixel data.
impl WebGlAtlasWriter for Pixmap {
    fn width(&self) -> u16 {
        self.width()
    }

    fn height(&self) -> u16 {
        self.height()
    }

    fn write_to_atlas(
        &self,
        gl: &WebGl2RenderingContext,
        atlas_texture: &WebGlTexture,
        offset: [u16; 2],
        width: u16,
        height: u16,
    ) -> Result<(), WebGlError> {
        gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(atlas_texture));

        // Convert pixmap data to the format expected by WebGL
        let rgba_data = self.data_as_u8_slice();

        gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
            WebGl2RenderingContext::TEXTURE_2D,
            0,
            offset[0] as i32,
            offset[1] as i32,
            width as i32,
            height as i32,
            WebGl2RenderingContext::RGBA,
            WebGl2RenderingContext::UNSIGNED_BYTE,
            Some(rgba_data),
        )
        .map_js_error(WebGlOperation::DataTransfer(
            WebGlDataTransferOperation::TextureUpload,
        ))?;

        Ok(())
    }
}

/// Implementation for `Arc<Pixmap>`.
impl WebGlAtlasWriter for Arc<Pixmap> {
    fn width(&self) -> u16 {
        self.as_ref().width()
    }

    fn height(&self) -> u16 {
        self.as_ref().height()
    }

    fn write_to_atlas(
        &self,
        gl: &WebGl2RenderingContext,
        atlas_texture: &WebGlTexture,
        offset: [u16; 2],
        width: u16,
        height: u16,
    ) -> Result<(), WebGlError> {
        self.as_ref()
            .write_to_atlas(gl, atlas_texture, offset, width, height)
    }
}

/// Implementation for `WebGlTexture` - texture-to-texture copy.
impl WebGlAtlasWriter for WebGlTexture {
    fn width(&self) -> u16 {
        // WebGL textures don't expose their dimensions directly
        // This is a limitation - in practice, you'd need to track dimensions separately
        // For now, we'll require the caller to provide correct width/height parameters
        unreachable!("WebGlTexture width must be provided by caller")
    }

    fn height(&self) -> u16 {
        // WebGL textures don't expose their dimensions directly
        // This is a limitation - in practice, you'd need to track dimensions separately
        // For now, we'll require the caller to provide correct width/height parameters
        unreachable!("WebGlTexture height must be provided by caller")
    }

    fn write_to_atlas(
        &self,
        gl: &WebGl2RenderingContext,
        atlas_texture: &WebGlTexture,
        offset: [u16; 2],
        width: u16,
        height: u16,
    ) -> Result<(), WebGlError> {
        copy_to_texture(
            gl,
            |gl| {
                // Attach source texture to read framebuffer
                gl.framebuffer_texture_2d(
                    WebGl2RenderingContext::READ_FRAMEBUFFER,
                    WebGl2RenderingContext::COLOR_ATTACHMENT0,
                    WebGl2RenderingContext::TEXTURE_2D,
                    Some(self),
                    0,
                );
            },
            atlas_texture,
            [u32::from(offset[0]), u32::from(offset[1])],
            [u32::from(width), u32::from(height)],
        )
    }
}

impl WebGlAtlasWriter for WebGlTextureWithDimensions {
    fn width(&self) -> u16 {
        self.width
    }

    fn height(&self) -> u16 {
        self.height
    }

    fn write_to_atlas(
        &self,
        gl: &WebGl2RenderingContext,
        atlas_texture: &WebGlTexture,
        offset: [u16; 2],
        width: u16,
        height: u16,
    ) -> Result<(), WebGlError> {
        self.texture
            .write_to_atlas(gl, atlas_texture, offset, width, height)
    }
}

/// Copy from a source texture/framebuffer to an atlas texture.
pub(crate) fn copy_to_texture(
    gl: &WebGl2RenderingContext,
    source_setup: impl FnOnce(&WebGl2RenderingContext),
    dest_texture: &WebGlTexture,
    dest_offset: [u32; 2],
    copy_size: [u32; 2],
) -> Result<(), WebGlError> {
    gl.active_texture(WebGl2RenderingContext::TEXTURE0);
    let read_framebuffer = Framebuffer::new(gl)?;
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(dest_texture));

    // Bind the READ framebuffer
    gl.bind_framebuffer(
        WebGl2RenderingContext::READ_FRAMEBUFFER,
        Some(&read_framebuffer),
    );

    // Let the caller set up the source (attach texture/layer to read framebuffer)
    source_setup(gl);

    gl.copy_tex_sub_image_2d(
        WebGl2RenderingContext::TEXTURE_2D,
        0,
        dest_offset[0] as i32,
        dest_offset[1] as i32,
        0,
        0,
        copy_size[0] as i32,
        copy_size[1] as i32,
    );

    Ok(())
}
