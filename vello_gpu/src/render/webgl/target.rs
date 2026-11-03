// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Render targets.

use crate::RenderSize;
use crate::render::webgl::gl::resource::Framebuffer;
use crate::render::webgl::{
    InvalidTextureTargetReason, WebGlContextOperation, WebGlError, WebGlOperation, WebGlResultExt,
};
use alloc::rc::Rc;
use web_sys::{WebGl2RenderingContext, WebGlRenderbuffer, WebGlTexture};

/// Depth precision required to preserve painter order during opaque-strip rendering.
pub(crate) const REQUIRED_DEPTH_BITS: u32 = 24;

/// What [`WebGlRenderer::render`] renders into.
///
/// [`WebGlRenderer::render`]: crate::WebGlRenderer::render
#[derive(Debug, Clone, Copy, Default)]
pub enum WebGlRenderTarget<'a> {
    /// The drawing buffer of the canvas.
    #[default]
    DrawingBuffer,
    /// An offscreen texture.
    Texture(&'a WebGlTextureTarget),
}

impl WebGlRenderTarget<'_> {
    pub(crate) fn size(&self, gl: &WebGl2RenderingContext) -> RenderSize {
        match self {
            Self::DrawingBuffer => RenderSize {
                width: gl.drawing_buffer_width().try_into().unwrap(),
                height: gl.drawing_buffer_height().try_into().unwrap(),
            },
            Self::Texture(target) => target.size(),
        }
    }
}

/// A texture to render into, created with [`WebGlRenderer::create_texture_target`].
///
/// It is meant to be reused across renders.
///
/// [`WebGlRenderer::create_texture_target`]: crate::WebGlRenderer::create_texture_target
#[derive(Debug, Clone)]
pub struct WebGlTextureTarget {
    pub(crate) texture: WebGlTexture,
    pub(crate) framebuffer: Rc<Framebuffer>,
    width: u16,
    height: u16,
    pub(crate) use_depth_buffer: bool,
}

impl WebGlTextureTarget {
    pub(crate) fn new(
        gl: &WebGl2RenderingContext,
        texture: &WebGlTexture,
        depth: Option<&WebGlRenderbuffer>,
        width: u16,
        height: u16,
    ) -> Result<Self, WebGlError> {
        let framebuffer = create_framebuffer_for_texture(gl, texture)?;
        gl.framebuffer_renderbuffer(
            WebGl2RenderingContext::FRAMEBUFFER,
            WebGl2RenderingContext::DEPTH_ATTACHMENT,
            WebGl2RenderingContext::RENDERBUFFER,
            depth,
        );
        let result = validate_bound_texture_target(gl, depth.is_some());
        gl.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, None);
        result?;

        Ok(Self {
            texture: texture.clone(),
            framebuffer: Rc::new(framebuffer),
            width,
            height,
            use_depth_buffer: depth.is_some(),
        })
    }

    pub(crate) fn for_atlas(
        gl: &WebGl2RenderingContext,
        texture: &WebGlTexture,
        width: u16,
        height: u16,
    ) -> Result<Self, WebGlError> {
        // Unlike `new`, we don't do validation here since we know our atlas framebuffers
        // are valid.
        let framebuffer = create_framebuffer_for_texture(gl, texture)?;
        gl.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, None);

        Ok(Self {
            texture: texture.clone(),
            framebuffer: Rc::new(framebuffer),
            width,
            height,
            use_depth_buffer: false,
        })
    }

    /// The width of the texture.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// The height of the texture.
    pub fn height(&self) -> u16 {
        self.height
    }

    fn size(&self) -> RenderSize {
        RenderSize {
            width: self.width,
            height: self.height,
        }
    }
}

pub(crate) fn validate_bound_texture_target(
    gl: &WebGl2RenderingContext,
    use_depth_buffer: bool,
) -> Result<(), WebGlError> {
    let status = gl.check_framebuffer_status(WebGl2RenderingContext::FRAMEBUFFER);
    if status != WebGl2RenderingContext::FRAMEBUFFER_COMPLETE {
        return Err(WebGlError::InvalidTextureTarget(
            InvalidTextureTargetReason::IncompleteFramebuffer { status },
        ));
    }

    if use_depth_buffer {
        let actual_bits = gl
            .get_framebuffer_attachment_parameter(
                WebGl2RenderingContext::FRAMEBUFFER,
                WebGl2RenderingContext::DEPTH_ATTACHMENT,
                WebGl2RenderingContext::FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE,
            )
            .map_js_error(WebGlOperation::Context(WebGlContextOperation::Query))?
            .as_f64()
            .unwrap() as u32;

        if actual_bits < REQUIRED_DEPTH_BITS {
            return Err(WebGlError::InvalidTextureTarget(
                InvalidTextureTargetReason::InsufficientDepthBuffer {
                    actual_bits,
                    required_bits: REQUIRED_DEPTH_BITS,
                },
            ));
        }
    }

    Ok(())
}

/// Create a framebuffer for a texture.
pub(crate) fn create_framebuffer_for_texture(
    gl: &WebGl2RenderingContext,
    texture: &WebGlTexture,
) -> Result<Framebuffer, WebGlError> {
    let framebuffer = Framebuffer::new(gl)?;
    gl.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(&framebuffer));

    gl.framebuffer_texture_2d(
        WebGl2RenderingContext::FRAMEBUFFER,
        WebGl2RenderingContext::COLOR_ATTACHMENT0,
        WebGl2RenderingContext::TEXTURE_2D,
        Some(texture),
        0,
    );

    Ok(framebuffer)
}
