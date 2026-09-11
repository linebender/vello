// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::ops::Deref;

use crate::{WebGlError, WebGlOperation, WebGlResourceKind, WebGlShaderStage};

use super::{
    WebGl2RenderingContext, WebGlBuffer, WebGlFramebuffer, WebGlProgram, WebGlShader, WebGlTexture,
    WebGlVertexArrayObject,
};

pub(crate) trait GlResource {
    const KIND: WebGlResourceKind;

    fn create(gl: &WebGl2RenderingContext) -> Option<Self>
    where
        Self: Sized;

    fn delete(gl: &WebGl2RenderingContext, raw: &Self);
}

#[derive(Debug)]
pub(crate) struct Resource<T: GlResource> {
    gl: WebGl2RenderingContext,
    raw: T,
}

// `Resource` intentionally does not implement `Clone`. There should
// only be a single handle to a given resources, such that it has
// unique ownership and we don't end up deleting the same resource
// twice.
impl<T: GlResource> Resource<T> {
    pub(super) fn new(gl: &WebGl2RenderingContext) -> Result<Self, WebGlError> {
        let raw = T::create(gl).ok_or(WebGlError::OperationFailed {
            operation: WebGlOperation::ResourceCreation(T::KIND),
            message: None,
        })?;

        Ok(Self {
            gl: gl.clone(),
            raw,
        })
    }
}

impl<T: GlResource> Drop for Resource<T> {
    fn drop(&mut self) {
        T::delete(&self.gl, &self.raw);
    }
}

impl<T: GlResource> Deref for Resource<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

#[derive(Debug)]
pub(crate) struct WebGlVertexShader(WebGlShader);

impl Deref for WebGlVertexShader {
    type Target = WebGlShader;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug)]
pub(crate) struct WebGlFragmentShader(WebGlShader);

impl Deref for WebGlFragmentShader {
    type Target = WebGlShader;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GlResource for WebGlTexture {
    const KIND: WebGlResourceKind = WebGlResourceKind::Texture;

    fn create(gl: &WebGl2RenderingContext) -> Option<Self> {
        gl.create_texture()
    }

    fn delete(gl: &WebGl2RenderingContext, raw: &Self) {
        gl.delete_texture(Some(raw));
    }
}

impl GlResource for WebGlBuffer {
    const KIND: WebGlResourceKind = WebGlResourceKind::Buffer;

    fn create(gl: &WebGl2RenderingContext) -> Option<Self> {
        gl.create_buffer()
    }

    fn delete(gl: &WebGl2RenderingContext, raw: &Self) {
        gl.delete_buffer(Some(raw));
    }
}

impl GlResource for WebGlFramebuffer {
    const KIND: WebGlResourceKind = WebGlResourceKind::Framebuffer;

    fn create(gl: &WebGl2RenderingContext) -> Option<Self> {
        gl.create_framebuffer()
    }

    fn delete(gl: &WebGl2RenderingContext, raw: &Self) {
        gl.delete_framebuffer(Some(raw));
    }
}

impl GlResource for WebGlProgram {
    const KIND: WebGlResourceKind = WebGlResourceKind::Program;

    fn create(gl: &WebGl2RenderingContext) -> Option<Self> {
        gl.create_program()
    }

    fn delete(gl: &WebGl2RenderingContext, raw: &Self) {
        gl.delete_program(Some(raw));
    }
}

impl GlResource for WebGlVertexShader {
    const KIND: WebGlResourceKind = WebGlResourceKind::Shader(WebGlShaderStage::Vertex);

    fn create(gl: &WebGl2RenderingContext) -> Option<Self> {
        gl.create_shader(WebGl2RenderingContext::VERTEX_SHADER)
            .map(Self)
    }

    fn delete(gl: &WebGl2RenderingContext, raw: &Self) {
        gl.delete_shader(Some(raw));
    }
}

impl GlResource for WebGlFragmentShader {
    const KIND: WebGlResourceKind = WebGlResourceKind::Shader(WebGlShaderStage::Fragment);

    fn create(gl: &WebGl2RenderingContext) -> Option<Self> {
        gl.create_shader(WebGl2RenderingContext::FRAGMENT_SHADER)
            .map(Self)
    }

    fn delete(gl: &WebGl2RenderingContext, raw: &Self) {
        gl.delete_shader(Some(raw));
    }
}

impl GlResource for WebGlVertexArrayObject {
    const KIND: WebGlResourceKind = WebGlResourceKind::VertexArray;

    fn create(gl: &WebGl2RenderingContext) -> Option<Self> {
        gl.create_vertex_array()
    }

    fn delete(gl: &WebGl2RenderingContext, raw: &Self) {
        gl.delete_vertex_array(Some(raw));
    }
}

pub(crate) type Texture = Resource<WebGlTexture>;
pub(crate) type Buffer = Resource<WebGlBuffer>;
pub(crate) type Framebuffer = Resource<WebGlFramebuffer>;
pub(crate) type Program = Resource<WebGlProgram>;
pub(crate) type VertexShader = Resource<WebGlVertexShader>;
pub(crate) type FragmentShader = Resource<WebGlFragmentShader>;
pub(crate) type VertexArray = Resource<WebGlVertexArrayObject>;
