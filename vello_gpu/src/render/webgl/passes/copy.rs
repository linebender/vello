// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The copy pass.

use crate::copy::GpuCopyInstance;
use crate::render::webgl::WebGlError;
use crate::render::webgl::gl::resource::Program;
use crate::render::webgl::passes::{initialize_packed_u32_attribs, required_uniform_location};
use crate::render::webgl::resources::WebGlResources;
use vello_gpu_shaders::copy;
use web_sys::{WebGl2RenderingContext, WebGlUniformLocation};

#[derive(Debug)]
pub(crate) struct CopyUniforms {
    pub(crate) source_texture: WebGlUniformLocation,
}

impl WebGlResources {
    pub(crate) fn upload_copy_instances(
        &self,
        gl: &WebGl2RenderingContext,
        instances: &[GpuCopyInstance],
    ) {
        gl.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&self.copy_instance_buffer),
        );
        gl.buffer_data_with_u8_array(
            WebGl2RenderingContext::ARRAY_BUFFER,
            bytemuck::cast_slice(instances),
            WebGl2RenderingContext::DYNAMIC_DRAW,
        );
    }
}

pub(crate) fn get_copy_uniforms(
    gl: &WebGl2RenderingContext,
    program: &Program,
) -> Result<CopyUniforms, WebGlError> {
    Ok(CopyUniforms {
        source_texture: required_uniform_location(gl, program, copy::fragment::SOURCE_TEXTURE)?,
    })
}

/// Derive this from the number of fields on [`GpuCopyInstance`].
pub(crate) const COPY_ATTRIB_COUNT: u32 = 4;
pub(crate) const COPY_INSTANCE_STRIDE: i32 = size_of::<GpuCopyInstance>() as i32;

pub(crate) fn initialize_copy_vao(gl: &WebGl2RenderingContext, resources: &WebGlResources) {
    gl.bind_vertex_array(Some(&resources.copy_vao));
    gl.bind_buffer(
        WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
        Some(&resources.quad_index_buffer),
    );
    gl.bind_buffer(
        WebGl2RenderingContext::ARRAY_BUFFER,
        Some(&resources.copy_instance_buffer),
    );

    initialize_packed_u32_attribs(gl, COPY_ATTRIB_COUNT, COPY_INSTANCE_STRIDE);

    gl.bind_vertex_array(None);
}
