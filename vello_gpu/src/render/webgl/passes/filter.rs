// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The filter pass.

use crate::render::webgl::WebGlError;
use crate::render::webgl::gl::resource::Program;
use crate::render::webgl::passes::{
    WebGlRendererContext, checked_instance_count, initialize_packed_u32_attribs,
    required_uniform_location,
};
use crate::render::webgl::resources::{WebGlResources, create_data_texture};
use crate::{
    filter::{FilterContext, FilterInstanceData, FilterPassPlan},
    target::{FilterPassBindings, LayerTextureId},
};
use vello_gpu_shaders::filter as filter_shader;
use web_sys::{WebGl2RenderingContext, WebGlUniformLocation};

#[derive(Debug)]
pub(crate) struct FilterPassUniforms {
    filter_data: WebGlUniformLocation,
    source_texture: WebGlUniformLocation,
    original_texture: WebGlUniformLocation,
}

impl WebGlResources {
    /// Grow the filter data texture if needed.
    pub(crate) fn maybe_resize_filter_data_tex(
        &mut self,
        gl: &WebGl2RenderingContext,
        filter_context: &FilterContext,
    ) -> Result<(), WebGlError> {
        self.filter_data_texture
            .ensure_bytes(filter_context.as_bytes().len(), |format, width, height| {
                create_data_texture(gl, format, width, height)
            })?;

        Ok(())
    }

    /// Upload filter data to the texture.
    pub(crate) fn upload_filter_data_texture(
        &mut self,
        gl: &WebGl2RenderingContext,
        filter_context: &FilterContext,
    ) -> Result<(), WebGlError> {
        if filter_context.is_empty() {
            return Ok(());
        }

        self.filter_data_texture
            .upload_chunks(gl, &mut self.staging, [filter_context.as_bytes()])
    }

    fn upload_filter_instances(
        &self,
        gl: &WebGl2RenderingContext,
        instances: &[FilterInstanceData],
    ) {
        gl.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&self.filter_instance_buffer),
        );
        gl.buffer_data_with_u8_array(
            WebGl2RenderingContext::ARRAY_BUFFER,
            bytemuck::cast_slice(instances),
            WebGl2RenderingContext::DYNAMIC_DRAW,
        );
    }
}

pub(crate) fn get_filter_pass_uniforms(
    gl: &WebGl2RenderingContext,
    program: &Program,
) -> Result<FilterPassUniforms, WebGlError> {
    let filter_data = required_uniform_location(gl, program, filter_shader::fragment::FILTER_DATA)?;
    let source_texture =
        required_uniform_location(gl, program, filter_shader::fragment::SOURCE_TEXTURE)?;
    let original_texture =
        required_uniform_location(gl, program, filter_shader::fragment::ORIGINAL_TEXTURE)?;
    Ok(FilterPassUniforms {
        filter_data,
        source_texture,
        original_texture,
    })
}

/// Derive this from the number of fields on [`FilterInstanceData`].
pub(crate) const FILTER_ATTRIB_COUNT: u32 = 9;
pub(crate) const FILTER_INSTANCE_STRIDE: i32 = size_of::<FilterInstanceData>() as i32;

pub(crate) fn initialize_filter_vao(gl: &WebGl2RenderingContext, resources: &WebGlResources) {
    gl.bind_vertex_array(Some(&resources.filter_vao));
    gl.bind_buffer(
        WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
        Some(&resources.quad_index_buffer),
    );
    gl.bind_buffer(
        WebGl2RenderingContext::ARRAY_BUFFER,
        Some(&resources.filter_instance_buffer),
    );

    initialize_packed_u32_attribs(gl, FILTER_ATTRIB_COUNT, FILTER_INSTANCE_STRIDE);

    gl.bind_vertex_array(None);
}

impl WebGlRendererContext<'_> {
    pub(crate) fn filter_pass_inner(
        &mut self,
        plan: &FilterPassPlan,
        bindings: FilterPassBindings,
    ) -> Result<(), WebGlError> {
        if let Some(copy_pass) = plan.copy_pass() {
            self.resources.upload_copy_instances(self.gl, copy_pass);
            self.gl.disable(WebGl2RenderingContext::BLEND);
            self.gl.disable(WebGl2RenderingContext::SCISSOR_TEST);
            self.gl.disable(WebGl2RenderingContext::DEPTH_TEST);
            self.gl.depth_mask(false);
            self.gl.bind_vertex_array(Some(&self.resources.copy_vao));
            self.gl.use_program(Some(&self.programs.copy_program));
            self.gl.bind_framebuffer(
                WebGl2RenderingContext::FRAMEBUFFER,
                Some(self.resources.scratch_framebuffer()),
            );
            let scratch_size = self.texture_size();
            self.gl.viewport(
                0,
                0,
                i32::from(scratch_size.width()),
                i32::from(scratch_size.height()),
            );
            self.gl.active_texture(WebGl2RenderingContext::TEXTURE0);
            self.gl.bind_texture(
                WebGl2RenderingContext::TEXTURE_2D,
                Some(self.resources.layer_texture(bindings.target())),
            );
            self.gl
                .uniform1i(Some(&self.programs.copy_uniforms.source_texture), 0);
            self.draw_instanced_quads(checked_instance_count(copy_pass.len())?);
        }

        self.gl.use_program(Some(&self.programs.filter_program));
        self.gl.bind_vertex_array(Some(&self.resources.filter_vao));

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(self.resources.filter_data_texture.texture()),
        );
        self.gl
            .uniform1i(Some(&self.programs.filter_uniforms.filter_data), 0);

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE2);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(self.resources.scratch_binding_texture()),
        );
        self.gl
            .uniform1i(Some(&self.programs.filter_uniforms.original_texture), 2);

        for (step_index, instances) in plan.steps().enumerate() {
            self.do_filter_instance_pass(
                instances,
                bindings.input(step_index),
                bindings.output(step_index),
            )?;
        }

        self.gl.bind_vertex_array(None);

        Ok(())
    }

    fn do_filter_instance_pass(
        &self,
        instances: &[FilterInstanceData],
        input: LayerTextureId,
        output: LayerTextureId,
    ) -> Result<(), WebGlError> {
        self.gl.disable(WebGl2RenderingContext::BLEND);
        self.gl.disable(WebGl2RenderingContext::SCISSOR_TEST);
        self.gl.disable(WebGl2RenderingContext::DEPTH_TEST);
        self.gl.depth_mask(false);
        self.gl.bind_framebuffer(
            WebGl2RenderingContext::FRAMEBUFFER,
            Some(self.resources.layer_framebuffer(output)),
        );
        let target_texture_size = self.texture_size();
        self.gl.viewport(
            0,
            0,
            i32::from(target_texture_size.width()),
            i32::from(target_texture_size.height()),
        );

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE1);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(self.resources.layer_texture(input)),
        );
        self.gl
            .uniform1i(Some(&self.programs.filter_uniforms.source_texture), 1);

        // TODO: Filter instances ideally should be uploaded once for the whole round (or even once
        // globally), not per pass.
        self.resources.upload_filter_instances(self.gl, instances);
        self.draw_instanced_quads(checked_instance_count(instances.len())?);

        Ok(())
    }
}
