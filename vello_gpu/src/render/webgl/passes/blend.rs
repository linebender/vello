// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The blend pass.

use crate::render::webgl::WebGlError;
use crate::render::webgl::gl::resource::Program;
use crate::render::webgl::passes::{
    WebGlRendererContext, checked_instance_count, initialize_packed_u32_attribs,
    required_uniform_location,
};
use crate::render::webgl::resources::WebGlResources;
use crate::util::RangedSlice;
use crate::{
    blend::{BlendStrip, GpuBlendInstance},
    schedule::round::BlendOp,
    target::{BlendPassBindings, TextureParity},
};
use vello_gpu_shaders::blend;
use web_sys::{WebGl2RenderingContext, WebGlUniformLocation};

#[derive(Debug)]
pub(crate) struct BlendUniforms {
    layer_texture_0: WebGlUniformLocation,
    layer_texture_1: WebGlUniformLocation,
    alphas_texture: WebGlUniformLocation,
}

impl WebGlResources {
    fn upload_blend_instances(&self, gl: &WebGl2RenderingContext, instances: &[GpuBlendInstance]) {
        gl.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&self.blend_instance_buffer),
        );
        gl.buffer_data_with_u8_array(
            WebGl2RenderingContext::ARRAY_BUFFER,
            bytemuck::cast_slice(instances),
            WebGl2RenderingContext::DYNAMIC_DRAW,
        );
    }
}

pub(crate) fn get_blend_uniforms(
    gl: &WebGl2RenderingContext,
    program: &Program,
) -> Result<BlendUniforms, WebGlError> {
    Ok(BlendUniforms {
        layer_texture_0: required_uniform_location(gl, program, blend::fragment::LAYER_TEXTURE_0)?,
        layer_texture_1: required_uniform_location(gl, program, blend::fragment::LAYER_TEXTURE_1)?,
        alphas_texture: required_uniform_location(gl, program, blend::fragment::ALPHAS_TEXTURE)?,
    })
}

/// Derive this from the number of fields on [`GpuBlendInstance`].
pub(crate) const BLEND_ATTRIB_COUNT: u32 = 8;
pub(crate) const BLEND_INSTANCE_STRIDE: i32 = size_of::<GpuBlendInstance>() as i32;

pub(crate) fn initialize_blend_vao(gl: &WebGl2RenderingContext, resources: &WebGlResources) {
    gl.bind_vertex_array(Some(&resources.blend_vao));
    gl.bind_buffer(
        WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
        Some(&resources.quad_index_buffer),
    );
    gl.bind_buffer(
        WebGl2RenderingContext::ARRAY_BUFFER,
        Some(&resources.blend_instance_buffer),
    );

    initialize_packed_u32_attribs(gl, BLEND_ATTRIB_COUNT, BLEND_INSTANCE_STRIDE);

    gl.bind_vertex_array(None);
}

impl WebGlRendererContext<'_> {
    pub(crate) fn blend_pass_inner(
        &mut self,
        blends: RangedSlice<'_, BlendOp>,
        blend_strips: &[BlendStrip],
        bindings: BlendPassBindings,
    ) -> Result<(), WebGlError> {
        let texture_size = self.texture_size();
        self.gl.disable(WebGl2RenderingContext::BLEND);
        self.gl.disable(WebGl2RenderingContext::SCISSOR_TEST);
        self.gl.disable(WebGl2RenderingContext::DEPTH_TEST);
        self.gl.depth_mask(false);
        self.gl.bind_vertex_array(Some(&self.resources.blend_vao));

        self.scratch_buffers.blend_instances.clear();
        for blend in blends.iter() {
            if let Some(range) = &blend.clip_strips {
                self.scratch_buffers.blend_instances.extend(
                    blend_strips[usize::try_from(range.start).unwrap()
                        ..usize::try_from(range.end).unwrap()]
                        .iter()
                        .copied()
                        .map(|strip| GpuBlendInstance::new(blend, Some(strip), texture_size)),
                );
            } else {
                self.scratch_buffers
                    .blend_instances
                    .push(GpuBlendInstance::new(blend, None, texture_size));
            }
        }
        self.resources
            .upload_blend_instances(self.gl, &self.scratch_buffers.blend_instances);
        let instance_count = checked_instance_count(self.scratch_buffers.blend_instances.len())?;

        self.gl.bind_framebuffer(
            WebGl2RenderingContext::FRAMEBUFFER,
            Some(self.resources.scratch_framebuffer()),
        );
        self.gl.viewport(
            0,
            0,
            i32::from(texture_size.width()),
            i32::from(texture_size.height()),
        );
        self.gl.use_program(Some(&self.programs.blend_program));

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE2);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(self.resources.alphas_texture.texture()),
        );
        self.gl
            .uniform1i(Some(&self.programs.blend_uniforms.alphas_texture), 2);

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(
                self.resources
                    .layer_texture(bindings.layer_id(TextureParity::Even)),
            ),
        );
        self.gl
            .uniform1i(Some(&self.programs.blend_uniforms.layer_texture_0), 0);

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE1);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(
                self.resources
                    .layer_texture(bindings.layer_id(TextureParity::Odd)),
            ),
        );
        self.gl
            .uniform1i(Some(&self.programs.blend_uniforms.layer_texture_1), 1);

        self.draw_instanced_quads(instance_count);

        self.scratch_buffers.copy_instances.clear();
        self.scratch_buffers.copy_instances.extend(
            self.scratch_buffers
                .blend_instances
                .iter()
                .copied()
                .map(GpuBlendInstance::copy_from_scratch),
        );
        self.resources
            .upload_copy_instances(self.gl, &self.scratch_buffers.copy_instances);

        self.gl.disable(WebGl2RenderingContext::BLEND);
        self.gl.disable(WebGl2RenderingContext::SCISSOR_TEST);
        self.gl.disable(WebGl2RenderingContext::DEPTH_TEST);
        self.gl.depth_mask(false);
        self.gl.bind_framebuffer(
            WebGl2RenderingContext::FRAMEBUFFER,
            Some(self.resources.layer_framebuffer(bindings.target())),
        );
        self.gl.viewport(
            0,
            0,
            i32::from(texture_size.width()),
            i32::from(texture_size.height()),
        );
        self.gl.bind_vertex_array(Some(&self.resources.copy_vao));
        self.gl.use_program(Some(&self.programs.copy_program));

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(self.resources.scratch_binding_texture()),
        );
        self.gl
            .uniform1i(Some(&self.programs.copy_uniforms.source_texture), 0);

        self.draw_instanced_quads(instance_count);

        self.gl.bind_vertex_array(None);

        Ok(())
    }
}
