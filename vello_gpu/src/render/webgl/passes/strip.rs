// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The strip pass.

use crate::draw::{EXTERNAL_TEXTURE_SLOT_COUNT, ExternalTextureBindings, ExternalTextureRun};
use crate::render::webgl::gl::resource::Program;
use crate::render::webgl::passes::{
    WebGlRendererContext, checked_instance_count, required_uniform_location,
};
use crate::render::webgl::resources::{WebGlResources, create_external_sampler};
use crate::render::webgl::{WebGlError, WebGlOperation, WebGlShaderInterfaceOperation};
use crate::util::RangedSlice;
use crate::{
    GpuStrip, RenderError,
    paint::{ExternalSampler, TextureSourceId},
    target::{DrawPassTarget, LayerTextureId},
};
use hashbrown::hash_map::Entry;
use vello_gpu_shaders::render;
use web_sys::{WebGl2RenderingContext, WebGlTexture, WebGlUniformLocation};

/// First texture unit from which the strip program samples external textures.
pub(crate) const EXTERNAL_TEXTURE_UNIT_START: u32 = 4;

/// Uniform locations for `strip_program`.
#[derive(Debug)]
pub(crate) struct StripUniforms {
    /// Config uniform block index for vertex shader.
    config_vs_block_index: u32,
    /// Config uniform block index for fragment shader.
    config_fs_block_index: u32,
    /// Alphas texture location.
    alphas_texture: WebGlUniformLocation,
    /// Layer input texture location.
    layer_input_texture: WebGlUniformLocation,
    /// Encoded paints texture location for fragment shader.
    encoded_paints_texture_fs: WebGlUniformLocation,
    /// Encoded paints texture location for vertex shader.
    encoded_paints_texture_vs: WebGlUniformLocation,
    /// Gradient texture location.
    gradient_texture: WebGlUniformLocation,
    /// External texture locations, indexed by shader slot.
    external_textures: [WebGlUniformLocation; EXTERNAL_TEXTURE_SLOT_COUNT],
}

impl WebGlResources {
    /// Uploads two strip slices (opaque then alpha) into a single GPU buffer.
    fn upload_strip_pair(
        &mut self,
        gl: &WebGl2RenderingContext,
        opaque_strips: &[GpuStrip],
        alpha_strips: RangedSlice<'_, GpuStrip>,
    ) {
        if opaque_strips.is_empty() && alpha_strips.len() == 0 {
            return;
        }

        gl.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&self.strips_buffer),
        );

        let opaque_len = size_of_val(opaque_strips);
        let alpha_len = alpha_strips.len() * size_of::<GpuStrip>();
        let total_len = opaque_len + alpha_len;

        // Allocate buffer, then write both slices via bufferSubData.
        // We don't want to pay for concatenating the two slices. It's better to
        // simply write twice.
        gl.buffer_data_with_i32(
            WebGl2RenderingContext::ARRAY_BUFFER,
            total_len as i32,
            WebGl2RenderingContext::DYNAMIC_DRAW,
        );
        if !opaque_strips.is_empty() {
            gl.buffer_sub_data_with_i32_and_u8_array(
                WebGl2RenderingContext::ARRAY_BUFFER,
                0,
                bytemuck::cast_slice(opaque_strips),
            );
        }
        let mut offset = opaque_len;

        // TODO: Maybe concatenate first and upload once?
        for strips in alpha_strips.slices() {
            let bytes = bytemuck::cast_slice(strips);
            gl.buffer_sub_data_with_i32_and_u8_array(
                WebGl2RenderingContext::ARRAY_BUFFER,
                offset as i32,
                bytes,
            );
            offset += bytes.len();
        }
    }
}
/// Get the  uniform locations for the `render_strips` program.
pub(crate) fn get_strip_uniforms(
    gl: &WebGl2RenderingContext,
    program: &Program,
) -> Result<StripUniforms, WebGlError> {
    let config_vs_name = render::vertex::CONFIG;
    let config_vs_block_index = gl.get_uniform_block_index(program, config_vs_name);

    let config_fs_name = render::fragment::CONFIG;
    let config_fs_block_index = gl.get_uniform_block_index(program, config_fs_name);

    if config_vs_block_index == WebGl2RenderingContext::INVALID_INDEX {
        return Err(WebGlError::OperationFailed {
            operation: WebGlOperation::ShaderInterface(WebGlShaderInterfaceOperation::UniformBlock),
            message: Some(config_vs_name.into()),
        });
    }

    if config_fs_block_index == WebGl2RenderingContext::INVALID_INDEX {
        return Err(WebGlError::OperationFailed {
            operation: WebGlOperation::ShaderInterface(WebGlShaderInterfaceOperation::UniformBlock),
            message: Some(config_fs_name.into()),
        });
    }

    // Bind uniform blocks to binding points.
    gl.uniform_block_binding(program, config_vs_block_index, 0);
    gl.uniform_block_binding(program, config_fs_block_index, 0);

    // Get texture uniform locations.
    let alphas_texture_name = render::fragment::ALPHAS_TEXTURE;
    let layer_input_texture_name = render::fragment::LAYER_INPUT_TEXTURE;
    let encoded_paints_texture_fs_name = render::fragment::ENCODED_PAINTS_TEXTURE;
    let encoded_paints_texture_vs_name = render::vertex::ENCODED_PAINTS_TEXTURE;
    let gradient_texture_name = render::fragment::GRADIENT_TEXTURE;
    // TODO: Change it so this is based on `EXTERNAL_TEXTURE_SLOT_COUNT`.
    let external_texture_names = [render::fragment::EXTERNAL_TEXTURE_0];

    Ok(StripUniforms {
        config_vs_block_index,
        config_fs_block_index,
        alphas_texture: required_uniform_location(gl, program, alphas_texture_name)?,
        layer_input_texture: required_uniform_location(gl, program, layer_input_texture_name)?,
        encoded_paints_texture_fs: required_uniform_location(
            gl,
            program,
            encoded_paints_texture_fs_name,
        )?,
        encoded_paints_texture_vs: required_uniform_location(
            gl,
            program,
            encoded_paints_texture_vs_name,
        )?,
        gradient_texture: required_uniform_location(gl, program, gradient_texture_name)?,
        external_textures: [required_uniform_location(
            gl,
            program,
            external_texture_names[0],
        )?],
    })
}

pub(crate) const STRIP_STRIDE: i32 = size_of::<GpuStrip>() as i32;
pub(crate) const STRIP_ATTR_COUNT: i32 = STRIP_STRIDE / 4;
const _: () = assert!(
    STRIP_STRIDE == 24,
    "GpuStrip layout must match strip vertex stride"
);

/// Initialize strip VAO.
pub(crate) fn initialize_strip_vao(gl: &WebGl2RenderingContext, resources: &WebGlResources) {
    gl.bind_vertex_array(Some(&resources.strip_vao));
    gl.bind_buffer(
        WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
        Some(&resources.quad_index_buffer),
    );
    gl.bind_buffer(
        WebGl2RenderingContext::ARRAY_BUFFER,
        Some(&resources.strips_buffer),
    );

    for i in 0..STRIP_ATTR_COUNT {
        let location = i as u32;
        let offset = i * 4;

        gl.enable_vertex_attrib_array(location);
        gl.vertex_attrib_i_pointer_with_i32(
            location,
            1,
            WebGl2RenderingContext::UNSIGNED_INT,
            STRIP_STRIDE,
            offset,
        );

        gl.vertex_attrib_divisor(location, 1);
    }

    gl.bind_vertex_array(None);
}

impl WebGlRendererContext<'_> {
    /// Draw `count` strip instances starting at `first_instance`, split into one draw per
    /// external texture run.
    ///
    /// Strip attribute offsets are restored to the start of the buffer before returning.
    fn draw_strips(
        &mut self,
        external_texture_runs: &[ExternalTextureRun],
        first_instance: i32,
        count: i32,
    ) -> Result<(), WebGlError> {
        if count == 0 {
            return Ok(());
        }

        // We validate it here once so that we don't need to validate it
        // again in each call of `draw_strip_range` and `set_strip_attrib_offset`.
        first_instance
            .checked_add(count)
            .and_then(|end_instance| end_instance.checked_mul(STRIP_STRIDE))
            .ok_or(WebGlError::TooManyInstances)?;

        if external_texture_runs.is_empty() {
            self.draw_strip_range(first_instance, count);
        } else {
            // A run covers the strips from its start up to the next run's start, or up to the
            // end of the draw for the last run.
            for (i, run) in external_texture_runs.iter().enumerate() {
                let start = i32::try_from(run.strips_start).unwrap();
                let end = external_texture_runs
                    .get(i + 1)
                    .map_or(count, |next| i32::try_from(next.strips_start).unwrap());
                self.bind_external_textures(&run.bindings)?;
                self.draw_strip_range(first_instance + start, end - start);
            }
            self.bind_external_textures(&ExternalTextureBindings::EMPTY)?;
        }

        self.set_strip_attrib_offset(0);

        Ok(())
    }

    /// Draw `count` strip instances starting at `first_instance`.
    ///
    /// WebGL2 has no base-instance argument, so the vertex attributes are offset to the first
    /// requested instance before issuing the instanced draw.
    fn draw_strip_range(&self, first_instance: i32, count: i32) {
        debug_assert!(count >= 0, "strip runs must be ordered by start index");
        if count == 0 {
            return;
        }
        self.set_strip_attrib_offset(first_instance);
        self.draw_instanced_quads(count);
    }

    /// Bind the external texture slots.
    fn bind_external_textures(
        &mut self,
        bindings: &ExternalTextureBindings,
    ) -> Result<(), WebGlError> {
        for (slot, texture_source) in bindings.as_array().into_iter().enumerate() {
            let texture_unit = EXTERNAL_TEXTURE_UNIT_START + u32::try_from(slot).unwrap();

            self.gl
                .active_texture(WebGl2RenderingContext::TEXTURE0 + texture_unit);
            let (texture, sampler_key): (&WebGlTexture, ExternalSampler) = match texture_source {
                Some(TextureSourceId::Atlas(atlas_id)) => (
                    &self.resources.atlas_textures[atlas_id.as_u32() as usize],
                    ExternalSampler::DEFAULT,
                ),
                Some(TextureSourceId::External(texture_id, sampler)) => (
                    &self
                        .texture_bindings
                        .get(texture_id)
                        .ok_or(RenderError::MissingTextureBinding(texture_id))?
                        .texture,
                    sampler,
                ),
                None => (
                    &self.resources.placeholder_external_texture,
                    ExternalSampler::DEFAULT,
                ),
            };
            self.gl
                .bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(texture));
            let sampler = match self.resources.external_samplers.entry(sampler_key) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    entry.insert(create_external_sampler(self.gl, sampler_key)?)
                }
            };

            self.gl.bind_sampler(texture_unit, Some(sampler));
        }

        Ok(())
    }

    /// Point the strip vertex attributes at the instance at `first_instance`.
    fn set_strip_attrib_offset(&self, first_instance: i32) {
        let byte_offset = first_instance * STRIP_STRIDE;
        for i in 0..STRIP_ATTR_COUNT {
            self.gl.vertex_attrib_i_pointer_with_i32(
                i as u32,
                1,
                WebGl2RenderingContext::UNSIGNED_INT,
                STRIP_STRIDE,
                i * 4 + byte_offset,
            );
        }
    }

    /// Render strips to the specified render target.
    pub(crate) fn strip_pass_inner(
        &mut self,
        opaque_strips: &[GpuStrip],
        alpha_strips: RangedSlice<'_, GpuStrip>,
        external_texture_runs: &[ExternalTextureRun],
        target: DrawPassTarget,
        child_layer_texture: Option<LayerTextureId>,
    ) -> Result<(), WebGlError> {
        let opaque_count = opaque_strips.len();
        let alpha_count = alpha_strips.len();
        if opaque_count == 0 && alpha_count == 0 {
            return Ok(());
        }
        match &target {
            DrawPassTarget::Root(_) => {
                self.gl.bind_framebuffer(
                    WebGl2RenderingContext::FRAMEBUFFER,
                    self.resources.root_framebuffer(),
                );
                let width = self.resources.render_size.width;
                let height = self.resources.render_size.height;
                self.gl.viewport(0, 0, i32::from(width), i32::from(height));

                self.gl.bind_buffer_base(
                    WebGl2RenderingContext::UNIFORM_BUFFER,
                    self.programs.strip_uniforms.config_vs_block_index,
                    Some(&self.resources.view_config_buffer),
                );
                self.gl.bind_buffer_base(
                    WebGl2RenderingContext::UNIFORM_BUFFER,
                    self.programs.strip_uniforms.config_fs_block_index,
                    Some(&self.resources.view_config_buffer),
                );
            }
            DrawPassTarget::Layer(id) => {
                self.gl.bind_framebuffer(
                    WebGl2RenderingContext::FRAMEBUFFER,
                    Some(self.resources.layer_framebuffer(*id)),
                );
                let size = self.texture_size();
                self.gl
                    .viewport(0, 0, i32::from(size.width()), i32::from(size.height()));

                let buf = &self.resources.layer_config_buffer;
                self.gl.bind_buffer_base(
                    WebGl2RenderingContext::UNIFORM_BUFFER,
                    self.programs.strip_uniforms.config_vs_block_index,
                    Some(buf),
                );
                self.gl.bind_buffer_base(
                    WebGl2RenderingContext::UNIFORM_BUFFER,
                    self.programs.strip_uniforms.config_fs_block_index,
                    Some(buf),
                );
            }
        };

        self.gl.disable(WebGl2RenderingContext::SCISSOR_TEST);

        // Use the strip program.
        self.gl.use_program(Some(&self.programs.strip_program));

        // Set up attributes.
        self.gl.bind_vertex_array(Some(&self.resources.strip_vao));

        // Bind textures.
        self.gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(self.resources.alphas_texture.texture()),
        );
        self.gl
            .uniform1i(Some(&self.programs.strip_uniforms.alphas_texture), 0);

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE1);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(child_layer_texture.map_or_else(
                || &self.resources.placeholder_external_texture,
                |id| self.resources.layer_texture(id),
            )),
        );
        self.gl
            .uniform1i(Some(&self.programs.strip_uniforms.layer_input_texture), 1);

        // Bind encoded paints texture for image metadata
        self.gl.active_texture(WebGl2RenderingContext::TEXTURE2);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(self.resources.encoded_paints_texture.texture()),
        );
        self.gl.uniform1i(
            Some(&self.programs.strip_uniforms.encoded_paints_texture_fs),
            2,
        );
        self.gl.uniform1i(
            Some(&self.programs.strip_uniforms.encoded_paints_texture_vs),
            2,
        );

        // Bind gradient texture for gradient rendering
        self.gl.active_texture(WebGl2RenderingContext::TEXTURE3);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(self.resources.gradient_texture.texture()),
        );
        self.gl
            .uniform1i(Some(&self.programs.strip_uniforms.gradient_texture), 3);

        // External textures are rebound per run while drawing. Start from the placeholder so the
        // sampler is valid for draws that don't reference one.
        self.bind_external_textures(&ExternalTextureBindings::EMPTY)?;
        for (slot, uniform) in self
            .programs
            .strip_uniforms
            .external_textures
            .iter()
            .enumerate()
        {
            self.gl.uniform1i(
                Some(uniform),
                i32::try_from(EXTERNAL_TEXTURE_UNIT_START).unwrap() + i32::try_from(slot).unwrap(),
            );
        }

        // TODO: Today, we only support early-z rejection on the final view. If we wanted to support
        // intermediate layers, we would require separate depth buffers for each target. We can explore
        // that possibility in the future.
        let enable_opaque = self.use_depth_buffer && target.enable_opaque();

        self.resources
            .upload_strip_pair(self.gl, opaque_strips, alpha_strips);
        let opaque_count = checked_instance_count(opaque_count)?;
        let alpha_count = checked_instance_count(alpha_count)?;

        if enable_opaque {
            self.gl.enable(WebGl2RenderingContext::DEPTH_TEST);
            self.gl.depth_mask(true);
            self.gl.depth_func(WebGl2RenderingContext::LEQUAL);

            // Clear depth buffer on first use per frame.
            if !self.resources.depth_cleared_this_frame {
                self.resources.depth_cleared_this_frame = true;
                self.gl.clear_depth(1.0);
                self.gl.clear(WebGl2RenderingContext::DEPTH_BUFFER_BIT);
            }

            // Opaque pass: front-to-back, depth test ON, depth write ON, blend OFF.
            if opaque_count > 0 {
                self.gl.enable(WebGl2RenderingContext::DEPTH_TEST);
                self.gl.depth_mask(true);
                self.gl.disable(WebGl2RenderingContext::BLEND);
                self.draw_strips(external_texture_runs, 0, opaque_count)?;
            }

            // Alpha pass: back-to-front, depth test ON, depth write OFF, blend ON.
            if alpha_count > 0 {
                self.gl.enable(WebGl2RenderingContext::DEPTH_TEST);
                self.gl.depth_mask(false);
                self.gl.enable(WebGl2RenderingContext::BLEND);
                self.draw_strips(external_texture_runs, opaque_count, alpha_count)?;
            }

            // Restore state.
            self.gl.disable(WebGl2RenderingContext::DEPTH_TEST);
            self.gl.depth_mask(true);
            self.gl.enable(WebGl2RenderingContext::BLEND);
        } else {
            // Intermediate target: single draw with blending, no depth.
            debug_assert_eq!(
                opaque_count, 0,
                "opaque strips require the final view's depth attachment"
            );
            self.gl.disable(WebGl2RenderingContext::DEPTH_TEST);
            self.gl.depth_mask(false);
            self.gl.enable(WebGl2RenderingContext::BLEND);
            let total_count = opaque_count
                .checked_add(alpha_count)
                .ok_or(WebGlError::TooManyInstances)?;
            self.draw_strips(external_texture_runs, 0, total_count)?;
        }

        // Clean up.
        self.gl.bind_vertex_array(None);

        Ok(())
    }
}
