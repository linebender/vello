// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Execution of the scheduled render passes.

pub(crate) mod blend;
pub(crate) mod clear;
pub(crate) mod copy;
pub(crate) mod filter;
pub(crate) mod strip;

use crate::draw::ExternalTextureRun;
use crate::render::webgl::bindings::WebGlTextureBindings;
use crate::render::webgl::gl::resource::Program;
use crate::render::webgl::init::ShaderCompilationStats;
use crate::render::webgl::passes::blend::BlendUniforms;
use crate::render::webgl::passes::copy::CopyUniforms;
use crate::render::webgl::passes::filter::FilterPassUniforms;
use crate::render::webgl::passes::strip::StripUniforms;
use crate::render::webgl::resources::WebGlResources;
use crate::render::webgl::{WebGlError, WebGlOperation, WebGlShaderInterfaceOperation};
use crate::util::RangedSlice;
use crate::{
    ClearSettings, GpuStrip,
    blend::BlendStrip,
    filter::FilterPassPlan,
    render::common::ScratchBuffers,
    schedule::{Backend, round::BlendOp},
    target::{
        BlendPassBindings, DrawPassBindings, DrawPassTarget, FilterPassBindings, LayerTextureId,
        RootTarget,
    },
};
use vello_common::color::AlphaColor;
use vello_common::geometry::{RectU16, SizeU16};
use web_sys::{WebGl2RenderingContext, WebGlUniformLocation};

/// Contains the compiled WebGL programs.
#[derive(Debug)]
pub(crate) struct WebGlPrograms {
    /// Program for rendering strips.
    pub(crate) strip_program: Program,
    /// Uniform locations for the strip program
    pub(crate) strip_uniforms: StripUniforms,
    /// Program for filter passes.
    pub(crate) filter_program: Program,
    /// Uniform locations for the filter program.
    pub(crate) filter_uniforms: FilterPassUniforms,
    /// Program for performing blending.
    pub(crate) blend_program: Program,
    /// Uniform locations for the blend program.
    pub(crate) blend_uniforms: BlendUniforms,
    /// Program for performing copies.
    pub(crate) copy_program: Program,
    /// Uniform locations for the copy program.
    pub(crate) copy_uniforms: CopyUniforms,
    /// Shader compilation stats.
    pub(crate) shader_compilation_stats: ShaderCompilationStats,
}

pub(crate) fn required_uniform_location(
    gl: &WebGl2RenderingContext,
    program: &Program,
    name: &'static str,
) -> Result<WebGlUniformLocation, WebGlError> {
    gl.get_uniform_location(program, name)
        .ok_or_else(|| WebGlError::OperationFailed {
            operation: WebGlOperation::ShaderInterface(WebGlShaderInterfaceOperation::Uniform),
            message: Some(name.into()),
        })
}

pub(crate) fn initialize_packed_u32_attribs(gl: &WebGl2RenderingContext, count: u32, stride: i32) {
    for loc in 0..count {
        gl.enable_vertex_attrib_array(loc);
        gl.vertex_attrib_i_pointer_with_i32(
            loc,
            1,
            WebGl2RenderingContext::UNSIGNED_INT,
            stride,
            i32::try_from(loc).unwrap() * size_of::<u32>() as i32,
        );
        gl.vertex_attrib_divisor(loc, 1);
    }
}

/// Context for WebGL rendering operations.
// TODO: Improve buffer management. Currently a single buffer is used per resource, which means that
// the GPU must finish drawing before the next `upload_strip_pair` can be executed (effectively pausing
// execution). Investigate a buffer pool or creating a new buffer per pass.
pub(crate) struct WebGlRendererContext<'a> {
    pub(crate) programs: &'a WebGlPrograms,
    pub(crate) resources: &'a mut WebGlResources,
    pub(crate) gl: &'a WebGl2RenderingContext,
    pub(crate) texture_bindings: &'a WebGlTextureBindings,
    pub(crate) scratch_buffers: &'a mut ScratchBuffers,
    pub(crate) use_depth_buffer: bool,
}

pub(crate) fn checked_instance_count(value: usize) -> Result<i32, WebGlError> {
    i32::try_from(value).map_err(|_| WebGlError::TooManyInstances)
}

impl WebGlRendererContext<'_> {
    /// Draw `instance_count` quad instances using the currently bound VAO.
    fn draw_instanced_quads(&self, instance_count: i32) {
        // We use `drawElementsInstanced` instead of `drawArraysInstanced` due to a bug on
        // older Chrome versions where integer vertex attributes aren't handled correctly.
        // See https://github.com/linebender/vello/pull/1819 for more information.
        self.gl.draw_elements_instanced_with_i32(
            WebGl2RenderingContext::TRIANGLE_STRIP,
            4,
            WebGl2RenderingContext::UNSIGNED_SHORT,
            0,
            instance_count,
        );
    }

    fn texture_size(&self) -> SizeU16 {
        self.resources.texture_size
    }
}

impl Backend for WebGlRendererContext<'_> {
    type Error = WebGlError;

    fn opaque_draw_pass(
        &mut self,
        strips: &[GpuStrip],
        external_texture_runs: &[ExternalTextureRun],
    ) -> Result<(), Self::Error> {
        self.strip_pass_inner(
            strips,
            RangedSlice::empty(),
            external_texture_runs,
            DrawPassTarget::Root(RootTarget::UserSurface),
            None,
        )
    }

    fn draw_pass(
        &mut self,
        strips: RangedSlice<'_, GpuStrip>,
        external_texture_runs: &[ExternalTextureRun],
        bindings: DrawPassBindings,
    ) -> Result<(), Self::Error> {
        self.strip_pass_inner(
            &[],
            strips,
            external_texture_runs,
            bindings.target,
            bindings.child,
        )
    }

    fn blend_pass(
        &mut self,
        blends: RangedSlice<'_, BlendOp>,
        blend_strips: &[BlendStrip],
        bindings: BlendPassBindings,
    ) -> Result<(), Self::Error> {
        self.blend_pass_inner(blends, blend_strips, bindings)
    }

    fn filter_pass(
        &mut self,
        plan: &FilterPassPlan,
        bindings: FilterPassBindings,
    ) -> Result<(), Self::Error> {
        self.filter_pass_inner(plan, bindings)
    }

    fn clear_pass(&mut self, target: LayerTextureId, rects: &[RectU16]) -> Result<(), Self::Error> {
        self.clear_pass_inner(
            DrawPassTarget::Layer(target),
            ClearSettings::Rects {
                color: AlphaColor::TRANSPARENT,
                rects,
            },
        );

        Ok(())
    }
}

impl DrawPassTarget {
    pub(crate) fn negate_ndc(self) -> bool {
        // Only negate if we are rendering to the main frame buffer.
        matches!(self, Self::Root(RootTarget::UserSurface))
    }
}
