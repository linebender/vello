/// Contains the WebGL programs and resources for rendering.
#[derive(Debug)]
pub(crate) struct WebGlPrograms {
    /// Program for rendering strips.
    strip_program: Program,
    /// Uniform locations for the strip program
    strip_uniforms: StripUniforms,
    /// Program for filter passes.
    filter_program: Program,
    /// Uniform locations for the filter program.
    filter_uniforms: FilterPassUniforms,
    /// Program for performing blending.
    blend_program: Program,
    /// Uniform locations for the blend program.
    blend_uniforms: BlendUniforms,
    /// Program for performing copies.
    copy_program: Program,
    /// Uniform locations for the copy program.
    copy_uniforms: CopyUniforms,
    /// Shader compilation stats.
    shader_compilation_stats: ShaderCompilationStats,
    /// WebGL resources for rendering.
    pub(crate) resources: WebGlResources,
    /// Dimensions of the rendering target.
    render_size: RenderSize,
    /// Whether the last config buffer upload had NDC Y negation enabled.
    negate_ndc: bool,
    /// Offset of the scene within the rendering target.
    scene_offset: (i32, i32),
    /// Scratch buffer for staging encoded paints texture data.
    encoded_paints_data: Vec<u8>,
    /// Scratch buffer for staging filter data texture data.
    filter_data: Vec<u8>,
}


fn required_uniform_location(
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

fn initialize_packed_u32_attribs(gl: &WebGl2RenderingContext, count: u32, stride: i32) {
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
struct WebGlRendererContext<'a> {
    programs: &'a mut WebGlPrograms,
    gl: &'a WebGl2RenderingContext,
    texture_bindings: &'a WebGlTextureBindings,
    scratch_buffers: &'a mut ScratchBuffers,
    use_depth_buffer: bool,
}

fn checked_instance_count(value: usize) -> Result<i32, WebGlError> {
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
        self.programs.resources.texture_size
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
    fn negate_ndc(self) -> bool {
        // Only negate if we are rendering to the main frame buffer.
        matches!(self, Self::Root(RootTarget::UserSurface))
    }
}

