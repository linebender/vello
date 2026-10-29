#[derive(Debug)]
struct CopyUniforms {
    source_texture: WebGlUniformLocation,
}

    fn upload_copy_instances(&self, gl: &WebGl2RenderingContext, instances: &[GpuCopyInstance]) {
        gl.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&self.resources.copy_instance_buffer),
        );
        gl.buffer_data_with_u8_array(
            WebGl2RenderingContext::ARRAY_BUFFER,
            bytemuck::cast_slice(instances),
            WebGl2RenderingContext::DYNAMIC_DRAW,
        );
    }

fn get_copy_uniforms(
    gl: &WebGl2RenderingContext,
    program: &Program,
) -> Result<CopyUniforms, WebGlError> {
    Ok(CopyUniforms {
        source_texture: required_uniform_location(gl, program, copy::fragment::SOURCE_TEXTURE)?,
    })
}

/// Derive this from the number of fields on [`GpuCopyInstance`].
const COPY_ATTRIB_COUNT: u32 = 4;
const COPY_INSTANCE_STRIDE: i32 = size_of::<GpuCopyInstance>() as i32;

fn initialize_copy_vao(gl: &WebGl2RenderingContext, resources: &WebGlResources) {
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

