// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! WebGL rendering module for the sparse strips CPU/GPU rendering engine.
//!
//! Although wgpu can support webgl as a backend, using webgl via wgpu incurs a ~2.5MB binary size
//! increase. This backend avoids the binary size increase by using the browser's webgl context and
//! skipping wgpu. The downside is the requirement to maintain two graphics backends, a wgpu backend
//! and a webgl backend.
//!
//! For shader development, glsl shaders are automatically generated from the `wgsl` shader via
//! build.rs.

use crate::{RenderSize, Scene};
use alloc::vec::Vec;
use web_sys::{WebGl2RenderingContext, WebGlBuffer, WebGlProgram, WebGlShader, WebGlTexture};

const VERTEX_SHADER_SOURCE: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/shaders/sparse_strip_renderer.vert"
));
const FRAGMENT_SHADER_SOURCE: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/shaders/sparse_strip_renderer.frag"
));

/// WebGL-based renderer for CPU/GPU hybrid rendering.
///
/// It matches the API pattern of the existing Renderer for easier integration.
#[derive(Debug)]
pub struct WebGLRenderer {
    gl: WebGl2RenderingContext,
    program: Option<WebGlProgram>,
    strips_buffer: Option<WebGlBuffer>,
    config_buffer: Option<WebGlBuffer>,
    alpha_texture: Option<WebGlTexture>,
    render_size: RenderSize,
    strips_count: i32,
    alpha_data: Vec<u8>,
}

impl WebGLRenderer {
    /// Creates a new renderer
    pub fn new(gl: &WebGl2RenderingContext) -> Self {
        let gl = gl.clone();
        let program = create_shader_program(&gl);
        let strips_buffer = gl.create_buffer();
        let config_buffer = gl.create_buffer();
        let alpha_texture = gl.create_texture();

        Self {
            gl,
            program,
            render_size: RenderSize {
                width: 0,
                height: 0,
            },
            strips_buffer,
            config_buffer,
            alpha_texture,
            strips_count: 1,
            alpha_data: Vec::new(),
        }
    }

    /// Prepares the scene for rendering.
    ///
    /// Processes the scene data and uploads it to the GPU.
    pub fn prepare(&mut self, scene: &Scene, render_size: &RenderSize) {
        self.render_size = render_size.clone();

        // Setup config uniform buffer
        if let Some(program) = &self.program {
            self.gl.use_program(Some(program));

            let config_block_index = self.gl.get_uniform_block_index(program, "ConfigBlock");
            if config_block_index != WebGl2RenderingContext::INVALID_INDEX {
                self.gl
                    .uniform_block_binding(program, config_block_index, 0);
                if let Some(buffer) = &self.config_buffer {
                    self.gl
                        .bind_buffer(WebGl2RenderingContext::UNIFORM_BUFFER, Some(buffer));
                    self.gl.bind_buffer_base(
                        WebGl2RenderingContext::UNIFORM_BUFFER,
                        0,
                        Some(buffer),
                    );

                    let max_texture_dimension_2d = self
                        .gl
                        .get_parameter(WebGl2RenderingContext::MAX_TEXTURE_SIZE)
                        .unwrap()
                        .as_f64()
                        .unwrap() as u32;

                    let alphas_tex_width_bits = max_texture_dimension_2d.trailing_zeros();

                    // Config data as 4 u32 values
                    let config_data = [
                        render_size.width,
                        render_size.height,
                        vello_common::tile::Tile::HEIGHT as u32,
                        alphas_tex_width_bits,
                    ];

                    let array = js_sys::Uint32Array::from(config_data.as_ref());
                    self.gl.buffer_data_with_array_buffer_view(
                        WebGl2RenderingContext::UNIFORM_BUFFER,
                        &array,
                        WebGl2RenderingContext::STATIC_DRAW,
                    );
                }
            }
        }

        // Get render data from scene
        let render_data = scene.prepare_render_data();

        // Process alpha data for texture
        let alpha_len = render_data.alphas.len();
        if alpha_len > 0 {
            // Get max texture size
            let max_texture_dimension_2d = self
                .gl
                .get_parameter(WebGl2RenderingContext::MAX_TEXTURE_SIZE)
                .unwrap()
                .as_f64()
                .unwrap() as u32;

            // Calculate texture dimensions needed
            // There are 16 1-byte alpha values per texel (4 channels x 4 bytes each)
            let texture_width = max_texture_dimension_2d;
            let texture_height = (alpha_len as u32 + 15) / 16 / texture_width + 1;

            // Resize alpha data buffer to hold packed values
            let texture_size = (texture_width * texture_height * 16) as usize;
            self.alpha_data.resize(texture_size, 0);
            assert!(
                texture_height <= max_texture_dimension_2d,
                "Alpha texture height exceeds max texture dimensions"
            );
            // Copy alpha data to our buffer
            self.alpha_data[0..alpha_len].copy_from_slice(&render_data.alphas);

            // Setup texture
            if let Some(texture) = &self.alpha_texture {
                self.gl.active_texture(WebGl2RenderingContext::TEXTURE0);
                self.gl
                    .bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(texture));
                self.gl.tex_parameteri(
                    WebGl2RenderingContext::TEXTURE_2D,
                    WebGl2RenderingContext::TEXTURE_MIN_FILTER,
                    WebGl2RenderingContext::NEAREST as i32,
                );
                self.gl.tex_parameteri(
                    WebGl2RenderingContext::TEXTURE_2D,
                    WebGl2RenderingContext::TEXTURE_MAG_FILTER,
                    WebGl2RenderingContext::NEAREST as i32,
                );
                self.gl.tex_parameteri(
                    WebGl2RenderingContext::TEXTURE_2D,
                    WebGl2RenderingContext::TEXTURE_WRAP_S,
                    WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
                );
                self.gl.tex_parameteri(
                    WebGl2RenderingContext::TEXTURE_2D,
                    WebGl2RenderingContext::TEXTURE_WRAP_T,
                    WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
                );

                // Pack alpha values into RGBA uint32 texture
                let texture_size_in_pixels = (texture_width * texture_height) as usize;
                let texture_size_in_bytes = texture_size_in_pixels * 16; // 16 bytes per texel (4 uint32s)

                let alpha_data_as_u32 =
                    bytemuck::cast_slice::<u8, u32>(&self.alpha_data[0..texture_size_in_bytes]);
                let packed_array = js_sys::Uint32Array::from(alpha_data_as_u32);

                self.gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
                    WebGl2RenderingContext::TEXTURE_2D,
                    0,
                    WebGl2RenderingContext::RGBA32UI as i32,
                    texture_width as i32,
                    texture_height as i32,
                    0,
                    WebGl2RenderingContext::RGBA_INTEGER,
                    WebGl2RenderingContext::UNSIGNED_INT,
                    Some(&packed_array)
                ).expect("Failed to write alpha texture");
            }
        }

        if let Some(buffer) = &self.strips_buffer {
            self.gl
                .bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(buffer));

            if !render_data.strips.is_empty() {
                let strips_array =
                    js_sys::Uint8Array::from(bytemuck::cast_slice(&render_data.strips));
                self.gl.buffer_data_with_array_buffer_view(
                    WebGl2RenderingContext::ARRAY_BUFFER,
                    &strips_array,
                    WebGl2RenderingContext::STATIC_DRAW,
                );

                self.strips_count = render_data.strips.len() as i32;
            } else {
                self.strips_count = 0;
            }
        }

        if let Some(program) = &self.program {
            let texture_loc = self
                .gl
                .get_uniform_location(program, "alphas_texture")
                .unwrap();
            self.gl.uniform1i(Some(&texture_loc), 0);
        }
    }

    /// Renders the scene to the WebGL context.
    pub fn render(&self, _scene: &Scene) {
        // Set viewport based on render size
        self.gl.viewport(
            0,
            0,
            self.render_size.width as i32,
            self.render_size.height as i32,
        );

        self.gl.clear_color(0.0, 0.0, 0.0, 1.0);
        self.gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);

        if let Some(program) = &self.program {
            self.gl.use_program(Some(program));

            let buffer = self.strips_buffer.as_ref().unwrap();
            self.gl
                .bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(buffer));

            // Setup xy attribute
            let xy_loc = self.gl.get_attrib_location(program, "xy") as u32;
            self.gl.enable_vertex_attrib_array(xy_loc);
            self.gl.vertex_attrib_i_pointer_with_i32(
                xy_loc,
                1,
                WebGl2RenderingContext::UNSIGNED_INT,
                16, // stride (4 uint32s)
                0,
            );
            self.gl.vertex_attrib_divisor(xy_loc, 1);

            // Setup widths attribute
            let widths_loc = self.gl.get_attrib_location(program, "widths") as u32;
            self.gl.enable_vertex_attrib_array(widths_loc);
            self.gl.vertex_attrib_i_pointer_with_i32(
                widths_loc,
                1,
                WebGl2RenderingContext::UNSIGNED_INT,
                16, // stride (4 uint32s)
                4,
            );
            self.gl.vertex_attrib_divisor(widths_loc, 1);

            // Setup cols attribute (location 2)
            let col_loc = self.gl.get_attrib_location(program, "col") as u32;
            self.gl.enable_vertex_attrib_array(col_loc);
            self.gl.vertex_attrib_i_pointer_with_i32(
                col_loc,
                1,
                WebGl2RenderingContext::UNSIGNED_INT,
                16, // stride (4 uint32s)
                8,
            );
            self.gl.vertex_attrib_divisor(col_loc, 1);

            // Setup rgba attribute (location 3)
            let rgba_loc = self.gl.get_attrib_location(program, "rgba") as u32;
            self.gl.enable_vertex_attrib_array(rgba_loc);
            self.gl.vertex_attrib_i_pointer_with_i32(
                rgba_loc,
                1,
                WebGl2RenderingContext::UNSIGNED_INT,
                16, // stride (4 uint32s)
                12,
            );
            self.gl.vertex_attrib_divisor(rgba_loc, 1);

            // Activate alpha texture
            if let Some(texture) = &self.alpha_texture {
                self.gl.active_texture(WebGl2RenderingContext::TEXTURE0);
                self.gl
                    .bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(texture));
            }

            self.gl.enable(WebGl2RenderingContext::BLEND);
            self.gl.blend_func(
                WebGl2RenderingContext::ONE,
                WebGl2RenderingContext::ONE_MINUS_SRC_ALPHA,
            );

            self.gl.draw_arrays_instanced(
                WebGl2RenderingContext::TRIANGLE_STRIP,
                0,
                4,
                self.strips_count,
            );
        }
    }

    /// Resize the renderer.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.render_size = RenderSize { width, height };
    }
}

/// Helper for creating a WebGL2 shader program.
fn create_shader_program(gl: &WebGl2RenderingContext) -> Option<WebGlProgram> {
    let compile_shader = |shader_type: u32, source: &str| -> Option<WebGlShader> {
        let shader = gl.create_shader(shader_type)?;
        gl.shader_source(&shader, source);
        gl.compile_shader(&shader);

        if !gl
            .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
            .as_bool()
            .unwrap_or(false)
        {
            gl.delete_shader(Some(&shader));
            return None;
        }

        Some(shader)
    };

    let vertex_shader =
        compile_shader(WebGl2RenderingContext::VERTEX_SHADER, VERTEX_SHADER_SOURCE)?;

    let fragment_shader = compile_shader(
        WebGl2RenderingContext::FRAGMENT_SHADER,
        FRAGMENT_SHADER_SOURCE,
    )?;

    let program = gl.create_program()?;
    gl.attach_shader(&program, &vertex_shader);
    gl.attach_shader(&program, &fragment_shader);
    gl.link_program(&program);

    if !gl
        .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        gl.delete_shader(Some(&vertex_shader));
        gl.delete_shader(Some(&fragment_shader));
        gl.delete_program(Some(&program));
        return None;
    }

    gl.delete_shader(Some(&vertex_shader));
    gl.delete_shader(Some(&fragment_shader));

    Some(program)
}
