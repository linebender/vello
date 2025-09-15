// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Native WebGL2 rendering module for the sparse strips CPU/GPU rendering engine.
//!
//! This module provides identical functionality as the [`render_wgpu`] module, however the graphics
//! context is the browser's native [`WebGl2RenderingContext`]. Hence, this module is only available
//! when targeting `wasm32` with the "webgl" feature flag active.
//!
//! The main benefit of this module, is binary size. Omitting `wgpu` saves approximately 3mb of
//! binary size (when targeting WebGL2).
//!
//! Maintaining this backend should be continually re-evaluated, as once the majority of users can
//! leverage WebGPU, we can remove this backend without the binary size increasing.
//!  - WebGPU usage: <https://caniuse.com/webgpu>

#![expect(
    clippy::cast_possible_truncation,
    reason = "We temporarily ignore those because the casts\
only break in edge cases, and some of them are also only related to conversions from f64 to f32."
)]

use crate::{
    GpuStrip, RenderError, RenderSettings, RenderSize,
    gradient_cache::GradientRampCache,
    image_cache::{ImageCache, ImageResource},
    render::{
        Config,
        common::{
            GPU_ENCODED_IMAGE_SIZE_TEXELS, GPU_LINEAR_GRADIENT_SIZE_TEXELS,
            GPU_RADIAL_GRADIENT_SIZE_TEXELS, GPU_SWEEP_GRADIENT_SIZE_TEXELS, GpuEncodedImage,
            GpuEncodedPaint, GpuLinearGradient, GpuRadialGradient, GpuSweepGradient,
            pack_image_offset, pack_image_size, pack_quality_and_extend_modes,
            pack_radial_kind_and_swapped, pack_texture_width_and_extend_mode,
        },
    },
    scene::Scene,
    schedule::{LoadOp, RendererBackend, Scheduler},
};

use alloc::vec;
use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};
use core::{fmt::Debug, mem};
use vello_common::{
    coarse::WideTile,
    encode::{EncodedKind, EncodedPaint, MAX_GRADIENT_LUT_SIZE, RadialKind},
    kurbo::Affine,
    paint::ImageSource,
    tile::Tile,
};
use vello_sparse_shaders::{clear_slots, render_strips};
use web_sys::wasm_bindgen::{JsCast, JsValue};
use web_sys::{
    WebGl2RenderingContext, WebGlBuffer, WebGlFramebuffer, WebGlProgram, WebGlTexture,
    WebGlUniformLocation, WebGlVertexArrayObject,
};

/// Placeholder value for uninitialized GPU encoded paints.
const GPU_PAINT_PLACEHOLDER: GpuEncodedPaint = GpuEncodedPaint::LinearGradient(GpuLinearGradient {
    texture_width_and_extend_mode: 0,
    gradient_start: 0,
    transform: [0.0; 6],
});

/// Query the WebGL context for the max texture size.
fn get_max_texture_dimension_2d(gl: &WebGl2RenderingContext) -> u32 {
    gl.get_parameter(WebGl2RenderingContext::MAX_TEXTURE_SIZE)
        .unwrap()
        .as_f64()
        .unwrap() as u32
}

/// Vello Hybrid's WebGL2 Renderer.
#[derive(Debug)]
pub struct WebGlRenderer {
    /// Programs for rendering.
    programs: WebGlPrograms,
    /// Scheduler for scheduling draws.
    scheduler: Scheduler,
    /// WebGL context.
    gl: WebGl2RenderingContext,
    /// Image cache for storing images atlas allocations.
    image_cache: ImageCache,
    /// Encoded paints for storing encoded paints.
    encoded_paints: Vec<GpuEncodedPaint>,
    /// Stores the index (offset) of the encoded paints in the encoded paints texture.
    paint_idxs: Vec<u32>,
    /// Gradient cache for storing gradient ramps.
    gradient_cache: GradientRampCache,
}

impl WebGlRenderer {
    /// Creates a new WebGL2 renderer
    pub fn new(canvas: &web_sys::HtmlCanvasElement) -> Self {
        Self::new_with(canvas, RenderSettings::default())
    }

    /// Creates a new WebGL2 renderer with specific settings.
    pub fn new_with(canvas: &web_sys::HtmlCanvasElement, settings: RenderSettings) -> Self {
        super::common::maybe_warn_about_webgl_feature_conflict();

        // The WebGL context must be created with anti-aliasing disabled such that we can blit the
        // view framebuffer onto the default framebuffer. This technique is required for the code
        // that converts the WebGPU coordinate system into the WebGL coordinate system, adapted from
        // the `wgpu` library. The coordinate space is fixed via two steps:
        //   1. naga adds a coordinate transform to the glsl vertex shaders – however Y axis remains
        //      flipped.
        //   2. A view framebuffer is used as an intermediate render target. The final result is
        //      blit onto the  default framebuffer reflected to fix the flipped Y axis.
        // Anti-aliasing causes the blit operation to fail.
        let context_options = js_sys::Object::new();
        js_sys::Reflect::set(&context_options, &"antialias".into(), &JsValue::FALSE).unwrap();

        let gl = canvas
            .get_context_with_context_options("webgl2", &context_options)
            .expect("WebGL2 context to be available")
            .unwrap()
            .dyn_into::<WebGl2RenderingContext>()
            .expect("Context to be a WebGL2 context");

        #[cfg(debug_assertions)]
        {
            // If a WebGL context already exists on this canvas, it will be returned instead of
            // creating a new one with the correct context_options set. A cached context with
            // antialias enabled will cause vello_hybrid to fail silently. This assertion catches
            // that error early.
            let context_attributes = gl.get_context_attributes().unwrap();
            let antialias = js_sys::Reflect::get(&context_attributes, &"antialias".into())
                .unwrap()
                .as_bool()
                .unwrap();
            debug_assert!(
                !antialias,
                "WebGL context must be created with `antialias: false` for vello_hybrid to work correctly."
            );
        }

        let max_texture_dimension_2d = get_max_texture_dimension_2d(&gl);
        let total_slots: usize = (max_texture_dimension_2d / u32::from(Tile::HEIGHT)) as usize;
        let image_cache = ImageCache::new(max_texture_dimension_2d, max_texture_dimension_2d);
        // Estimate the maximum number of gradient cache entries based on the max texture dimension
        // and the maximum gradient LUT size - worst case scenario.
        let max_gradient_cache_size =
            max_texture_dimension_2d * max_texture_dimension_2d / MAX_GRADIENT_LUT_SIZE as u32;
        let gradient_cache = GradientRampCache::new(max_gradient_cache_size, settings.level);

        Self {
            programs: WebGlPrograms::new(gl.clone(), total_slots),
            scheduler: Scheduler::new(total_slots),
            gl,
            image_cache,
            encoded_paints: Vec::new(),
            paint_idxs: Vec::new(),
            gradient_cache,
        }
    }

    /// Render `scene` using WebGL2
    ///
    /// This method creates GPU resources as needed and schedules potentially multiple draw calls.
    pub fn render(&mut self, scene: &Scene, render_size: &RenderSize) -> Result<(), RenderError> {
        debug_assert_eq!(
            RenderSize {
                width: self.gl.drawing_buffer_width() as u32,
                height: self.gl.drawing_buffer_height() as u32
            },
            *render_size,
            "Render size must match drawing buffer size"
        );

        self.prepare_gpu_encoded_paints(&scene.encoded_paints);
        // TODO: For the time being, we upload the entire alpha buffer as one big chunk. As a future
        // refinement, we could have a bounded alpha buffer, and break draws when the alpha
        // buffer fills.
        self.programs.prepare(
            &self.gl,
            &mut self.gradient_cache,
            &self.encoded_paints,
            scene.strip_generator.alpha_buf(),
            render_size,
            &self.paint_idxs,
        );
        let mut ctx = WebGlRendererContext {
            programs: &mut self.programs,
            gl: &self.gl,
        };
        self.scheduler.do_scene(&mut ctx, scene, &self.paint_idxs)?;

        // Blit the view framebuffer to the default framebuffer (canvas element), reflecting the
        // image along the Y axis to complete the WebGPU to WebGL2 coordinate transform.
        self.gl.bind_framebuffer(
            WebGl2RenderingContext::READ_FRAMEBUFFER,
            Some(&self.programs.resources.view_framebuffer),
        );
        #[cfg(debug_assertions)]
        {
            let status = self
                .gl
                .check_framebuffer_status(WebGl2RenderingContext::READ_FRAMEBUFFER);
            debug_assert_eq!(
                status,
                WebGl2RenderingContext::FRAMEBUFFER_COMPLETE,
                "read framebuffer not complete"
            );
        }

        self.gl
            .bind_framebuffer(WebGl2RenderingContext::DRAW_FRAMEBUFFER, None);

        #[cfg(debug_assertions)]
        {
            let status = self
                .gl
                .check_framebuffer_status(WebGl2RenderingContext::DRAW_FRAMEBUFFER);
            debug_assert_eq!(
                status,
                WebGl2RenderingContext::FRAMEBUFFER_COMPLETE,
                "write framebuffer not complete"
            );
        }

        self.gl.blit_framebuffer(
            0,
            render_size.height as i32,
            render_size.width as i32,
            0,
            0,
            0,
            render_size.width as i32,
            render_size.height as i32,
            WebGl2RenderingContext::COLOR_BUFFER_BIT,
            WebGl2RenderingContext::LINEAR,
        );

        #[cfg(debug_assertions)]
        {
            // `get_error` cause synchronous stalls on the calling thread. It's best practice in
            // release to omit this call.
            // Reference:
            //   https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices#avoid_blocking_api_calls_in_production
            let error = self.gl.get_error();
            if error != WebGl2RenderingContext::NO_ERROR {
                panic!("WebGL error {error}");
            }
        }
        Ok(())
    }

    /// Upload image to cache and atlas in one step. Returns the `ImageId`.
    ///
    /// This is the WebGL analogue of the wgpu Renderer's `upload_image` method.
    /// It allocates space in the image cache and uploads the image data to the atlas texture.
    pub fn upload_image<T: WebGlAtlasWriter>(
        &mut self,
        writer: &T,
    ) -> vello_common::paint::ImageId {
        let width = writer.width();
        let height = writer.height();
        let image_id = self.image_cache.allocate(width, height);
        let image_resource = self
            .image_cache
            .get(image_id)
            .expect("Image resource not found");
        let offset = [
            image_resource.offset[0] as u32,
            image_resource.offset[1] as u32,
        ];

        writer.write_to_atlas(
            &self.gl,
            &self.programs.resources.atlas_texture,
            offset,
            width,
            height,
        );

        image_id
    }

    /// Get a reference to the underlying WebGL context.
    ///
    /// This allows direct access to WebGL operations for advanced use cases like texture creation.
    pub fn gl_context(&self) -> &WebGl2RenderingContext {
        &self.gl
    }

    /// Destroy an image from the cache and clear the allocated slot in the atlas.
    pub fn destroy_image(&mut self, image_id: vello_common::paint::ImageId) {
        if let Some(image_resource) = self.image_cache.deallocate(image_id) {
            self.clear_atlas_region(
                [
                    image_resource.offset[0] as u32,
                    image_resource.offset[1] as u32,
                ],
                image_resource.width as u32,
                image_resource.height as u32,
            );
        }
    }

    /// Clear a specific region of the atlas texture with uninitialized data.
    fn clear_atlas_region(&mut self, offset: [u32; 2], width: u32, height: u32) {
        // Rgba8Unorm is 4 bytes per pixel
        let uninitialized_data = vec![0_u8; (width * height * 4) as usize];

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&self.programs.resources.atlas_texture),
        );

        self.gl
            .tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
                WebGl2RenderingContext::TEXTURE_2D,
                0,                // mip level
                offset[0] as i32, // x offset
                offset[1] as i32, // y offset
                width as i32,
                height as i32,
                WebGl2RenderingContext::RGBA,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                Some(&uninitialized_data),
            )
            .unwrap();
    }

    fn prepare_gpu_encoded_paints(&mut self, encoded_paints: &[EncodedPaint]) {
        self.encoded_paints
            .resize_with(encoded_paints.len(), || GPU_PAINT_PLACEHOLDER);
        self.paint_idxs.resize(encoded_paints.len() + 1, 0);

        let mut current_idx = 0;
        for (encoded_paint_idx, paint) in encoded_paints.iter().enumerate() {
            self.paint_idxs[encoded_paint_idx] = current_idx;
            match paint {
                EncodedPaint::Image(img) => {
                    if let ImageSource::OpaqueId(image_id) = img.source {
                        let image_resource: Option<&ImageResource> = self.image_cache.get(image_id);
                        if let Some(image_resource) = image_resource {
                            let gpu_image = self.encode_image_paint(img, image_resource);
                            self.encoded_paints[encoded_paint_idx] = gpu_image;
                            current_idx += GPU_ENCODED_IMAGE_SIZE_TEXELS;
                        }
                    }
                }
                EncodedPaint::Gradient(gradient) => {
                    let (gradient_start, gradient_width) =
                        self.gradient_cache.get_or_create_ramp(gradient);
                    let gpu_gradient =
                        self.encode_gradient_paint(gradient, gradient_width, gradient_start);
                    let gradient_size_texels = match &gpu_gradient {
                        GpuEncodedPaint::LinearGradient(_) => GPU_LINEAR_GRADIENT_SIZE_TEXELS,
                        GpuEncodedPaint::RadialGradient(_) => GPU_RADIAL_GRADIENT_SIZE_TEXELS,
                        GpuEncodedPaint::SweepGradient(_) => GPU_SWEEP_GRADIENT_SIZE_TEXELS,
                        _ => unreachable!("encode_gradient_for_gpu only returns gradient types"),
                    };
                    self.encoded_paints[encoded_paint_idx] = gpu_gradient;
                    current_idx += gradient_size_texels;
                }
                EncodedPaint::BlurredRoundedRect(_blurred_rect) => {
                    // TODO: Blurred rounded rectangles are not yet supported
                    log::warn!(
                        "Blurred rounded rectangles are not yet supported in sparse strips hybrid renderer"
                    );
                }
            }
        }
        self.paint_idxs[encoded_paints.len()] = current_idx;
    }

    fn encode_image_paint(
        &self,
        image: &vello_common::encode::EncodedImage,
        image_resource: &ImageResource,
    ) -> GpuEncodedPaint {
        let image_transform = image.transform * Affine::translate((-0.5, -0.5));
        let transform = image_transform.as_coeffs().map(|x| x as f32);
        let image_size = pack_image_size(image_resource.width, image_resource.height);
        let image_offset = pack_image_offset(image_resource.offset[0], image_resource.offset[1]);
        let quality_and_extend_modes = pack_quality_and_extend_modes(
            image.extends.0 as u32,
            image.extends.1 as u32,
            image.quality as u32,
        );

        GpuEncodedPaint::Image(GpuEncodedImage {
            quality_and_extend_modes,
            image_size,
            image_offset,
            transform,
            _padding: [0, 0, 0],
        })
    }

    fn encode_gradient_paint(
        &self,
        gradient: &vello_common::encode::EncodedGradient,
        gradient_width: u32,
        gradient_start: u32,
    ) -> GpuEncodedPaint {
        let gradient_transform = gradient.transform * Affine::translate((-0.5, -0.5));
        let transform = gradient_transform.as_coeffs().map(|x| x as f32);
        let extend_mode = match gradient.pad {
            true => 0,
            false => 1,
        };
        let texture_width_and_extend_mode =
            pack_texture_width_and_extend_mode(gradient_width, extend_mode);

        match &gradient.kind {
            EncodedKind::Linear(_) => GpuEncodedPaint::LinearGradient(GpuLinearGradient {
                texture_width_and_extend_mode,
                gradient_start,
                transform,
            }),
            EncodedKind::Radial(radial) => {
                let (kind, bias, scale, fp0, fp1, fr1, f_focal_x, f_is_swapped, scaled_r0_squared) =
                    match radial {
                        RadialKind::Radial { bias, scale } => {
                            (0, *bias, *scale, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
                        }
                        RadialKind::Strip { scaled_r0_squared } => {
                            (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, *scaled_r0_squared)
                        }
                        RadialKind::Focal {
                            focal_data,
                            fp0,
                            fp1,
                        } => (
                            2,
                            *fp0,
                            *fp1,
                            *fp0,
                            *fp1,
                            focal_data.fr1,
                            focal_data.f_focal_x,
                            focal_data.f_is_swapped as u32,
                            0.0,
                        ),
                    };
                GpuEncodedPaint::RadialGradient(GpuRadialGradient {
                    texture_width_and_extend_mode,
                    gradient_start,
                    transform,
                    kind_and_f_is_swapped: pack_radial_kind_and_swapped(kind, f_is_swapped),
                    bias,
                    scale,
                    fp0,
                    fp1,
                    fr1,
                    f_focal_x,
                    scaled_r0_squared,
                })
            }
            EncodedKind::Sweep(sweep) => GpuEncodedPaint::SweepGradient(GpuSweepGradient {
                texture_width_and_extend_mode,
                gradient_start,
                transform,
                start_angle: sweep.start_angle,
                inv_angle_delta: sweep.inv_angle_delta,
                _padding: [0, 0],
            }),
        }
    }
}

/// Contains the WebGL programs and resources for rendering.
#[derive(Debug)]
struct WebGlPrograms {
    /// Program for rendering wide tile commands.
    strip_program: WebGlProgram,
    /// Uniform locations for the strip program
    strip_uniforms: StripUniforms,
    /// Program for clearing slots in slot textures.
    clear_program: WebGlProgram,
    /// Uniform locations for the `clear_program`.
    clear_uniforms: ClearUniforms,
    /// WebGL resources for rendering.
    resources: WebGlResources,
    /// Dimensions of the rendering target.
    render_size: RenderSize,
    /// Scratch buffer for staging alpha texture data.
    alpha_data: Vec<u8>,
    /// Scratch buffer for staging encoded paints texture data.
    encoded_paints_data: Vec<u8>,
}

/// Uniform locations for `strip_program`.
#[derive(Debug)]
struct StripUniforms {
    /// Config uniform block index for vertex shader.
    config_vs_block_index: u32,
    /// Config uniform block index for fragment shader.
    config_fs_block_index: u32,
    /// Alphas texture location.
    alphas_texture: WebGlUniformLocation,
    /// Clip input texture location.
    clip_input_texture: WebGlUniformLocation,
    /// Atlas texture location.
    atlas_texture: WebGlUniformLocation,
    /// Encoded paints texture location for fragment shader.
    encoded_paints_texture_fs: WebGlUniformLocation,
    /// Encoded paints texture location for vertex shader.
    encoded_paints_texture_vs: WebGlUniformLocation,
    /// Gradient texture location.
    gradient_texture: WebGlUniformLocation,
}

/// Uniform locations for `clear_program`.
#[derive(Debug)]
struct ClearUniforms {
    /// Config uniform block index.
    config_block_index: u32,
}

/// Contains all WebGL resources needed for rendering.
#[derive(Debug)]
struct WebGlResources {
    /// VAO for strip rendering.
    strip_vao: WebGlVertexArrayObject,
    /// Buffer for [`GpuStrip`] data.
    strips_buffer: WebGlBuffer,
    /// Texture for alpha values (used by both view and slot rendering).
    alphas_texture: WebGlTexture,
    /// Height of alpha texture.
    alpha_texture_height: u32,
    /// Texture for atlas data
    // TODO: Support more than one atlas texture
    atlas_texture: WebGlTexture,
    /// Encoded paints texture for image metadata.
    encoded_paints_texture: WebGlTexture,
    /// Height of encoded paints texture.
    encoded_paints_texture_height: u32,
    /// Gradient texture for gradient ramp data.
    gradient_texture: WebGlTexture,
    /// Height of gradient texture.
    gradient_texture_height: u32,

    /// Config buffer for rendering wide tile commands into the view texture.
    view_config_buffer: WebGlBuffer,
    /// Config buffer for rendering wide tile commands into a slot texture.
    slot_config_buffer: WebGlBuffer,

    /// Buffer for slot indices used in `clear_slots`.
    clear_slot_indices_buffer: WebGlBuffer,
    /// VAO for clear slots program.
    clear_vao: WebGlVertexArrayObject,
    /// Config buffer for clear program.
    clear_config_buffer: WebGlBuffer,

    /// Intermediate surface texture for the main view.
    view_texture: WebGlTexture,
    /// Framebuffer for the vfiew texture.
    view_framebuffer: WebGlFramebuffer,

    /// Slot textures.
    slot_textures: [WebGlTexture; 2],
    /// Framebuffers for slot textures.
    slot_framebuffers: [WebGlFramebuffer; 2],

    /// Cached result from querying `WebGl2RenderingContext::MAX_TEXTURE_SIZE` which is a blocking
    /// WebGL call.
    max_texture_dimension_2d: u32,
}

/// Config for the clear slots pipeline.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct ClearSlotsConfig {
    /// Width of a slot.
    pub slot_width: u32,
    /// Height of a slot.
    pub slot_height: u32,
    /// Total height of the texture.
    pub texture_height: u32,
    /// Padding for alignment.
    pub _padding: u32,
}

impl WebGlPrograms {
    /// Creates programs and initializes resources.
    fn new(gl: WebGl2RenderingContext, slot_count: usize) -> Self {
        let strip_program = create_shader_program(
            &gl,
            render_strips::VERTEX_SOURCE,
            render_strips::FRAGMENT_SOURCE,
        );
        let clear_program = create_shader_program(
            &gl,
            clear_slots::VERTEX_SOURCE,
            clear_slots::FRAGMENT_SOURCE,
        );

        let strip_uniforms = get_strip_uniforms(&gl, &strip_program);
        let clear_uniforms = get_clear_uniforms(&gl, &clear_program);

        let resources = create_webgl_resources(&gl, slot_count);

        initialize_strip_vao(&gl, &resources);
        initialize_clear_vao(&gl, &resources);

        let alpha_data = vec![0; (resources.max_texture_dimension_2d << 4) as usize];
        let encoded_paints_data = vec![0; (resources.max_texture_dimension_2d << 4) as usize];

        gl.enable(WebGl2RenderingContext::BLEND);
        gl.blend_func(
            WebGl2RenderingContext::ONE,
            WebGl2RenderingContext::ONE_MINUS_SRC_ALPHA,
        );

        Self {
            strip_program,
            clear_program,
            strip_uniforms,
            clear_uniforms,
            resources,
            render_size: RenderSize {
                width: 0,
                height: 0,
            },
            alpha_data,
            encoded_paints_data,
        }
    }

    /// Prepare resources for rendering.
    fn prepare(
        &mut self,
        gl: &WebGl2RenderingContext,
        gradient_cache: &mut GradientRampCache,
        encoded_paints: &[GpuEncodedPaint],
        alphas: &[u8],
        render_size: &RenderSize,
        paint_idxs: &[u32],
    ) {
        let max_texture_dimension_2d = self.resources.max_texture_dimension_2d;

        self.maybe_resize_alphas_tex(max_texture_dimension_2d, alphas);
        self.maybe_resize_encoded_paints_tex(max_texture_dimension_2d, paint_idxs);
        self.maybe_update_config_buffer(gl, max_texture_dimension_2d, render_size);

        self.upload_alpha_texture(gl, alphas);
        self.upload_encoded_paints_texture(gl, encoded_paints);

        if gradient_cache.has_changed() {
            self.maybe_resize_gradient_tex(gl, max_texture_dimension_2d, gradient_cache);
            self.upload_gradient_texture(gl, gradient_cache);
            gradient_cache.mark_synced();
        }

        self.clear_view_framebuffer(gl);
    }

    /// Update the alpha texture size if needed.
    fn maybe_resize_alphas_tex(&mut self, max_texture_dimension_2d: u32, alphas: &[u8]) {
        let required_alpha_height = (alphas.len() as u32)
            // There are 16 1-byte alpha values per texel.
            .div_ceil(max_texture_dimension_2d << 4);

        let current_alpha_height = self.resources.alpha_texture_height;
        if required_alpha_height > current_alpha_height {
            // We need to resize the alpha texture to fit the new alpha data.
            assert!(
                required_alpha_height <= max_texture_dimension_2d,
                "Alpha texture height exceeds max texture dimensions"
            );

            // Resize the alpha texture staging buffer.
            let required_alpha_size = (max_texture_dimension_2d * required_alpha_height) << 4;
            self.alpha_data.resize(required_alpha_size as usize, 0);

            // Track the new height.
            self.resources.alpha_texture_height = required_alpha_height;
        }
    }

    /// Update the encoded paints texture size if needed.
    fn maybe_resize_encoded_paints_tex(
        &mut self,
        max_texture_dimension_2d: u32,
        paint_idxs: &[u32],
    ) {
        let required_texels = paint_idxs.last().unwrap();
        let required_encoded_paints_height = required_texels.div_ceil(max_texture_dimension_2d);
        let current_encoded_paints_height = self.resources.encoded_paints_texture_height;
        if required_encoded_paints_height > current_encoded_paints_height {
            assert!(
                required_encoded_paints_height <= max_texture_dimension_2d,
                "Encoded paints texture height exceeds max texture dimensions"
            );

            let required_encoded_paints_size =
                (max_texture_dimension_2d * required_encoded_paints_height) << 4;
            self.encoded_paints_data
                .resize(required_encoded_paints_size as usize, 0);
            self.resources.encoded_paints_texture_height = required_encoded_paints_height;
        }
    }

    /// Update the gradient texture size if needed.
    fn maybe_resize_gradient_tex(
        &mut self,
        _gl: &WebGl2RenderingContext,
        max_texture_dimension_2d: u32,
        gradient_cache: &GradientRampCache,
    ) {
        if gradient_cache.is_empty() {
            return;
        }

        let gradient_data_size = gradient_cache.luts_size();
        // Each texel is RGBA8, so 4 bytes per texel
        let required_gradient_height =
            (gradient_data_size as u32).div_ceil(max_texture_dimension_2d * 4);

        let current_gradient_height = self.resources.gradient_texture_height;
        if required_gradient_height > current_gradient_height {
            assert!(
                required_gradient_height <= max_texture_dimension_2d,
                "Gradient texture height exceeds max texture dimensions"
            );

            self.resources.gradient_texture_height = required_gradient_height;
        }
    }

    /// Update config buffer if dimensions changed.
    fn maybe_update_config_buffer(
        &mut self,
        gl: &WebGl2RenderingContext,
        max_texture_dimension_2d: u32,
        new_render_size: &RenderSize,
    ) {
        if self.render_size != *new_render_size {
            // Update view config buffer
            {
                let config = Config {
                    width: new_render_size.width,
                    height: new_render_size.height,
                    strip_height: Tile::HEIGHT.into(),
                    alphas_tex_width_bits: max_texture_dimension_2d.trailing_zeros(),
                };

                gl.bind_buffer(
                    WebGl2RenderingContext::UNIFORM_BUFFER,
                    Some(&self.resources.view_config_buffer),
                );
                let config_data = bytemuck::bytes_of(&config);
                gl.buffer_data_with_u8_array(
                    WebGl2RenderingContext::UNIFORM_BUFFER,
                    config_data,
                    WebGl2RenderingContext::STATIC_DRAW,
                );
            }

            // Update slot config buffer.
            {
                let slot_config = Config {
                    width: u32::from(WideTile::WIDTH),
                    height: u32::from(Tile::HEIGHT)
                        * (max_texture_dimension_2d / u32::from(Tile::HEIGHT)),
                    strip_height: Tile::HEIGHT.into(),
                    alphas_tex_width_bits: max_texture_dimension_2d.trailing_zeros(),
                };

                gl.bind_buffer(
                    WebGl2RenderingContext::UNIFORM_BUFFER,
                    Some(&self.resources.slot_config_buffer),
                );
                let slot_config_data = bytemuck::bytes_of(&slot_config);
                gl.buffer_data_with_u8_array(
                    WebGl2RenderingContext::UNIFORM_BUFFER,
                    slot_config_data,
                    WebGl2RenderingContext::STATIC_DRAW,
                );
            }

            // Update clear config buffer.
            // TODO: This can be done once, and doesn't need to be done on every `prepare` call.
            {
                let clear_config = ClearSlotsConfig {
                    slot_width: u32::from(WideTile::WIDTH),
                    slot_height: u32::from(Tile::HEIGHT)
                        * (max_texture_dimension_2d / u32::from(Tile::HEIGHT)),
                    texture_height: Tile::HEIGHT.into(),
                    _padding: 0,
                };

                gl.bind_buffer(
                    WebGl2RenderingContext::UNIFORM_BUFFER,
                    Some(&self.resources.clear_config_buffer),
                );
                let clear_config_data = bytemuck::bytes_of(&clear_config);
                gl.buffer_data_with_u8_array(
                    WebGl2RenderingContext::UNIFORM_BUFFER,
                    clear_config_data,
                    WebGl2RenderingContext::STATIC_DRAW,
                );
            }

            // Resize the view texture.
            gl.bind_texture(
                WebGl2RenderingContext::TEXTURE_2D,
                Some(&self.resources.view_texture),
            );
            gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
                WebGl2RenderingContext::TEXTURE_2D,
                0,
                WebGl2RenderingContext::RGBA8 as i32,
                new_render_size.width as i32,
                new_render_size.height as i32,
                0,
                WebGl2RenderingContext::RGBA,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                None,
            )
            .unwrap();

            self.render_size = new_render_size.clone();
        }
    }

    /// Upload alpha data to the texture.
    fn upload_alpha_texture(&mut self, gl: &WebGl2RenderingContext, alphas: &[u8]) {
        if !alphas.is_empty() {
            let alpha_texture_width = self.resources.max_texture_dimension_2d;
            let alpha_texture_height = self.resources.alpha_texture_height;

            debug_assert!(
                alphas.len() <= (alpha_texture_width * alpha_texture_height * 16) as usize,
                "Alpha texture dimensions are too small to fit the alpha data"
            );

            // After this copy to `self.alpha_data`, there may be stale trailing alpha values. These
            // are not sampled, so can be left as-is.
            // TODO: Apply the same optimization as gradient texture upload - use alphas directly
            // instead of copying to staging buffer, by taking alphas from strip generator as a vec,
            // resizing appropriately, truncating, and restoring back to the generator.
            self.alpha_data[0..alphas.len()].copy_from_slice(alphas);
            gl.active_texture(WebGl2RenderingContext::TEXTURE0);
            gl.bind_texture(
                WebGl2RenderingContext::TEXTURE_2D,
                Some(&self.resources.alphas_texture),
            );

            // Pack alpha values into RGBA uint32 texture
            let alpha_data_as_u32 = bytemuck::cast_slice::<u8, u32>(&self.alpha_data);
            let packed_array = js_sys::Uint32Array::from(alpha_data_as_u32);

            gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
                WebGl2RenderingContext::TEXTURE_2D,
                0,
                WebGl2RenderingContext::RGBA32UI as i32,
                alpha_texture_width as i32,
                alpha_texture_height as i32,
                0,
                WebGl2RenderingContext::RGBA_INTEGER,
                WebGl2RenderingContext::UNSIGNED_INT,
                Some(&packed_array),
            )
            .unwrap();
        }
    }

    /// Upload encoded paints to the texture.
    fn upload_encoded_paints_texture(
        &mut self,
        gl: &WebGl2RenderingContext,
        encoded_paints: &[GpuEncodedPaint],
    ) {
        if !encoded_paints.is_empty() {
            let encoded_paints_texture_width = self.resources.max_texture_dimension_2d;
            let encoded_paints_texture_height = self.resources.encoded_paints_texture_height;

            GpuEncodedPaint::serialize_to_buffer(encoded_paints, &mut self.encoded_paints_data);

            gl.active_texture(WebGl2RenderingContext::TEXTURE0);
            gl.bind_texture(
                WebGl2RenderingContext::TEXTURE_2D,
                Some(&self.resources.encoded_paints_texture),
            );

            // Pack encoded paints into RGBA uint32 texture
            let encoded_paints_data_as_u32 =
                bytemuck::cast_slice::<u8, u32>(&self.encoded_paints_data);
            let packed_array = js_sys::Uint32Array::from(encoded_paints_data_as_u32);

            gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
                WebGl2RenderingContext::TEXTURE_2D,
                0,
                WebGl2RenderingContext::RGBA32UI as i32,
                encoded_paints_texture_width as i32,
                encoded_paints_texture_height as i32,
                0,
                WebGl2RenderingContext::RGBA_INTEGER,
                WebGl2RenderingContext::UNSIGNED_INT,
                Some(&packed_array),
            )
            .unwrap();
        }
    }

    /// Upload gradient data to the texture.
    fn upload_gradient_texture(
        &mut self,
        gl: &WebGl2RenderingContext,
        gradient_cache: &mut GradientRampCache,
    ) {
        if gradient_cache.is_empty() {
            return;
        }

        let gradient_texture_width = self.resources.max_texture_dimension_2d;
        let gradient_texture_height = self.resources.gradient_texture_height;
        let total_capacity = (gradient_texture_width * gradient_texture_height * 4) as usize;

        // Take ownership of the luts to avoid copying, then resize for texture padding.
        let mut luts = gradient_cache.take_luts();
        luts.resize(total_capacity, 0);

        gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&self.resources.gradient_texture),
        );

        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
            WebGl2RenderingContext::TEXTURE_2D,
            0,
            WebGl2RenderingContext::RGBA8 as i32,
            gradient_texture_width as i32,
            gradient_texture_height as i32,
            0,
            WebGl2RenderingContext::RGBA,
            WebGl2RenderingContext::UNSIGNED_BYTE,
            Some(&luts),
        )
        .unwrap();

        // Restore the luts back to the cache.
        gradient_cache.restore_luts(luts);
    }

    /// Clear the view framebuffer.
    fn clear_view_framebuffer(&mut self, gl: &WebGl2RenderingContext) {
        gl.bind_framebuffer(
            WebGl2RenderingContext::FRAMEBUFFER,
            Some(&self.resources.view_framebuffer),
        );
        gl.clear_color(0.0, 0.0, 0.0, 0.0);
        gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);
    }

    /// Upload strip data to GPU.
    fn upload_strips(&mut self, gl: &WebGl2RenderingContext, strips: &[GpuStrip]) {
        if strips.is_empty() {
            return;
        }

        gl.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&self.resources.strips_buffer),
        );
        let strips_data = bytemuck::cast_slice(strips);
        gl.buffer_data_with_u8_array(
            WebGl2RenderingContext::ARRAY_BUFFER,
            strips_data,
            WebGl2RenderingContext::DYNAMIC_DRAW,
        );
    }
}

/// Create a WebGL shader program from vertex and fragment sources.
fn create_shader_program(
    gl: &WebGl2RenderingContext,
    vertex_src: &str,
    fragment_src: &str,
) -> WebGlProgram {
    // Compile vertex shader.
    let vertex_shader = gl
        .create_shader(WebGl2RenderingContext::VERTEX_SHADER)
        .unwrap();
    gl.shader_source(&vertex_shader, vertex_src);
    gl.compile_shader(&vertex_shader);

    if !gl
        .get_shader_parameter(&vertex_shader, WebGl2RenderingContext::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        let info = gl
            .get_shader_info_log(&vertex_shader)
            .unwrap_or_else(|| "Unknown error creating vertex shader".into());
        panic!("Failed to compile vertex shader: {info}");
    }

    // Compile fragment shader.
    let fragment_shader = gl
        .create_shader(WebGl2RenderingContext::FRAGMENT_SHADER)
        .unwrap();
    gl.shader_source(&fragment_shader, fragment_src);
    gl.compile_shader(&fragment_shader);

    if !gl
        .get_shader_parameter(&fragment_shader, WebGl2RenderingContext::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        let info = gl
            .get_shader_info_log(&fragment_shader)
            .unwrap_or_else(|| "Unknown error creating fragment shader".into());
        panic!("Failed to compile fragment shader: {info}");
    }

    // Create and link the program.
    let program = gl.create_program().unwrap();
    gl.attach_shader(&program, &vertex_shader);
    gl.attach_shader(&program, &fragment_shader);
    gl.link_program(&program);

    if !gl
        .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        let info = gl
            .get_program_info_log(&program)
            .unwrap_or_else(|| "Unknown error creating program".into());
        panic!("Failed to link program: {info}");
    }

    gl.delete_shader(Some(&vertex_shader));
    gl.delete_shader(Some(&fragment_shader));

    program
}

/// Get the  uniform locations for the `render_strips` program.
fn get_strip_uniforms(gl: &WebGl2RenderingContext, program: &WebGlProgram) -> StripUniforms {
    let config_vs_name = render_strips::vertex::CONFIG;
    let config_vs_block_index = gl.get_uniform_block_index(program, config_vs_name);

    let config_fs_name = render_strips::fragment::CONFIG;
    let config_fs_block_index = gl.get_uniform_block_index(program, config_fs_name);

    debug_assert_ne!(
        config_vs_block_index,
        WebGl2RenderingContext::INVALID_INDEX,
        "invalid uniform index"
    );
    debug_assert_ne!(
        config_fs_block_index,
        WebGl2RenderingContext::INVALID_INDEX,
        "invalid uniform index"
    );

    // Bind uniform blocks to binding points.
    gl.uniform_block_binding(program, config_vs_block_index, 0);
    gl.uniform_block_binding(program, config_fs_block_index, 0);

    // Get texture uniform locations.
    let alphas_texture_name = render_strips::fragment::ALPHAS_TEXTURE;
    let clip_input_texture_name = render_strips::fragment::CLIP_INPUT_TEXTURE;
    let atlas_texture_name = render_strips::fragment::ATLAS_TEXTURE;
    let encoded_paints_texture_fs_name = render_strips::fragment::ENCODED_PAINTS_TEXTURE;
    let encoded_paints_texture_vs_name = render_strips::vertex::ENCODED_PAINTS_TEXTURE;
    let gradient_texture_name = render_strips::fragment::GRADIENT_TEXTURE;

    StripUniforms {
        config_vs_block_index,
        config_fs_block_index,
        alphas_texture: gl
            .get_uniform_location(program, alphas_texture_name)
            .unwrap(),
        clip_input_texture: gl
            .get_uniform_location(program, clip_input_texture_name)
            .unwrap(),
        atlas_texture: gl
            .get_uniform_location(program, atlas_texture_name)
            .unwrap(),
        encoded_paints_texture_fs: gl
            .get_uniform_location(program, encoded_paints_texture_fs_name)
            .unwrap(),
        encoded_paints_texture_vs: gl
            .get_uniform_location(program, encoded_paints_texture_vs_name)
            .unwrap(),
        gradient_texture: gl
            .get_uniform_location(program, gradient_texture_name)
            .unwrap(),
    }
}

/// Get the uniform locations for the `clear_slots` program.
fn get_clear_uniforms(gl: &WebGl2RenderingContext, program: &WebGlProgram) -> ClearUniforms {
    let config_name = clear_slots::vertex::CONFIG;
    let config_block_index = gl.get_uniform_block_index(program, config_name);

    debug_assert_ne!(
        config_block_index,
        WebGl2RenderingContext::INVALID_INDEX,
        "invalid uniform index"
    );

    // Bind uniform block to binding point.
    gl.uniform_block_binding(program, config_block_index, 0);

    ClearUniforms { config_block_index }
}

/// Create all WebGL resources needed for rendering.
fn create_webgl_resources(gl: &WebGl2RenderingContext, slot_count: usize) -> WebGlResources {
    let strip_vao = gl.create_vertex_array().unwrap();
    let clear_vao = gl.create_vertex_array().unwrap();

    let strips_buffer = gl.create_buffer().unwrap();
    let view_config_buffer = gl.create_buffer().unwrap();
    let slot_config_buffer = gl.create_buffer().unwrap();
    let clear_slot_indices_buffer = gl.create_buffer().unwrap();
    let clear_config_buffer = gl.create_buffer().unwrap();

    // Create and configure alpha texture.
    let alphas_texture = gl.create_texture().unwrap();
    {
        gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&alphas_texture));
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MIN_FILTER,
            WebGl2RenderingContext::NEAREST as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MAG_FILTER,
            WebGl2RenderingContext::NEAREST as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_S,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_T,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
    }

    // Create and configure atlas texture.
    let atlas_texture = gl.create_texture().unwrap();
    {
        gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&atlas_texture));
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MIN_FILTER,
            WebGl2RenderingContext::LINEAR as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MAG_FILTER,
            WebGl2RenderingContext::LINEAR as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_S,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_T,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );

        // Initialize with empty texture data
        let max_texture_dimension_2d = get_max_texture_dimension_2d(gl);
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
            WebGl2RenderingContext::TEXTURE_2D,
            0,
            WebGl2RenderingContext::RGBA8 as i32,
            max_texture_dimension_2d as i32,
            max_texture_dimension_2d as i32,
            0,
            WebGl2RenderingContext::RGBA,
            WebGl2RenderingContext::UNSIGNED_BYTE,
            None,
        )
        .unwrap();
    }

    // Create and configure encoded paints texture.
    let encoded_paints_texture = gl.create_texture().unwrap();
    {
        gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&encoded_paints_texture),
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MIN_FILTER,
            WebGl2RenderingContext::NEAREST as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MAG_FILTER,
            WebGl2RenderingContext::NEAREST as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_S,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_T,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
    }

    // Create and configure gradient texture.
    let gradient_texture = gl.create_texture().unwrap();
    {
        gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&gradient_texture));
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MIN_FILTER,
            WebGl2RenderingContext::LINEAR as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MAG_FILTER,
            WebGl2RenderingContext::LINEAR as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_S,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_T,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
    }

    // Create and configure view texture.
    let view_texture = gl.create_texture().unwrap();
    {
        gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&view_texture));
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MIN_FILTER,
            WebGl2RenderingContext::LINEAR as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MAG_FILTER,
            WebGl2RenderingContext::LINEAR as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_S,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_T,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
    };
    // Create framebuffer for the view texture.
    let view_framebuffer = create_framebuffer_for_texture(gl, &view_texture);

    // Create slot textures and framebuffers.
    let slot_textures: [WebGlTexture; 2] = [
        create_slot_texture(gl, slot_count),
        create_slot_texture(gl, slot_count),
    ];

    let slot_framebuffers: [WebGlFramebuffer; 2] = [
        create_framebuffer_for_texture(gl, &slot_textures[0]),
        create_framebuffer_for_texture(gl, &slot_textures[1]),
    ];

    let max_texture_dimension_2d = get_max_texture_dimension_2d(gl);

    WebGlResources {
        strip_vao,
        strips_buffer,
        alphas_texture,
        alpha_texture_height: 0,
        atlas_texture,
        encoded_paints_texture,
        encoded_paints_texture_height: 0,
        gradient_texture,
        gradient_texture_height: 0,
        view_config_buffer,
        slot_config_buffer,
        clear_slot_indices_buffer,
        clear_vao,
        clear_config_buffer,
        slot_textures,
        slot_framebuffers,
        view_texture,
        view_framebuffer,
        max_texture_dimension_2d,
    }
}

/// Create a texture for slot rendering.
fn create_slot_texture(gl: &WebGl2RenderingContext, slot_count: usize) -> WebGlTexture {
    let texture = gl.create_texture().unwrap();
    gl.active_texture(WebGl2RenderingContext::TEXTURE0);
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture));
    gl.tex_parameteri(
        WebGl2RenderingContext::TEXTURE_2D,
        WebGl2RenderingContext::TEXTURE_MIN_FILTER,
        WebGl2RenderingContext::NEAREST_MIPMAP_LINEAR as i32,
    );
    gl.tex_parameteri(
        WebGl2RenderingContext::TEXTURE_2D,
        WebGl2RenderingContext::TEXTURE_MAG_FILTER,
        WebGl2RenderingContext::LINEAR as i32,
    );
    gl.tex_parameteri(
        WebGl2RenderingContext::TEXTURE_2D,
        WebGl2RenderingContext::TEXTURE_WRAP_S,
        WebGl2RenderingContext::REPEAT as i32,
    );
    gl.tex_parameteri(
        WebGl2RenderingContext::TEXTURE_2D,
        WebGl2RenderingContext::TEXTURE_WRAP_T,
        WebGl2RenderingContext::REPEAT as i32,
    );
    gl.tex_parameteri(
        WebGl2RenderingContext::TEXTURE_2D,
        WebGl2RenderingContext::TEXTURE_MAX_LEVEL,
        0,
    );

    gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
        WebGl2RenderingContext::TEXTURE_2D,
        0,
        WebGl2RenderingContext::RGBA8 as i32,
        u32::from(WideTile::WIDTH) as i32,
        (u32::from(Tile::HEIGHT) * slot_count as u32) as i32,
        0,
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        None,
    )
    .unwrap();

    texture
}

/// Create a framebuffer for a texture.
fn create_framebuffer_for_texture(
    gl: &WebGl2RenderingContext,
    texture: &WebGlTexture,
) -> WebGlFramebuffer {
    let framebuffer = gl.create_framebuffer().unwrap();
    gl.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(&framebuffer));

    gl.framebuffer_texture_2d(
        WebGl2RenderingContext::FRAMEBUFFER,
        WebGl2RenderingContext::COLOR_ATTACHMENT0,
        WebGl2RenderingContext::TEXTURE_2D,
        Some(texture),
        0,
    );

    framebuffer
}

/// Initialize strip VAO.
fn initialize_strip_vao(gl: &WebGl2RenderingContext, resources: &WebGlResources) {
    gl.bind_vertex_array(Some(&resources.strip_vao));
    gl.bind_buffer(
        WebGl2RenderingContext::ARRAY_BUFFER,
        Some(&resources.strips_buffer),
    );

    let stride = mem::size_of::<GpuStrip>() as i32;
    debug_assert_eq!(stride, 20, "expected stride of 20");

    // Configure attributes.
    for i in 0..5 {
        let location = i as u32;
        let offset = i * 4;

        gl.enable_vertex_attrib_array(location);
        gl.vertex_attrib_i_pointer_with_i32(
            location,
            1,
            WebGl2RenderingContext::UNSIGNED_INT,
            stride,
            offset,
        );

        gl.vertex_attrib_divisor(location, 1);
    }

    gl.bind_vertex_array(None);
}

/// Initialize clear VAO.
fn initialize_clear_vao(gl: &WebGl2RenderingContext, resources: &WebGlResources) {
    gl.bind_vertex_array(Some(&resources.clear_vao));
    gl.bind_buffer(
        WebGl2RenderingContext::ARRAY_BUFFER,
        Some(&resources.clear_slot_indices_buffer),
    );

    // Configure attributes.
    let slot_idx_loc = 0;
    gl.enable_vertex_attrib_array(slot_idx_loc);
    gl.vertex_attrib_i_pointer_with_i32(
        slot_idx_loc,
        1,
        WebGl2RenderingContext::UNSIGNED_INT,
        4,
        0,
    );
    gl.vertex_attrib_divisor(slot_idx_loc, 1);

    gl.bind_vertex_array(None);
}

/// Context for WebGL rendering operations.
// TODO: Improve buffer management. Currently a single buffer is used per resource, which means that
// the GPU must finish drawing before the next `upload_strips` can be executed (effectively pausing
// execution). Investigate a buffer pool or creating a new buffer per pass.
struct WebGlRendererContext<'a> {
    programs: &'a mut WebGlPrograms,
    gl: &'a WebGl2RenderingContext,
}

impl WebGlRendererContext<'_> {
    /// Render the strips to either the view or a slot texture (depending on `ix`).
    fn do_strip_render_pass(&mut self, strips: &[GpuStrip], ix: usize, load: LoadOp) {
        debug_assert!(ix < 3, "Invalid texture index");
        if strips.is_empty() {
            return;
        }
        self.programs.upload_strips(self.gl, strips);

        // Bind the appropriate framebuffer.
        if ix == 2 {
            self.gl.bind_framebuffer(
                WebGl2RenderingContext::FRAMEBUFFER,
                Some(&self.programs.resources.view_framebuffer),
            );
            // Set viewport to match view framebuffer.
            let width = self.programs.render_size.width;
            let height = self.programs.render_size.height;
            self.gl.viewport(0, 0, width as i32, height as i32);

            // Use view config buffer for rendering to the main view.
            self.gl.bind_buffer_base(
                WebGl2RenderingContext::UNIFORM_BUFFER,
                self.programs.strip_uniforms.config_vs_block_index,
                Some(&self.programs.resources.view_config_buffer),
            );
            self.gl.bind_buffer_base(
                WebGl2RenderingContext::UNIFORM_BUFFER,
                self.programs.strip_uniforms.config_fs_block_index,
                Some(&self.programs.resources.view_config_buffer),
            );
        } else {
            self.gl.bind_framebuffer(
                WebGl2RenderingContext::FRAMEBUFFER,
                Some(&self.programs.resources.slot_framebuffers[ix]),
            );
            // Set viewport to match slot framebuffer.
            // TODO: Remove the slot height texture calculation.
            let total_slots: usize = (self.programs.resources.max_texture_dimension_2d
                / u32::from(Tile::HEIGHT)) as usize;
            // Set viewport to match slot texture.
            let height = u32::from(Tile::HEIGHT) * total_slots as u32;
            self.gl
                .viewport(0, 0, i32::from(WideTile::WIDTH), height as i32);

            // Use slot config buffer for rendering to a slot texture.
            self.gl.bind_buffer_base(
                WebGl2RenderingContext::UNIFORM_BUFFER,
                self.programs.strip_uniforms.config_vs_block_index,
                Some(&self.programs.resources.slot_config_buffer),
            );
            self.gl.bind_buffer_base(
                WebGl2RenderingContext::UNIFORM_BUFFER,
                self.programs.strip_uniforms.config_fs_block_index,
                Some(&self.programs.resources.slot_config_buffer),
            );
        }

        // Clear framebuffer if requested.
        if matches!(load, LoadOp::Clear) {
            self.gl.clear_color(0.0, 0.0, 0.0, 0.0);
            self.gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);
        }

        // Use the strip program.
        self.gl.use_program(Some(&self.programs.strip_program));

        // Set up attributes.
        self.gl
            .bind_vertex_array(Some(&self.programs.resources.strip_vao));

        // Bind textures.
        self.gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&self.programs.resources.alphas_texture),
        );
        self.gl
            .uniform1i(Some(&self.programs.strip_uniforms.alphas_texture), 0);

        // Bound clip textures are dependent on `ix`:
        // - ix=0 or ix=2: use slot_texture[1]
        // - ix=1: use slot_texture[0]
        self.gl.active_texture(WebGl2RenderingContext::TEXTURE1);
        let clip_texture_idx = if ix == 1 { 0 } else { 1 };
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&self.programs.resources.slot_textures[clip_texture_idx]),
        );
        self.gl
            .uniform1i(Some(&self.programs.strip_uniforms.clip_input_texture), 1);

        // Bind atlas texture for image rendering
        self.gl.active_texture(WebGl2RenderingContext::TEXTURE2);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&self.programs.resources.atlas_texture),
        );
        self.gl
            .uniform1i(Some(&self.programs.strip_uniforms.atlas_texture), 2);

        // Bind encoded paints texture for image metadata
        self.gl.active_texture(WebGl2RenderingContext::TEXTURE3);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&self.programs.resources.encoded_paints_texture),
        );
        self.gl.uniform1i(
            Some(&self.programs.strip_uniforms.encoded_paints_texture_fs),
            3,
        );
        self.gl.uniform1i(
            Some(&self.programs.strip_uniforms.encoded_paints_texture_vs),
            3,
        );

        // Bind gradient texture for gradient rendering
        self.gl.active_texture(WebGl2RenderingContext::TEXTURE4);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&self.programs.resources.gradient_texture),
        );
        self.gl
            .uniform1i(Some(&self.programs.strip_uniforms.gradient_texture), 4);

        // Draw.
        self.gl.draw_arrays_instanced(
            WebGl2RenderingContext::TRIANGLE_STRIP,
            0,
            4,
            strips.len() as i32,
        );

        // Clean up.
        self.gl.bind_vertex_array(None);
    }

    /// Clear specific slots from a slot texture.
    fn do_clear_slots_render_pass(&mut self, ix: usize, slot_indices: &[u32]) {
        if slot_indices.is_empty() {
            return;
        }

        // Upload slot indices.
        self.gl.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&self.programs.resources.clear_slot_indices_buffer),
        );
        let slot_indices_data = bytemuck::cast_slice(slot_indices);
        self.gl.buffer_data_with_u8_array(
            WebGl2RenderingContext::ARRAY_BUFFER,
            slot_indices_data,
            WebGl2RenderingContext::STATIC_DRAW,
        );

        // Bind framebuffer and setup viewport.
        self.gl.bind_framebuffer(
            WebGl2RenderingContext::FRAMEBUFFER,
            Some(&self.programs.resources.slot_framebuffers[ix]),
        );
        // TODO: Remove the slot height texture calculation.
        let total_slots: usize =
            (self.programs.resources.max_texture_dimension_2d / u32::from(Tile::HEIGHT)) as usize;
        let height = u32::from(Tile::HEIGHT) * total_slots as u32;
        self.gl
            .viewport(0, 0, i32::from(WideTile::WIDTH), height as i32);

        // Setup clear program.
        self.gl.use_program(Some(&self.programs.clear_program));

        // Set up attributes.
        self.gl
            .bind_vertex_array(Some(&self.programs.resources.clear_vao));

        // Set up clear config.
        self.gl.bind_buffer_base(
            WebGl2RenderingContext::UNIFORM_BUFFER,
            self.programs.clear_uniforms.config_block_index,
            Some(&self.programs.resources.clear_config_buffer),
        );

        // Draw.
        self.gl.draw_arrays_instanced(
            WebGl2RenderingContext::TRIANGLE_STRIP,
            0,
            4,
            slot_indices.len() as i32,
        );

        // Clean up.
        self.gl.bind_vertex_array(None);
    }
}

impl RendererBackend for WebGlRendererContext<'_> {
    /// Clear specific slots in a texture
    fn clear_slots(&mut self, texture_index: usize, slots: &[u32]) {
        self.do_clear_slots_render_pass(texture_index, slots);
    }

    /// Execute a render pass for strips.
    fn render_strips(&mut self, strips: &[GpuStrip], target_index: usize, load_op: LoadOp) {
        self.do_strip_render_pass(strips, target_index, load_op);
    }
}

/// Trait for types that can write image data directly to the atlas texture in WebGL.
///
/// This allows efficient uploading from different sources:
/// - `Pixmap`: Direct upload using raw pixel data
/// - `ImageData`: Browser `ImageData` objects
/// - Custom implementations for other image sources
pub trait WebGlAtlasWriter {
    /// Get the width of the image.
    fn width(&self) -> u32;
    /// Get the height of the image.
    fn height(&self) -> u32;

    /// Write image data to the atlas texture at the specified offset.
    fn write_to_atlas(
        &self,
        gl: &WebGl2RenderingContext,
        atlas_texture: &WebGlTexture,
        offset: [u32; 2],
        width: u32,
        height: u32,
    );
}

/// Implementation for `Pixmap` - direct upload using raw pixel data
impl WebGlAtlasWriter for crate::Pixmap {
    fn width(&self) -> u32 {
        self.width() as u32
    }

    fn height(&self) -> u32 {
        self.height() as u32
    }

    fn write_to_atlas(
        &self,
        gl: &WebGl2RenderingContext,
        atlas_texture: &WebGlTexture,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        // Bind the atlas texture
        gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(atlas_texture));

        // Convert pixmap data to the format expected by WebGL
        let rgba_data = self.data_as_u8_slice();

        // Upload the image data to the specific region of the atlas texture
        gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
            WebGl2RenderingContext::TEXTURE_2D,
            0,                // mip level
            offset[0] as i32, // x offset
            offset[1] as i32, // y offset
            width as i32,
            height as i32,
            WebGl2RenderingContext::RGBA,
            WebGl2RenderingContext::UNSIGNED_BYTE,
            Some(rgba_data),
        )
        .unwrap();
    }
}

/// Implementation for `Arc<Pixmap>`
impl WebGlAtlasWriter for alloc::sync::Arc<crate::Pixmap> {
    fn width(&self) -> u32 {
        self.as_ref().width() as u32
    }

    fn height(&self) -> u32 {
        self.as_ref().height() as u32
    }

    fn write_to_atlas(
        &self,
        gl: &WebGl2RenderingContext,
        atlas_texture: &WebGlTexture,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        self.as_ref()
            .write_to_atlas(gl, atlas_texture, offset, width, height);
    }
}

/// Implementation for `WebGlTexture` - texture-to-texture copy
impl WebGlAtlasWriter for WebGlTexture {
    fn width(&self) -> u32 {
        // WebGL textures don't expose their dimensions directly
        // This is a limitation - in practice, you'd need to track dimensions separately
        // For now, we'll require the caller to provide correct width/height parameters
        unreachable!("WebGlTexture width must be provided by caller")
    }

    fn height(&self) -> u32 {
        // WebGL textures don't expose their dimensions directly
        // This is a limitation - in practice, you'd need to track dimensions separately
        // For now, we'll require the caller to provide correct width/height parameters
        unreachable!("WebGlTexture height must be provided by caller")
    }

    fn write_to_atlas(
        &self,
        gl: &WebGl2RenderingContext,
        atlas_texture: &WebGlTexture,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        // Create a temporary framebuffer for the source texture
        let framebuffer = gl.create_framebuffer().unwrap();
        gl.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(&framebuffer));

        // Attach the source texture to the framebuffer
        gl.framebuffer_texture_2d(
            WebGl2RenderingContext::FRAMEBUFFER,
            WebGl2RenderingContext::COLOR_ATTACHMENT0,
            WebGl2RenderingContext::TEXTURE_2D,
            Some(self),
            0, // mip level
        );

        // Verify framebuffer is complete
        let status = gl.check_framebuffer_status(WebGl2RenderingContext::FRAMEBUFFER);
        if status != WebGl2RenderingContext::FRAMEBUFFER_COMPLETE {
            panic!("Framebuffer not complete: {status}");
        }

        // Bind the atlas texture as the target
        gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(atlas_texture));

        // Copy from framebuffer to atlas texture
        gl.copy_tex_sub_image_2d(
            WebGl2RenderingContext::TEXTURE_2D,
            0,                // mip level
            offset[0] as i32, // x offset in atlas
            offset[1] as i32, // y offset in atlas
            0,                // x coordinate in source framebuffer
            0,                // y coordinate in source framebuffer
            width as i32,
            height as i32,
        );

        // Clean up
        gl.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, None);
        gl.delete_framebuffer(Some(&framebuffer));
    }
}

/// Wrapper for `WebGlTexture` with known dimensions to work around WebGL API limitations
#[derive(Debug)]
pub struct WebGlTextureWithDimensions {
    /// The WebGL texture.
    pub texture: WebGlTexture,
    /// The width of the texture.
    pub width: u32,
    /// The height of the texture.
    pub height: u32,
}

impl WebGlAtlasWriter for WebGlTextureWithDimensions {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn write_to_atlas(
        &self,
        gl: &WebGl2RenderingContext,
        atlas_texture: &WebGlTexture,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        self.texture
            .write_to_atlas(gl, atlas_texture, offset, width, height);
    }
}
