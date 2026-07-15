// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Native WebGL2 rendering module for the sparse strips CPU/GPU rendering engine.
//!
//! This module provides identical functionality as the [`wgpu`] module, however the graphics
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

use crate::draw::ExternalTextureRun;
use crate::render::common::IMAGE_PADDING;
use crate::util::RangedSlice;
use crate::{
    GpuStrip, LayersConfig, RenderError, RenderSettings, RenderSize, Resources,
    blend::{GpuBlendInstance, gpu_blend_instance},
    copy::GpuCopyInstance,
    filter::{FilterContext, FilterInstanceData, FilterPassPlan},
    gradient_cache::GradientRampCache,
    paint::PaintResolver,
    render::{
        Config,
        common::{
            DeviceLimits, GPU_BLURRED_ROUNDED_RECT_SIZE_TEXELS, GPU_ENCODED_IMAGE_SIZE_TEXELS,
            GPU_LINEAR_GRADIENT_SIZE_TEXELS, GPU_RADIAL_GRADIENT_SIZE_TEXELS,
            GPU_SWEEP_GRADIENT_SIZE_TEXELS, GpuBlurredRoundedRect, GpuEncodedImage,
            GpuEncodedPaint, GpuLinearGradient, GpuRadialGradient, GpuSweepGradient,
            ScratchBuffers, ScratchTexture, pack_image_offset, pack_image_params, pack_image_size,
            pack_radial_kind_and_swapped, pack_texture_width_and_extend_mode, pack_tint,
        },
    },
    scene::Scene,
    schedule::{RendererBackend, Schedule, ScheduleStorage, round::BlendOp},
    target::{
        DrawPassTarget, FilterTexturePair, LayerTextureId, LayerTexturePair, RootRenderTarget,
        TextureParity,
    },
};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
#[cfg(feature = "text")]
use glifo::{GLYPH_PADDING, PendingClearRect};
use resource::{Buffer, FragmentShader, Framebuffer, Program, Texture, VertexArray, VertexShader};
use vello_common::image_cache::{ImageCache, ImageResource};
use vello_common::multi_atlas::{AtlasConfig, AtlasId};
use vello_common::{
    encode::{
        EncodedBlurredRoundedRectangle, EncodedGradient, EncodedKind, EncodedPaint,
        MAX_GRADIENT_LUT_SIZE, RadialKind,
    },
    geometry::{RectU16, SizeU16},
    paint::{ImageId, ImageSource},
    peniko::{self},
    pixmap::Pixmap,
    tile::Tile,
};
use vello_sparse_shaders::{blend, copy, filter as filter_shader, render};
use web_sys::wasm_bindgen::{JsCast, JsValue};
use web_sys::{
    HtmlCanvasElement, WebGl2RenderingContext, WebGlBuffer, WebGlFramebuffer, WebGlProgram,
    WebGlShader, WebGlTexture, WebGlUniformLocation, WebGlVertexArrayObject,
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

fn get_max_texture_array_layers(gl: &WebGl2RenderingContext) -> u32 {
    gl.get_parameter(WebGl2RenderingContext::MAX_ARRAY_TEXTURE_LAYERS)
        .unwrap()
        .as_f64()
        .unwrap() as u32
}

/// Vello Hybrid's WebGL2 Renderer.
#[derive(Debug)]
pub struct WebGlRenderer {
    /// Programs for rendering.
    pub(super) programs: WebGlPrograms,
    /// WebGL context.
    pub(crate) gl: WebGl2RenderingContext,
    /// Encoded paints for storing encoded paints.
    encoded_paints: Vec<GpuEncodedPaint>,
    /// Stores the index (offset) of the encoded paints in the encoded paints texture.
    paint_idxs: Vec<u32>,
    /// Gradient cache for storing gradient ramps.
    gradient_cache: GradientRampCache,
    dummy_image_cache: Option<ImageCache>,
    schedule_storage: ScheduleStorage,
    scratch: ScratchBuffers,
    layer_config: LayersConfig,
}

impl WebGlRenderer {
    /// Creates a new WebGL2 renderer
    pub fn new(canvas: &HtmlCanvasElement) -> Self {
        Self::new_with(canvas, RenderSettings::default())
    }

    /// Creates a new WebGL2 renderer with specific settings.
    pub fn new_with(canvas: &HtmlCanvasElement, settings: RenderSettings) -> Self {
        #[allow(
            clippy::assertions_on_constants,
            reason = "intentional guard against non-wasm32 use"
        )]
        {
            debug_assert!(
                cfg!(target_arch = "wasm32"),
                "`WebGlRenderer` can only be constructed when targeting `wasm32`",
            );
        }
        super::common::maybe_warn_about_webgl_feature_conflict();

        // We do our own anti-aliasing, so no need to enable it in the WebGL
        // context.
        let context_options = js_sys::Object::new();
        js_sys::Reflect::set(&context_options, &"antialias".into(), &JsValue::FALSE).unwrap();
        // Vello only supports 24+ bit depth buffers. If the hardware falls back to a 16 bit depth buffer,
        // correctness issues will arise. For all intents and purposes, a device manufactured in the past 10 years
        // should support 24+ bit depth buffers (certainly those within the realm of what we consider "supported" devices)
        // but:
        //
        // Relevant code for default depth buffer behaviour can be found here:
        // - Chromium defaults to 24 bit with no fallback: https://github.com/chromium/chromium/blob/86bafb3aab8e999690d310b201d0b5489f512b08/third_party/blink/renderer/platform/graphics/gpu/drawing_buffer.cc#L1376-L1400
        // - Firefox defaults to 24 bit with no fallback: https://github.com/mozilla/gecko-dev/blob/5836a062726f715fda621338a17b51aff30d0a8c/gfx/gl/MozFramebuffer.cpp#L155-L161
        // - Safari defaults to 24 bit _with 16 bit_ fallback: https://github.com/WebKit/WebKit/blob/a6d6c154bbee0643f5ad1e55c071558c0df9aef7/Source/WebCore/platform/graphics/angle/GraphicsContextGLANGLE.cpp#L393-L416
        //
        // TODO: The above understanding is encoded in a below assertion, but this should be encapsulated within a
        // "this device can run Vello correctly" check function.
        js_sys::Reflect::set(&context_options, &"depth".into(), &JsValue::TRUE).unwrap();

        let gl = canvas
            .get_context_with_context_options("webgl2", &context_options)
            .expect("WebGL2 context to be available")
            .unwrap()
            .dyn_into::<WebGl2RenderingContext>()
            .expect("Context to be a WebGL2 context");

        let cloned_gl = gl.clone();
        let _state_guard = WebGlStateGuard::with_config(
            &cloned_gl,
            WebGlStateConfig {
                framebuffer: true,
                ..Default::default()
            },
        );

        // Note: It is not entirely clear whether we really _have_ to ensure anti-aliasing is disabled.
        // This code is inherited from a similar snippet in wgpu
        // (https://github.com/gfx-rs/wgpu/blob/56e4a389ddd02403e232beef3d3ff305625e6485/wgpu-hal/src/gles/web.rs#L101-L106),
        // which itself seems to have been copied from the older `gfx` crate, where it was first introduced
        // in https://github.com/gfx-rs/gfx/pull/2554/changes#diff-a47711d61df7a43fe6dd99c39b936d17ff817cbc2238d7e3ae6698ffde9b88f7R79,
        // without any comment on why.
        // From my (Laurenz) testing, tests seem to work even when anti-aliasing is enabled,
        // but Andrew previously got errors similar to the ones outlined in
        // https://github.com/gfx-rs/wgpu/issues/5263. Therefore, we just leave it as is for now.
        #[cfg(debug_assertions)]
        {
            // If a WebGL context already exists on this canvas, it will be returned instead of
            // creating a new one with the correct context_options set.
            // See this comment for why we still care about non-antialiased context:
            // https://github.com/linebender/vello/pull/1546/changes#r3008692535
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

        let mut settings = settings;
        let device_limits = DeviceLimits {
            max_texture_dimension_2d: get_max_texture_dimension_2d(&gl),
            max_texture_array_layers: get_max_texture_array_layers(&gl),
        };
        settings.memory.normalize(&device_limits, 1);
        assert!(
            gl.get_parameter(WebGl2RenderingContext::DEPTH_BITS)
                .unwrap()
                .as_f64()
                .unwrap()
                >= 24.0,
            "Depth buffer must be at least 24 bits"
        );
        let image_cache = ImageCache::new_with_config(settings.memory.image_atlas_config);
        let max_texture_dimension_2d = device_limits.max_texture_dimension_2d;

        // Estimate the maximum number of gradient cache entries based on the max texture dimension
        // and the maximum gradient LUT size - worst case scenario.
        let max_gradient_cache_size =
            max_texture_dimension_2d * max_texture_dimension_2d / MAX_GRADIENT_LUT_SIZE as u32;
        let gradient_cache = GradientRampCache::new(max_gradient_cache_size, settings.level);
        let layer_config = settings.memory.layers_config;
        Self {
            programs: WebGlPrograms::new(gl.clone(), &image_cache, layer_config),
            gl,
            encoded_paints: Vec::new(),
            paint_idxs: Vec::new(),
            gradient_cache,
            dummy_image_cache: Some(ImageCache::new_dummy()),
            schedule_storage: ScheduleStorage::default(),
            scratch: ScratchBuffers::default(),
            layer_config,
        }
    }

    /// Render `scene` using WebGL2
    ///
    /// This method creates GPU resources as needed and schedules potentially multiple draw calls.
    pub fn render(
        &mut self,
        scene: &Scene,
        resources: &mut Resources,
        render_size: &RenderSize,
    ) -> Result<(), RenderError> {
        debug_assert_eq!(
            RenderSize {
                width: self.gl.drawing_buffer_width() as u32,
                height: self.gl.drawing_buffer_height() as u32
            },
            *render_size,
            "Render size must match drawing buffer size"
        );

        #[cfg(feature = "text")]
        {
            resources.before_render(
                self,
                |renderer, glyph_renderer, atlas_count, atlas_config, atlas_id| {
                    renderer
                        .render_to_atlas(glyph_renderer, atlas_count, atlas_config, atlas_id)
                        .expect("Failed to render glyphs to atlas");
                },
                |renderer, image_cache, upload, dst_x, dst_y| {
                    renderer.write_to_atlas(
                        image_cache,
                        upload.image_id,
                        &upload.pixmap,
                        Some([dst_x, dst_y]),
                    );
                },
            );
        }

        self.render_scene(
            scene,
            &mut resources.image_cache,
            render_size,
            true,
            RootRenderTarget::UserSurface,
        )?;

        #[cfg(feature = "text")]
        {
            resources.after_render(self, |renderer, rect| {
                clear_atlas_region(renderer, rect);
            });
        }

        Ok(())
    }

    /// Render a `scene` directly into an atlas layer.
    ///
    /// This renders the scene's content into the specified atlas layer, which can then
    /// be sampled as an image in subsequent render passes. This is useful for rendering
    /// vector content (e.g., glyphs) into the atlas for later use as cached images.
    ///
    /// The scene should be sized to the atlas layer dimensions
    /// ([`AtlasConfig::atlas_size`]), with content positioned at the allocated offset
    /// coordinates from `ImageCache::allocate`.
    ///
    /// This method creates its own command encoder and submits immediately,
    /// ensuring atlas content is committed before any subsequent
    /// [`render`](Self::render) call (the two methods share GPU resources that
    /// are staged by `queue.write_*` and only applied on the next `queue.submit`).
    #[doc(hidden)]
    pub fn render_to_atlas(
        &mut self,
        scene: &Scene,
        atlas_count: u32,
        atlas_config: AtlasConfig,
        atlas_id: AtlasId,
    ) -> Result<(), RenderError> {
        self.programs
            .maybe_resize_atlas_texture_array(&self.gl, atlas_count);

        let (atlas_width, atlas_height) = atlas_config.atlas_size;
        let atlas_render_size = RenderSize {
            width: atlas_width,
            height: atlas_height,
        };

        let atlas_framebuffer = self
            .programs
            .resources
            .atlas_render_framebuffer
            .take()
            .unwrap_or_else(|| Framebuffer::new(&self.gl));
        self.gl.bind_framebuffer(
            WebGl2RenderingContext::FRAMEBUFFER,
            Some(&atlas_framebuffer),
        );
        self.gl.framebuffer_texture_layer(
            WebGl2RenderingContext::FRAMEBUFFER,
            WebGl2RenderingContext::COLOR_ATTACHMENT0,
            Some(&self.programs.resources.atlas_texture_array.texture),
            0,
            atlas_id.as_u32() as i32,
        );

        // Set the view framebuffer override so the scheduler renders to the
        // atlas layer instead of the default framebuffer.
        self.programs.resources.view_framebuffer_override = Some(atlas_framebuffer);

        // Swap in the stub atlas texture array to avoid binding the real atlas
        // texture as a shader input while it is also the render target.
        core::mem::swap(
            &mut self.programs.resources.atlas_texture_array,
            &mut self.programs.resources.stub_atlas_texture_array,
        );

        // TODO: Explore using an option instead of a dummy image cache.
        let mut dummy_image_cache = self
            .dummy_image_cache
            .take()
            .expect("dummy image cache must exist");
        let result = self.render_scene(
            scene,
            &mut dummy_image_cache,
            &atlas_render_size,
            false,
            RootRenderTarget::AtlasLayer,
        );
        self.dummy_image_cache = Some(dummy_image_cache);

        // Restore the real atlas texture array.
        core::mem::swap(
            &mut self.programs.resources.atlas_texture_array,
            &mut self.programs.resources.stub_atlas_texture_array,
        );

        // Restore the default view framebuffer and cache the atlas FBO for reuse.
        self.programs.resources.atlas_render_framebuffer =
            self.programs.resources.view_framebuffer_override.take();

        result
    }

    /// Shared render pipeline: prepares GPU resources, runs the scheduler, and
    /// maintains caches.
    ///
    /// When `clear` is true the view framebuffer is cleared to transparent black
    /// before drawing. This must happen *after* `prepare` (which may create/resize
    /// the framebuffer attachment). Atlas renders skip the clear so previously
    /// rendered atlas content is preserved.
    pub(crate) fn render_scene(
        &mut self,
        scene: &Scene,
        image_cache: &mut ImageCache,
        render_size: &RenderSize,
        clear: bool,
        root_output_target: RootRenderTarget,
    ) -> Result<(), RenderError> {
        let mut encoded_paints = scene.encoded_paints.borrow_mut();
        let original_scene_paint_count = encoded_paints.len();

        self.prepare_gpu_encoded_paints(&encoded_paints, image_cache);
        self.programs
            .maybe_resize_atlas_texture_array(&self.gl, image_cache.atlas_count() as u32);

        let required_texture_size = self
            .layer_config
            .required_intermediate_texture_size(&scene.recorder)?;

        // We currently only grow and never shrink textures, so max it with whatever
        // we had in the previous run.
        let texture_size = self
            .programs
            .resources
            .texture_size
            .max(required_texture_size);

        let paint_resolver = PaintResolver::new(&encoded_paints, &self.paint_idxs);
        let schedule = Schedule::try_new(
            &mut self.schedule_storage,
            scene,
            root_output_target,
            paint_resolver,
            texture_size,
            self.layer_config,
        )?;
        self.programs
            .prepare_intermediate_textures(&self.gl, &schedule, texture_size);

        // TODO: For the time being, we upload the entire alpha buffer as one big chunk. As a future
        // refinement, we could have a bounded alpha buffer, and break draws when the alpha
        // buffer fills.
        self.programs.prepare(
            &self.gl,
            &mut self.gradient_cache,
            &self.encoded_paints,
            &mut scene.strip_storage.borrow_mut().alphas,
            render_size,
            &self.paint_idxs,
            &self.schedule_storage.filter_context,
        );
        if clear {
            self.programs.clear_view_framebuffer(&self.gl);
        }
        self.programs.resources.depth_cleared_this_frame = false;
        let mut ctx = WebGlRendererContext {
            programs: &mut self.programs,
            gl: &self.gl,
            scratch: &mut self.scratch,
        };
        crate::schedule::execute(
            &mut ctx,
            &mut self.schedule_storage,
            schedule,
            root_output_target,
        );

        // See: https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices#use_invalidateframebuffer
        // We want to indicate to the GPU driver that we won't read the depth buffer again
        // until the next clear. This enables the GPU to avoid storing depth tiles back to VRAM.
        if self.programs.resources.depth_cleared_this_frame {
            self.gl.bind_framebuffer(
                WebGl2RenderingContext::FRAMEBUFFER,
                self.programs.resources.view_framebuffer_override.as_deref(),
            );
            self.gl
                .invalidate_framebuffer(
                    WebGl2RenderingContext::FRAMEBUFFER,
                    &self.programs.resources.depth_attachment_array,
                )
                .unwrap();
        }

        encoded_paints.truncate(original_scene_paint_count);
        self.gradient_cache.maintain();

        Ok(())
    }

    /// Get a reference to the underlying WebGL context.
    ///
    /// This allows direct access to WebGL operations for advanced use cases like texture creation.
    pub fn gl_context(&self) -> &WebGl2RenderingContext {
        &self.gl
    }

    /// Upload image to cache and atlas in one step. Returns the `ImageId`.
    ///
    /// This is the WebGL analogue of the wgpu Renderer's `upload_image` method.
    /// It allocates space in the image cache and uploads the image data to the atlas texture.
    pub fn upload_image<T: WebGlAtlasWriter>(
        &mut self,
        resources: &mut Resources,
        writer: &T,
    ) -> ImageId {
        self.upload_image_with(&mut resources.image_cache, writer, IMAGE_PADDING)
    }

    pub(crate) fn upload_image_with<T: WebGlAtlasWriter>(
        &mut self,
        image_cache: &mut ImageCache,
        writer: &T,
        padding: u16,
    ) -> ImageId {
        let width = writer.width();
        let height = writer.height();
        let image_id = image_cache.allocate(width, height, padding).unwrap();
        self.write_to_atlas(image_cache, image_id, writer, None);
        image_id
    }

    /// Write pixel data to an existing atlas allocation.
    ///
    /// Unlike [`upload_image`](Self::upload_image), this does not allocate space in the image
    /// cache. The `image_id` must have been previously allocated (e.g. via
    /// `ImageCache::allocate`). This is useful for uploading CPU-side pixel data (such as
    /// bitmap font glyphs) to a pre-allocated atlas region.
    ///
    /// If `offset_override` is `Some`, the provided offset is used instead of the
    /// allocator-assigned position. Pass `None` to use the default atlas offset.
    pub(crate) fn write_to_atlas<T: WebGlAtlasWriter>(
        &mut self,
        image_cache: &ImageCache,
        image_id: ImageId,
        writer: &T,
        offset_override: Option<[u32; 2]>,
    ) {
        let image_resource = image_cache.get(image_id).expect("Image resource not found");

        self.programs
            .maybe_resize_atlas_texture_array(&self.gl, image_cache.atlas_count() as u32);
        let offset = offset_override.unwrap_or([
            image_resource.offset[0] as u32,
            image_resource.offset[1] as u32,
        ]);
        writer.write_to_atlas_layer(
            &self.gl,
            &self.programs.resources.atlas_texture_array.texture,
            image_resource.atlas_id.as_u32(),
            offset,
            writer.width(),
            writer.height(),
        );
    }

    /// Destroy an image from the cache and clear the allocated slot in the atlas.
    pub fn destroy_image(&mut self, resources: &mut Resources, image_id: ImageId) {
        if let Some(image_resource) = resources.image_cache.deallocate(image_id) {
            let padding = image_resource.padding as u32;
            self.clear_atlas_region(
                image_resource.atlas_id,
                [
                    image_resource.offset[0] as u32 - padding,
                    image_resource.offset[1] as u32 - padding,
                ],
                image_resource.width as u32 + padding * 2,
                image_resource.height as u32 + padding * 2,
            );
        }
    }

    /// Returns a reference to the underlying atlas texture array.
    ///
    /// This is a 2D array texture (`TextureViewDimension::D2Array`) containing all
    /// atlas layers used by the image cache. Each layer holds cached image data
    /// (e.g., rasterised glyphs) that the renderer samples during draw calls.
    pub fn atlas_texture(&self) -> &WebGlTexture {
        &self.programs.resources.atlas_texture_array.texture
    }

    /// Clear a specific region of the atlas texture array.
    fn clear_atlas_region(&mut self, atlas_id: AtlasId, offset: [u32; 2], width: u32, height: u32) {
        let _state_guard = WebGlStateGuard::for_clear_atlas_region(&self.gl);
        let temp_framebuffer = Framebuffer::new(&self.gl);

        // Bind our temporary framebuffer
        self.gl
            .bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(&temp_framebuffer));

        // Attach the specific atlas layer to the framebuffer
        self.gl.framebuffer_texture_layer(
            WebGl2RenderingContext::FRAMEBUFFER,
            WebGl2RenderingContext::COLOR_ATTACHMENT0,
            Some(&self.programs.resources.atlas_texture_array.texture),
            0,
            atlas_id.as_u32() as i32,
        );

        // Set viewport to match the atlas texture dimensions
        let atlas_size = &self.programs.resources.atlas_texture_array.size;
        self.gl
            .viewport(0, 0, atlas_size.width as i32, atlas_size.height as i32);

        // Enable scissor test and set scissor rectangle to our region
        self.gl.enable(WebGl2RenderingContext::SCISSOR_TEST);
        self.gl.scissor(
            offset[0] as i32,
            offset[1] as i32,
            width as i32,
            height as i32,
        );

        // Clear the region to transparent (0, 0, 0, 0)
        self.gl.clear_color(0.0, 0.0, 0.0, 0.0);
        self.gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);
    }

    fn prepare_gpu_encoded_paints(
        &mut self,
        encoded_paints: &[EncodedPaint],
        image_cache: &ImageCache,
    ) {
        self.encoded_paints
            .resize_with(encoded_paints.len(), || GPU_PAINT_PLACEHOLDER);
        self.paint_idxs.resize(encoded_paints.len() + 1, 0);

        let mut current_idx = 0;
        for (encoded_paint_idx, paint) in encoded_paints.iter().enumerate() {
            self.paint_idxs[encoded_paint_idx] = current_idx;
            match paint {
                EncodedPaint::Image(img) => {
                    if let ImageSource::OpaqueId { id: image_id, .. } = img.source {
                        let image_resource: Option<&ImageResource> = image_cache.get(image_id);
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
                EncodedPaint::ExternalTexture(_external_texture) => {
                    // TODO: External textures are not yet supported.
                    log::warn!("External textures are not yet supported in the WebGL backend");
                    current_idx += GPU_ENCODED_IMAGE_SIZE_TEXELS;
                }
                EncodedPaint::BlurredRoundedRect(blurred_rect) => {
                    self.encoded_paints[encoded_paint_idx] =
                        Self::encode_blurred_rounded_rect_paint(blurred_rect);
                    current_idx += GPU_BLURRED_ROUNDED_RECT_SIZE_TEXELS;
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
        let transform = image.transform.as_coeffs().map(|x| x as f32);
        let image_size = pack_image_size(image_resource.width, image_resource.height);
        let image_offset = pack_image_offset(image_resource.offset[0], image_resource.offset[1]);
        let image_params = pack_image_params(
            image.sampler.quality as u32,
            image.sampler.x_extend as u32,
            image.sampler.y_extend as u32,
            image_resource.atlas_id.as_u32(),
        );
        let (tint, tint_mode) = pack_tint(image.tint);

        GpuEncodedPaint::Image(GpuEncodedImage {
            image_params,
            image_size,
            image_offset,
            transform,
            tint,
            tint_mode,
            image_padding: image_resource.padding as u32,
        })
    }

    fn encode_gradient_paint(
        &self,
        gradient: &EncodedGradient,
        gradient_width: u32,
        gradient_start: u32,
    ) -> GpuEncodedPaint {
        let transform = gradient.transform.as_coeffs().map(|x| x as f32);
        let extend_mode = match gradient.extend {
            peniko::Extend::Pad => 0,
            peniko::Extend::Repeat => 1,
            peniko::Extend::Reflect => 2,
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

    fn encode_blurred_rounded_rect_paint(rect: &EncodedBlurredRoundedRectangle) -> GpuEncodedPaint {
        GpuEncodedPaint::BlurredRoundedRect(GpuBlurredRoundedRect {
            transform: rect.transform.as_coeffs().map(|x| x as f32),
            color: rect.color.as_premul_rgba8().to_u32(),
            invert: u32::from(rect.invert),
            params0: [
                rect.exponent,
                rect.recip_exponent,
                rect.scale,
                rect.std_dev_inv,
            ],
            params1: [rect.min_edge, rect.w, rect.h, rect.r1],
            size: [rect.width, rect.height],
            _padding1: [0, 0],
        })
    }
}

#[cfg(feature = "text")]
fn clear_atlas_region(renderer: &mut WebGlRenderer, rect: &PendingClearRect) {
    // TODO: Similarly to wgpu, maybe this can be done in a more effective
    // way?
    let padding = u32::from(GLYPH_PADDING);
    let offset = [
        u32::from(rect.x).saturating_sub(padding),
        u32::from(rect.y).saturating_sub(padding),
    ];
    let width = u32::from(rect.width) + padding * 2;
    let height = u32::from(rect.height) + padding * 2;
    renderer.clear_atlas_region(AtlasId::new(rect.page_index), offset, width, height);
}

/// Contains the WebGL programs and resources for rendering.
#[derive(Debug)]
pub(crate) struct WebGlPrograms {
    /// Program for rendering wide tile commands.
    strip_program: Program,
    /// Uniform locations for the strip program
    strip_uniforms: StripUniforms,
    /// Program for filter passes.
    filter_program: Program,
    /// Uniform locations for the filter program.
    filter_uniforms: FilterPassUniforms,
    /// Program for resolving non-default blend layers into scratch.
    blend_program: Program,
    /// Uniform locations for the blend program.
    blend_uniforms: BlendUniforms,
    /// Program for copying between intermediate textures.
    copy_program: Program,
    /// Uniform locations for the blend copy program.
    copy_uniforms: CopyUniforms,
    /// WebGL resources for rendering.
    pub(crate) resources: WebGlResources,
    /// Dimensions of the rendering target.
    render_size: RenderSize,
    /// Whether the last config buffer upload had NDC Y negation enabled.
    negate_ndc: bool,
    /// Scratch buffer for staging encoded paints texture data.
    encoded_paints_data: Vec<u8>,
    /// Scratch buffer for staging filter data texture data.
    filter_data: Vec<u8>,
}

#[derive(Debug)]
struct FilterPassUniforms {
    filter_data: WebGlUniformLocation,
    in_tex: WebGlUniformLocation,
    original_texture: WebGlUniformLocation,
}

#[derive(Debug)]
struct BlendUniforms {
    layer_texture_0: WebGlUniformLocation,
    layer_texture_1: WebGlUniformLocation,
}

#[derive(Debug)]
struct CopyUniforms {
    source_texture: WebGlUniformLocation,
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
    /// Layer input texture location.
    layer_input_texture: WebGlUniformLocation,
    /// Atlas texture location.
    atlas_texture_array: WebGlUniformLocation,
    /// Encoded paints texture location for fragment shader.
    encoded_paints_texture_fs: WebGlUniformLocation,
    /// Encoded paints texture location for vertex shader.
    encoded_paints_texture_vs: WebGlUniformLocation,
    /// Gradient texture location.
    gradient_texture: WebGlUniformLocation,
    /// External texture location.
    external_texture: WebGlUniformLocation,
}

/// Contains all WebGL resources needed for rendering.
#[derive(Debug)]
pub(crate) struct WebGlResources {
    /// VAO for strip rendering.
    strip_vao: VertexArray,
    /// Buffer for [`GpuStrip`] data.
    strips_buffer: Buffer,
    /// Texture for alpha values.
    alphas_texture: Texture,
    /// Height of alpha texture.
    alpha_texture_height: u32,
    /// Texture array for atlas data (multiple atlases supported)
    pub(crate) atlas_texture_array: WebGlTextureArray,
    /// Encoded paints texture for image metadata.
    encoded_paints_texture: Texture,
    /// Height of encoded paints texture.
    encoded_paints_texture_height: u32,
    /// Gradient texture for gradient ramp data.
    gradient_texture: Texture,
    /// Placeholder texture bound to the strip shader's `external_texture` sampler.
    placeholder_external_texture: Texture,
    /// Height of gradient texture.
    gradient_texture_height: u32,

    /// Config buffer for rendering wide tile commands into the view texture.
    view_config_buffer: Buffer,

    pub(crate) view_framebuffer_override: Option<Framebuffer>,
    /// Whether the depth buffer has been cleared this frame.
    depth_cleared_this_frame: bool,
    /// Pre-allocated JS array for `invalidateFramebuffer` calls.
    depth_attachment_array: js_sys::Array,

    /// Cached result from querying `WebGl2RenderingContext::MAX_TEXTURE_SIZE` which is a blocking
    /// WebGL call.
    max_texture_dimension_2d: u32,

    /// Dimensions of the intermediate layer and scratch textures.
    texture_size: SizeU16,

    /// Placeholder 1x1 atlas texture array, used during `render_to_atlas` to avoid
    /// binding the real atlas texture while it is also the render target.
    stub_atlas_texture_array: WebGlTextureArray,

    /// Cached framebuffer for rendering into an atlas layer in `render_to_atlas`.
    /// Reused to avoid create/delete overhead on every call.
    atlas_render_framebuffer: Option<Framebuffer>,

    /// RGBA32UI texture storing filter parameters.
    filter_data_texture: Texture,
    /// Current height of filter data texture.
    filter_data_texture_height: u32,
    /// Per-instance vertex data buffer for filter draws.
    filter_instance_buffer: Buffer,
    /// VAO for filter rendering.
    filter_vao: VertexArray,
    /// Per-instance vertex data buffer for blend draws.
    blend_instance_buffer: Buffer,
    /// VAO for blend rendering.
    blend_vao: VertexArray,
    /// Per-instance vertex data buffer for copy draws.
    copy_instance_buffer: Buffer,
    /// VAO for copy rendering.
    copy_vao: VertexArray,
    /// Scratch texture for blend and certain filter operations.
    scratch_texture: Option<ScratchTexture<WebGlIntermediateTexture>>,
    /// Config buffer for rendering strips into a layer texture.
    layer_config_buffers: [Buffer; 2],
    /// Layer texture pages grouped by parity.
    layer_textures: [Vec<WebGlIntermediateTexture>; 2],
}

#[derive(Debug)]
struct WebGlIntermediateTexture {
    texture: Texture,
    framebuffer: Framebuffer,
}

impl WebGlIntermediateTexture {
    fn binding_texture(&self) -> &Texture {
        &self.texture
    }

    fn framebuffer(&self) -> &Framebuffer {
        &self.framebuffer
    }
}

impl WebGlResources {
    fn layer_texture(&self, id: LayerTextureId) -> &Texture {
        self.layer_textures[id.texture_parity.get_parity()]
            .get(usize::from(id.page_index))
            .expect("vello_hybrid attempted to use a missing layer texture")
            .binding_texture()
    }

    fn layer_framebuffer(&self, id: LayerTextureId) -> &Framebuffer {
        self.layer_textures[id.texture_parity.get_parity()]
            .get(usize::from(id.page_index))
            .expect("vello_hybrid attempted to use a missing layer texture")
            .framebuffer()
    }

    fn scratch_binding_texture(&self) -> &Texture {
        self.scratch_texture
            .as_ref()
            .map_or(&self.placeholder_external_texture, |texture| {
                texture.get().binding_texture()
            })
    }

    fn scratch_framebuffer(&self) -> &Framebuffer {
        self.scratch_texture
            .as_ref()
            .expect("vello_hybrid attempted to use a missing scratch texture")
            .get()
            .framebuffer()
    }
}

impl WebGlPrograms {
    /// Creates programs and initializes resources.
    fn new(
        gl: WebGl2RenderingContext,
        image_cache: &ImageCache,
        layer_config: LayersConfig,
    ) -> Self {
        let strip_program =
            create_shader_program(&gl, render::VERTEX_SOURCE, render::FRAGMENT_SOURCE);
        let filter_program = create_shader_program(
            &gl,
            filter_shader::VERTEX_SOURCE,
            filter_shader::FRAGMENT_SOURCE,
        );
        let filter_uniforms = get_filter_pass_uniforms(&gl, &filter_program);
        let blend_program =
            create_shader_program(&gl, blend::VERTEX_SOURCE, blend::FRAGMENT_SOURCE);
        let blend_uniforms = get_blend_uniforms(&gl, &blend_program);
        let copy_program = create_shader_program(&gl, copy::VERTEX_SOURCE, copy::FRAGMENT_SOURCE);
        let copy_uniforms = get_copy_uniforms(&gl, &copy_program);

        let strip_uniforms = get_strip_uniforms(&gl, &strip_program);

        let resources = create_webgl_resources(&gl, image_cache, layer_config);

        initialize_strip_vao(&gl, &resources);
        initialize_filter_vao(&gl, &resources);
        initialize_blend_vao(&gl, &resources);
        initialize_copy_vao(&gl, &resources);

        let encoded_paints_data = vec![0; (resources.max_texture_dimension_2d << 4) as usize];

        gl.enable(WebGl2RenderingContext::BLEND);
        gl.blend_func(
            WebGl2RenderingContext::ONE,
            WebGl2RenderingContext::ONE_MINUS_SRC_ALPHA,
        );

        Self {
            strip_program,
            filter_program,
            filter_uniforms,
            blend_program,
            blend_uniforms,
            copy_program,
            copy_uniforms,
            strip_uniforms,
            resources,
            render_size: RenderSize {
                width: 0,
                height: 0,
            },
            negate_ndc: false,
            encoded_paints_data,
            filter_data: Vec::new(),
        }
    }

    /// Prepare resources for rendering.
    fn prepare(
        &mut self,
        gl: &WebGl2RenderingContext,
        gradient_cache: &mut GradientRampCache,
        encoded_paints: &[GpuEncodedPaint],
        alphas: &mut Vec<u8>,
        render_size: &RenderSize,
        paint_idxs: &[u32],
        filter_context: &FilterContext,
    ) {
        let max_texture_dimension_2d = self.resources.max_texture_dimension_2d;

        self.maybe_resize_alphas_tex(max_texture_dimension_2d, alphas.len());
        self.maybe_resize_encoded_paints_tex(max_texture_dimension_2d, paint_idxs);
        self.maybe_resize_filter_data_tex(filter_context);
        self.maybe_update_config_buffer(gl, max_texture_dimension_2d, render_size);

        self.upload_alpha_texture(gl, alphas);
        self.upload_encoded_paints_texture(gl, encoded_paints);
        self.upload_filter_data_texture(gl, filter_context);

        if gradient_cache.has_changed() {
            self.maybe_resize_gradient_tex(gl, max_texture_dimension_2d, gradient_cache);
            self.upload_gradient_texture(gl, gradient_cache);
            gradient_cache.mark_synced();
        }
    }

    fn prepare_intermediate_textures(
        &mut self,
        gl: &WebGl2RenderingContext,
        schedule: &Schedule,
        texture_size: SizeU16,
    ) {
        let current_size = self.resources.texture_size;
        let layer_pages = schedule.layer_page_counts();
        let scratch_required = schedule.scratch_texture();

        for (index, textures) in self.resources.layer_textures.iter_mut().enumerate() {
            let required_page_count = layer_pages[index];

            // Note: Currently, `texture_size` only grows across frames, so if this condition
            // is true it means that the new required size is larger than what we currently have.
            if current_size != texture_size {
                // TODO: Drop old layer explicitly before creating new one to ensure WebGL
                // driver has the chance to reduce peak memory usage instead of having to keep both
                // textures alive at the same time?
                for texture in textures.iter_mut() {
                    *texture = create_intermediate_texture(gl, texture_size);
                }
                upload_layer_config_buffer(
                    gl,
                    &self.resources.layer_config_buffers[index],
                    texture_size,
                    self.resources.max_texture_dimension_2d,
                );
            }

            textures.extend(
                (textures.len()..required_page_count)
                    .map(|_| create_intermediate_texture(gl, texture_size)),
            );
        }

        let recreate_scratch = (self.resources.scratch_texture.is_some()
            && current_size != texture_size)
            || (self.resources.scratch_texture.is_none() && scratch_required);

        if recreate_scratch {
            self.resources.scratch_texture = Some(ScratchTexture::new(
                create_intermediate_texture(gl, texture_size),
            ));
        }

        self.resources.texture_size = texture_size;
    }

    /// Resize atlas texture array to accommodate more atlases.
    fn maybe_resize_atlas_texture_array(
        &mut self,
        gl: &WebGl2RenderingContext,
        required_atlas_count: u32,
    ) {
        let WebGlTextureSize {
            width,
            height,
            depth_or_array_layers: current_atlas_count,
        } = self.resources.atlas_texture_array.size();
        if required_atlas_count > current_atlas_count {
            // Create new texture array with more layers
            let new_atlas_texture_array =
                create_atlas_texture_array(gl, width, height, required_atlas_count);

            // Copy existing atlas data from old texture array to new one
            self.copy_atlas_texture_data(gl, &new_atlas_texture_array, current_atlas_count);

            // Replace the old resources
            self.resources.atlas_texture_array = new_atlas_texture_array;
            // Cached FBOs were attached to the old texture; drop them so we recreate on next use.
            self.resources.atlas_render_framebuffer = None;
        }
    }

    /// Copy texture data from the old atlas texture array to a new one.
    /// This is necessary when resizing the texture array to preserve existing atlas data.
    fn copy_atlas_texture_data(
        &self,
        gl: &WebGl2RenderingContext,
        new_atlas_texture_array: &WebGlTextureArray,
        layer_count_to_copy: u32,
    ) {
        let WebGlTextureSize { width, height, .. } = self.resources.atlas_texture_array.size();

        // Copy each layer from the old atlas to the new one
        for layer in 0..layer_count_to_copy {
            copy_to_texture_array_layer(
                gl,
                |gl| {
                    // Attach source layer to READ framebuffer
                    gl.framebuffer_texture_layer(
                        WebGl2RenderingContext::READ_FRAMEBUFFER,
                        WebGl2RenderingContext::COLOR_ATTACHMENT0,
                        Some(&self.resources.atlas_texture_array.texture),
                        0,
                        layer as i32,
                    );
                },
                &new_atlas_texture_array.texture,
                layer,
                [0, 0],
                [width, height],
            );
        }
    }

    fn maybe_resize_filter_data_tex(&mut self, filter_context: &FilterContext) {
        let max_texture_dimension_2d = self.resources.max_texture_dimension_2d;

        let Some(required_height) =
            filter_context.required_filter_data_height(max_texture_dimension_2d)
        else {
            return;
        };

        if required_height > self.resources.filter_data_texture_height {
            let required_size = (max_texture_dimension_2d * required_height) << 4;
            self.filter_data.resize(required_size as usize, 0);
            self.resources.filter_data_texture_height = required_height;
        }
    }

    fn upload_filter_data_texture(
        &mut self,
        gl: &WebGl2RenderingContext,
        filter_context: &FilterContext,
    ) {
        if filter_context.is_empty() {
            return;
        }

        let width = self.resources.max_texture_dimension_2d;
        let height = self.resources.filter_data_texture_height;
        filter_context.serialize_to_buffer(&mut self.filter_data);
        gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&self.resources.filter_data_texture),
        );
        let data_as_u32 = bytemuck::cast_slice::<u8, u32>(&self.filter_data);
        let packed_array = js_sys::Uint32Array::from(data_as_u32);
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
            WebGl2RenderingContext::TEXTURE_2D,
            0,
            WebGl2RenderingContext::RGBA32UI as i32,
            width as i32,
            height as i32,
            0,
            WebGl2RenderingContext::RGBA_INTEGER,
            WebGl2RenderingContext::UNSIGNED_INT,
            Some(&packed_array),
        )
        .unwrap();
    }

    fn upload_filter_instances(
        &self,
        gl: &WebGl2RenderingContext,
        instances: &[FilterInstanceData],
    ) {
        gl.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&self.resources.filter_instance_buffer),
        );
        gl.buffer_data_with_u8_array(
            WebGl2RenderingContext::ARRAY_BUFFER,
            bytemuck::cast_slice(instances),
            WebGl2RenderingContext::DYNAMIC_DRAW,
        );
    }

    fn upload_blend_instances(&self, gl: &WebGl2RenderingContext, instances: &[GpuBlendInstance]) {
        gl.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&self.resources.blend_instance_buffer),
        );
        gl.buffer_data_with_u8_array(
            WebGl2RenderingContext::ARRAY_BUFFER,
            bytemuck::cast_slice(instances),
            WebGl2RenderingContext::DYNAMIC_DRAW,
        );
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

    /// Update the alpha texture size if needed.
    fn maybe_resize_alphas_tex(&mut self, max_texture_dimension_2d: u32, alphas_len: usize) {
        let required_alpha_height = (alphas_len as u32)
            // There are 16 1-byte alpha values per texel.
            .div_ceil(max_texture_dimension_2d << 4);

        let current_alpha_height = self.resources.alpha_texture_height;
        if required_alpha_height > current_alpha_height {
            // We need to resize the alpha texture to fit the new alpha data.
            assert!(
                required_alpha_height <= max_texture_dimension_2d,
                "Alpha texture height exceeds max texture dimensions"
            );

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
        // Only negate if we are rendering to the main frame buffer.
        let negate_ndc = self.resources.view_framebuffer_override.is_none();

        // TODO: Collect all attributes that influence the config buffer into a
        // single struct and compare that, such that we cannot forget to update the
        // condition in case we add new fields in the future.
        if self.render_size != *new_render_size || self.negate_ndc != negate_ndc {
            let config = Config {
                width: new_render_size.width,
                height: new_render_size.height,
                strip_height: u32::from(Tile::HEIGHT),
                alphas_tex_width_bits: max_texture_dimension_2d.trailing_zeros(),
                encoded_paints_tex_width_bits: max_texture_dimension_2d.trailing_zeros(),
                strip_offset_x: 0,
                strip_offset_y: 0,
                negate_ndc: u32::from(negate_ndc),
            };

            gl.bind_buffer(
                WebGl2RenderingContext::UNIFORM_BUFFER,
                Some(&self.resources.view_config_buffer),
            );
            gl.buffer_data_with_u8_array(
                WebGl2RenderingContext::UNIFORM_BUFFER,
                bytemuck::bytes_of(&config),
                WebGl2RenderingContext::STATIC_DRAW,
            );

            self.render_size = new_render_size.clone();
            self.negate_ndc = negate_ndc;
        }
    }

    /// Upload alpha data to the texture.
    fn upload_alpha_texture(&mut self, gl: &WebGl2RenderingContext, alphas: &mut Vec<u8>) {
        if alphas.is_empty() {
            return;
        }

        let alpha_texture_width = self.resources.max_texture_dimension_2d;
        let alpha_texture_height = self.resources.alpha_texture_height;
        let total_size = alpha_texture_width as usize * alpha_texture_height as usize * 16;

        let original_len = alphas.len();

        // Temporarily pad the length of the alphas to the texture size before uploading.
        alphas.resize(total_size, 0);

        upload_data_to_rgba32_texture(
            gl,
            &self.resources.alphas_texture,
            bytemuck::cast_slice::<u8, u32>(alphas),
            alpha_texture_width,
            alpha_texture_height,
        );

        // Truncate back to the original size.
        alphas.truncate(original_len);
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

            upload_data_to_rgba32_texture(
                gl,
                &self.resources.encoded_paints_texture,
                bytemuck::cast_slice::<u8, u32>(&self.encoded_paints_data),
                encoded_paints_texture_width,
                encoded_paints_texture_height,
            );
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
        let old_luts_len = luts.len();
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
        luts.truncate(old_luts_len);
        gradient_cache.restore_luts(luts);
    }

    /// Clear the view framebuffer.
    // TODO: Investigate adding tests for the clear_view behavior.
    fn clear_view_framebuffer(&mut self, gl: &WebGl2RenderingContext) {
        gl.bind_framebuffer(
            WebGl2RenderingContext::FRAMEBUFFER,
            self.resources.view_framebuffer_override.as_deref(),
        );
        gl.clear_color(0.0, 0.0, 0.0, 0.0);
        gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);
    }

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
            Some(&self.resources.strips_buffer),
        );

        let opaque_bytes: &[u8] = bytemuck::cast_slice(opaque_strips);
        let alpha_len = alpha_strips.len() * size_of::<GpuStrip>();
        let total_len = opaque_bytes.len() + alpha_len;

        // Allocate buffer, then write both slices via bufferSubData.
        // We don't want to pay for concatenating the two slices. It's better to
        // simply write twice.
        gl.buffer_data_with_i32(
            WebGl2RenderingContext::ARRAY_BUFFER,
            total_len as i32,
            WebGl2RenderingContext::DYNAMIC_DRAW,
        );
        if !opaque_bytes.is_empty() {
            gl.buffer_sub_data_with_i32_and_u8_array(
                WebGl2RenderingContext::ARRAY_BUFFER,
                0,
                opaque_bytes,
            );
        }
        let mut offset = opaque_bytes.len();
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

/// RAII guard for WebGL state management.
/// Automatically saves state on creation and restores it on drop.
/// Only saves/restores the state specified in the configuration.
pub(crate) struct WebGlStateGuard {
    gl: WebGl2RenderingContext,
    config: WebGlStateConfig,
    original_framebuffer: Option<WebGlFramebuffer>,
    original_read_framebuffer: Option<WebGlFramebuffer>,
    original_active_texture: Option<u32>,
    original_texture_2d: Option<WebGlTexture>,
    original_texture_2d_array: Option<WebGlTexture>,
    original_pixel_pack_buffer: Option<WebGlBuffer>,
    scissor_enabled: bool,
    viewport: [i32; 4],
}

impl WebGlStateGuard {
    /// Create a new state guard with custom configuration.
    pub(crate) fn with_config(gl: &WebGl2RenderingContext, config: WebGlStateConfig) -> Self {
        // Save current framebuffer binding if requested
        let original_framebuffer = if config.framebuffer {
            gl.get_parameter(WebGl2RenderingContext::FRAMEBUFFER_BINDING)
                .ok()
                .and_then(|v| v.dyn_into::<WebGlFramebuffer>().ok())
        } else {
            None
        };

        // Save current read framebuffer binding if requested
        let original_read_framebuffer = if config.read_framebuffer {
            gl.get_parameter(WebGl2RenderingContext::READ_FRAMEBUFFER_BINDING)
                .ok()
                .and_then(|v| v.dyn_into::<WebGlFramebuffer>().ok())
        } else {
            None
        };

        let original_active_texture = if config.active_texture {
            gl.get_parameter(WebGl2RenderingContext::ACTIVE_TEXTURE)
                .ok()
                .and_then(|v| v.as_f64())
                .map(|v| v as u32)
        } else {
            None
        };

        let original_texture_2d = if config.texture_2d {
            gl.get_parameter(WebGl2RenderingContext::TEXTURE_BINDING_2D)
                .ok()
                .and_then(|v| v.dyn_into::<WebGlTexture>().ok())
        } else {
            None
        };

        // Save current 2D array texture binding if requested
        let original_texture_2d_array = if config.texture_2d_array {
            gl.get_parameter(WebGl2RenderingContext::TEXTURE_BINDING_2D_ARRAY)
                .ok()
                .and_then(|v| v.dyn_into::<WebGlTexture>().ok())
        } else {
            None
        };

        let original_pixel_pack_buffer = if config.pixel_pack_buffer {
            gl.get_parameter(WebGl2RenderingContext::PIXEL_PACK_BUFFER_BINDING)
                .ok()
                .and_then(|v| v.dyn_into::<WebGlBuffer>().ok())
        } else {
            None
        };

        // Save current scissor test state if requested
        let scissor_enabled = if config.scissor {
            gl.get_parameter(WebGl2RenderingContext::SCISSOR_TEST)
                .unwrap()
                .as_bool()
                .unwrap_or(false)
        } else {
            false
        };

        // Save current viewport if requested
        let viewport = if config.viewport {
            let viewport_js = gl
                .get_parameter(WebGl2RenderingContext::VIEWPORT)
                .unwrap()
                .dyn_into::<js_sys::Int32Array>()
                .unwrap();
            let viewport_vec = viewport_js.to_vec();
            [
                viewport_vec[0],
                viewport_vec[1],
                viewport_vec[2],
                viewport_vec[3],
            ]
        } else {
            [0, 0, 0, 0]
        };

        Self {
            gl: gl.clone(),
            config,
            original_framebuffer,
            original_read_framebuffer,
            original_active_texture,
            original_texture_2d,
            original_texture_2d_array,
            original_pixel_pack_buffer,
            scissor_enabled,
            viewport,
        }
    }

    /// Create a state guard for clearing an atlas region operations.
    fn for_clear_atlas_region(gl: &WebGl2RenderingContext) -> Self {
        Self::with_config(
            gl,
            WebGlStateConfig {
                framebuffer: true,
                scissor: true,
                viewport: true,
                ..Default::default()
            },
        )
    }

    /// Create a state guard for texture copying operations.
    fn for_texture_copy(gl: &WebGl2RenderingContext) -> Self {
        Self::with_config(
            gl,
            WebGlStateConfig {
                read_framebuffer: true,
                active_texture: true,
                texture_2d_array: true,
                ..Default::default()
            },
        )
    }
}

impl Drop for WebGlStateGuard {
    /// Restore WebGL state when the guard goes out of scope.
    /// Only restores state that was configured to be saved.
    fn drop(&mut self) {
        // Restore scissor test state if it was saved
        if self.config.scissor {
            if self.scissor_enabled {
                self.gl.enable(WebGl2RenderingContext::SCISSOR_TEST);
            } else {
                self.gl.disable(WebGl2RenderingContext::SCISSOR_TEST);
            }
        }

        // Restore viewport if it was saved
        if self.config.viewport {
            self.gl.viewport(
                self.viewport[0],
                self.viewport[1],
                self.viewport[2],
                self.viewport[3],
            );
        }

        // Restore original framebuffer if it was saved
        if self.config.framebuffer {
            self.gl.bind_framebuffer(
                WebGl2RenderingContext::FRAMEBUFFER,
                self.original_framebuffer.as_ref(),
            );
        }

        // Restore original read framebuffer if it was saved
        if self.config.read_framebuffer {
            self.gl.bind_framebuffer(
                WebGl2RenderingContext::READ_FRAMEBUFFER,
                self.original_read_framebuffer.as_ref(),
            );
        }

        if self.config.active_texture
            && let Some(active_texture) = self.original_active_texture
        {
            self.gl.active_texture(active_texture);
        }

        if self.config.texture_2d {
            self.gl.bind_texture(
                WebGl2RenderingContext::TEXTURE_2D,
                self.original_texture_2d.as_ref(),
            );
        }

        // Restore original 2D array texture binding if it was saved
        if self.config.texture_2d_array {
            self.gl.bind_texture(
                WebGl2RenderingContext::TEXTURE_2D_ARRAY,
                self.original_texture_2d_array.as_ref(),
            );
        }

        if self.config.pixel_pack_buffer {
            self.gl.bind_buffer(
                WebGl2RenderingContext::PIXEL_PACK_BUFFER,
                self.original_pixel_pack_buffer.as_ref(),
            );
        }
    }
}
/// Configuration for which WebGL state to save/restore.
#[derive(Debug, Default)]
pub(crate) struct WebGlStateConfig {
    /// Save/restore framebuffer binding (`FRAMEBUFFER_BINDING`)
    pub(crate) framebuffer: bool,
    /// Save/restore read framebuffer binding (`READ_FRAMEBUFFER_BINDING`)
    pub(crate) read_framebuffer: bool,
    /// Save/restore active texture unit (`ACTIVE_TEXTURE`)
    pub(crate) active_texture: bool,
    /// Save/restore 2D texture binding (`TEXTURE_BINDING_2D`)
    pub(crate) texture_2d: bool,
    /// Save/restore 2D array texture binding (`TEXTURE_BINDING_2D_ARRAY`)
    pub(crate) texture_2d_array: bool,
    /// Save/restore pixel pack buffer binding (`PIXEL_PACK_BUFFER_BINDING`)
    pub(crate) pixel_pack_buffer: bool,
    /// Save/restore scissor test state
    pub(crate) scissor: bool,
    /// Save/restore viewport
    pub(crate) viewport: bool,
}

/// Create a WebGL shader program from vertex and fragment sources.
fn create_shader_program(
    gl: &WebGl2RenderingContext,
    vertex_src: &str,
    fragment_src: &str,
) -> Program {
    // Compile vertex shader.
    let vertex_shader = VertexShader::new(gl);
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
    let fragment_shader = FragmentShader::new(gl);
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
    let program = Program::new(gl);
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

    program
}

/// Get the  uniform locations for the `render_strips` program.
fn get_strip_uniforms(gl: &WebGl2RenderingContext, program: &Program) -> StripUniforms {
    let config_vs_name = render::vertex::CONFIG;
    let config_vs_block_index = gl.get_uniform_block_index(program, config_vs_name);

    let config_fs_name = render::fragment::CONFIG;
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
    let alphas_texture_name = render::fragment::ALPHAS_TEXTURE;
    let layer_input_texture_name = render::fragment::LAYER_INPUT_TEXTURE;
    let atlas_texture_array_name = render::fragment::ATLAS_TEXTURE_ARRAY;
    let encoded_paints_texture_fs_name = render::fragment::ENCODED_PAINTS_TEXTURE;
    let encoded_paints_texture_vs_name = render::vertex::ENCODED_PAINTS_TEXTURE;
    let gradient_texture_name = render::fragment::GRADIENT_TEXTURE;
    let external_texture_name = render::fragment::EXTERNAL_TEXTURE;

    StripUniforms {
        config_vs_block_index,
        config_fs_block_index,
        alphas_texture: gl
            .get_uniform_location(program, alphas_texture_name)
            .unwrap(),
        layer_input_texture: gl
            .get_uniform_location(program, layer_input_texture_name)
            .unwrap(),
        atlas_texture_array: gl
            .get_uniform_location(program, atlas_texture_array_name)
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
        external_texture: gl
            .get_uniform_location(program, external_texture_name)
            .unwrap(),
    }
}

fn get_filter_pass_uniforms(gl: &WebGl2RenderingContext, program: &Program) -> FilterPassUniforms {
    let filter_data = gl
        .get_uniform_location(program, filter_shader::fragment::FILTER_DATA)
        .unwrap();
    let in_tex = gl
        .get_uniform_location(program, filter_shader::fragment::IN_TEX)
        .unwrap();
    let original_texture = gl
        .get_uniform_location(program, filter_shader::fragment::ORIGINAL_TEXTURE)
        .unwrap();
    FilterPassUniforms {
        filter_data,
        in_tex,
        original_texture,
    }
}

fn get_blend_uniforms(gl: &WebGl2RenderingContext, program: &Program) -> BlendUniforms {
    BlendUniforms {
        layer_texture_0: gl
            .get_uniform_location(program, blend::fragment::LAYER_TEXTURE_0)
            .unwrap(),
        layer_texture_1: gl
            .get_uniform_location(program, blend::fragment::LAYER_TEXTURE_1)
            .unwrap(),
    }
}

fn get_copy_uniforms(gl: &WebGl2RenderingContext, program: &Program) -> CopyUniforms {
    CopyUniforms {
        source_texture: gl
            .get_uniform_location(program, copy::fragment::SOURCE_TEXTURE)
            .unwrap(),
    }
}

/// Vertex attribute layout for [`FilterInstanceData`].
const FILTER_ATTRIBS: [(i32, i32); 9] = [
    (2, 0),  // src_min
    (2, 8),  // src_max
    (2, 16), // dest_min
    (2, 24), // dest_max
    (2, 32), // dest_atlas_size
    (1, 40), // filter_data_offset
    (2, 44), // original_min
    (2, 52), // original_max
    (1, 60), // other_data
];

const FILTER_INSTANCE_STRIDE: i32 = size_of::<FilterInstanceData>() as i32;

fn initialize_filter_vao(gl: &WebGl2RenderingContext, resources: &WebGlResources) {
    gl.bind_vertex_array(Some(&resources.filter_vao));
    gl.bind_buffer(
        WebGl2RenderingContext::ARRAY_BUFFER,
        Some(&resources.filter_instance_buffer),
    );

    for (loc, &(components, offset)) in FILTER_ATTRIBS.iter().enumerate() {
        let loc = loc as u32;
        gl.enable_vertex_attrib_array(loc);
        gl.vertex_attrib_i_pointer_with_i32(
            loc,
            components,
            WebGl2RenderingContext::UNSIGNED_INT,
            FILTER_INSTANCE_STRIDE,
            offset,
        );
        gl.vertex_attrib_divisor(loc, 1);
    }

    gl.bind_vertex_array(None);
}

const BLEND_ATTRIB_COUNT: u32 = 8;
const BLEND_INSTANCE_STRIDE: i32 = size_of::<GpuBlendInstance>() as i32;

const COPY_ATTRIB_COUNT: u32 = 4;
const COPY_INSTANCE_STRIDE: i32 = size_of::<GpuCopyInstance>() as i32;

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

fn initialize_blend_vao(gl: &WebGl2RenderingContext, resources: &WebGlResources) {
    gl.bind_vertex_array(Some(&resources.blend_vao));
    gl.bind_buffer(
        WebGl2RenderingContext::ARRAY_BUFFER,
        Some(&resources.blend_instance_buffer),
    );

    initialize_packed_u32_attribs(gl, BLEND_ATTRIB_COUNT, BLEND_INSTANCE_STRIDE);

    gl.bind_vertex_array(None);
}

fn initialize_copy_vao(gl: &WebGl2RenderingContext, resources: &WebGlResources) {
    gl.bind_vertex_array(Some(&resources.copy_vao));
    gl.bind_buffer(
        WebGl2RenderingContext::ARRAY_BUFFER,
        Some(&resources.copy_instance_buffer),
    );

    initialize_packed_u32_attribs(gl, COPY_ATTRIB_COUNT, COPY_INSTANCE_STRIDE);

    gl.bind_vertex_array(None);
}

/// Create a texture with nearest neighbor sampling and clamp-to-edge wrapping.
pub(crate) fn create_texture(gl: &WebGl2RenderingContext) -> Texture {
    create_texture_inner(gl, WebGl2RenderingContext::TEXTURE_2D)
}

/// Create a texture array with nearest neighbor sampling and
/// clamp-to-edge wrapping.
fn create_texture_array(gl: &WebGl2RenderingContext) -> Texture {
    create_texture_inner(gl, WebGl2RenderingContext::TEXTURE_2D_ARRAY)
}

fn create_texture_inner(gl: &WebGl2RenderingContext, target: u32) -> Texture {
    let texture = Texture::new(gl);
    gl.active_texture(WebGl2RenderingContext::TEXTURE0);
    gl.bind_texture(target, Some(&texture));
    // The filter and wrap modes are irrelevant because the shader
    // (`render.wgsl`) exclusively uses `textureLoad`, which bypasses
    // the sampler entirely.
    gl.tex_parameteri(
        target,
        WebGl2RenderingContext::TEXTURE_MIN_FILTER,
        WebGl2RenderingContext::NEAREST as i32,
    );
    gl.tex_parameteri(
        target,
        WebGl2RenderingContext::TEXTURE_MAG_FILTER,
        WebGl2RenderingContext::NEAREST as i32,
    );
    gl.tex_parameteri(
        target,
        WebGl2RenderingContext::TEXTURE_WRAP_S,
        WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
    );
    gl.tex_parameteri(
        target,
        WebGl2RenderingContext::TEXTURE_WRAP_T,
        WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
    );
    // Also only to be defensive, in theory this shouldn't be necessary since we use
    // `NEAREST` for both filters.
    gl.tex_parameteri(target, WebGl2RenderingContext::TEXTURE_MAX_LEVEL, 0);

    texture
}

/// Create a 1x1 RGBA8 placeholder texture.
fn create_placeholder_texture(gl: &WebGl2RenderingContext) -> Texture {
    let texture = create_texture(gl);
    gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
        WebGl2RenderingContext::TEXTURE_2D,
        0,
        WebGl2RenderingContext::RGBA8 as i32,
        1,
        1,
        0,
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        Some(&[0, 0, 0, 0]),
    )
    .unwrap();
    texture
}

fn upload_layer_config_buffer(
    gl: &WebGl2RenderingContext,
    buffer: &Buffer,
    size: SizeU16,
    max_texture_dimension_2d: u32,
) {
    let config = Config {
        width: u32::from(size.width()),
        height: u32::from(size.height()),
        strip_height: u32::from(Tile::HEIGHT),
        alphas_tex_width_bits: max_texture_dimension_2d.trailing_zeros(),
        encoded_paints_tex_width_bits: max_texture_dimension_2d.trailing_zeros(),
        strip_offset_x: 0,
        strip_offset_y: 0,
        // Always use y-down when rendering to layer textures.
        negate_ndc: 0,
    };
    gl.bind_buffer(WebGl2RenderingContext::UNIFORM_BUFFER, Some(buffer));
    gl.buffer_data_with_u8_array(
        WebGl2RenderingContext::UNIFORM_BUFFER,
        bytemuck::bytes_of(&config),
        WebGl2RenderingContext::STATIC_DRAW,
    );
}

/// Create all WebGL resources needed for rendering.
fn create_webgl_resources(
    gl: &WebGl2RenderingContext,
    image_cache: &ImageCache,
    layer_config: LayersConfig,
) -> WebGlResources {
    let strip_vao = VertexArray::new(gl);
    let filter_vao = VertexArray::new(gl);
    let blend_vao = VertexArray::new(gl);
    let copy_vao = VertexArray::new(gl);
    let filter_instance_buffer = Buffer::new(gl);
    let blend_instance_buffer = Buffer::new(gl);
    let copy_instance_buffer = Buffer::new(gl);

    let strips_buffer = Buffer::new(gl);
    let view_config_buffer = Buffer::new(gl);
    let max_texture_dimension_2d = get_max_texture_dimension_2d(gl);
    let texture_size = layer_config.initial_intermediate_texture_size();
    let layer_config_buffers = core::array::from_fn(|_| {
        let buffer = Buffer::new(gl);
        upload_layer_config_buffer(gl, &buffer, texture_size, max_texture_dimension_2d);
        buffer
    });

    // Create and configure alpha texture.
    let alphas_texture = create_texture(gl);

    let AtlasConfig {
        atlas_size: (atlas_width, atlas_height),
        initial_atlas_count,
        ..
    } = image_cache.atlas_manager().config();
    let atlas_texture_array =
        create_atlas_texture_array(gl, *atlas_width, *atlas_height, *initial_atlas_count as u32);

    // Create a 1x1 stub atlas texture array for use during render_to_atlas.
    // This avoids binding the real atlas as a shader input while it is the render target.
    let stub_atlas_texture_array = create_atlas_texture_array(gl, 1, 1, 1);

    // Create and configure encoded paints texture.
    let encoded_paints_texture = create_texture(gl);

    // Create and configure gradient texture.
    let gradient_texture = create_texture(gl);
    let placeholder_external_texture = create_placeholder_texture(gl);

    let layer_textures: [Vec<WebGlIntermediateTexture>; 2] = core::array::from_fn(|_| Vec::new());
    let scratch_texture = None;
    let filter_data_texture = create_texture(gl);

    WebGlResources {
        strip_vao,
        strips_buffer,
        alphas_texture,
        alpha_texture_height: 0,
        atlas_texture_array,
        encoded_paints_texture,
        encoded_paints_texture_height: 0,
        gradient_texture,
        placeholder_external_texture,
        gradient_texture_height: 0,
        view_config_buffer,
        view_framebuffer_override: None,
        depth_cleared_this_frame: false,
        // Note: we use DEPTH (not DEPTH_ATTACHMENT) because we render to the default
        // framebuffer. If we ever support non-default framebuffers, this must change
        // to DEPTH_ATTACHMENT.
        depth_attachment_array: js_sys::Array::of1(&WebGl2RenderingContext::DEPTH.into()),
        max_texture_dimension_2d,
        texture_size,
        stub_atlas_texture_array,
        atlas_render_framebuffer: None,
        filter_data_texture,
        filter_data_texture_height: 0,
        filter_instance_buffer,
        filter_vao,
        blend_instance_buffer,
        blend_vao,
        copy_instance_buffer,
        copy_vao,
        scratch_texture,
        layer_config_buffers,
        layer_textures,
    }
}

fn create_intermediate_texture(
    gl: &WebGl2RenderingContext,
    size: SizeU16,
) -> WebGlIntermediateTexture {
    let texture = create_layer_texture(gl, size);
    let framebuffer = create_framebuffer_for_texture(gl, &texture);
    WebGlIntermediateTexture {
        texture,
        framebuffer,
    }
}

/// Create an atlas texture array.
pub(crate) fn create_atlas_texture_array(
    gl: &WebGl2RenderingContext,
    width: u32,
    height: u32,
    layer_count: u32,
) -> WebGlTextureArray {
    let atlas_texture = create_texture_array(gl);

    // Initialize with empty texture array data
    gl.tex_image_3d_with_opt_u8_array(
        WebGl2RenderingContext::TEXTURE_2D_ARRAY,
        0,
        WebGl2RenderingContext::RGBA8 as i32,
        width as i32,
        height as i32,
        layer_count as i32,
        0,
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        None,
    )
    .unwrap();

    WebGlTextureArray::new(atlas_texture, width, height, layer_count)
}

/// Create a texture for layer rendering.
fn create_layer_texture(gl: &WebGl2RenderingContext, size: SizeU16) -> Texture {
    let texture = create_texture(gl);
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

    gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
        WebGl2RenderingContext::TEXTURE_2D,
        0,
        WebGl2RenderingContext::RGBA8 as i32,
        i32::from(size.width()),
        i32::from(size.height()),
        0,
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        None,
    )
    .unwrap();

    texture
}

/// Create a framebuffer for a texture.
pub(crate) fn create_framebuffer_for_texture(
    gl: &WebGl2RenderingContext,
    texture: &WebGlTexture,
) -> Framebuffer {
    let framebuffer = Framebuffer::new(gl);
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

const STRIP_STRIDE: i32 = size_of::<GpuStrip>() as i32;
const STRIP_ATTR_COUNT: i32 = STRIP_STRIDE / 4;
const _: () = assert!(
    STRIP_STRIDE == 24,
    "GpuStrip layout must match strip vertex stride"
);

/// Initialize strip VAO.
fn initialize_strip_vao(gl: &WebGl2RenderingContext, resources: &WebGlResources) {
    gl.bind_vertex_array(Some(&resources.strip_vao));
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

/// Context for WebGL rendering operations.
// TODO: Improve buffer management. Currently a single buffer is used per resource, which means that
// the GPU must finish drawing before the next `upload_strip_pair` can be executed (effectively pausing
// execution). Investigate a buffer pool or creating a new buffer per pass.
struct WebGlRendererContext<'a> {
    programs: &'a mut WebGlPrograms,
    gl: &'a WebGl2RenderingContext,
    scratch: &'a mut ScratchBuffers,
}

impl WebGlRendererContext<'_> {
    /// Render strips to the specified render target.
    fn strip_pass_inner(
        &mut self,
        opaque_strips: &[GpuStrip],
        alpha_strips: RangedSlice<'_, GpuStrip>,
        target: DrawPassTarget,
        child_layer_texture: Option<LayerTextureId>,
    ) {
        let opaque_count = opaque_strips.len();
        let alpha_count = alpha_strips.len();
        if opaque_count == 0 && alpha_count == 0 {
            return;
        }
        match &target {
            DrawPassTarget::Root(_) => {
                self.gl.bind_framebuffer(
                    WebGl2RenderingContext::FRAMEBUFFER,
                    self.programs.resources.view_framebuffer_override.as_deref(),
                );
                let width = self.programs.render_size.width;
                let height = self.programs.render_size.height;
                self.gl.viewport(0, 0, width as i32, height as i32);

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
            }
            DrawPassTarget::Layer(id) => {
                self.gl.bind_framebuffer(
                    WebGl2RenderingContext::FRAMEBUFFER,
                    Some(self.programs.resources.layer_framebuffer(*id)),
                );
                let size = self.texture_size();
                self.gl
                    .viewport(0, 0, i32::from(size.width()), i32::from(size.height()));

                let buf =
                    &self.programs.resources.layer_config_buffers[id.texture_parity.get_parity()];
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

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE1);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(child_layer_texture.map_or_else(
                || &self.programs.resources.placeholder_external_texture,
                |id| self.programs.resources.layer_texture(id),
            )),
        );
        self.gl
            .uniform1i(Some(&self.programs.strip_uniforms.layer_input_texture), 1);

        // Bind atlas texture array for image rendering
        self.gl.active_texture(WebGl2RenderingContext::TEXTURE2);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D_ARRAY,
            Some(&self.programs.resources.atlas_texture_array.texture),
        );
        self.gl
            .uniform1i(Some(&self.programs.strip_uniforms.atlas_texture_array), 2);

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

        // We don't support external textures in our WebGL backend yet; instead we bind a
        // placeholder so the shader's sampler binding is satisfied.
        self.gl.active_texture(WebGl2RenderingContext::TEXTURE5);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&self.programs.resources.placeholder_external_texture),
        );
        self.gl
            .uniform1i(Some(&self.programs.strip_uniforms.external_texture), 5);

        // TODO: Today, we only support early-z rejection on the final view. If we wanted to support
        // intermediate layers, we would require separate depth buffers for each target. We can explore
        // that possibility in the future.
        let enable_opaque = target.enable_opaque();

        self.programs
            .upload_strip_pair(self.gl, opaque_strips, alpha_strips);
        let opaque_count = opaque_count as i32;
        let alpha_count = alpha_count as i32;

        if enable_opaque {
            self.gl.enable(WebGl2RenderingContext::DEPTH_TEST);
            self.gl.depth_func(WebGl2RenderingContext::LEQUAL);

            // Clear depth buffer on first use per frame.
            if !self.programs.resources.depth_cleared_this_frame {
                self.programs.resources.depth_cleared_this_frame = true;
                self.gl.clear_depth(1.0);
                self.gl.clear(WebGl2RenderingContext::DEPTH_BUFFER_BIT);
            }

            // Opaque pass: front-to-back, depth test ON, depth write ON, blend OFF.
            if opaque_count > 0 {
                self.gl.depth_mask(true);
                self.gl.disable(WebGl2RenderingContext::BLEND);
                self.gl.draw_arrays_instanced(
                    WebGl2RenderingContext::TRIANGLE_STRIP,
                    0,
                    4,
                    opaque_count,
                );
            }

            // Alpha pass: back-to-front, depth test ON, depth write OFF, blend ON.
            if alpha_count > 0 {
                // Rebind attribute pointers with offset to start at the alpha portion
                // of the buffer.
                let alpha_byte_offset = opaque_count * STRIP_STRIDE;
                for i in 0..STRIP_ATTR_COUNT {
                    self.gl.vertex_attrib_i_pointer_with_i32(
                        i as u32,
                        1,
                        WebGl2RenderingContext::UNSIGNED_INT,
                        STRIP_STRIDE,
                        i * 4 + alpha_byte_offset,
                    );
                }

                self.gl.depth_mask(false);
                self.gl.enable(WebGl2RenderingContext::BLEND);
                self.gl.draw_arrays_instanced(
                    WebGl2RenderingContext::TRIANGLE_STRIP,
                    0,
                    4,
                    alpha_count,
                );

                // Restore attribute offsets to base for subsequent passes.
                for i in 0..STRIP_ATTR_COUNT {
                    self.gl.vertex_attrib_i_pointer_with_i32(
                        i as u32,
                        1,
                        WebGl2RenderingContext::UNSIGNED_INT,
                        STRIP_STRIDE,
                        i * 4,
                    );
                }
            }

            // Restore state.
            self.gl.disable(WebGl2RenderingContext::DEPTH_TEST);
            self.gl.depth_mask(true);
            self.gl.enable(WebGl2RenderingContext::BLEND);
        } else {
            // Intermediate target: single draw with blending, no depth.
            self.gl.draw_arrays_instanced(
                WebGl2RenderingContext::TRIANGLE_STRIP,
                0,
                4,
                opaque_count + alpha_count,
            );
        }

        // Clean up.
        self.gl.bind_vertex_array(None);
    }

    fn texture_size(&self) -> SizeU16 {
        self.programs.resources.texture_size
    }

    fn blend_pass_inner(
        &mut self,
        blends: RangedSlice<'_, BlendOp>,
        parent_texture_parity: TextureParity,
        texture_pair: LayerTexturePair,
    ) {
        if blends.len() == 0 {
            return;
        }

        let texture_size = self.texture_size();
        self.gl.disable(WebGl2RenderingContext::BLEND);
        self.gl.disable(WebGl2RenderingContext::SCISSOR_TEST);
        self.gl.disable(WebGl2RenderingContext::DEPTH_TEST);
        self.gl.depth_mask(false);
        self.gl
            .bind_vertex_array(Some(&self.programs.resources.blend_vao));

        let resources = &self.programs.resources;
        self.scratch.blend_instances.clear();
        self.scratch.blend_instances.extend(
            blends
                .iter()
                .copied()
                .filter(|blend| !blend.blend_bbox.is_empty())
                .map(|blend| {
                    resources.layer_texture(blend.parent_region.texture.target);
                    resources.layer_texture(blend.child_region.texture.target);
                    gpu_blend_instance(blend, texture_size)
                }),
        );
        if self.scratch.blend_instances.is_empty() {
            self.gl.bind_vertex_array(None);
            self.gl.disable(WebGl2RenderingContext::SCISSOR_TEST);
            self.gl.depth_mask(true);
            self.gl.enable(WebGl2RenderingContext::BLEND);
            return;
        }

        self.programs
            .upload_blend_instances(self.gl, &self.scratch.blend_instances);
        let instance_count = i32::try_from(self.scratch.blend_instances.len()).unwrap();

        self.gl.bind_framebuffer(
            WebGl2RenderingContext::FRAMEBUFFER,
            Some(self.programs.resources.scratch_framebuffer()),
        );
        self.gl.viewport(
            0,
            0,
            i32::from(texture_size.width()),
            i32::from(texture_size.height()),
        );
        self.gl.use_program(Some(&self.programs.blend_program));

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE2);
        self.gl
            .bind_texture(WebGl2RenderingContext::TEXTURE_2D, None);

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(
                self.programs
                    .resources
                    .layer_texture(texture_pair.layer_id(TextureParity::Even)),
            ),
        );
        self.gl
            .uniform1i(Some(&self.programs.blend_uniforms.layer_texture_0), 0);

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE1);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(
                self.programs
                    .resources
                    .layer_texture(texture_pair.layer_id(TextureParity::Odd)),
            ),
        );
        self.gl
            .uniform1i(Some(&self.programs.blend_uniforms.layer_texture_1), 1);

        self.gl
            .draw_arrays_instanced(WebGl2RenderingContext::TRIANGLE_STRIP, 0, 4, instance_count);

        self.scratch.copy_instances.clear();
        self.scratch.copy_instances.extend(
            self.scratch
                .blend_instances
                .iter()
                .copied()
                .map(|instance| instance.copy_from_scratch(texture_size)),
        );
        self.programs
            .upload_copy_instances(self.gl, &self.scratch.copy_instances);

        self.gl.bind_framebuffer(
            WebGl2RenderingContext::FRAMEBUFFER,
            Some(
                self.programs
                    .resources
                    .layer_framebuffer(texture_pair.layer_id(parent_texture_parity)),
            ),
        );
        self.gl.viewport(
            0,
            0,
            i32::from(texture_size.width()),
            i32::from(texture_size.height()),
        );
        self.gl
            .bind_vertex_array(Some(&self.programs.resources.copy_vao));
        self.gl.use_program(Some(&self.programs.copy_program));

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(self.programs.resources.scratch_binding_texture()),
        );
        self.gl
            .uniform1i(Some(&self.programs.copy_uniforms.source_texture), 0);

        self.gl
            .draw_arrays_instanced(WebGl2RenderingContext::TRIANGLE_STRIP, 0, 4, instance_count);

        self.gl.bind_vertex_array(None);
        self.gl.disable(WebGl2RenderingContext::SCISSOR_TEST);
        self.gl.depth_mask(true);
        self.gl.enable(WebGl2RenderingContext::BLEND);
    }

    fn filter_pass_inner(&mut self, plan: &FilterPassPlan, textures: FilterTexturePair) {
        if plan.is_empty() {
            return;
        }

        self.gl.disable(WebGl2RenderingContext::BLEND);
        self.gl.disable(WebGl2RenderingContext::SCISSOR_TEST);
        self.gl.disable(WebGl2RenderingContext::DEPTH_TEST);
        self.gl.depth_mask(false);

        let copy_pass = plan.copy_pass();
        if !copy_pass.is_empty() {
            self.programs.upload_copy_instances(self.gl, copy_pass);
            self.gl
                .bind_vertex_array(Some(&self.programs.resources.copy_vao));
            self.gl.use_program(Some(&self.programs.copy_program));
            self.gl.bind_framebuffer(
                WebGl2RenderingContext::FRAMEBUFFER,
                Some(self.programs.resources.scratch_framebuffer()),
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
                Some(self.programs.resources.layer_texture(textures.original())),
            );
            self.gl
                .uniform1i(Some(&self.programs.copy_uniforms.source_texture), 0);
            self.gl.draw_arrays_instanced(
                WebGl2RenderingContext::TRIANGLE_STRIP,
                0,
                4,
                i32::try_from(copy_pass.len()).unwrap(),
            );
        }

        self.gl.use_program(Some(&self.programs.filter_program));
        self.gl
            .bind_vertex_array(Some(&self.programs.resources.filter_vao));

        self.gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&self.programs.resources.filter_data_texture),
        );
        self.gl
            .uniform1i(Some(&self.programs.filter_uniforms.filter_data), 0);
        self.gl.active_texture(WebGl2RenderingContext::TEXTURE2);
        self.gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(self.programs.resources.scratch_binding_texture()),
        );
        self.gl
            .uniform1i(Some(&self.programs.filter_uniforms.original_texture), 2);

        for (step_index, instances) in plan.steps().enumerate() {
            self.do_filter_instance_pass(
                instances,
                textures.input(step_index),
                textures.output(step_index),
            );
        }

        self.gl.bind_vertex_array(None);
        self.gl.depth_mask(true);
        self.gl.enable(WebGl2RenderingContext::BLEND);
    }

    fn do_filter_instance_pass(
        &self,
        instances: &[FilterInstanceData],
        input: LayerTextureId,
        output: LayerTextureId,
    ) {
        if instances.is_empty() {
            return;
        }

        self.gl.bind_framebuffer(
            WebGl2RenderingContext::FRAMEBUFFER,
            Some(self.programs.resources.layer_framebuffer(output)),
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
            Some(self.programs.resources.layer_texture(input)),
        );
        self.gl
            .uniform1i(Some(&self.programs.filter_uniforms.in_tex), 1);

        self.programs.upload_filter_instances(self.gl, instances);
        self.gl.draw_arrays_instanced(
            WebGl2RenderingContext::TRIANGLE_STRIP,
            0,
            4,
            i32::try_from(instances.len()).unwrap(),
        );
    }

    fn clear_pass_inner(&self, target: LayerTextureId, rects: &[RectU16]) {
        if rects.is_empty() {
            return;
        }

        self.prepare_clear_rects(target);
        for rect in rects.iter().copied().filter(|rect| !rect.is_empty()) {
            self.clear_rect(rect);
        }
        self.finish_clear_rects();
        self.gl.enable(WebGl2RenderingContext::BLEND);
    }

    fn prepare_clear_rects(&self, target: LayerTextureId) {
        let size = self.texture_size();
        self.gl.disable(WebGl2RenderingContext::BLEND);
        self.gl.clear_color(0.0, 0.0, 0.0, 0.0);
        let framebuffer = self.programs.resources.layer_framebuffer(target);
        self.gl
            .bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(framebuffer));
        self.gl
            .viewport(0, 0, i32::from(size.width()), i32::from(size.height()));
        self.gl.enable(WebGl2RenderingContext::SCISSOR_TEST);
    }

    fn clear_rect(&self, rect: RectU16) {
        self.gl.scissor(
            i32::from(rect.x0),
            i32::from(rect.y0),
            i32::from(rect.width()),
            i32::from(rect.height()),
        );
        self.gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);
    }

    fn finish_clear_rects(&self) {
        self.gl.disable(WebGl2RenderingContext::SCISSOR_TEST);
    }
}

impl RendererBackend for WebGlRendererContext<'_> {
    fn opaque_pass(&mut self, strips: &[GpuStrip]) {
        self.strip_pass_inner(
            strips,
            RangedSlice::empty(),
            DrawPassTarget::Root(RootRenderTarget::UserSurface),
            None,
        );
    }

    fn draw_pass(
        &mut self,
        strips: RangedSlice<'_, GpuStrip>,
        _external_texture_runs: &[ExternalTextureRun],
        target: DrawPassTarget,
        child_layer_texture: Option<LayerTextureId>,
    ) {
        self.strip_pass_inner(&[], strips, target, child_layer_texture);
    }

    fn blend_pass(
        &mut self,
        blends: RangedSlice<'_, BlendOp>,
        parent_texture_parity: TextureParity,
        texture_pair: LayerTexturePair,
    ) {
        self.blend_pass_inner(blends, parent_texture_parity, texture_pair);
    }

    fn filter_pass(&mut self, plan: &FilterPassPlan, textures: FilterTexturePair) {
        self.filter_pass_inner(plan, textures);
    }

    fn clear_pass(&mut self, target: LayerTextureId, rects: &[RectU16]) {
        self.clear_pass_inner(target, rects);
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

    /// Write image data to a specific layer of an atlas texture array at the specified offset.
    fn write_to_atlas_layer(
        &self,
        gl: &WebGl2RenderingContext,
        atlas_texture_array: &WebGlTexture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    );
}

/// Implementation for `Pixmap` - direct upload using raw pixel data.
impl WebGlAtlasWriter for Pixmap {
    fn width(&self) -> u32 {
        self.width() as u32
    }

    fn height(&self) -> u32 {
        self.height() as u32
    }

    fn write_to_atlas_layer(
        &self,
        gl: &WebGl2RenderingContext,
        atlas_texture_array: &WebGlTexture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        // Bind the atlas texture array
        gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D_ARRAY,
            Some(atlas_texture_array),
        );

        // Convert pixmap data to the format expected by WebGL
        let rgba_data = self.data_as_u8_slice();

        // Upload the image data to the specific layer and region of the atlas texture array
        gl.tex_sub_image_3d_with_opt_u8_array(
            WebGl2RenderingContext::TEXTURE_2D_ARRAY,
            0,
            offset[0] as i32,
            offset[1] as i32,
            layer as i32,
            width as i32,
            height as i32,
            1,
            WebGl2RenderingContext::RGBA,
            WebGl2RenderingContext::UNSIGNED_BYTE,
            Some(rgba_data),
        )
        .unwrap();
    }
}

/// Implementation for `Arc<Pixmap>`.
impl WebGlAtlasWriter for Arc<Pixmap> {
    fn width(&self) -> u32 {
        self.as_ref().width() as u32
    }

    fn height(&self) -> u32 {
        self.as_ref().height() as u32
    }

    fn write_to_atlas_layer(
        &self,
        gl: &WebGl2RenderingContext,
        atlas_texture_array: &WebGlTexture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        self.as_ref()
            .write_to_atlas_layer(gl, atlas_texture_array, layer, offset, width, height);
    }
}

/// Implementation for `WebGlTexture` - texture-to-texture copy.
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

    fn write_to_atlas_layer(
        &self,
        gl: &WebGl2RenderingContext,
        atlas_texture_array: &WebGlTexture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        copy_to_texture_array_layer(
            gl,
            |gl| {
                // Attach source texture to read framebuffer
                gl.framebuffer_texture_2d(
                    WebGl2RenderingContext::READ_FRAMEBUFFER,
                    WebGl2RenderingContext::COLOR_ATTACHMENT0,
                    WebGl2RenderingContext::TEXTURE_2D,
                    Some(self),
                    0,
                );
            },
            atlas_texture_array,
            layer,
            offset,
            [width, height],
        );
    }
}

/// Wrapper for `WebGlTexture` with known dimensions.
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

    fn write_to_atlas_layer(
        &self,
        gl: &WebGl2RenderingContext,
        atlas_texture_array: &WebGlTexture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        self.texture
            .write_to_atlas_layer(gl, atlas_texture_array, layer, offset, width, height);
    }
}

/// Wrapper for `WebGlTexture` array with known dimensions.
#[derive(Debug)]
pub(crate) struct WebGlTextureArray {
    /// The WebGL texture array.
    texture: Texture,
    /// The size of the texture array.
    size: WebGlTextureSize,
}

impl WebGlTextureArray {
    /// Create a new WebGL texture array wrapper.
    fn new(texture: Texture, width: u32, height: u32, depth_or_array_layers: u32) -> Self {
        Self {
            texture,
            size: WebGlTextureSize {
                width,
                height,
                depth_or_array_layers,
            },
        }
    }

    /// Get the size of the texture array, similar to WGPU's `texture.size()`.
    fn size(&self) -> WebGlTextureSize {
        self.size
    }
}

/// Size information for WebGL texture arrays, similar to WGPU's `Extent3d`.
#[derive(Debug, Clone, Copy)]
struct WebGlTextureSize {
    /// The width of the texture.
    width: u32,
    /// The height of the texture.
    height: u32,
    /// The number of layers in the texture array.
    depth_or_array_layers: u32,
}

/// Helper function to copy from a source texture/framebuffer to a destination texture array layer.
fn copy_to_texture_array_layer(
    gl: &WebGl2RenderingContext,
    source_setup: impl FnOnce(&WebGl2RenderingContext),
    dest_texture_array: &WebGlTexture,
    dest_layer: u32,
    dest_offset: [u32; 2],
    copy_size: [u32; 2],
) {
    let _state_guard = WebGlStateGuard::for_texture_copy(gl);
    let read_framebuffer = Framebuffer::new(gl);

    // Bind destination texture array
    gl.active_texture(WebGl2RenderingContext::TEXTURE0);
    gl.bind_texture(
        WebGl2RenderingContext::TEXTURE_2D_ARRAY,
        Some(dest_texture_array),
    );

    // Bind the READ framebuffer
    gl.bind_framebuffer(
        WebGl2RenderingContext::READ_FRAMEBUFFER,
        Some(&read_framebuffer),
    );

    // Let the caller set up the source (attach texture/layer to read framebuffer)
    source_setup(gl);

    // Copy from READ framebuffer to destination array layer
    gl.copy_tex_sub_image_3d(
        WebGl2RenderingContext::TEXTURE_2D_ARRAY,
        0,
        dest_offset[0] as i32,
        dest_offset[1] as i32,
        dest_layer as i32,
        0,
        0,
        copy_size[0] as i32,
        copy_size[1] as i32,
    );
}

// Upload the data to the currently bound texture assuming a RGBA32UI format.
fn upload_data_to_rgba32_texture(
    gl: &WebGl2RenderingContext,
    texture: &WebGlTexture,
    data: &[u32],
    texture_width: u32,
    texture_height: u32,
) {
    gl.active_texture(WebGl2RenderingContext::TEXTURE0);
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(texture));

    // Safety: This calling `Uint32Array::view` is unsafe because it provides a view into
    // WASM linear memory, and any additional allocations might invalidate that view.
    // In our case, this is not an issue because we only use this view once for uploading
    // data to the GPU below, and no allocations happen between that.
    // The `tex_image_2d` method is synchronous in the sense that once it returns, it is guaranteed
    // that all necessary data has already been read, so any allocations that happen
    // after this block don't affect this anymore.
    //
    // See also: https://wikis.khronos.org/opengl/Synchronization
    // >> There are several OpenGL functions that can pull data directly from client-side memory,
    // >> or push data directly into client-side memory. Functions like `glTexSubImage2D`,
    // >> `glReadPixels`, `glBufferSubData` and so forth.
    //
    // >> Because OpenGL is defined to be synchronous, when any of these functions have
    // >> returned, they must have finished with the client memory. When `glReadPixels` returns,
    // >> the pixel data is in your client memory (unless you are reading into a buffer object).
    // >> When `glBufferSubData` returns, you can immediately modify or delete whatever memory
    // >> pointer you gave it, as OpenGL has already read as much as it wants.
    let packed_array = unsafe { js_sys::Uint32Array::view(data) };

    gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
        WebGl2RenderingContext::TEXTURE_2D,
        0,
        WebGl2RenderingContext::RGBA32UI as i32,
        texture_width as i32,
        texture_height as i32,
        0,
        WebGl2RenderingContext::RGBA_INTEGER,
        WebGl2RenderingContext::UNSIGNED_INT,
        Some(&packed_array),
    )
    .unwrap();
}

pub(crate) mod resource {
    use core::ops::Deref;

    use super::{
        WebGl2RenderingContext, WebGlBuffer, WebGlFramebuffer, WebGlProgram, WebGlShader,
        WebGlTexture, WebGlVertexArrayObject,
    };

    pub(crate) trait GlResource {
        const LABEL: &'static str;

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
        pub(super) fn new(gl: &WebGl2RenderingContext) -> Self {
            let raw =
                T::create(gl).unwrap_or_else(|| panic!("failed to create WebGL {}", T::LABEL));
            Self {
                gl: gl.clone(),
                raw,
            }
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
        const LABEL: &'static str = "texture";

        fn create(gl: &WebGl2RenderingContext) -> Option<Self> {
            gl.create_texture()
        }

        fn delete(gl: &WebGl2RenderingContext, raw: &Self) {
            gl.delete_texture(Some(raw));
        }
    }

    impl GlResource for WebGlBuffer {
        const LABEL: &'static str = "buffer";

        fn create(gl: &WebGl2RenderingContext) -> Option<Self> {
            gl.create_buffer()
        }

        fn delete(gl: &WebGl2RenderingContext, raw: &Self) {
            gl.delete_buffer(Some(raw));
        }
    }

    impl GlResource for WebGlFramebuffer {
        const LABEL: &'static str = "framebuffer";

        fn create(gl: &WebGl2RenderingContext) -> Option<Self> {
            gl.create_framebuffer()
        }

        fn delete(gl: &WebGl2RenderingContext, raw: &Self) {
            gl.delete_framebuffer(Some(raw));
        }
    }

    impl GlResource for WebGlProgram {
        const LABEL: &'static str = "program";

        fn create(gl: &WebGl2RenderingContext) -> Option<Self> {
            gl.create_program()
        }

        fn delete(gl: &WebGl2RenderingContext, raw: &Self) {
            gl.delete_program(Some(raw));
        }
    }

    impl GlResource for WebGlVertexShader {
        const LABEL: &'static str = "vertex shader";

        fn create(gl: &WebGl2RenderingContext) -> Option<Self> {
            gl.create_shader(WebGl2RenderingContext::VERTEX_SHADER)
                .map(Self)
        }

        fn delete(gl: &WebGl2RenderingContext, raw: &Self) {
            gl.delete_shader(Some(raw));
        }
    }

    impl GlResource for WebGlFragmentShader {
        const LABEL: &'static str = "fragment shader";

        fn create(gl: &WebGl2RenderingContext) -> Option<Self> {
            gl.create_shader(WebGl2RenderingContext::FRAGMENT_SHADER)
                .map(Self)
        }

        fn delete(gl: &WebGl2RenderingContext, raw: &Self) {
            gl.delete_shader(Some(raw));
        }
    }

    impl GlResource for WebGlVertexArrayObject {
        const LABEL: &'static str = "vertex array";

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
}
