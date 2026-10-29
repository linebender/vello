// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Native WebGL2 backend for Vello GPU's Sparse Strips renderer.
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

mod atlas;
mod bindings;
mod error;
mod gl;
mod init;
mod passes;
#[cfg(feature = "probe")]
pub(crate) mod probe;
mod resources;
mod target;

pub use atlas::WebGlAtlasWriter;
pub use bindings::{WebGlTextureBindings, WebGlTextureWithDimensions};
#[cfg(feature = "probe")]
pub use error::WebGlProbeOperation;
pub(crate) use error::WebGlResultExt;
pub use error::{
    IncompatibleContextReason, IncompatiblePrecompiledShadersReason, InvalidTextureTargetReason,
    WebGlContextOperation, WebGlDataTransferOperation, WebGlError, WebGlOperation,
    WebGlResourceKind, WebGlShaderInterfaceOperation, WebGlShaderProgramOperation,
    WebGlShaderStage,
};
pub use init::{ShaderCompilationStats, WebGlRendererInit, WebGlRendererInitStatus};
pub use target::{WebGlRenderTarget, WebGlTextureTarget};

use crate::render::common::{IMAGE_PADDING, SharedCore};
use crate::{
    ClearSettings, RenderError, RenderSize, Resources, TargetInit,
    scene::Scene,
    schedule::IntermediateTextureAllocations,
    target::{DrawPassTarget, RootTarget},
};
use alloc::vec::Vec;
#[cfg(feature = "text")]
use glifo::PendingClearRect;
use passes::{WebGlPrograms, WebGlRendererContext};
use vello_common::color::{AlphaColor, Srgb};
use vello_common::image_cache::ImageCache;
use vello_common::multi_atlas::{AtlasConfig, AtlasId};
use vello_common::{
    geometry::{RectU16, SizeU16},
    paint::ImageId,
};
use web_sys::{WebGl2RenderingContext, WebGlRenderbuffer, WebGlTexture};

/// Vello GPU's WebGL2 renderer.
///
/// Note that any operation of [`WebGlRenderer`] might modify the global state of the underlying
/// WebGL context; state like the active texture unit and texture bindings are therefore not
/// guaranteed to be restored to their original value after each method call.
///
/// # Context loss
/// Methods on [`WebGlRenderer`] are **not** guaranteed to return an error if the WebGL context
/// has been lost; therefore, if not handled properly, rendering or other operations might silently
/// fail. To prevent this, clients should create their own listener for the WebGL context lost
/// event and react to it appropriately.
#[derive(Debug)]
pub struct WebGlRenderer {
    /// Programs for rendering.
    pub(super) programs: WebGlPrograms,
    /// WebGL context.
    pub(crate) gl: WebGl2RenderingContext,
    /// Backend-independent renderer state.
    core: SharedCore,
}

/// Options for [`WebGlRenderer::render`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct WebGlRenderOptions {
    /// The color that the target is cleared to before drawing.
    // Note that in WebGL, the contents of drawing buffers are not guaranteed to be preserved
    // unless `preserveDrawingBuffer` has been set to `true` (which is highly discouraged).
    // Therefore, for the WebGL backend we currently don't expose more configurable clear
    // options.
    pub clear_color: AlphaColor<Srgb>,
    /// The position of the scene's origin in the target.
    pub offset: (i32, i32),
}

impl Default for WebGlRenderOptions {
    fn default() -> Self {
        Self {
            clear_color: AlphaColor::TRANSPARENT,
            offset: (0, 0),
        }
    }
}

/// Current allocation state of the WebGL image/glyph atlas textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtlasTextureInfo {
    /// Width of each atlas texture.
    pub width: u16,
    /// Height of each atlas texture.
    pub height: u16,
    /// Number of atlas textures available to the image cache.
    pub texture_count: u32,
}

impl WebGlRenderer {
    /// Returns the shaders compiled while creating this renderer and the time spent.
    ///
    /// `elapsed` is measured until the renderer is finished, so when finishing is deferred (for
    /// example, by polling readiness or doing other work first) it is an upper bound on the actual
    /// compilation time.
    pub fn shader_compilation_stats(&self) -> ShaderCompilationStats {
        self.programs.shader_compilation_stats
    }

    /// Create a target for rendering into `texture`.
    ///
    /// `texture` must be `RGBA8` and `depth` `DEPTH_COMPONENT24`, and `width` and `height` must
    /// match the dimensions of both. The same `depth` can be used for any number of targets of
    /// the same size. Like the drawing buffer, the first row of the texture is the bottom row of
    /// the image.
    pub fn create_texture_target(
        &self,
        texture: &WebGlTexture,
        depth: Option<&WebGlRenderbuffer>,
        width: u16,
        height: u16,
    ) -> Result<WebGlTextureTarget, WebGlError> {
        WebGlTextureTarget::new(&self.gl, texture, depth, width, height)
    }

    /// Render `scene` into `target` using WebGL2
    ///
    /// This method creates GPU resources as needed and schedules potentially multiple draw calls.
    ///
    /// Every [`TextureId`] referenced by the scene must have a binding; this returns
    /// [`WebGlError::Render`] containing [`RenderError::MissingTextureBinding`] otherwise. See
    /// [`WebGlTextureBindings::insert`] for the requirements on the bound textures.
    ///
    /// To render without any texture bindings, you can pass an empty [`WebGlTextureBindings`].
    ///
    /// [`TextureId`]: vello_common::TextureId
    pub fn render(
        &mut self,
        scene: &Scene,
        resources: &mut Resources,
        target: WebGlRenderTarget<'_>,
        texture_bindings: &WebGlTextureBindings,
        options: WebGlRenderOptions,
    ) -> Result<(), WebGlError> {
        #[cfg(feature = "text")]
        resources.before_render(
            self,
            |renderer, glyph_renderer, atlas_count, atlas_config, atlas_id| {
                renderer.render_to_atlas(
                    glyph_renderer,
                    atlas_count,
                    atlas_config,
                    atlas_id,
                    texture_bindings,
                )
            },
            |renderer, image_cache, upload, dst_x, dst_y| {
                renderer.write_to_atlas(
                    image_cache,
                    upload.image_id,
                    &upload.pixmap,
                    Some([dst_x, dst_y]),
                )
            },
        )?;

        let render_size = target.size(&self.gl);
        let target_init = TargetInit::Clear(ClearSettings::Viewport {
            color: options.clear_color,
        });
        match target {
            WebGlRenderTarget::DrawingBuffer => self.render_scene(
                scene,
                &resources.image_cache,
                &render_size,
                target_init,
                RootTarget::UserSurface,
                texture_bindings,
                None,
                options.offset,
            )?,
            WebGlRenderTarget::Texture(target) => self.render_offscreen(target, |renderer| {
                renderer.render_scene(
                    scene,
                    &resources.image_cache,
                    &render_size,
                    target_init,
                    RootTarget::UserSurface,
                    texture_bindings,
                    Some(&target.texture),
                    options.offset,
                )
            })?,
        }

        #[cfg(feature = "text")]
        // TODO: We should sort the rectangles once by atlas
        // page and then clear per atlas page instead of per rect.
        resources.after_render(self, clear_atlas_region)?;

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
    /// `texture_bindings` provides [externally bound textures](`WebGlTextureBindings`)
    /// referenced by the scene. Pass `&WebGlTextureBindings::new()` if the scene does
    /// not use any.
    #[doc(hidden)]
    pub fn render_to_atlas(
        &mut self,
        scene: &Scene,
        atlas_count: u32,
        atlas_config: AtlasConfig,
        atlas_id: AtlasId,
        texture_bindings: &WebGlTextureBindings,
    ) -> Result<(), WebGlError> {
        self.programs
            .maybe_create_atlas_textures(&self.gl, atlas_count)?;

        let (atlas_width, atlas_height) = atlas_config.atlas_size;
        let atlas_render_size = RenderSize {
            width: atlas_width,
            height: atlas_height,
        };
        let target = self.programs.resources.atlas_targets[atlas_id.as_u32() as usize].clone();
        // TODO: Explore using an option instead of a dummy image cache.
        let dummy_image_cache = self
            .core
            .dummy_image_cache
            .take()
            .expect("dummy image cache must exist");
        let result = self.render_offscreen(&target, |renderer| {
            renderer.render_scene(
                scene,
                &dummy_image_cache,
                &atlas_render_size,
                TargetInit::SrcOver,
                RootTarget::AtlasLayer,
                texture_bindings,
                Some(&target.texture),
                (0, 0),
            )
        });
        self.core.dummy_image_cache = Some(dummy_image_cache);

        result
    }

    /// Run `render` with `target` as the root target.
    ///
    /// Note that the method might update the currently bound framebuffer.
    fn render_offscreen(
        &mut self,
        target: &WebGlTextureTarget,
        render: impl FnOnce(&mut Self) -> Result<(), WebGlError>,
    ) -> Result<(), WebGlError> {
        let previous_target = self.programs.resources.root_target.replace(target.clone());
        let result = render(self);
        self.programs.resources.root_target = previous_target;
        result
    }

    fn current_allocations(&self) -> IntermediateTextureAllocations {
        IntermediateTextureAllocations {
            layer_pages: self
                .programs
                .resources
                .layer_textures
                .each_ref()
                .map(Vec::len),
            scratch: self.programs.resources.scratch_texture.is_some(),
        }
    }

    /// Shared render pipeline: prepares GPU resources, runs the scheduler, and
    /// maintains caches.
    ///
    pub(crate) fn render_scene(
        &mut self,
        scene: &Scene,
        image_cache: &ImageCache,
        render_size: &RenderSize,
        target_init: TargetInit<'_>,
        root_output_target: RootTarget,
        texture_bindings: &WebGlTextureBindings,
        render_target_texture: Option<&WebGlTexture>,
        scene_offset: (i32, i32),
    ) -> Result<(), WebGlError> {
        let use_depth_buffer = self.programs.resources.root_uses_depth_buffer();

        let current_allocations = self.current_allocations();
        let schedule = self.core.prepare(
            scene,
            image_cache,
            root_output_target,
            RectU16::visible_scene_bounds(
                scene.recorder.scene_size,
                SizeU16::from_wh(render_size.width, render_size.height),
                scene_offset,
            ),
            use_depth_buffer,
            self.programs.resources.texture_size,
            current_allocations,
            |id, image, region| {
                texture_bindings.encode_paint(id, image, region, render_target_texture)
            },
        )?;
        self.programs
            .prepare_intermediate_textures(&self.gl, &schedule)?;

        // TODO: For the time being, we upload the entire alpha buffer as one big chunk. As a future
        // refinement, we could have a bounded alpha buffer, and break draws when the alpha
        // buffer fills.
        self.programs.prepare(
            &self.gl,
            &mut self.core.gradient_cache,
            &self.core.encoded_paints,
            &mut scene.strip_storage.borrow_mut().alphas,
            render_size,
            scene_offset,
            &self.core.paint_idxs,
            &self.core.schedule_storage.filter_context,
            root_output_target,
        )?;
        self.programs.resources.depth_cleared_this_frame = false;
        let mut ctx = WebGlRendererContext {
            programs: &mut self.programs,
            gl: &self.gl,
            texture_bindings,
            scratch_buffers: &mut self.core.scratch_buffers,
            use_depth_buffer,
        };
        if let TargetInit::Clear(clear) = target_init {
            ctx.clear_pass_inner(DrawPassTarget::Root(root_output_target), clear);
        }
        crate::schedule::execute(
            &mut ctx,
            &mut self.core.schedule_storage,
            schedule,
            root_output_target,
        )?;

        self.gl.bind_framebuffer(
            WebGl2RenderingContext::FRAMEBUFFER,
            self.programs.resources.root_framebuffer(),
        );
        if self.programs.resources.depth_cleared_this_frame {
            // See: https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices#use_invalidateframebuffer
            // We want to indicate to the GPU driver that we won't read the depth buffer again
            // until the next clear. This enables the GPU to avoid storing depth tiles back to VRAM.
            self.gl
                .invalidate_framebuffer(
                    WebGl2RenderingContext::FRAMEBUFFER,
                    if self.programs.resources.root_target.is_some() {
                        &self.programs.resources.offscreen_depth_attachment_array
                    } else {
                        &self.programs.resources.default_depth_attachment_array
                    },
                )
                .map_js_error(WebGlOperation::DataTransfer(
                    WebGlDataTransferOperation::FramebufferInvalidation,
                ))?;
        }

        self.core.gradient_cache.maintain();

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
    ///
    /// # Errors
    /// Returns [`WebGlError::FullImageSupportDisabled`] if the `full_image_support` feature
    /// is disabled, or an error if allocating or uploading the image fails.
    #[track_caller]
    pub fn upload_image<T: WebGlAtlasWriter>(
        &mut self,
        resources: &mut Resources,
        writer: &T,
    ) -> Result<ImageId, WebGlError> {
        self.upload_image_with(&mut resources.image_cache, writer, IMAGE_PADDING)
    }

    #[track_caller]
    pub(crate) fn upload_image_with<T: WebGlAtlasWriter>(
        &mut self,
        image_cache: &mut ImageCache,
        writer: &T,
        padding: u16,
    ) -> Result<ImageId, WebGlError> {
        if !cfg!(feature = "full_image_support") {
            return Err(WebGlError::FullImageSupportDisabled);
        }

        let width = writer.width();
        let height = writer.height();
        let image_id = image_cache
            .allocate(width, height, padding)
            .map_err(RenderError::AtlasError)?;

        if let Err(err) = self.write_to_atlas(image_cache, image_id, writer, None) {
            image_cache.deallocate(image_id);

            return Err(err);
        }

        Ok(image_id)
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
        offset_override: Option<[u16; 2]>,
    ) -> Result<(), WebGlError> {
        let image_resource = image_cache
            .get(image_id)
            .ok_or(RenderError::MissingImage(image_id))?;

        self.programs
            .maybe_create_atlas_textures(&self.gl, image_cache.atlas_count() as u32)?;
        let offset = offset_override.unwrap_or(image_resource.offset);
        writer.write_to_atlas(
            &self.gl,
            &self.programs.resources.atlas_textures[image_resource.atlas_id.as_u32() as usize],
            offset,
            writer.width(),
            writer.height(),
        )?;

        Ok(())
    }

    /// Destroy an image from the cache and clear the allocated slot in the atlas.
    pub fn destroy_image(
        &mut self,
        resources: &mut Resources,
        image_id: ImageId,
    ) -> Result<(), WebGlError> {
        if let Some(image_resource) = resources.image_cache.get(image_id) {
            let padding = image_resource.padding;
            self.clear_atlas_region(
                image_resource.atlas_id,
                [
                    image_resource.offset[0] - padding,
                    image_resource.offset[1] - padding,
                ],
                image_resource.width + padding * 2,
                image_resource.height + padding * 2,
            )?;

            let _ = resources.image_cache.deallocate(image_id);
        }

        Ok(())
    }

    /// Returns an individual image atlas texture.
    pub fn atlas_texture(&self, atlas_id: AtlasId) -> &WebGlTexture {
        self.programs
            .resources
            .atlas_textures
            .get(atlas_id.as_u32() as usize)
            .map(|texture| &**texture)
            .unwrap()
    }

    /// Returns the current allocation state of the image/glyph atlas texture.
    pub fn atlas_info(&self) -> AtlasTextureInfo {
        let resources = &self.programs.resources;
        AtlasTextureInfo {
            width: resources.atlas_size.0,
            height: resources.atlas_size.1,
            texture_count: u32::try_from(resources.atlas_textures.len()).unwrap(),
        }
    }

    /// Clear a specific region of an atlas texture.
    fn clear_atlas_region(
        &mut self,
        atlas_id: AtlasId,
        offset: [u16; 2],
        width: u16,
        height: u16,
    ) -> Result<(), WebGlError> {
        self.gl.bind_framebuffer(
            WebGl2RenderingContext::FRAMEBUFFER,
            Some(&self.programs.resources.atlas_targets[atlas_id.as_u32() as usize].framebuffer),
        );

        // Set viewport to match the atlas texture dimensions
        let (atlas_width, atlas_height) = self.programs.resources.atlas_size;
        self.gl
            .viewport(0, 0, i32::from(atlas_width), i32::from(atlas_height));

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

        Ok(())
    }
}

#[cfg(feature = "text")]
fn clear_atlas_region(
    renderer: &mut WebGlRenderer,
    rect: &PendingClearRect,
) -> Result<(), WebGlError> {
    // TODO: Similarly to wgpu, maybe this can be done in a more effective
    // way?
    renderer.clear_atlas_region(
        AtlasId::new(rect.page_index),
        [rect.x, rect.y],
        rect.width,
        rect.height,
    )?;

    Ok(())
}
