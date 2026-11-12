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

#[cfg(feature = "text")]
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

pub use bindings::WebGlTextureBindings;
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

#[cfg(feature = "text")]
use crate::RenderError;
use crate::render::common::ScenePreparer;
use crate::{
    ClearSettings, RenderSize, Resources, TargetInit,
    scene::Scene,
    schedule::IntermediateTextureAllocations,
    target::{DrawPassTarget, RootTarget},
};
use alloc::vec::Vec;
#[cfg(feature = "text")]
use atlas::write_pixmap_to_atlas;
#[cfg(feature = "text")]
use glifo::PendingClearRect;
use passes::{WebGlPrograms, WebGlRendererContext};
use resources::{WebGlResources, create_webgl_resources};
use vello_common::color::{AlphaColor, Srgb};
use vello_common::geometry::{RectU16, SizeU16};
use vello_common::image_cache::ImageCache;
use vello_common::multi_atlas::{AtlasConfig, AtlasId};
#[cfg(feature = "text")]
use vello_common::{paint::ImageId, pixmap::Pixmap};
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
    /// GPU resources for rendering.
    pub(super) resources: WebGlResources,
    /// WebGL context.
    pub(crate) gl: WebGl2RenderingContext,
    /// Encodes scenes and schedules their rendering.
    scene_preparer: ScenePreparer,
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

    /// Reset the renderer and `resources` and free the memory they grew for earlier scenes.
    ///
    /// Unlike [`Scene::reset`](crate::Scene::reset), this is not meant to be called every
    /// frame. Scenes recorded before this call, including ones not yet rendered, must be recorded
    /// again; rendering them afterwards may draw the wrong glyphs.
    pub fn reset_and_shrink(&mut self, resources: &mut Resources) -> Result<(), WebGlError> {
        let webgl_resources = create_webgl_resources(
            &self.gl,
            &resources.image_cache,
            self.scene_preparer.layers_config,
            self.resources.resource_texture_dimension_2d,
            self.resources.drawing_buffer_uses_depth_buffer,
        )?;
        self.resources = webgl_resources;
        self.scene_preparer.reset_and_shrink();
        resources.clear();

        Ok(())
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
    /// [`RenderError::MissingTextureBinding`]: crate::RenderError::MissingTextureBinding
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
                    [dst_x, dst_y],
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
    #[cfg_attr(
        not(feature = "text"),
        expect(dead_code, reason = "only glyph caching renders into the atlas")
    )]
    pub(crate) fn render_to_atlas(
        &mut self,
        scene: &Scene,
        atlas_count: u32,
        atlas_config: AtlasConfig,
        atlas_id: AtlasId,
        texture_bindings: &WebGlTextureBindings,
    ) -> Result<(), WebGlError> {
        self.resources
            .maybe_create_atlas_textures(&self.gl, atlas_count)?;

        let (atlas_width, atlas_height) = atlas_config.atlas_size;
        let atlas_render_size = RenderSize {
            width: atlas_width,
            height: atlas_height,
        };
        let target = self.resources.atlas_targets[atlas_id.as_u32() as usize].clone();
        // TODO: Explore using an option instead of a dummy image cache.
        let dummy_image_cache = self
            .scene_preparer
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
        self.scene_preparer.dummy_image_cache = Some(dummy_image_cache);

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
        let previous_target = self.resources.root_target.replace(target.clone());
        let result = render(self);
        self.resources.root_target = previous_target;
        result
    }

    fn current_allocations(&self) -> IntermediateTextureAllocations {
        IntermediateTextureAllocations {
            layer_pages: self.resources.layer_textures.each_ref().map(Vec::len),
            scratch: self.resources.scratch_texture.is_some(),
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
        let use_depth_buffer = self.resources.root_uses_depth_buffer();
        self.scene_preparer.gradient_cache.maintain();

        let current_allocations = self.current_allocations();
        let schedule = self.scene_preparer.prepare(
            scene,
            image_cache,
            root_output_target,
            RectU16::visible_scene_bounds(
                scene.recorder.scene_size,
                SizeU16::from_wh(render_size.width, render_size.height),
                scene_offset,
            ),
            use_depth_buffer,
            self.resources.texture_size,
            current_allocations,
            |id, image, region| {
                texture_bindings.encode_paint(id, image, region, render_target_texture)
            },
        )?;
        self.resources
            .prepare_intermediate_textures(&self.gl, &schedule)?;

        // TODO: For the time being, we upload the entire alpha buffer as one big chunk. As a future
        // refinement, we could have a bounded alpha buffer, and break draws when the alpha
        // buffer fills.
        self.resources.prepare(
            &self.gl,
            &mut self.scene_preparer.gradient_cache,
            &self.scene_preparer.encoded_paints,
            &mut scene.strip_storage.borrow_mut().alphas,
            render_size,
            scene_offset,
            &self.scene_preparer.paint_idxs,
            &self.scene_preparer.schedule_storage.filter_context,
            root_output_target,
        )?;
        self.resources.depth_cleared_this_frame = false;
        let mut ctx = WebGlRendererContext {
            programs: &self.programs,
            resources: &mut self.resources,
            gl: &self.gl,
            texture_bindings,
            scratch_buffers: &mut self.scene_preparer.scratch_buffers,
            use_depth_buffer,
        };
        if let TargetInit::Clear(clear) = target_init {
            ctx.clear_pass_inner(DrawPassTarget::Root(root_output_target), clear);
        }
        crate::schedule::execute(
            &mut ctx,
            &mut self.scene_preparer.schedule_storage,
            schedule,
            root_output_target,
        )?;

        self.gl.bind_framebuffer(
            WebGl2RenderingContext::FRAMEBUFFER,
            self.resources.root_framebuffer(),
        );
        if self.resources.depth_cleared_this_frame {
            // See: https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices#use_invalidateframebuffer
            // We want to indicate to the GPU driver that we won't read the depth buffer again
            // until the next clear. This enables the GPU to avoid storing depth tiles back to VRAM.
            self.gl
                .invalidate_framebuffer(
                    WebGl2RenderingContext::FRAMEBUFFER,
                    if self.resources.root_target.is_some() {
                        &self.resources.offscreen_depth_attachment_array
                    } else {
                        &self.resources.default_depth_attachment_array
                    },
                )
                .map_js_error(WebGlOperation::DataTransfer(
                    WebGlDataTransferOperation::FramebufferInvalidation,
                ))?;
        }

        Ok(())
    }

    /// Get a reference to the underlying WebGL context.
    ///
    /// This allows direct access to WebGL operations for advanced use cases like texture creation.
    pub fn gl_context(&self) -> &WebGl2RenderingContext {
        &self.gl
    }

    #[cfg(feature = "text")]
    fn write_to_atlas(
        &mut self,
        image_cache: &ImageCache,
        image_id: ImageId,
        pixmap: &Pixmap,
        offset: [u16; 2],
    ) -> Result<(), WebGlError> {
        let image_resource = image_cache
            .get(image_id)
            .ok_or(RenderError::MissingAtlasEntry(image_id))?;

        self.resources
            .maybe_create_atlas_textures(&self.gl, image_cache.atlas_count() as u32)?;
        write_pixmap_to_atlas(
            &self.gl,
            pixmap,
            &self.resources.atlas_textures[image_resource.atlas_id.as_u32() as usize],
            offset,
        )
    }

    #[cfg(feature = "text")]
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
            Some(&self.resources.atlas_targets[atlas_id.as_u32() as usize].framebuffer),
        );

        // Set viewport to match the atlas texture dimensions
        let (atlas_width, atlas_height) = self.resources.atlas_size;
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
