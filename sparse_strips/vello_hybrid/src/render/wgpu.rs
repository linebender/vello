// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU rendering module for the sparse strips CPU/GPU rendering engine.
//!
//! This module provides the GPU-side implementation of the hybrid rendering system.
//! It handles:
//! - GPU resource management (buffers, textures, pipelines)
//! - Surface/window management and presentation
//! - Shader execution and rendering
//!
//! The hybrid approach combines CPU-side path processing with efficient GPU rendering
//! to balance flexibility and performance.

#![expect(
    clippy::cast_possible_truncation,
    reason = "We temporarily ignore those because the casts\
only break in edge cases, and some of them are also only related to conversions from f64 to f32."
)]

use crate::draw::ExternalTextureRun;
use crate::render::common::IMAGE_PADDING;
use crate::schedule::execute::TextureRequirements;
use crate::util::{Int16Size, RangedSlice};
use crate::{
    GpuStrip, RenderError, RenderSettings, RenderSize, Resources,
    blend::{BLEND_SCRATCH_INDEX, GpuBlendInstance, gpu_blend_instance},
    copy::GpuCopyInstance,
    filter::{FilterContext, FilterInstanceData, FilterPassPlan},
    gradient_cache::GradientRampCache,
    paint::PaintResolver,
    render::{
        Config,
        common::{
            GPU_BLURRED_ROUNDED_RECT_SIZE_TEXELS, GPU_ENCODED_IMAGE_SIZE_TEXELS,
            GPU_LINEAR_GRADIENT_SIZE_TEXELS, GPU_RADIAL_GRADIENT_SIZE_TEXELS,
            GPU_SWEEP_GRADIENT_SIZE_TEXELS, GpuBlurredRoundedRect, GpuClearInstance,
            GpuEncodedImage, GpuEncodedPaint, GpuLinearGradient, GpuRadialGradient,
            GpuSweepGradient, ScratchBuffers, normalize_atlas_config, pack_image_offset,
            pack_image_params, pack_image_size, pack_radial_kind_and_swapped,
            pack_texture_width_and_extend_mode, pack_tint,
        },
    },
    scene::Scene,
    schedule::{RendererBackend, Schedule, ScheduleStorage, round::BlendOp},
    target::{
        DrawPassTarget, IntermediateTextureSizes, RootRenderTarget, TextureIndex, TextureTarget,
    },
};
use alloc::vec::Vec;
use alloc::{sync::Arc, vec};
use core::{fmt::Debug, num::NonZeroU64};
use hashbrown::{HashMap, hash_map::Entry};
use vello_common::image_cache::{ImageCache, ImageResource};
use vello_common::multi_atlas::{AtlasConfig, AtlasId};
use vello_common::{
    TextureId,
    encode::{
        EncodedBlurredRoundedRectangle, EncodedExternalTexture, EncodedGradient, EncodedKind,
        EncodedPaint, MAX_GRADIENT_LUT_SIZE, RadialKind,
    },
    geometry::RectU16,
    paint::ImageSource,
    peniko::{self},
    pixmap::Pixmap,
    tile::Tile,
};
use wgpu::{
    BindGroup, BindGroupLayout, BlendState, Buffer, ColorTargetState, ColorWrites, CommandEncoder,
    Device, Extent3d, PipelineCompilationOptions, Queue, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, Sampler, Texture, TextureView as WgpuTextureView,
    TextureViewDescriptor, util::DeviceExt,
};

/// Placeholder value for uninitialized GPU encoded paints.
const GPU_PAINT_PLACEHOLDER: GpuEncodedPaint = GpuEncodedPaint::LinearGradient(GpuLinearGradient {
    texture_width_and_extend_mode: 0,
    gradient_start: 0,
    transform: [0.0; 6],
});

const EXTERNAL_IMAGE_SOURCE_FLAG: u32 = 1 << 14;

/// Options for the renderer
#[derive(Debug)]
pub struct RenderTargetConfig {
    /// Format of the rendering target
    pub format: wgpu::TextureFormat,
    /// Width of the rendering target
    pub width: u32,
    /// Height of the rendering target
    pub height: u32,
}

/// Runtime bindings for [externally owned textures](`TextureId`) sampled by texture-rect draws.
#[derive(Debug, Default, Clone)]
pub struct TextureBindings {
    views: HashMap<TextureId, WgpuTextureView>,
}

impl TextureBindings {
    /// Create an empty binding map.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a texture binding.
    ///
    /// The [`WgpuTextureView`] must fit the following binding type.
    ///
    /// ```ignore
    /// wgpu::BindGroupLayoutEntry {
    ///     binding: 1,
    ///     visibility: wgpu::ShaderStages::FRAGMENT,
    ///     ty: wgpu::BindingType::Texture {
    ///         sample_type: wgpu::TextureSampleType::Float { filterable: false },
    ///         view_dimension: wgpu::TextureViewDimension::D2,
    ///         multisampled: false,
    ///     },
    ///     count: None,
    /// }
    /// ```
    ///
    /// This means the view must be a non-array 2D view of a float-sampleable texture (e.g. integer
    /// formats are rejected by wgpu at bind time), the underlying texture must include
    /// [`wgpu::TextureUsages::TEXTURE_BINDING`], and only mip level 0 is read.
    #[inline]
    pub fn insert(&mut self, texture_id: TextureId, view: WgpuTextureView) {
        self.views.insert(texture_id, view);
    }

    /// Get a texture binding.
    #[inline]
    fn get(&self, texture_id: TextureId) -> Option<&WgpuTextureView> {
        self.views.get(&texture_id)
    }

    /// Remove a texture binding.
    ///
    /// This returns the removed [`WgpuTextureView`] binding if it existed.
    #[inline]
    pub fn remove(&mut self, texture_id: TextureId) -> Option<WgpuTextureView> {
        self.views.remove(&texture_id)
    }
}

/// Vello Hybrid's Renderer.
#[derive(Debug)]
pub struct Renderer {
    /// Programs for rendering.
    programs: Programs,
    /// Encoded paints for storing encoded paints.
    encoded_paints: Vec<GpuEncodedPaint>,
    /// Stores the index (offset) of the encoded paints in the encoded paints texture.
    paint_idxs: Vec<u32>,
    /// Gradient cache for storing gradient ramps.
    gradient_cache: GradientRampCache,
    dummy_image_cache: Option<ImageCache>,
    schedule_storage: ScheduleStorage,
    scratch: ScratchBuffers,
}

impl Renderer {
    /// Creates a new renderer.
    pub fn new(device: &Device, render_target_config: &RenderTargetConfig) -> Self {
        Self::new_with(device, render_target_config, RenderSettings::default())
    }

    /// Creates a new renderer with specific settings.
    pub fn new_with(
        device: &Device,
        render_target_config: &RenderTargetConfig,
        settings: RenderSettings,
    ) -> Self {
        super::common::maybe_warn_about_webgl_feature_conflict();

        let mut settings = settings;
        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        // When targeting wasm32 with a WebGL/GLES backend, we need to set
        // `initial_atlas_count` to 2. In WGPU's GLES backend, heuristics are used to decide
        // whether a texture should be treated as D2 or D2Array. However, this can cause a
        // mismatch: when depth_or_array_layers == 1, the backend assumes the texture is D2,
        // even if it was actually created as a D2Array. This issue only occurs with the GLES
        // backend.
        //
        // @see https://github.com/gfx-rs/wgpu/blob/61e5124eb9530d3b3865556a7da4fd320d03ddc5/wgpu-hal/src/gles/mod.rs#L470-L517
        // TODO: Can we somehow dynamically detect whether the WebGL backend was chosen, so that the
        // wgpu backend isn't affected by this?
        #[cfg(target_arch = "wasm32")]
        let min_initial_atlas_count = 2;
        #[cfg(not(target_arch = "wasm32"))]
        let min_initial_atlas_count = 1;
        normalize_atlas_config(
            &mut settings.atlas_config,
            max_texture_dimension_2d,
            device.limits().max_texture_array_layers,
            min_initial_atlas_count,
        );
        let image_cache = ImageCache::new_with_config(settings.atlas_config);
        // Estimate the maximum number of gradient cache entries based on the max texture dimension
        // and the maximum gradient LUT size - worst case scenario.
        let max_gradient_cache_size =
            max_texture_dimension_2d * max_texture_dimension_2d / MAX_GRADIENT_LUT_SIZE as u32;
        let gradient_cache = GradientRampCache::new(max_gradient_cache_size, settings.level);

        Self {
            programs: Programs::new(device, &image_cache, render_target_config),
            gradient_cache,
            encoded_paints: Vec::new(),
            paint_idxs: Vec::new(),
            dummy_image_cache: Some(ImageCache::new_dummy()),
            schedule_storage: ScheduleStorage::default(),
            scratch: ScratchBuffers::default(),
        }
    }

    /// Render `scene`.
    ///
    /// Every [`TextureId`] referenced by the scene must have a binding; this returns
    /// [`RenderError::MissingTextureBinding`] otherwise. See [`TextureBindings::insert`] for the
    /// requirements on the bound texture views.
    ///
    /// To render without any texture bindings, you can pass an empty [`TextureBindings`].
    pub fn render(
        &mut self,
        scene: &Scene,
        resources: &mut Resources,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        render_size: &RenderSize,
        view: &WgpuTextureView,
        texture_bindings: &TextureBindings,
    ) -> Result<(), RenderError> {
        #[cfg(feature = "text")]
        {
            resources.before_render(
                self,
                |renderer, glyph_renderer, atlas_count, atlas_config, atlas_id| {
                    renderer
                        .render_to_atlas(
                            glyph_renderer,
                            atlas_count,
                            atlas_config,
                            device,
                            queue,
                            atlas_id,
                            texture_bindings,
                        )
                        .expect("Failed to render glyphs to atlas");
                },
                |renderer, image_cache, upload, dst_x, dst_y| {
                    renderer.write_to_atlas(
                        image_cache,
                        device,
                        queue,
                        encoder,
                        upload.image_id,
                        &upload.pixmap,
                        Some([dst_x, dst_y]),
                    );
                },
            );
        }

        let mut encoded_paints = scene.encoded_paints.borrow_mut();
        let scene_paint_count = encoded_paints.len();

        let result = self.render_scene(
            scene,
            device,
            queue,
            encoder,
            render_size,
            view,
            &resources.image_cache,
            &encoded_paints,
            true,
            RootRenderTarget::UserSurface,
            texture_bindings,
        );

        encoded_paints.truncate(scene_paint_count);
        #[cfg(feature = "text")]
        resources.after_render(self, |renderer, rect| {
            renderer.clear_atlas_region(
                encoder,
                AtlasId::new(rect.page_index),
                [rect.x as u32, rect.y as u32],
                rect.width as u32,
                rect.height as u32,
            );
        });
        result
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
    ///
    /// `texture_bindings` provides [externally bound textures](`TextureBindings`)
    /// referenced by the scene. Pass `&TextureBindings::new()` if the scene does
    /// not use any.
    #[doc(hidden)]
    pub fn render_to_atlas(
        &mut self,
        scene: &Scene,
        atlas_count: u32,
        atlas_config: AtlasConfig,
        device: &Device,
        queue: &Queue,
        atlas_id: AtlasId,
        texture_bindings: &TextureBindings,
    ) -> Result<(), RenderError> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render to Atlas Encoder"),
        });

        Programs::maybe_resize_atlas_texture_array(
            device,
            &mut encoder,
            &mut self.programs.resources,
            &self.programs.atlas_bind_group_layout,
            atlas_count,
        );

        let (atlas_width, atlas_height) = atlas_config.atlas_size;
        let atlas_render_size = RenderSize {
            width: atlas_width,
            height: atlas_height,
        };

        let layer_view =
            self.programs
                .resources
                .atlas_texture_array
                .create_view(&TextureViewDescriptor {
                    label: Some("Atlas Layer Render View"),
                    format: Some(wgpu::TextureFormat::Rgba8Unorm),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: atlas_id.as_u32(),
                    array_layer_count: Some(1),
                    usage: None,
                });

        // Swap in the stub atlas bind group to avoid the read-write conflict:
        // the real atlas texture is used as the render target (COLOR_TARGET), so it
        // cannot also be bound as a shader resource (TEXTURE_BINDING) in the same pass.
        core::mem::swap(
            &mut self.programs.resources.atlas_bind_group,
            &mut self.programs.resources.stub_atlas_bind_group,
        );

        let encoded_paints = scene.encoded_paints.borrow();
        let dummy_image_cache = self
            .dummy_image_cache
            .take()
            .expect("dummy image cache must exist");
        let result = self.render_scene(
            scene,
            device,
            queue,
            &mut encoder,
            &atlas_render_size,
            &layer_view,
            &dummy_image_cache,
            &encoded_paints,
            false,
            RootRenderTarget::AtlasLayer,
            texture_bindings,
        );
        self.dummy_image_cache = Some(dummy_image_cache);

        // Restore the real atlas bind group.
        core::mem::swap(
            &mut self.programs.resources.atlas_bind_group,
            &mut self.programs.resources.stub_atlas_bind_group,
        );

        // Submit immediately so the atlas content is committed before subsequent
        // render() calls overwrite the shared alpha/config/paint resources.
        queue.submit(Some(encoder.finish()));

        result
    }

    /// Shared render pipeline: prepares GPU resources, runs the scheduler against
    /// the provided `view` at `render_size`, and maintains caches.
    ///
    /// When `clear` is true the render target is cleared to transparent black
    /// before drawing (normal frame rendering).
    fn render_scene(
        &mut self,
        scene: &Scene,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        render_size: &RenderSize,
        view: &WgpuTextureView,
        image_cache: &ImageCache,
        encoded_paints: &[EncodedPaint],
        clear: bool,
        root_output_target: RootRenderTarget,
        texture_bindings: &TextureBindings,
    ) -> Result<(), RenderError> {
        self.programs.depth_cleared_this_frame = false;
        self.prepare_gpu_encoded_paints(encoded_paints, image_cache, texture_bindings)?;
        let paint_resolver = PaintResolver::new(encoded_paints, &self.paint_idxs);
        let schedule = Schedule::try_new(
            &mut self.schedule_storage,
            scene,
            root_output_target,
            paint_resolver,
            self.programs.resources.texture_sizes,
        )?;
        // TODO: For the time being, we upload the entire alpha buffer as one big chunk. As a future
        // refinement, we could have a bounded alpha buffer, and break draws when the alpha
        // buffer fills.
        self.programs.prepare(
            device,
            queue,
            &mut self.gradient_cache,
            &self.encoded_paints,
            &mut scene.strip_storage.borrow_mut().alphas,
            render_size,
            &self.paint_idxs,
            &self.schedule_storage.filter_context,
        );
        self.programs
            .prepare_intermediate_textures(device, TextureRequirements::new(scene));

        if clear {
            Self::clear_view(encoder, view);
        }
        let mut ctx = RendererContext {
            programs: &mut self.programs,
            device,
            queue,
            encoder,
            view,
            texture_bindings,
            external_paint_source_bind_groups: HashMap::new(),
            scratch: &mut self.scratch,
        };
        crate::schedule::execute(
            &mut ctx,
            &mut self.schedule_storage,
            schedule,
            root_output_target,
        );
        self.gradient_cache.maintain();

        Ok(())
    }

    /// Clear the view to transparent black.
    // TODO: Investigate adding tests for the clear_view behavior.
    fn clear_view(encoder: &mut CommandEncoder, view: &WgpuTextureView) {
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Clear View"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
    }

    /// Upload image to cache and atlas in one step. Returns the `ImageId`.
    ///
    /// It's used when an image is not already in the cache.
    ///
    /// This is a convenience method that:
    /// 1. Reserves space in the image cache
    /// 2. Writes the image data directly to the atlas
    /// 3. Returns the `ImageId` for use in rendering
    pub fn upload_image<T: AtlasWriter>(
        &mut self,
        resources: &mut Resources,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        writer: &T,
    ) -> vello_common::paint::ImageId {
        self.upload_image_with(
            &mut resources.image_cache,
            device,
            queue,
            encoder,
            writer,
            IMAGE_PADDING,
        )
    }

    pub(crate) fn upload_image_with<T: AtlasWriter>(
        &mut self,
        image_cache: &mut ImageCache,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        writer: &T,
        padding: u16,
    ) -> vello_common::paint::ImageId {
        let width = writer.width();
        let height = writer.height();
        let image_id = image_cache.allocate(width, height, padding).unwrap();
        self.write_to_atlas(image_cache, device, queue, encoder, image_id, writer, None);
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
    pub(crate) fn write_to_atlas<T: AtlasWriter>(
        &mut self,
        image_cache: &ImageCache,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        image_id: vello_common::paint::ImageId,
        writer: &T,
        offset_override: Option<[u32; 2]>,
    ) {
        let image_resource = image_cache.get(image_id).expect("Image resource not found");

        Programs::maybe_resize_atlas_texture_array(
            device,
            encoder,
            &mut self.programs.resources,
            &self.programs.atlas_bind_group_layout,
            image_cache.atlas_count() as u32,
        );
        let offset = offset_override.unwrap_or([
            image_resource.offset[0] as u32,
            image_resource.offset[1] as u32,
        ]);
        writer.write_to_atlas_layer(
            device,
            queue,
            encoder,
            &self.programs.resources.atlas_texture_array,
            image_resource.atlas_id.as_u32(),
            offset,
            writer.width(),
            writer.height(),
        );
    }

    /// Destroy an image from the cache and clear the allocated slot in the atlas.
    pub fn destroy_image(
        &mut self,
        resources: &mut Resources,
        encoder: &mut CommandEncoder,
        image_id: vello_common::paint::ImageId,
    ) {
        if let Some(image_resource) = resources.image_cache.deallocate(image_id) {
            let padding = image_resource.padding as u32;

            self.clear_atlas_region(
                encoder,
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
    pub fn atlas_texture(&self) -> &Texture {
        &self.programs.resources.atlas_texture_array
    }

    /// Clear a specific region of the atlas texture.
    fn clear_atlas_region(
        &mut self,
        encoder: &mut CommandEncoder,
        atlas_id: AtlasId,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        // Create a texture view for the specific atlas layer
        let layer_view =
            self.programs
                .resources
                .atlas_texture_array
                .create_view(&TextureViewDescriptor {
                    label: Some("Atlas Layer Clear View"),
                    format: Some(wgpu::TextureFormat::Rgba8Unorm),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: atlas_id.as_u32(),
                    array_layer_count: Some(1),
                    // Inherit usage from the texture
                    usage: None,
                });

        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Clear Atlas Region"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &layer_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Don't clear entire texture, just the scissor region
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        // Set scissor rectangle to limit clearing to specific region
        render_pass.set_scissor_rect(offset[0], offset[1], width, height);
        // Use atlas clear pipeline to render transparent pixels
        render_pass.set_pipeline(&self.programs.atlas_clear_pipeline);
        // Draw fullscreen quad
        render_pass.draw(0..4, 0..1);
    }

    fn prepare_gpu_encoded_paints(
        &mut self,
        encoded_paints: &[EncodedPaint],
        image_cache: &ImageCache,
        texture_bindings: &TextureBindings,
    ) -> Result<(), RenderError> {
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
                            let image_paint = self.encode_image_paint(img, image_resource);
                            self.encoded_paints[encoded_paint_idx] = image_paint;
                            current_idx += GPU_ENCODED_IMAGE_SIZE_TEXELS;
                        }
                    }
                }
                EncodedPaint::ExternalTexture(img) => {
                    if texture_bindings.get(img.texture_id).is_none() {
                        return Err(RenderError::MissingTextureBinding(img.texture_id));
                    }
                    let image_paint = self.encode_external_texture_paint(img);
                    self.encoded_paints[encoded_paint_idx] = image_paint;
                    current_idx += GPU_ENCODED_IMAGE_SIZE_TEXELS;
                }
                EncodedPaint::Gradient(gradient) => {
                    let (gradient_start, gradient_width) =
                        self.gradient_cache.get_or_create_ramp(gradient);
                    let gradient_paint: GpuEncodedPaint =
                        self.encode_gradient_paint(gradient, gradient_width, gradient_start);
                    let gradient_size_texels = match &gradient_paint {
                        GpuEncodedPaint::LinearGradient(_) => GPU_LINEAR_GRADIENT_SIZE_TEXELS,
                        GpuEncodedPaint::RadialGradient(_) => GPU_RADIAL_GRADIENT_SIZE_TEXELS,
                        GpuEncodedPaint::SweepGradient(_) => GPU_SWEEP_GRADIENT_SIZE_TEXELS,
                        _ => unreachable!("encode_gradient_for_gpu only returns gradient types"),
                    };
                    self.encoded_paints[encoded_paint_idx] = gradient_paint;
                    current_idx += gradient_size_texels;
                }
                EncodedPaint::BlurredRoundedRect(blurred_rect) => {
                    self.encoded_paints[encoded_paint_idx] =
                        Self::encode_blurred_rounded_rect_paint(blurred_rect);
                    current_idx += GPU_BLURRED_ROUNDED_RECT_SIZE_TEXELS;
                }
            }
        }
        self.paint_idxs[encoded_paints.len()] = current_idx;
        Ok(())
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

    fn encode_external_texture_paint(&self, image: &EncodedExternalTexture) -> GpuEncodedPaint {
        let transform = image.transform.as_coeffs().map(|x| x as f32);
        let region = image.source_region;
        let image_size = pack_image_size(region.width(), region.height());
        let image_offset = pack_image_offset(region.x0, region.y0);
        let image_params = pack_image_params(
            image.sampler.quality as u32,
            image.sampler.x_extend as u32,
            image.sampler.y_extend as u32,
            0,
        ) | EXTERNAL_IMAGE_SOURCE_FLAG;
        let (tint, tint_mode) = pack_tint(image.tint);

        GpuEncodedPaint::Image(GpuEncodedImage {
            image_params,
            image_size,
            image_offset,
            transform,
            tint,
            tint_mode,
            image_padding: 0,
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

/// Defines the GPU resources and pipelines for rendering.
#[derive(Debug)]
struct Programs {
    /// Pipelines for rendering strips to intermediate targets (depth test OFF, depth write OFF,
    /// blending ON).
    /// The first pipeline should be used for color attachments in the native pixel format,
    /// the second for color attachments in RGBA8.
    intermediate_strip_pipelines: [RenderPipeline; 2],
    /// Alpha pipelines for rendering strips to Output targets (depth test ON, depth write OFF, blending ON).
    alpha_strip_pipelines: [RenderPipeline; 2],
    /// Opaque pipelines for rendering strips to Output targets (depth test ON, depth write ON, blending OFF).
    opaque_strip_pipelines: [RenderPipeline; 2],
    /// Depth texture for early-z rejection on the Output target.
    depth_texture: Texture,
    /// View for the depth texture.
    depth_texture_view: WgpuTextureView,
    /// Whether the depth buffer has been cleared this frame.
    depth_cleared_this_frame: bool,
    /// Bind group layout for strip draws
    strip_bind_group_layout: BindGroupLayout,
    /// Bind group layout for encoded paints
    encoded_paints_bind_group_layout: BindGroupLayout,
    /// Bind group layout for gradient texture
    gradient_bind_group_layout: BindGroupLayout,
    /// Bind group layout for atlas textures
    atlas_bind_group_layout: BindGroupLayout,
    /// Bind group layout for filter data texture.
    filter_bind_group_layout: BindGroupLayout,
    /// Bind group layouts for filter input and original textures.
    filter_input_bind_group_layouts: [BindGroupLayout; 2],
    /// Sampler used for filter input textures.
    filter_sampler: Sampler,
    /// Bind group layout for blend operations that sample layer textures.
    blend_layer_bind_group_layout: BindGroupLayout,
    /// Bind group layout for copying scratch results back to layer textures.
    blend_copy_bind_group_layout: BindGroupLayout,
    /// Pipeline for applying filter effects.
    filter_pipeline: RenderPipeline,
    /// Pipeline for clearing rectangular regions in intermediate textures.
    clear_pipeline: RenderPipeline,
    /// Pipeline for clearing atlas regions.
    atlas_clear_pipeline: RenderPipeline,
    /// Pipeline for resolving non-default blend layers into scratch.
    blend_pipeline: RenderPipeline,
    /// Pipeline for copying resolved blend scratch back into a layer texture.
    blend_copy_pipeline: RenderPipeline,
    /// GPU resources for rendering (created during prepare)
    resources: GpuResources,
    /// Dimensions of the rendering target
    render_size: RenderSize,
    /// Scratch buffer for staging encoded paints texture data.
    encoded_paints_data: Vec<u8>,
    /// Scratch buffer for staging filter data texture data.
    filter_data: Vec<u8>,
}

/// Contains all GPU resources needed for rendering
#[derive(Debug)]
struct GpuResources {
    /// Buffer for [`GpuStrip`] data
    strips_buffer: Buffer,
    /// Texture for alpha values.
    alphas_texture: Texture,
    /// Textures for atlas data (multiple atlases supported)
    atlas_texture_array: Texture,
    /// View for atlas texture array
    atlas_texture_array_view: WgpuTextureView,
    /// Bind group for paint sources: an atlas textures as texture array plus an external texture.
    atlas_bind_group: BindGroup,
    /// Transparent 1x1 placeholder texture in case no external texture is bound by the user.
    placeholder_external_texture_view: WgpuTextureView,
    /// Texture for encoded paints
    encoded_paints_texture: Texture,
    /// Bind group for encoded paints
    encoded_paints_bind_group: BindGroup,
    /// Texture for gradient lookup table
    gradient_texture: Texture,
    /// Bind group for gradient texture
    gradient_bind_group: BindGroup,
    /// Texture holding serialized `GpuFilterData` for all filter layers.
    filter_data_texture: Texture,
    /// Bind group for the filter data texture.
    filter_base_bind_group: BindGroup,
    /// Dimensions of the intermediate layer and scratch textures.
    texture_sizes: IntermediateTextureSizes,
    /// Config buffer for rendering wide tile commands into the view texture.
    view_config_buffer: Buffer,
    /// Config buffer for rendering strips into a layer texture.
    layer_config_buffers: [Buffer; 2],

    /// Bind groups for rendering to the root target while sampling layer atlas textures.
    root_layer_bind_groups: [BindGroup; 2],
    /// Bind groups for rendering into layer atlas textures.
    layer_bind_groups: [BindGroup; 2],
    /// Placeholder paint-source bind group with a 1x1 dummy atlas texture, used during
    /// `render_to_atlas` to avoid a read-write conflict on the real atlas texture.
    stub_atlas_bind_group: BindGroup,

    /// Layer atlas texture slots.
    layer_textures: [IntermediateTexture; 2],
    /// Dummy layer atlas texture used when creating fixed-shape bind groups.
    dummy_layer_texture: WgpuIntermediateTexture,
    /// Filter bind groups for sampling layer atlas textures.
    layer_filter_input_bind_groups: [BindGroup; 2],
    /// Filter bind group for sampling original layer atlas textures.
    filter_layer_textures_bind_group: BindGroup,
    /// Scratch texture slots used for filter ping-ponging and blend scratch.
    scratch_textures: [IntermediateTexture; 2],
    /// Dummy scratch texture used when creating fixed-shape bind groups.
    dummy_scratch_texture: WgpuIntermediateTexture,
    /// Filter bind groups for sampling scratch textures.
    scratch_input_bind_groups: [BindGroup; 2],

    /// Bind group for blend operations that sample layer atlas textures.
    blend_layer_bind_group: BindGroup,
    /// Bind group for copying blend scratch back into layer atlas textures.
    blend_copy_bind_group: BindGroup,
}

type IntermediateTexture = Option<WgpuIntermediateTexture>;

#[derive(Debug)]
struct WgpuIntermediateTexture {
    _texture: Texture,
    view: WgpuTextureView,
}

impl WgpuIntermediateTexture {
    fn new(texture: Texture) -> Self {
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
        }
    }
}

impl GpuResources {
    fn layer_binding_views(&self) -> [&WgpuTextureView; 2] {
        [
            self.layer_binding_view(TextureIndex::Even),
            self.layer_binding_view(TextureIndex::Odd),
        ]
    }

    fn layer_binding_view(&self, index: TextureIndex) -> &WgpuTextureView {
        self.layer_textures[index.get_index()]
            .as_ref()
            .map_or(&self.dummy_layer_texture.view, |texture| &texture.view)
    }

    fn scratch_binding_view(&self, index: TextureIndex) -> &WgpuTextureView {
        self.scratch_textures[index.get_index()]
            .as_ref()
            .map_or(&self.dummy_scratch_texture.view, |texture| &texture.view)
    }

    fn layer_view(&self, index: TextureIndex) -> &WgpuTextureView {
        &self.layer_textures[index.get_index()]
            .as_ref()
            .expect("vello_hybrid attempted to use a missing layer texture")
            .view
    }

    fn scratch_view(&self, index: TextureIndex) -> &WgpuTextureView {
        &self.scratch_textures[index.get_index()]
            .as_ref()
            .expect("vello_hybrid attempted to use a missing scratch texture")
            .view
    }

    fn texture_target_view(&self, target: TextureTarget) -> &WgpuTextureView {
        match target {
            TextureTarget::Layer(_) => self.layer_view(target.index()),
            TextureTarget::Scratch(_) => self.scratch_view(target.index()),
        }
    }

    fn has_intermediate_textures(&self, requirements: TextureRequirements) -> bool {
        self.layer_textures
            .iter()
            .zip(requirements.layer_textures)
            .all(|(texture, required)| texture.is_some() == required)
            && self
                .scratch_textures
                .iter()
                .zip(requirements.scratch_textures)
                .all(|(texture, required)| texture.is_some() == required)
    }
}

const SIZE_OF_CONFIG: NonZeroU64 = NonZeroU64::new(size_of::<Config>() as u64).unwrap();

impl GpuStrip {
    /// Vertex attributes for the strip
    pub fn vertex_attributes() -> [wgpu::VertexAttribute; 6] {
        wgpu::vertex_attr_array![
            0 => Uint32,
            1 => Uint32,
            2 => Uint32,
            3 => Uint32,
            4 => Uint32,
            5 => Uint32,
        ]
    }
}

impl Programs {
    fn new(
        device: &Device,
        image_cache: &ImageCache,
        render_target_config: &RenderTargetConfig,
    ) -> Self {
        let strip_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Strip Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let atlas_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Paint Source Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let encoded_paints_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Encoded Paints Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let gradient_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gradient Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let strip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Strip Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_sparse_shaders::wgsl::RENDER.into()),
        });

        let clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Clear Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_sparse_shaders::wgsl::CLEAR.into()),
        });

        let strip_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Strip Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&strip_bind_group_layout),
                    Some(&atlas_bind_group_layout),
                    Some(&encoded_paints_bind_group_layout),
                    Some(&gradient_bind_group_layout),
                ],
                immediate_size: 0,
            });

        let clear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Clear Pipeline Layout"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });

        let depth_format = wgpu::TextureFormat::Depth24Plus;
        let strip_formats = [render_target_config.format, wgpu::TextureFormat::Rgba8Unorm];

        let strip_vertex_state = wgpu::VertexBufferLayout {
            array_stride: size_of::<GpuStrip>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &GpuStrip::vertex_attributes(),
        };

        let create_strip_pipelines =
            |label, blend, depth_stencil: Option<wgpu::DepthStencilState>| -> [RenderPipeline; 2] {
                core::array::from_fn(|i| {
                    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some(label),
                        layout: Some(&strip_pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &strip_shader,
                            entry_point: Some("vs_main"),
                            buffers: core::slice::from_ref(&strip_vertex_state),
                            compilation_options: PipelineCompilationOptions::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &strip_shader,
                            entry_point: Some("fs_main"),
                            targets: &[Some(ColorTargetState {
                                format: strip_formats[i],
                                blend,
                                write_mask: ColorWrites::ALL,
                            })],
                            compilation_options: PipelineCompilationOptions::default(),
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleStrip,
                            ..Default::default()
                        },
                        depth_stencil: depth_stencil.clone(),
                        multisample: wgpu::MultisampleState::default(),
                        multiview_mask: None,
                        cache: None,
                    })
                })
            };

        let depth_stencil = |depth_write_enabled| wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: Some(depth_write_enabled),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        // Intermediate pipelines: depth test OFF, depth write OFF, blending ON.
        let intermediate_strip_pipelines = create_strip_pipelines(
            "Strip Intermediate Pipeline",
            Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            None,
        );
        // Alpha pipelines: depth test ON (LessEqual), depth write OFF, blending ON.
        let alpha_strip_pipelines = create_strip_pipelines(
            "Strip Alpha Pipeline",
            Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            Some(depth_stencil(false)),
        );
        // Opaque pipelines: depth test ON (LessEqual), depth write ON, blending OFF.
        let opaque_strip_pipelines =
            create_strip_pipelines("Strip Opaque Pipeline", None, Some(depth_stencil(true)));

        let clear_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Clear Pipeline"),
            layout: Some(&clear_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &clear_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<GpuClearInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Uint32x2,
                        1 => Uint32x2,
                        2 => Uint32x2,
                    ],
                }],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &clear_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    // No blending needed for clearing
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Create atlas clear pipeline
        let atlas_clear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Atlas Clear Pipeline Layout"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
        let atlas_clear_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Atlas Clear Pipeline"),
            layout: Some(&atlas_clear_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &clear_shader,
                // Use a different vertex shader entry point
                entry_point: Some("vs_main_fullscreen"),
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &clear_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let filter_texture_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let filter_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Filter Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
        let filter_input_bind_group_layouts = [
            // Input texture and linear sampler.
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Filter Input Bind Group Layout"),
                entries: &[
                    filter_texture_entry,
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            }),
            // The original texture.
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Filter Layer Textures Bind Group Layout"),
                entries: &[
                    filter_texture_entry,
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            }),
        ];

        let filter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Filter Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_sparse_shaders::wgsl::FILTER.into()),
        });
        let filter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Filter Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&filter_bind_group_layout),
                    Some(&filter_input_bind_group_layouts[0]),
                    Some(&filter_input_bind_group_layouts[1]),
                ],
                immediate_size: 0,
            });
        let filter_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Filter Pipeline"),
            layout: Some(&filter_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &filter_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<FilterInstanceData>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Uint32x2,
                        1 => Uint32x2,
                        2 => Uint32x2,
                        3 => Uint32x2,
                        4 => Uint32x2,
                        5 => Uint32,
                        6 => Uint32x2,
                        7 => Uint32x2,
                        8 => Uint32,
                    ],
                }],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &filter_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

        let blend_layer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blend Layer Textures Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
        let blend_copy_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blend Copy Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
        let empty_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Empty Bind Group Layout"),
                entries: &[],
            });
        let blend_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blend Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_sparse_shaders::wgsl::BLEND.into()),
        });
        let blend_copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Copy Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_sparse_shaders::wgsl::COPY.into()),
        });
        let blend_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Blend Pipeline Layout"),
                bind_group_layouts: &[Some(&blend_layer_bind_group_layout)],
                immediate_size: 0,
            });
        let blend_copy_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Copy Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&empty_bind_group_layout),
                    Some(&blend_copy_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let blend_vertex_state = wgpu::VertexBufferLayout {
            array_stride: size_of::<GpuBlendInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![
                0 => Uint32,
                1 => Uint32,
                2 => Uint32,
                3 => Uint32,
                4 => Uint32,
                5 => Uint32,
                6 => Uint32,
                7 => Uint32,
            ],
        };
        let copy_vertex_state = wgpu::VertexBufferLayout {
            array_stride: size_of::<GpuCopyInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![
                0 => Uint32,
                1 => Uint32,
                2 => Uint32,
                3 => Uint32,
            ],
        };
        let create_blend_pipeline =
            |label,
             shader_module: &wgpu::ShaderModule,
             layout,
             vertex_state: &wgpu::VertexBufferLayout<'_>| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module: shader_module,
                        entry_point: Some("vs_main"),
                        buffers: core::slice::from_ref(vertex_state),
                        compilation_options: PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader_module,
                        entry_point: Some("fs_main"),
                        targets: &[Some(ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        })],
                        compilation_options: PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
            };
        let blend_pipeline = create_blend_pipeline(
            "Blend Pipeline",
            &blend_shader,
            &blend_pipeline_layout,
            &blend_vertex_state,
        );
        let blend_copy_pipeline = create_blend_pipeline(
            "Copy Pipeline",
            &blend_copy_shader,
            &blend_copy_pipeline_layout,
            &copy_vertex_state,
        );

        let intermediate_texture_size =
            u16::try_from(device.limits().max_texture_dimension_2d.min(4096)).unwrap();
        let texture_sizes = IntermediateTextureSizes::uniform(Int16Size::new(
            intermediate_texture_size,
            intermediate_texture_size,
        ));
        let filter_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Filter Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let layer_textures: [IntermediateTexture; 2] = core::array::from_fn(|_| None);
        let dummy_layer_texture = WgpuIntermediateTexture::new(Self::create_intermediate_texture(
            device,
            Int16Size::new(1, 1),
            "Layer Placeholder Texture",
        ));
        let layer_binding_views = [&dummy_layer_texture.view, &dummy_layer_texture.view];
        let layer_filter_input_bind_groups = layer_binding_views.map(|view| {
            create_filter_input_bind_group(
                device,
                &filter_input_bind_group_layouts[0],
                &filter_sampler,
                view,
            )
        });
        let filter_layer_textures_bind_group = create_filter_layer_textures_bind_group(
            device,
            &filter_input_bind_group_layouts[1],
            layer_binding_views,
        );
        let scratch_textures: [IntermediateTexture; 2] = core::array::from_fn(|_| None);
        let dummy_scratch_texture =
            WgpuIntermediateTexture::new(Self::create_intermediate_texture(
                device,
                Int16Size::new(1, 1),
                "Scratch Placeholder Texture",
            ));
        let scratch_binding_views = [&dummy_scratch_texture.view, &dummy_scratch_texture.view];
        let scratch_input_bind_groups = scratch_binding_views.map(|view| {
            create_filter_input_bind_group(
                device,
                &filter_input_bind_group_layouts[0],
                &filter_sampler,
                view,
            )
        });
        let layer_config_buffers = core::array::from_fn(|index| {
            let texture_index = TextureIndex::from_index(index);
            let size = texture_sizes.size(TextureTarget::layer(texture_index));
            Self::create_config_buffer_for_size(
                device,
                u32::from(size.width()),
                u32::from(size.height()),
                device.limits().max_texture_dimension_2d,
                0,
            )
        });

        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        const INITIAL_ALPHA_TEXTURE_HEIGHT: u32 = 1;
        let alphas_texture = Self::create_alphas_texture(
            device,
            max_texture_dimension_2d,
            INITIAL_ALPHA_TEXTURE_HEIGHT,
        );
        let view_config_buffer = Self::create_config_buffer(
            device,
            &RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },
            max_texture_dimension_2d,
        );

        let AtlasConfig {
            atlas_size: (atlas_width, atlas_height),
            initial_atlas_count,
            ..
        } = image_cache.atlas_manager().config();
        let (atlas_texture_array, atlas_texture_array_view) = Self::create_atlas_texture_array(
            device,
            *atlas_width,
            *atlas_height,
            *initial_atlas_count as u32,
        );
        let placeholder_external_texture_view = Self::create_placeholder_external_texture(device);
        let atlas_bind_group = Self::create_paint_source_bind_group(
            device,
            &atlas_bind_group_layout,
            &atlas_texture_array_view,
            &placeholder_external_texture_view,
        );

        // Create a 1x1 stub atlas texture array for use during render_to_atlas.
        // This avoids the read-write conflict that occurs when the real atlas is both
        // a shader input (bind group) and render target in the same pass.
        let (_stub_atlas_texture, stub_atlas_view) =
            Self::create_atlas_texture_array(device, 1, 1, 1);
        let stub_atlas_bind_group = Self::create_paint_source_bind_group(
            device,
            &atlas_bind_group_layout,
            &stub_atlas_view,
            &placeholder_external_texture_view,
        );

        const INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT: u32 = 1;
        let encoded_paints_data = vec![
            0;
            ((max_texture_dimension_2d * INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT) << 4)
                as usize
        ];
        let encoded_paints_texture = Self::create_encoded_paints_texture(
            device,
            max_texture_dimension_2d,
            INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT,
        );
        let encoded_paints_bind_group = Self::create_encoded_paints_bind_group(
            device,
            &encoded_paints_bind_group_layout,
            &encoded_paints_texture.create_view(&TextureViewDescriptor::default()),
        );

        const INITIAL_GRADIENT_TEXTURE_HEIGHT: u32 = 1;
        let gradient_texture = Self::create_gradient_texture(
            device,
            max_texture_dimension_2d,
            INITIAL_GRADIENT_TEXTURE_HEIGHT,
        );
        let gradient_bind_group = Self::create_gradient_bind_group(
            device,
            &gradient_bind_group_layout,
            &gradient_texture.create_view(&TextureViewDescriptor::default()),
        );

        // TODO: We really should deduplicate handling of this this with encoded paints texture.
        const INITIAL_FILTER_TEXTURE_HEIGHT: u32 = 1;
        let filter_data =
            vec![0_u8; ((max_texture_dimension_2d * INITIAL_FILTER_TEXTURE_HEIGHT) << 4) as usize];
        let filter_data_texture = Self::create_filter_data_texture(
            device,
            max_texture_dimension_2d,
            INITIAL_FILTER_TEXTURE_HEIGHT,
        );
        let filter_base_bind_group = Self::create_filter_base_bind_group(
            device,
            &filter_bind_group_layout,
            &filter_data_texture.create_view(&TextureViewDescriptor::default()),
        );
        let root_layer_bind_groups = Self::create_root_layer_bind_groups(
            device,
            &strip_bind_group_layout,
            &alphas_texture.create_view(&TextureViewDescriptor::default()),
            &view_config_buffer,
            layer_binding_views,
        );
        let layer_bind_groups = Self::create_layer_bind_groups(
            device,
            &strip_bind_group_layout,
            &alphas_texture.create_view(&TextureViewDescriptor::default()),
            &layer_config_buffers,
            layer_binding_views,
        );
        let blend_layer_bind_group = Self::create_blend_layer_bind_group(
            device,
            &blend_layer_bind_group_layout,
            layer_binding_views,
        );
        let blend_copy_bind_group = Self::create_blend_copy_bind_group(
            device,
            &blend_copy_bind_group_layout,
            scratch_binding_views[BLEND_SCRATCH_INDEX.get_index()],
        );

        let resources = GpuResources {
            strips_buffer: Self::create_strips_buffer(device, 0),
            layer_textures,
            dummy_layer_texture,
            layer_filter_input_bind_groups,
            filter_layer_textures_bind_group,
            scratch_textures,
            dummy_scratch_texture,
            scratch_input_bind_groups,
            blend_layer_bind_group,
            blend_copy_bind_group,
            layer_config_buffers,
            root_layer_bind_groups,
            layer_bind_groups,
            alphas_texture,
            atlas_texture_array,
            atlas_texture_array_view,
            atlas_bind_group,
            placeholder_external_texture_view,
            stub_atlas_bind_group,
            encoded_paints_texture,
            encoded_paints_bind_group,
            gradient_texture,
            gradient_bind_group,
            filter_data_texture,
            filter_base_bind_group,
            texture_sizes,
            view_config_buffer,
        };

        let depth_texture = Self::create_depth_texture(
            device,
            render_target_config.width,
            render_target_config.height,
        );
        let depth_texture_view = depth_texture.create_view(&TextureViewDescriptor::default());

        Self {
            intermediate_strip_pipelines,
            alpha_strip_pipelines,
            opaque_strip_pipelines,
            depth_texture,
            depth_texture_view,
            depth_cleared_this_frame: false,
            strip_bind_group_layout,
            encoded_paints_bind_group_layout,
            gradient_bind_group_layout,
            atlas_bind_group_layout,
            filter_bind_group_layout,
            filter_pipeline,
            blend_pipeline,
            blend_copy_pipeline,
            resources,
            encoded_paints_data,
            filter_data,
            render_size: RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },
            clear_pipeline,
            atlas_clear_pipeline,
            filter_input_bind_group_layouts: filter_input_bind_group_layouts.clone(),
            filter_sampler,
            blend_layer_bind_group_layout: blend_layer_bind_group_layout.clone(),
            blend_copy_bind_group_layout: blend_copy_bind_group_layout.clone(),
        }
    }

    #[inline]
    fn create_depth_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    fn create_strips_buffer(device: &Device, required_strips_size: u64) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Strips Buffer"),
            size: required_strips_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_intermediate_texture(
        device: &Device,
        size: Int16Size,
        label: &'static str,
    ) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: u32::from(size.width()),
                height: u32::from(size.height()),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    fn prepare_intermediate_textures(
        &mut self,
        device: &Device,
        requirements: TextureRequirements,
    ) {
        if self.resources.has_intermediate_textures(requirements) {
            return;
        }

        self.resources.layer_textures = core::array::from_fn(|index| {
            let texture_index = TextureIndex::from_index(index);
            if requirements.layer_textures[texture_index.get_index()] {
                let size = self
                    .resources
                    .texture_sizes
                    .size(TextureTarget::layer(texture_index));
                Some(WgpuIntermediateTexture::new(
                    Self::create_intermediate_texture(device, size, "Layer Atlas Texture"),
                ))
            } else {
                None
            }
        });
        self.resources.scratch_textures = core::array::from_fn(|index| {
            let texture_index = TextureIndex::from_index(index);
            if requirements.scratch_textures[texture_index.get_index()] {
                let size = self
                    .resources
                    .texture_sizes
                    .size(TextureTarget::scratch(texture_index));
                Some(WgpuIntermediateTexture::new(
                    Self::create_intermediate_texture(device, size, "Scratch Texture"),
                ))
            } else {
                None
            }
        });

        let (
            layer_filter_input_bind_groups,
            filter_layer_textures_bind_group,
            scratch_input_bind_groups,
            root_layer_bind_groups,
            layer_bind_groups,
            blend_layer_bind_group,
            blend_copy_bind_group,
        ) = {
            let layer_binding_views = self.resources.layer_binding_views();
            let scratch_binding_views = [
                self.resources.scratch_binding_view(TextureIndex::Even),
                self.resources.scratch_binding_view(TextureIndex::Odd),
            ];
            let alphas_texture_view = self
                .resources
                .alphas_texture
                .create_view(&TextureViewDescriptor::default());

            (
                layer_binding_views.map(|view| {
                    create_filter_input_bind_group(
                        device,
                        &self.filter_input_bind_group_layouts[0],
                        &self.filter_sampler,
                        view,
                    )
                }),
                create_filter_layer_textures_bind_group(
                    device,
                    &self.filter_input_bind_group_layouts[1],
                    layer_binding_views,
                ),
                scratch_binding_views.map(|view| {
                    create_filter_input_bind_group(
                        device,
                        &self.filter_input_bind_group_layouts[0],
                        &self.filter_sampler,
                        view,
                    )
                }),
                Self::create_root_layer_bind_groups(
                    device,
                    &self.strip_bind_group_layout,
                    &alphas_texture_view,
                    &self.resources.view_config_buffer,
                    layer_binding_views,
                ),
                Self::create_layer_bind_groups(
                    device,
                    &self.strip_bind_group_layout,
                    &alphas_texture_view,
                    &self.resources.layer_config_buffers,
                    layer_binding_views,
                ),
                Self::create_blend_layer_bind_group(
                    device,
                    &self.blend_layer_bind_group_layout,
                    layer_binding_views,
                ),
                Self::create_blend_copy_bind_group(
                    device,
                    &self.blend_copy_bind_group_layout,
                    scratch_binding_views[BLEND_SCRATCH_INDEX.get_index()],
                ),
            )
        };

        self.resources.layer_filter_input_bind_groups = layer_filter_input_bind_groups;
        self.resources.filter_layer_textures_bind_group = filter_layer_textures_bind_group;
        self.resources.scratch_input_bind_groups = scratch_input_bind_groups;
        self.resources.root_layer_bind_groups = root_layer_bind_groups;
        self.resources.layer_bind_groups = layer_bind_groups;
        self.resources.blend_layer_bind_group = blend_layer_bind_group;
        self.resources.blend_copy_bind_group = blend_copy_bind_group;
    }

    fn create_blend_layer_bind_group(
        device: &Device,
        layout: &BindGroupLayout,
        layer_texture_views: [&WgpuTextureView; 2],
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blend Layer Textures Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(layer_texture_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(layer_texture_views[1]),
                },
            ],
        })
    }

    fn create_blend_copy_bind_group(
        device: &Device,
        layout: &BindGroupLayout,
        scratch_texture_view: &WgpuTextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blend Copy Bind Group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(scratch_texture_view),
            }],
        })
    }

    fn create_config_buffer(
        device: &Device,
        render_size: &RenderSize,
        alpha_texture_width: u32,
    ) -> Buffer {
        Self::create_config_buffer_for_size(
            device,
            render_size.width,
            render_size.height,
            alpha_texture_width,
            0,
        )
    }

    fn create_config_buffer_for_size(
        device: &Device,
        width: u32,
        height: u32,
        alpha_texture_width: u32,
        negate_ndc: u32,
    ) -> Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::bytes_of(&Config {
                width,
                height,
                strip_height: Tile::HEIGHT.into(),
                alphas_tex_width_bits: alpha_texture_width.trailing_zeros(),
                encoded_paints_tex_width_bits: alpha_texture_width.trailing_zeros(),
                strip_offset_x: 0,
                strip_offset_y: 0,
                negate_ndc,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn create_alphas_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Alpha Texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn create_atlas_texture_array(
        device: &Device,
        width: u32,
        height: u32,
        atlas_count: u32,
    ) -> (Texture, WgpuTextureView) {
        // See the comment in `Renderer::new_with`. On WASM, we need to set this to at
        // least 2 so it works with the wgpu WebGL backend.
        #[cfg(target_arch = "wasm32")]
        let depth_or_array_layers = atlas_count.max(2);
        #[cfg(not(target_arch = "wasm32"))]
        let depth_or_array_layers = atlas_count;

        let atlas_texture_array = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Atlas Texture Array"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let atlas_texture_array_view = atlas_texture_array.create_view(&TextureViewDescriptor {
            label: Some("Atlas Texture Array View"),
            format: None,
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(atlas_count),
            usage: None,
        });

        (atlas_texture_array, atlas_texture_array_view)
    }

    fn create_filter_data_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Filter Data Texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn create_filter_base_bind_group(
        device: &Device,
        filter_bind_group_layout: &BindGroupLayout,
        filter_texture_view: &WgpuTextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Filter Base Bind Group"),
            layout: filter_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(filter_texture_view),
            }],
        })
    }

    fn create_placeholder_external_texture(device: &Device) -> WgpuTextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Placeholder External Texture"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        texture.create_view(&TextureViewDescriptor::default())
    }

    fn create_paint_source_bind_group(
        device: &Device,
        atlas_bind_group_layout: &BindGroupLayout,
        atlas_texture_array_view: &WgpuTextureView,
        external_texture_view: &WgpuTextureView,
    ) -> BindGroup {
        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(atlas_texture_array_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(external_texture_view),
            },
        ];
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Paint Source Bind Group"),
            layout: atlas_bind_group_layout,
            entries: &entries,
        })
    }

    fn create_encoded_paints_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Encoded Paints Texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn create_encoded_paints_bind_group(
        device: &Device,
        encoded_paints_bind_group_layout: &BindGroupLayout,
        encoded_paints_texture_view: &WgpuTextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Encoded Paints Bind Group"),
            layout: encoded_paints_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(encoded_paints_texture_view),
            }],
        })
    }

    fn create_gradient_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Gradient Texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn create_gradient_bind_group(
        device: &Device,
        gradient_bind_group_layout: &BindGroupLayout,
        gradient_texture_view: &WgpuTextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gradient Bind Group"),
            layout: gradient_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(gradient_texture_view),
            }],
        })
    }

    fn create_layer_bind_groups(
        device: &Device,
        strip_bind_group_layout: &BindGroupLayout,
        alphas_texture_view: &WgpuTextureView,
        layer_config_buffers: &[Buffer; 2],
        layer_texture_views: [&WgpuTextureView; 2],
    ) -> [BindGroup; 2] {
        [
            Self::create_strip_bind_group(
                device,
                strip_bind_group_layout,
                alphas_texture_view,
                &layer_config_buffers[0],
                layer_texture_views[1],
            ),
            Self::create_strip_bind_group(
                device,
                strip_bind_group_layout,
                alphas_texture_view,
                &layer_config_buffers[1],
                layer_texture_views[0],
            ),
        ]
    }

    fn create_root_layer_bind_groups(
        device: &Device,
        strip_bind_group_layout: &BindGroupLayout,
        alphas_texture_view: &WgpuTextureView,
        view_config_buffer: &Buffer,
        layer_texture_views: [&WgpuTextureView; 2],
    ) -> [BindGroup; 2] {
        [
            Self::create_strip_bind_group(
                device,
                strip_bind_group_layout,
                alphas_texture_view,
                view_config_buffer,
                layer_texture_views[0],
            ),
            Self::create_strip_bind_group(
                device,
                strip_bind_group_layout,
                alphas_texture_view,
                view_config_buffer,
                layer_texture_views[1],
            ),
        ]
    }

    fn create_strip_bind_group(
        device: &Device,
        strip_bind_group_layout: &BindGroupLayout,
        alphas_texture_view: &WgpuTextureView,
        config_buffer: &Buffer,
        strip_texture_view: &WgpuTextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Strip Bind Group"),
            layout: strip_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(alphas_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(strip_texture_view),
                },
            ],
        })
    }

    /// Prepare GPU buffers for rendering, given alphas.
    ///
    /// Specifically, updates the alpha texture with `alphas` and the config buffer when
    /// the rendering size changes.
    fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        gradient_cache: &mut GradientRampCache,
        encoded_paints: &[GpuEncodedPaint],
        alphas: &mut Vec<u8>,
        new_render_size: &RenderSize,
        paint_idxs: &[u32],
        filter_context: &FilterContext,
    ) {
        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        self.maybe_resize_alphas_tex(device, max_texture_dimension_2d, alphas.len());
        self.maybe_resize_encoded_paints_tex(device, max_texture_dimension_2d, paint_idxs);
        self.maybe_resize_filter_tex(device, max_texture_dimension_2d, filter_context);
        self.maybe_update_config_buffer(device, queue, max_texture_dimension_2d, new_render_size);

        self.upload_alpha_texture(queue, alphas);
        self.upload_encoded_paints_texture(queue, encoded_paints);
        self.upload_filter_texture(queue, filter_context);

        if gradient_cache.has_changed() {
            self.maybe_resize_gradient_tex(device, max_texture_dimension_2d, gradient_cache);
            self.upload_gradient_texture(queue, gradient_cache);
            gradient_cache.mark_synced();
        }
    }

    fn maybe_resize_filter_tex(
        &mut self,
        device: &Device,
        max_texture_dimension_2d: u32,
        filter_context: &FilterContext,
    ) {
        let Some(required_filter_height) =
            filter_context.required_filter_data_height(max_texture_dimension_2d)
        else {
            return;
        };
        debug_assert!(
            self.resources.filter_data_texture.width() == max_texture_dimension_2d,
            "Filter texture width must match max texture dimensions"
        );
        let current_filter_height = self.resources.filter_data_texture.height();
        if required_filter_height > current_filter_height {
            let required_filter_size = (max_texture_dimension_2d * required_filter_height) << 4;
            self.filter_data.resize(required_filter_size as usize, 0);

            let filter_texture = Self::create_filter_data_texture(
                device,
                max_texture_dimension_2d,
                required_filter_height,
            );
            self.resources.filter_data_texture = filter_texture;
            self.resources.filter_base_bind_group = Self::create_filter_base_bind_group(
                device,
                &self.filter_bind_group_layout,
                &self
                    .resources
                    .filter_data_texture
                    .create_view(&TextureViewDescriptor::default()),
            );
        }
    }

    /// Update the alpha texture size if needed.
    fn maybe_resize_alphas_tex(
        &mut self,
        device: &Device,
        max_texture_dimension_2d: u32,
        alphas_len: usize,
    ) {
        let required_alpha_height = u32::try_from(alphas_len)
            .unwrap()
            // There are 16 1-byte alpha values per texel.
            .div_ceil(max_texture_dimension_2d << 4);
        debug_assert!(
            self.resources.alphas_texture.width() == max_texture_dimension_2d,
            "Alpha texture width must match max texture dimensions"
        );
        let current_alpha_height = self.resources.alphas_texture.height();
        if required_alpha_height > current_alpha_height {
            // We need to resize the alpha texture to fit the new alpha data.
            assert!(
                required_alpha_height <= max_texture_dimension_2d,
                "Alpha texture height exceeds max texture dimensions"
            );

            // The alpha texture encodes 16 1-byte alpha values per texel, with 4 alpha values packed in each channel
            let alphas_texture = Self::create_alphas_texture(
                device,
                max_texture_dimension_2d,
                required_alpha_height,
            );
            self.resources.alphas_texture = alphas_texture;

            self.resources.root_layer_bind_groups = Self::create_root_layer_bind_groups(
                device,
                &self.strip_bind_group_layout,
                &self
                    .resources
                    .alphas_texture
                    .create_view(&TextureViewDescriptor::default()),
                &self.resources.view_config_buffer,
                self.resources.layer_binding_views(),
            );
            self.resources.layer_bind_groups = Self::create_layer_bind_groups(
                device,
                &self.strip_bind_group_layout,
                &self
                    .resources
                    .alphas_texture
                    .create_view(&TextureViewDescriptor::default()),
                &self.resources.layer_config_buffers,
                self.resources.layer_binding_views(),
            );
        }
    }

    /// Update the encoded paints texture size if needed.
    fn maybe_resize_encoded_paints_tex(
        &mut self,
        device: &Device,
        max_texture_dimension_2d: u32,
        paint_idxs: &[u32],
    ) {
        let required_texels = paint_idxs.last().unwrap();
        let required_encoded_paints_height = required_texels.div_ceil(max_texture_dimension_2d);
        debug_assert!(
            self.resources.encoded_paints_texture.width() == max_texture_dimension_2d,
            "Encoded paints texture width must match max texture dimensions"
        );
        let current_encoded_paints_height = self.resources.encoded_paints_texture.height();
        if required_encoded_paints_height > current_encoded_paints_height {
            assert!(
                required_encoded_paints_height <= max_texture_dimension_2d,
                "Encoded paints texture height exceeds max texture dimensions"
            );
            let required_encoded_paints_size =
                (max_texture_dimension_2d * required_encoded_paints_height) << 4;
            self.encoded_paints_data
                .resize(required_encoded_paints_size as usize, 0);
            let encoded_paints_texture = Self::create_encoded_paints_texture(
                device,
                max_texture_dimension_2d,
                required_encoded_paints_height,
            );
            self.resources.encoded_paints_texture = encoded_paints_texture;

            // Since the encoded paints texture has changed, we need to update the strip bind groups.
            self.resources.encoded_paints_bind_group = Self::create_encoded_paints_bind_group(
                device,
                &self.encoded_paints_bind_group_layout,
                &self
                    .resources
                    .encoded_paints_texture
                    .create_view(&TextureViewDescriptor::default()),
            );
        }
    }

    /// Update the gradient texture size if needed.
    fn maybe_resize_gradient_tex(
        &mut self,
        device: &Device,
        max_texture_dimension_2d: u32,
        gradient_cache: &GradientRampCache,
    ) {
        let gradient_pixels = (gradient_cache.luts_size() / 4) as u32; // 4 bytes per RGBA8 pixel
        let required_gradient_height = gradient_pixels.div_ceil(max_texture_dimension_2d);
        debug_assert!(
            self.resources.gradient_texture.width() == max_texture_dimension_2d,
            "Gradient texture width must match max texture dimensions"
        );
        let current_gradient_height = self.resources.gradient_texture.height();
        if required_gradient_height > current_gradient_height {
            assert!(
                required_gradient_height <= max_texture_dimension_2d,
                "Gradient texture height exceeds max texture dimensions"
            );
            let gradient_texture = Self::create_gradient_texture(
                device,
                max_texture_dimension_2d,
                required_gradient_height,
            );
            self.resources.gradient_texture = gradient_texture;

            // Since the gradient texture has changed, we need to update the gradient bind group.
            self.resources.gradient_bind_group = Self::create_gradient_bind_group(
                device,
                &self.gradient_bind_group_layout,
                &self
                    .resources
                    .gradient_texture
                    .create_view(&TextureViewDescriptor::default()),
            );
        }
    }

    /// Update config buffer if dimensions changed.
    fn maybe_update_config_buffer(
        &mut self,
        device: &Device,
        queue: &Queue,
        max_texture_dimension_2d: u32,
        new_render_size: &RenderSize,
    ) {
        if self.render_size != *new_render_size {
            let config = Config {
                width: new_render_size.width,
                height: new_render_size.height,
                strip_height: Tile::HEIGHT.into(),
                alphas_tex_width_bits: max_texture_dimension_2d.trailing_zeros(),
                encoded_paints_tex_width_bits: max_texture_dimension_2d.trailing_zeros(),
                strip_offset_x: 0,
                strip_offset_y: 0,
                negate_ndc: 0,
            };
            let mut buffer = queue
                .write_buffer_with(&self.resources.view_config_buffer, 0, SIZE_OF_CONFIG)
                .expect("Buffer only ever holds `Config`");
            buffer.copy_from_slice(bytemuck::bytes_of(&config));

            self.depth_texture =
                Self::create_depth_texture(device, new_render_size.width, new_render_size.height);
            self.depth_texture_view = self
                .depth_texture
                .create_view(&TextureViewDescriptor::default());

            self.render_size = new_render_size.clone();
        }
    }

    /// Resize the texture array to accommodate more atlases.
    fn maybe_resize_atlas_texture_array(
        device: &Device,
        encoder: &mut CommandEncoder,
        resources: &mut GpuResources,
        atlas_bind_group_layout: &BindGroupLayout,
        required_atlas_count: u32,
    ) {
        let Extent3d {
            width,
            height,
            depth_or_array_layers: current_atlas_count,
        } = resources.atlas_texture_array.size();
        if required_atlas_count > current_atlas_count {
            // Create new texture array with more layers
            let (new_atlas_texture_array, new_atlas_texture_array_view) =
                Self::create_atlas_texture_array(device, width, height, required_atlas_count);

            // Copy existing atlas data from old texture array to new one
            Self::copy_atlas_texture_data(
                encoder,
                &resources.atlas_texture_array,
                &new_atlas_texture_array,
                current_atlas_count,
                width,
                height,
            );

            // Update the bind group with the new texture array view
            let new_atlas_bind_group = Self::create_paint_source_bind_group(
                device,
                atlas_bind_group_layout,
                &new_atlas_texture_array_view,
                &resources.placeholder_external_texture_view,
            );

            // Replace the old resources
            resources.atlas_texture_array = new_atlas_texture_array;
            resources.atlas_texture_array_view = new_atlas_texture_array_view;
            resources.atlas_bind_group = new_atlas_bind_group;
        }
    }

    fn create_external_paint_source_bind_group(
        &self,
        device: &Device,
        external_texture_view: &WgpuTextureView,
    ) -> BindGroup {
        Self::create_paint_source_bind_group(
            device,
            &self.atlas_bind_group_layout,
            &self.resources.atlas_texture_array_view,
            external_texture_view,
        )
    }

    /// Copy texture data from the old atlas texture array to a new one.
    /// This is necessary when resizing the texture array to preserve existing atlas data.
    fn copy_atlas_texture_data(
        encoder: &mut CommandEncoder,
        old_atlas_texture_array: &Texture,
        new_atlas_texture_array: &Texture,
        layer_count_to_copy: u32,
        width: u32,
        height: u32,
    ) {
        // Copy all layers from old texture array to new texture array
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: old_atlas_texture_array,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: new_atlas_texture_array,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: layer_count_to_copy,
            },
        );
    }

    /// Upload alpha data to the texture.
    fn upload_alpha_texture(&mut self, queue: &Queue, alphas: &mut Vec<u8>) {
        if alphas.is_empty() {
            return;
        }

        let texture_width = self.resources.alphas_texture.width();
        let texture_height = self.resources.alphas_texture.height();
        let total_size = texture_width as usize * texture_height as usize * 16;

        let original_len = alphas.len();

        // Temporarily pad the length of the alphas to the texture size before uploading.
        alphas.resize(total_size, 0);

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.resources.alphas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            alphas,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // 16 bytes per RGBA32Uint texel (4 u32s × 4 bytes each), which is equivalent to
                // a bit shift of 4.
                bytes_per_row: Some(texture_width << 4),
                rows_per_image: Some(texture_height),
            },
            Extent3d {
                width: texture_width,
                height: texture_height,
                depth_or_array_layers: 1,
            },
        );

        // Truncate back to the original size.
        alphas.truncate(original_len);
    }

    /// Upload encoded paints to the texture.
    fn upload_encoded_paints_texture(&mut self, queue: &Queue, encoded_paints: &[GpuEncodedPaint]) {
        let encoded_paints_texture = &self.resources.encoded_paints_texture;
        let encoded_paints_texture_width = encoded_paints_texture.width();
        let encoded_paints_texture_height = encoded_paints_texture.height();

        GpuEncodedPaint::serialize_to_buffer(encoded_paints, &mut self.encoded_paints_data);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: encoded_paints_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.encoded_paints_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // 16 bytes per RGBA32Uint texel (4 u32s × 4 bytes each), equivalent to bit shift of 4
                bytes_per_row: Some(encoded_paints_texture_width << 4),
                rows_per_image: Some(encoded_paints_texture_height),
            },
            Extent3d {
                width: encoded_paints_texture_width,
                height: encoded_paints_texture_height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn upload_filter_texture(&mut self, queue: &Queue, filter_context: &FilterContext) {
        if filter_context.is_empty() {
            return;
        }

        let filter_texture = &self.resources.filter_data_texture;
        let width = filter_texture.width();
        let height = filter_texture.height();

        filter_context.serialize_to_buffer(&mut self.filter_data);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: filter_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.filter_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width << 4),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Upload gradient data to the texture.
    fn upload_gradient_texture(&mut self, queue: &Queue, gradient_cache: &mut GradientRampCache) {
        let gradient_texture = &self.resources.gradient_texture;
        let gradient_texture_width = gradient_texture.width();
        let gradient_texture_height = gradient_texture.height();

        // Upload the gradient LUT data
        if !gradient_cache.is_empty() {
            let total_capacity = (gradient_texture_width * gradient_texture_height * 4) as usize;

            // Take ownership of the luts to avoid copying, then resize for texture padding
            let mut luts = gradient_cache.take_luts();
            let old_luts_len = luts.len();
            luts.resize(total_capacity, 0);

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: gradient_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &luts,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    // 4 bytes per RGBA8 pixel
                    bytes_per_row: Some(gradient_texture_width << 2),
                    rows_per_image: Some(gradient_texture_height),
                },
                Extent3d {
                    width: gradient_texture_width,
                    height: gradient_texture_height,
                    depth_or_array_layers: 1,
                },
            );

            // Restore the luts back to the cache
            luts.truncate(old_luts_len);
            gradient_cache.restore_luts(luts);
        }
    }

    /// Uploads two strip slices (opaque then alpha) into a single GPU buffer.
    fn upload_strip_pair(
        &mut self,
        device: &Device,
        queue: &Queue,
        opaque_strips: &[GpuStrip],
        alpha_strips: RangedSlice<'_, GpuStrip>,
    ) {
        let opaque_bytes = size_of_val(opaque_strips) as u64;
        let alpha_bytes = (alpha_strips.len() * size_of::<GpuStrip>()) as u64;
        let total = opaque_bytes + alpha_bytes;
        self.resources.strips_buffer = Self::create_strips_buffer(device, total);
        // TODO: Consider using a staging belt to avoid an extra staging buffer allocation.
        let mut buffer_view = queue
            .write_buffer_with(&self.resources.strips_buffer, 0, total.try_into().unwrap())
            .expect("Capacity handled in creation");
        buffer_view
            .slice(..opaque_bytes as usize)
            .copy_from_slice(bytemuck::cast_slice(opaque_strips));
        let mut offset = opaque_bytes as usize;
        for strips in alpha_strips.slices() {
            let bytes = bytemuck::cast_slice(strips);
            buffer_view
                .slice(offset..offset + bytes.len())
                .copy_from_slice(bytes);
            offset += bytes.len();
        }
    }
}

/// A struct containing references to the many objects needed to get work
/// scheduled onto the GPU.
struct RendererContext<'a> {
    programs: &'a mut Programs,
    device: &'a Device,
    queue: &'a Queue,
    encoder: &'a mut CommandEncoder,
    view: &'a WgpuTextureView,
    texture_bindings: &'a TextureBindings,
    external_paint_source_bind_groups: HashMap<TextureId, BindGroup>,
    scratch: &'a mut ScratchBuffers,
}

impl RendererContext<'_> {
    fn external_paint_source_bind_group_for_texture(
        &mut self,
        texture_id: TextureId,
    ) -> &BindGroup {
        match self.external_paint_source_bind_groups.entry(texture_id) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let texture_view = self
                    .texture_bindings
                    .get(texture_id)
                    .expect("external texture bindings were validated during paint preparation");
                let bind_group = self
                    .programs
                    .create_external_paint_source_bind_group(self.device, texture_view);
                entry.insert(bind_group)
            }
        }
    }

    /// Render the strips to the specified render target.
    fn strip_pass_inner(
        &mut self,
        opaque_strips: &[GpuStrip],
        alpha_strips: RangedSlice<'_, GpuStrip>,
        external_texture_runs: &[ExternalTextureRun],
        target: DrawPassTarget,
    ) {
        let opaque_count = opaque_strips.len();
        let alpha_count = alpha_strips.len();
        if opaque_count == 0 && alpha_count == 0 {
            return;
        }
        // TODO: We currently allocate a new strips buffer for each render pass. A more efficient
        // approach would be to re-use buffers or slices of a larger buffer.
        // Create bind groups for all external textures passed in by the user that are used this
        // pass.
        for run in external_texture_runs {
            self.external_paint_source_bind_group_for_texture(run.texture_id);
        }

        self.programs
            .upload_strip_pair(self.device, self.queue, opaque_strips, alpha_strips);
        let opaque_count = opaque_count as u32;
        let alpha_count = alpha_count as u32;

        let (view, bind_group): (&WgpuTextureView, &BindGroup) = match target {
            DrawPassTarget::Root(_) => (
                self.view,
                &self.programs.resources.root_layer_bind_groups[1],
            ),
            DrawPassTarget::Layer(texture_index) => (
                self.programs.resources.layer_view(texture_index),
                &self.programs.resources.layer_bind_groups[texture_index.get_index()],
            ),
        };

        let enable_opaque = target.enable_opaque();
        let pipeline_idx = usize::from(!enable_opaque);

        let depth_stencil_attachment = if enable_opaque {
            let depth_load = if self.programs.depth_cleared_this_frame {
                wgpu::LoadOp::Load
            } else {
                self.programs.depth_cleared_this_frame = true;
                wgpu::LoadOp::Clear(1.0)
            };
            Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.programs.depth_texture_view,
                depth_ops: Some(wgpu::Operations {
                    load: depth_load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            })
        } else {
            None
        };

        let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render to Texture Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_bind_group(2, &self.programs.resources.encoded_paints_bind_group, &[]);
        render_pass.set_bind_group(3, &self.programs.resources.gradient_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.programs.resources.strips_buffer.slice(..));

        if opaque_count > 0 {
            // Opaque pass
            debug_assert!(
                enable_opaque,
                "opaque strips require the final view depth attachment"
            );
            render_pass.set_pipeline(&self.programs.opaque_strip_pipelines[pipeline_idx]);
            render_pass.set_bind_group(1, &self.programs.resources.atlas_bind_group, &[]);
            render_pass.draw(0..4, 0..opaque_count);
        }

        if alpha_count > 0 {
            // Alpha pass
            if enable_opaque {
                render_pass.set_pipeline(&self.programs.alpha_strip_pipelines[pipeline_idx]);
            } else {
                render_pass.set_pipeline(&self.programs.intermediate_strip_pipelines[pipeline_idx]);
            }

            let alpha_start = opaque_count;
            if external_texture_runs.is_empty() {
                render_pass.set_bind_group(1, &self.programs.resources.atlas_bind_group, &[]);
                render_pass.draw(0..4, alpha_start..alpha_start + alpha_count);
            } else {
                // Each run is drawn with a different external texture binding. Runs go from
                // `run.strips_start` to the next run's `strips_start`; the last run goes to the end of
                // the strips buffer.
                for (i, run) in external_texture_runs.iter().enumerate() {
                    let paint_source_bind_group = self
                        .external_paint_source_bind_groups
                        .get(&run.texture_id)
                        .unwrap();
                    render_pass.set_bind_group(1, paint_source_bind_group, &[]);
                    let start = u32::try_from(run.strips_start).unwrap();
                    let end = external_texture_runs
                        .get(i + 1)
                        .map_or(alpha_count, |next| {
                            u32::try_from(next.strips_start).unwrap()
                        });
                    render_pass.draw(0..4, alpha_start + start..alpha_start + end);
                }
            }
        }
    }

    fn texture_size(&self, target: TextureTarget) -> Int16Size {
        self.programs.resources.texture_sizes.size(target)
    }

    fn blend_pass_inner(&mut self, blends: RangedSlice<'_, BlendOp>, texture_index: TextureIndex) {
        let parent_texture_size = self.texture_size(TextureTarget::layer(texture_index));
        let scratch_texture_size = self.texture_size(TextureTarget::scratch(TextureIndex::Even));
        if blends.len() == 0 {
            return;
        }
        let resources = &self.programs.resources;
        self.scratch.blend_instances.clear();
        self.scratch.blend_instances.extend(
            blends
                .iter()
                .copied()
                .filter(|blend| !blend.blend_bbox.is_empty())
                .map(|blend| {
                    debug_assert_eq!(blend.parent_region.texture.texture_index, texture_index);
                    resources.layer_view(blend.parent_region.texture.texture_index);
                    resources.layer_view(blend.child_region.texture.texture_index);
                    gpu_blend_instance(blend, scratch_texture_size)
                }),
        );
        if self.scratch.blend_instances.is_empty() {
            return;
        }

        let instance_count = u32::try_from(self.scratch.blend_instances.len()).unwrap();
        let instance_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Blend Instances Buffer"),
                contents: bytemuck::cast_slice(&self.scratch.blend_instances),
                usage: wgpu::BufferUsages::VERTEX,
            });

        {
            let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Blend To Scratch"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: self.programs.resources.scratch_view(BLEND_SCRATCH_INDEX),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            render_pass.set_pipeline(&self.programs.blend_pipeline);
            render_pass.set_bind_group(0, &self.programs.resources.blend_layer_bind_group, &[]);
            render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
            render_pass.draw(0..4, 0..instance_count);
        }

        self.scratch.copy_instances.clear();
        self.scratch.copy_instances.extend(
            self.scratch
                .blend_instances
                .iter()
                .copied()
                .map(|instance| instance.copy_from_parent_in_scratch(parent_texture_size)),
        );
        let copy_instance_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Blend Copy Instances Buffer"),
                    contents: bytemuck::cast_slice(&self.scratch.copy_instances),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        {
            let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Copy Blend Scratch To Layer"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: self.programs.resources.layer_view(texture_index),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            render_pass.set_pipeline(&self.programs.blend_copy_pipeline);
            render_pass.set_bind_group(1, &self.programs.resources.blend_copy_bind_group, &[]);
            render_pass.set_vertex_buffer(0, copy_instance_buffer.slice(..));
            render_pass.draw(0..4, 0..instance_count);
        }
    }

    fn filter_pass_inner(&mut self, plan: &FilterPassPlan, texture_index: TextureIndex) {
        if plan.is_empty() {
            return;
        }
        let resources = &self.programs.resources;
        for (step_index, instances) in plan.steps().enumerate() {
            let (input, output) = if step_index == 0 {
                (
                    TextureTarget::layer(texture_index),
                    TextureTarget::scratch(TextureIndex::Even),
                )
            } else if step_index % 2 == 1 {
                (
                    TextureTarget::scratch(TextureIndex::Even),
                    TextureTarget::scratch(TextureIndex::Odd),
                )
            } else {
                (
                    TextureTarget::scratch(TextureIndex::Odd),
                    TextureTarget::scratch(TextureIndex::Even),
                )
            };
            encode_filter_pass(
                self.device,
                self.encoder,
                resources,
                &self.programs.filter_pipeline,
                instances,
                input,
                output,
            );
        }

        let copy_back = plan.copy_back();
        let instance_count = u32::try_from(copy_back.len()).unwrap();
        let instance_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Filter Copy Instances Buffer"),
                contents: bytemuck::cast_slice(copy_back),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Copy Filter Scratch To Layer"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: self.programs.resources.layer_view(texture_index),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(&self.programs.blend_copy_pipeline);
        render_pass.set_bind_group(1, &self.programs.resources.blend_copy_bind_group, &[]);
        render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
        render_pass.draw(0..4, 0..instance_count);
    }

    fn do_clear_rects(&mut self, target: TextureTarget, rects: &[RectU16], label: &'static str) {
        let target_size = self.texture_size(target);
        self.scratch.clear_instances.clear();
        self.scratch.clear_instances.extend(
            rects
                .iter()
                .copied()
                .filter(|rect| !rect.is_empty())
                .map(|rect| {
                    gpu_clear_instance(
                        rect,
                        [
                            u32::from(target_size.width()),
                            u32::from(target_size.height()),
                        ],
                    )
                }),
        );
        self.do_clear_instances(target, label);
    }

    fn do_clear_instances(&mut self, target: TextureTarget, label: &'static str) {
        if self.scratch.clear_instances.is_empty() {
            return;
        }

        // Each recorded render pass needs stable vertex contents until command submission.
        let clear_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Clear Buffer"),
                contents: bytemuck::cast_slice(&self.scratch.clear_instances),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let resources = &self.programs.resources;
        let view = match target {
            TextureTarget::Layer(_) => resources.layer_view(target.index()),
            TextureTarget::Scratch(_) => resources.scratch_view(target.index()),
        };
        let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(&self.programs.clear_pipeline);
        render_pass.set_vertex_buffer(0, clear_buffer.slice(..));
        render_pass.draw(
            0..4,
            0..u32::try_from(self.scratch.clear_instances.len()).unwrap(),
        );
    }
}

impl RendererBackend for RendererContext<'_> {
    fn opaque_pass(&mut self, strips: &[GpuStrip]) {
        self.strip_pass_inner(
            strips,
            RangedSlice::empty(),
            &[],
            DrawPassTarget::Root(RootRenderTarget::UserSurface),
        );
    }

    fn draw_pass(
        &mut self,
        strips: RangedSlice<'_, GpuStrip>,
        external_texture_runs: &[ExternalTextureRun],
        target: DrawPassTarget,
    ) {
        self.strip_pass_inner(&[], strips, external_texture_runs, target);
    }

    fn blend_pass(&mut self, blends: RangedSlice<'_, BlendOp>, texture_index: TextureIndex) {
        self.blend_pass_inner(blends, texture_index);
    }

    fn filter_pass(&mut self, plan: &FilterPassPlan, texture_index: TextureIndex) {
        self.filter_pass_inner(plan, texture_index);
    }

    fn clear_pass(&mut self, target: TextureTarget, rects: &[RectU16]) {
        self.do_clear_rects(target, rects, "Clear Rects");
    }
}

fn encode_filter_pass(
    device: &Device,
    encoder: &mut CommandEncoder,
    resources: &GpuResources,
    filter_pipeline: &RenderPipeline,
    instances: &[FilterInstanceData],
    input: TextureTarget,
    output: TextureTarget,
) {
    if instances.is_empty() {
        return;
    }

    let input_bg = filter_input_bind_group(resources, input);
    let output_view = filter_output_view(resources, output);
    let instance_count = u32::try_from(instances.len()).unwrap();
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Filter Instances Buffer"),
        contents: bytemuck::cast_slice(instances),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("Apply Filter Pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: output_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: None,
    });
    render_pass.set_pipeline(filter_pipeline);
    render_pass.set_bind_group(0, &resources.filter_base_bind_group, &[]);
    render_pass.set_bind_group(1, input_bg, &[]);
    render_pass.set_bind_group(2, &resources.filter_layer_textures_bind_group, &[]);
    render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
    render_pass.draw(0..4, 0..instance_count);
}

fn filter_input_bind_group(resources: &GpuResources, texture: TextureTarget) -> &BindGroup {
    match texture {
        TextureTarget::Layer(_) => {
            assert!(
                resources.layer_textures[texture.index().get_index()].is_some(),
                "vello_hybrid attempted to sample a missing layer texture"
            );
            &resources.layer_filter_input_bind_groups[texture.index().get_index()]
        }
        TextureTarget::Scratch(_) => {
            assert!(
                resources.scratch_textures[texture.index().get_index()].is_some(),
                "vello_hybrid attempted to sample a missing scratch texture"
            );
            &resources.scratch_input_bind_groups[texture.index().get_index()]
        }
    }
}

fn filter_output_view(resources: &GpuResources, texture: TextureTarget) -> &WgpuTextureView {
    resources.texture_target_view(texture)
}

fn gpu_clear_instance(rect: RectU16, target_size: [u32; 2]) -> GpuClearInstance {
    GpuClearInstance {
        origin: [u32::from(rect.x0), u32::from(rect.y0)],
        size: [u32::from(rect.width()), u32::from(rect.height())],
        target_size,
    }
}

fn create_filter_input_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    sampler: &Sampler,
    texture_view: &WgpuTextureView,
) -> BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Filter Input Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn create_filter_layer_textures_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    layer_texture_views: [&WgpuTextureView; 2],
) -> BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Filter Layer Textures Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(layer_texture_views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(layer_texture_views[1]),
            },
        ],
    })
}

/// Trait for types that can write image data directly to the atlas texture.
///
/// This allows efficient uploading from different sources:
/// - `Pixmap`: Direct upload without intermediate texture
/// - `Texture`: Texture-to-texture copy
/// - Custom implementations for other image sources
pub trait AtlasWriter {
    /// Get the width of the image.
    fn width(&self) -> u32;
    /// Get the height of the image.
    fn height(&self) -> u32;

    /// Write image data to a specific layer of an atlas texture array at the specified offset.
    fn write_to_atlas_layer(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        atlas_texture: &Texture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    );
}

/// Implementation for `wgpu::Texture` - uses texture-to-texture copy
impl AtlasWriter for Texture {
    fn width(&self) -> u32 {
        self.width()
    }

    fn height(&self) -> u32 {
        self.height()
    }

    fn write_to_atlas_layer(
        &self,
        _device: &Device,
        _queue: &Queue,
        encoder: &mut CommandEncoder,
        atlas_texture: &Texture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: offset[0],
                    y: offset[1],
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Implementation for `Pixmap` - direct upload to atlas
impl AtlasWriter for Pixmap {
    fn width(&self) -> u32 {
        self.width() as u32
    }

    fn height(&self) -> u32 {
        self.height() as u32
    }

    fn write_to_atlas_layer(
        &self,
        _device: &Device,
        queue: &Queue,
        _encoder: &mut CommandEncoder,
        atlas_texture: &Texture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: offset[0],
                    y: offset[1],
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            self.data_as_u8_slice(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Implementation for `Arc<Pixmap>`
impl AtlasWriter for Arc<Pixmap> {
    fn width(&self) -> u32 {
        self.as_ref().width() as u32
    }

    fn height(&self) -> u32 {
        self.as_ref().height() as u32
    }

    fn write_to_atlas_layer(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        atlas_texture: &Texture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        self.as_ref().write_to_atlas_layer(
            device,
            queue,
            encoder,
            atlas_texture,
            layer,
            offset,
            width,
            height,
        );
    }
}
