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

use crate::render::common::IMAGE_PADDING;
use crate::{
    GpuStrip, RenderError, RenderSettings, RenderSize, Resources,
    filter::{FilterContext, FilterInstanceData, FilterPassState, FilterPassTarget},
    gradient_cache::GradientRampCache,
    render::{
        Config,
        common::{
            GPU_ENCODED_IMAGE_SIZE_TEXELS, GPU_LINEAR_GRADIENT_SIZE_TEXELS,
            GPU_RADIAL_GRADIENT_SIZE_TEXELS, GPU_SWEEP_GRADIENT_SIZE_TEXELS, GpuEncodedImage,
            GpuEncodedPaint, GpuLinearGradient, GpuRadialGradient, GpuSweepGradient,
            normalize_atlas_config, pack_image_offset, pack_image_params, pack_image_size,
            pack_radial_kind_and_swapped, pack_texture_width_and_extend_mode, pack_tint,
        },
    },
    scene::Scene,
    schedule::{
        LoadOp, RendererBackend, RootRenderTarget, Scheduler, SchedulerState, StripPassRenderTarget,
    },
};
use alloc::vec::Vec;
use alloc::{sync::Arc, vec};
use bytemuck::{Pod, Zeroable};
use core::{fmt::Debug, num::NonZeroU64};
#[cfg(feature = "text")]
use glifo::PendingClearRect;
use vello_common::image_cache::{ImageCache, ImageResource};
use vello_common::multi_atlas::{AtlasConfig, AtlasError, AtlasId};
use vello_common::render_graph::LayerId;
use vello_common::{
    coarse::WideTile,
    encode::{EncodedGradient, EncodedKind, EncodedPaint, MAX_GRADIENT_LUT_SIZE, RadialKind},
    paint::ImageSource,
    peniko,
    pixmap::Pixmap,
    tile::Tile,
};
use wgpu::{
    BindGroup, BindGroupLayout, BlendState, Buffer, ColorTargetState, ColorWrites, CommandEncoder,
    Device, Extent3d, PipelineCompilationOptions, Queue, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, Sampler, Texture, TextureView, TextureViewDescriptor,
    util::DeviceExt,
};

/// Placeholder value for uninitialized GPU encoded paints.
const GPU_PAINT_PLACEHOLDER: GpuEncodedPaint = GpuEncodedPaint::LinearGradient(GpuLinearGradient {
    texture_width_and_extend_mode: 0,
    gradient_start: 0,
    transform: [0.0; 6],
});

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

/// Vello Hybrid's Renderer.
#[derive(Debug)]
pub struct Renderer {
    /// Programs for rendering.
    programs: Programs,
    /// Scheduler for scheduling draws.
    scheduler: Scheduler,
    /// The state used by the scheduler.
    scheduler_state: SchedulerState,
    /// Encoded paints for storing encoded paints.
    encoded_paints: Vec<GpuEncodedPaint>,
    /// Stores the index (offset) of the encoded paints in the encoded paints texture.
    paint_idxs: Vec<u32>,
    /// Gradient cache for storing gradient ramps.
    gradient_cache: GradientRampCache,
    /// Context for GPU filter effects.
    filter_context: FilterContext,
    /// State used for constructing filter passes.
    filter_pass_state: FilterPassState,
    dummy_image_cache: Option<ImageCache>,
    #[cfg(feature = "text")]
    atlas_clear_scratch: Vec<u8>,
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
        let total_slots = (max_texture_dimension_2d / u32::from(Tile::HEIGHT)) as usize;
        let image_cache = ImageCache::new_with_config(settings.atlas_config);
        // Estimate the maximum number of gradient cache entries based on the max texture dimension
        // and the maximum gradient LUT size - worst case scenario.
        let max_gradient_cache_size =
            max_texture_dimension_2d * max_texture_dimension_2d / MAX_GRADIENT_LUT_SIZE as u32;
        let gradient_cache = GradientRampCache::new(max_gradient_cache_size, settings.level);

        let filter_context = FilterContext::new(settings.atlas_config);
        Self {
            programs: Programs::new(
                device,
                &image_cache,
                &filter_context.image_cache,
                render_target_config,
                total_slots,
            ),
            scheduler: Scheduler::new(total_slots),
            scheduler_state: SchedulerState::default(),
            gradient_cache,
            encoded_paints: Vec::new(),
            paint_idxs: Vec::new(),
            filter_context,
            filter_pass_state: FilterPassState::default(),
            dummy_image_cache: Some(ImageCache::new_dummy()),
            #[cfg(feature = "text")]
            atlas_clear_scratch: Vec::new(),
        }
    }

    fn prepare_filter_textures(
        &mut self,
        scene: &Scene,
        device: &Device,
        encoder: &mut CommandEncoder,
        image_cache: &mut ImageCache,
        encoded_paints: &mut Vec<EncodedPaint>,
    ) -> Result<(), AtlasError> {
        // TODO: Maybe we can do the clear implicitly when using the textures for the first time.
        if !self.filter_context.filter_textures.is_empty() {
            for view in &self.programs.resources.filter_atlas.views {
                let _pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("Clear Filter Atlas Texture"),
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
        }

        self.filter_context
            .deallocate_all_and_clear_context(image_cache);

        self.filter_context
            .prepare(&scene.render_graph, image_cache, encoded_paints)?;

        Programs::maybe_resize_atlas_texture_array(
            device,
            encoder,
            &mut self.programs.resources,
            &self.programs.atlas_bind_group_layout,
            image_cache.atlas_count() as u32,
        );
        self.programs.resources.filter_atlas.ensure_count(
            device,
            self.filter_context.image_cache.atlas_count() as u32,
            &self.programs.filter_input_bind_group_layouts[0],
            &self.programs.filter_input_bind_group_layouts[1],
        );

        Ok(())
    }

    /// Render `scene` into the provided command encoder.
    ///
    /// This method creates GPU resources as needed and schedules potentially multiple
    /// render passes.
    pub fn render(
        &mut self,
        scene: &Scene,
        resources: &mut Resources,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        render_size: &RenderSize,
        view: &TextureView,
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

        self.prepare_filter_textures(
            scene,
            device,
            encoder,
            &mut resources.image_cache,
            &mut encoded_paints,
        )?;

        // TODO: Passing `false` here because wgpu swapchain textures likely have
        // undefined initial content, making an explicit clear redundant in the common
        // case. Verify whether there are scenarios where wgpu would need a clear.
        let result = self.render_scene(
            scene,
            device,
            queue,
            encoder,
            render_size,
            view,
            &resources.image_cache,
            &encoded_paints,
            false,
            RootRenderTarget::UserSurface,
        );

        encoded_paints.truncate(scene_paint_count);
        #[cfg(feature = "text")]
        resources.after_render(self, |renderer, rect| {
            clear_atlas_region(queue, renderer, rect);
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
    #[doc(hidden)]
    pub fn render_to_atlas(
        &mut self,
        scene: &Scene,
        atlas_count: u32,
        atlas_config: AtlasConfig,
        device: &Device,
        queue: &Queue,
        atlas_id: AtlasId,
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
        view: &TextureView,
        image_cache: &ImageCache,
        encoded_paints: &[EncodedPaint],
        clear: bool,
        root_output_target: RootRenderTarget,
    ) -> Result<(), RenderError> {
        self.prepare_gpu_encoded_paints(encoded_paints, image_cache);
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
            &self.filter_context,
        );

        if clear {
            Self::clear_view(encoder, view);
        }
        let mut ctx = RendererContext {
            programs: &mut self.programs,
            device,
            queue,
            encoder,
            view,
            image_cache,
            filter_context: &self.filter_context,
            filter_pass_state: &mut self.filter_pass_state,
        };
        self.scheduler.do_scene(
            &mut self.scheduler_state,
            &mut ctx,
            scene,
            root_output_target,
            &self.paint_idxs,
            &self.filter_context,
            encoded_paints,
        )?;
        self.gradient_cache.maintain();

        Ok(())
    }

    /// Clear the view to transparent black.
    // TODO: Investigate adding tests for the clear_view behavior.
    fn clear_view(encoder: &mut CommandEncoder, view: &TextureView) {
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
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        image_id: vello_common::paint::ImageId,
    ) {
        if let Some(image_resource) = resources.image_cache.deallocate(image_id) {
            let padding = image_resource.padding as u32;

            self.clear_atlas_region(
                device,
                queue,
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
        _device: &Device,
        _queue: &Queue,
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
                            let image_paint = self.encode_image_paint(img, image_resource);
                            self.encoded_paints[encoded_paint_idx] = image_paint;
                            current_idx += GPU_ENCODED_IMAGE_SIZE_TEXELS;
                        }
                    }
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
}

#[cfg(feature = "text")]
fn clear_atlas_region(queue: &Queue, renderer: &mut Renderer, rect: &PendingClearRect) {
    // TODO: Can we optimize this more?
    let byte_count = rect.width as usize * rect.height as usize * 4;
    renderer.atlas_clear_scratch.resize(byte_count, 0);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: renderer.atlas_texture(),
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: rect.x as u32,
                y: rect.y as u32,
                z: rect.page_index,
            },
            aspect: wgpu::TextureAspect::All,
        },
        &renderer.atlas_clear_scratch[..byte_count],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(rect.width as u32 * 4),
            rows_per_image: None,
        },
        Extent3d {
            width: rect.width as u32,
            height: rect.height as u32,
            depth_or_array_layers: 1,
        },
    );
}

/// Defines the GPU resources and pipelines for rendering.
#[derive(Debug)]
struct Programs {
    /// Pipelines for rendering strips.
    /// The first pipeline should be used for color attachments in the native pixel format,
    /// the second for color attachments in RGBA8.
    strip_pipelines: [RenderPipeline; 2],
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
    /// Pipeline for applying filter effects.
    filter_pipeline: RenderPipeline,
    /// Bind group layouts for filter input.
    filter_input_bind_group_layouts: [BindGroupLayout; 2],
    /// Pipeline for clearing slots in slot textures.
    clear_pipeline: RenderPipeline,
    /// Pipeline for clearing atlas regions.
    atlas_clear_pipeline: RenderPipeline,
    /// GPU resources for rendering (created during prepare)
    resources: GpuResources,
    /// Dimensions of the rendering target
    render_size: RenderSize,
    /// Scratch buffer for staging encoded paints texture data.
    encoded_paints_data: Vec<u8>,
    /// Scratch buffer for staging filter data texture data.
    filter_data: Vec<u8>,
}

#[derive(Debug)]
struct FilterAtlasState {
    textures: Vec<Texture>,
    views: Vec<TextureView>,
    input_bind_groups: Vec<BindGroup>,
    original_bind_groups: Vec<BindGroup>,
    sampler: Sampler,
    atlas_size: (u32, u32),
}

impl FilterAtlasState {
    fn new(device: &Device, atlas_size: (u32, u32)) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Filter Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            textures: Vec::new(),
            views: Vec::new(),
            input_bind_groups: Vec::new(),
            original_bind_groups: Vec::new(),
            sampler,
            atlas_size,
        }
    }

    fn ensure_count(
        &mut self,
        device: &Device,
        required_count: u32,
        input_layout: &BindGroupLayout,
        original_layout: &BindGroupLayout,
    ) {
        let current_count = self.textures.len() as u32;

        if required_count <= current_count {
            return;
        }
        let (width, height) = self.atlas_size;

        for _ in current_count..required_count {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Filter Atlas Texture"),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor::default());
            let input_bg =
                create_filter_input_bind_group(device, input_layout, &self.sampler, &view);
            let original_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: original_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                }],
            });
            self.textures.push(texture);
            self.views.push(view);
            self.input_bind_groups.push(input_bg);
            self.original_bind_groups.push(original_bg);
        }
    }
}

/// Contains all GPU resources needed for rendering
#[derive(Debug)]
struct GpuResources {
    /// Buffer for [`GpuStrip`] data
    strips_buffer: Buffer,
    /// Texture for alpha values (used by both view and slot rendering)
    alphas_texture: Texture,
    /// Textures for atlas data (multiple atlases supported)
    atlas_texture_array: Texture,
    /// View for atlas texture array
    atlas_texture_array_view: TextureView,
    /// Bind group for atlas textures (as texture array)
    atlas_bind_group: BindGroup,
    /// Filter atlas textures and their associated views/bind groups.
    /// Lazily allocated: stays empty until the first scene with filters.
    filter_atlas: FilterAtlasState,
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

    /// Config buffer for rendering wide tile commands into the view texture.
    view_config_buffer: Buffer,
    /// Config buffer for rendering wide tile commands into a slot texture.
    slot_config_buffer: Buffer,

    /// Buffer for slot indices used in `clear_slots`
    clear_slot_indices_buffer: Buffer,
    /// Buffer holding `FilterInstanceData` for a single filter draw call.
    filter_instance_buffer: Buffer,
    // Bind groups for rendering with clip buffers
    slot_bind_groups: [BindGroup; 3],
    /// Slot texture views
    slot_texture_views: [TextureView; 2],

    /// Bind group for clear slots operation
    clear_bind_group: BindGroup,

    /// Placeholder atlas bind group with a 1x1 dummy texture, used during
    /// `render_to_atlas` to avoid a read-write conflict on the real atlas texture.
    stub_atlas_bind_group: BindGroup,
}

const SIZE_OF_CONFIG: NonZeroU64 = NonZeroU64::new(size_of::<Config>() as u64).unwrap();

/// Config for the clear slots pipeline
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct ClearSlotsConfig {
    /// Width of a slot
    pub slot_width: u32,
    /// Height of a slot
    pub slot_height: u32,
    /// Total height of the texture
    pub texture_height: u32,
    /// Padding for 16-byte alignment
    pub _padding: u32,
}

impl GpuStrip {
    /// Vertex attributes for the strip
    pub fn vertex_attributes() -> [wgpu::VertexAttribute; 5] {
        wgpu::vertex_attr_array![
            0 => Uint32,
            1 => Uint32,
            2 => Uint32,
            3 => Uint32,
            4 => Uint32,
        ]
    }
}

impl Programs {
    fn new(
        device: &Device,
        image_cache: &ImageCache,
        filter_texture_cache: &ImageCache,
        render_target_config: &RenderTargetConfig,
        slot_count: usize,
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
                label: Some("Atlas Texture Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                }],
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

        // Create bind group layout for clearing slots
        let clear_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Clear Slots Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let strip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Strip Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_sparse_shaders::wgsl::RENDER_STRIPS.into()),
        });

        let clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Clear Slots Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_sparse_shaders::wgsl::CLEAR_SLOTS.into()),
        });

        let strip_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Strip Pipeline Layout"),
                bind_group_layouts: &[
                    &strip_bind_group_layout,
                    &atlas_bind_group_layout,
                    &encoded_paints_bind_group_layout,
                    &gradient_bind_group_layout,
                ],
                immediate_size: 0,
            });

        let clear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Clear Slots Pipeline Layout"),
                bind_group_layouts: &[&clear_bind_group_layout],
                immediate_size: 0,
            });

        let strip_formats = [render_target_config.format, wgpu::TextureFormat::Rgba8Unorm];
        let strip_pipelines: [RenderPipeline; 2] = core::array::from_fn(|i| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Strip Pipeline"),
                layout: Some(&strip_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &strip_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: size_of::<GpuStrip>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &GpuStrip::vertex_attributes(),
                    }],
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &strip_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format: strip_formats[i],
                        blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
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
        });

        let clear_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Clear Slots Pipeline"),
            layout: Some(&clear_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &clear_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<u32>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Uint32,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &clear_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: render_target_config.format,
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
                label: Some("Filter Original Bind Group Layout"),
                entries: &[filter_texture_entry],
            }),
        ];

        let filter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Filter Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_sparse_shaders::wgsl::FILTERS.into()),
        });
        let filter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Filter Pipeline Layout"),
                bind_group_layouts: &[
                    &filter_bind_group_layout,
                    &filter_input_bind_group_layouts[0],
                    &filter_input_bind_group_layouts[1],
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

        let slot_texture_views: [TextureView; 2] = core::array::from_fn(|_| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("Slot Texture"),
                    size: Extent3d {
                        width: u32::from(WideTile::WIDTH),
                        height: u32::from(Tile::HEIGHT) * slot_count as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: render_target_config.format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor::default())
        });

        let clear_config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Clear Slots Config"),
            contents: bytemuck::bytes_of(&ClearSlotsConfig {
                slot_width: u32::from(WideTile::WIDTH),
                slot_height: u32::from(Tile::HEIGHT),
                texture_height: u32::from(Tile::HEIGHT) * slot_count as u32,
                _padding: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let clear_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Clear Slots Bind Group"),
            layout: &clear_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: clear_config_buffer.as_entire_binding(),
            }],
        });
        let clear_slot_indices_buffer = Self::create_clear_slot_indices_buffer(
            device,
            slot_count as u64 * size_of::<u32>() as u64,
        );

        let slot_config_buffer = Self::create_config_buffer(
            device,
            &RenderSize {
                width: u32::from(WideTile::WIDTH),
                height: u32::from(Tile::HEIGHT) * slot_count as u32,
            },
            device.limits().max_texture_dimension_2d,
        );

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
        let atlas_bind_group = Self::create_atlas_bind_group(
            device,
            &atlas_bind_group_layout,
            &atlas_texture_array_view,
        );

        // Create a 1x1 stub atlas texture array for use during render_to_atlas.
        // This avoids the read-write conflict that occurs when the real atlas is both
        // a shader input (bind group) and render target in the same pass.
        let (_stub_atlas_texture, stub_atlas_view) =
            Self::create_atlas_texture_array(device, 1, 1, 1);
        let stub_atlas_bind_group =
            Self::create_atlas_bind_group(device, &atlas_bind_group_layout, &stub_atlas_view);

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

        let AtlasConfig {
            atlas_size: (filter_atlas_width, filter_atlas_height),
            ..
        } = filter_texture_cache.atlas_manager().config();
        let filter_atlas_size = (*filter_atlas_width, *filter_atlas_height);

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

        let filter_atlas = FilterAtlasState::new(device, filter_atlas_size);

        let slot_bind_groups = Self::create_strip_bind_groups(
            device,
            &strip_bind_group_layout,
            &alphas_texture.create_view(&TextureViewDescriptor::default()),
            &slot_config_buffer,
            &view_config_buffer,
            &slot_texture_views,
        );

        let resources = GpuResources {
            strips_buffer: Self::create_strips_buffer(device, 0),
            clear_slot_indices_buffer,
            filter_instance_buffer: Self::create_filter_instance_buffer(
                device,
                size_of::<FilterInstanceData>() as u64,
            ),
            slot_texture_views,
            slot_config_buffer,
            slot_bind_groups,
            clear_bind_group,
            alphas_texture,
            atlas_texture_array,
            atlas_texture_array_view,
            atlas_bind_group,
            filter_atlas,
            stub_atlas_bind_group,
            encoded_paints_texture,
            encoded_paints_bind_group,
            gradient_texture,
            gradient_bind_group,
            filter_data_texture,
            filter_base_bind_group,
            view_config_buffer,
        };

        Self {
            strip_pipelines,
            strip_bind_group_layout,
            encoded_paints_bind_group_layout,
            gradient_bind_group_layout,
            atlas_bind_group_layout,
            filter_bind_group_layout,
            filter_pipeline,
            filter_input_bind_group_layouts,
            resources,
            encoded_paints_data,
            filter_data,
            render_size: RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },
            clear_pipeline,
            atlas_clear_pipeline,
        }
    }

    fn create_strips_buffer(device: &Device, required_strips_size: u64) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Strips Buffer"),
            size: required_strips_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_clear_slot_indices_buffer(device: &Device, required_size: u64) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Slot Indices Buffer"),
            size: required_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_filter_instance_buffer(device: &Device, required_size: u64) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Filter Instance Buffer"),
            size: required_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_config_buffer(
        device: &Device,
        render_size: &RenderSize,
        alpha_texture_width: u32,
    ) -> Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::bytes_of(&Config {
                width: render_size.width,
                height: render_size.height,
                strip_height: Tile::HEIGHT.into(),
                alphas_tex_width_bits: alpha_texture_width.trailing_zeros(),
                encoded_paints_tex_width_bits: alpha_texture_width.trailing_zeros(),
                strip_offset_x: 0,
                strip_offset_y: 0,
                negate_ndc: 0,
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
    ) -> (Texture, TextureView) {
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
        filter_texture_view: &TextureView,
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

    fn create_atlas_bind_group(
        device: &Device,
        atlas_bind_group_layout: &BindGroupLayout,
        atlas_texture_array_view: &TextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Atlas Bind Group"),
            layout: atlas_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(atlas_texture_array_view),
            }],
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
        encoded_paints_texture_view: &TextureView,
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
        gradient_texture_view: &TextureView,
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

    fn create_strip_bind_groups(
        device: &Device,
        strip_bind_group_layout: &BindGroupLayout,
        alphas_texture_view: &TextureView,
        strip_config_buffer: &Buffer,
        config_buffer: &Buffer,
        strip_texture_views: &[TextureView],
    ) -> [BindGroup; 3] {
        [
            Self::create_strip_bind_group(
                device,
                strip_bind_group_layout,
                alphas_texture_view,
                strip_config_buffer,
                &strip_texture_views[1],
            ),
            Self::create_strip_bind_group(
                device,
                strip_bind_group_layout,
                alphas_texture_view,
                strip_config_buffer,
                &strip_texture_views[0],
            ),
            Self::create_strip_bind_group(
                device,
                strip_bind_group_layout,
                alphas_texture_view,
                config_buffer,
                &strip_texture_views[1],
            ),
        ]
    }

    fn create_strip_bind_group(
        device: &Device,
        strip_bind_group_layout: &BindGroupLayout,
        alphas_texture_view: &TextureView,
        config_buffer: &Buffer,
        strip_texture_view: &TextureView,
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
        self.maybe_update_config_buffer(queue, max_texture_dimension_2d, new_render_size);

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

            // Since the alpha texture has changed, we need to update the clip bind groups.
            self.resources.slot_bind_groups = Self::create_strip_bind_groups(
                device,
                &self.strip_bind_group_layout,
                &self
                    .resources
                    .alphas_texture
                    .create_view(&TextureViewDescriptor::default()),
                &self.resources.slot_config_buffer,
                &self.resources.view_config_buffer,
                &self.resources.slot_texture_views,
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
            let new_atlas_bind_group = Self::create_atlas_bind_group(
                device,
                atlas_bind_group_layout,
                &new_atlas_texture_array_view,
            );

            // Replace the old resources
            resources.atlas_texture_array = new_atlas_texture_array;
            resources.atlas_texture_array_view = new_atlas_texture_array_view;
            resources.atlas_bind_group = new_atlas_bind_group;
        }
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

    /// Upload the strip data by creating and assigning a new `self.resources.strips_buffer`.
    fn upload_strips(&mut self, device: &Device, queue: &Queue, strips: &[GpuStrip]) {
        let required_strips_size = size_of_val(strips) as u64;
        self.resources.strips_buffer = Self::create_strips_buffer(device, required_strips_size);
        // TODO: Consider using a staging belt to avoid an extra staging buffer allocation.
        let mut buffer = queue
            .write_buffer_with(
                &self.resources.strips_buffer,
                0,
                required_strips_size.try_into().unwrap(),
            )
            .expect("Capacity handled in creation");
        buffer.copy_from_slice(bytemuck::cast_slice(strips));
    }
}

/// A struct containing references to the many objects needed to get work
/// scheduled onto the GPU.
struct RendererContext<'a> {
    programs: &'a mut Programs,
    device: &'a Device,
    queue: &'a Queue,
    encoder: &'a mut CommandEncoder,
    view: &'a TextureView,
    image_cache: &'a ImageCache,
    filter_context: &'a FilterContext,
    filter_pass_state: &'a mut FilterPassState,
}

impl RendererContext<'_> {
    /// Render the strips to the specified render target.
    fn do_strip_render_pass(
        &mut self,
        strips: &[GpuStrip],
        target: StripPassRenderTarget,
        load: wgpu::LoadOp<wgpu::Color>,
    ) {
        if strips.is_empty() {
            return;
        }
        // TODO: We currently allocate a new strips buffer for each render pass. A more efficient
        // approach would be to re-use buffers or slices of a larger buffer.
        self.programs.upload_strips(self.device, self.queue, strips);

        enum MaybeOwned<'a, T> {
            Borrowed(&'a T),
            Owned(T),
        }

        impl<T> AsRef<T> for MaybeOwned<'_, T> {
            fn as_ref(&self) -> &T {
                match self {
                    Self::Borrowed(r) => r,
                    Self::Owned(v) => v,
                }
            }
        }
        let (view, bind_group, scissor_rect): (
            &TextureView,
            MaybeOwned<'_, BindGroup>,
            Option<[u32; 4]>,
        ) = match target {
            StripPassRenderTarget::Root(_) => (
                self.view,
                MaybeOwned::Borrowed(&self.programs.resources.slot_bind_groups[2]),
                None,
            ),
            StripPassRenderTarget::FilterLayer(layer_id) => {
                let image_id = self
                    .filter_context
                    .filter_textures
                    .get(&layer_id)
                    .unwrap()
                    .initial_image_id;
                let resources = self.filter_context.image_cache.get(image_id).unwrap();

                let atlas_idx = resources.atlas_id.as_u32() as usize;
                let filter_atlas = &self.programs.resources.filter_atlas;

                let atlas_size = filter_atlas.textures[atlas_idx].size();
                let filter_textures = self.filter_context.filter_textures.get(&layer_id).unwrap();

                // Two offsets are needed:
                // 1. Account for the intermediate texture living at an offset within its atlas.
                // 2. Account for the filter layer bbox not starting at (0, 0).
                let strip_offset_x = resources.offset[0] as i32
                    - (filter_textures.bbox.x0() * WideTile::WIDTH) as i32;
                let strip_offset_y =
                    resources.offset[1] as i32 - (filter_textures.bbox.y0() * Tile::HEIGHT) as i32;

                // TODO: Cache this and bind group? See https://github.com/linebender/vello/pull/1494#discussion_r2937895891.
                let atlas_config_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Filter Strip Config Buffer"),
                            contents: bytemuck::bytes_of(&Config {
                                width: atlas_size.width,
                                height: atlas_size.height,
                                strip_height: Tile::HEIGHT.into(),
                                alphas_tex_width_bits: self
                                    .programs
                                    .resources
                                    .alphas_texture
                                    .size()
                                    .width
                                    .trailing_zeros(),
                                encoded_paints_tex_width_bits: self
                                    .programs
                                    .resources
                                    .alphas_texture
                                    .size()
                                    .width
                                    .trailing_zeros(),
                                strip_offset_x,
                                strip_offset_y,
                                negate_ndc: 0,
                            }),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Filter Strip Bind Group"),
                    layout: &self.programs.strip_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self
                                    .programs
                                    .resources
                                    .alphas_texture
                                    .create_view(&TextureViewDescriptor::default()),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: atlas_config_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                &self.programs.resources.slot_texture_views[1],
                            ),
                        },
                    ],
                });

                (
                    &filter_atlas.views[atlas_idx],
                    MaybeOwned::Owned(bind_group),
                    Some([
                        resources.offset[0] as u32,
                        resources.offset[1] as u32,
                        resources.width as u32,
                        resources.height as u32,
                    ]),
                )
            }
            StripPassRenderTarget::SlotTexture(idx) => (
                &self.programs.resources.slot_texture_views[idx as usize],
                MaybeOwned::Borrowed(&self.programs.resources.slot_bind_groups[idx as usize]),
                None,
            ),
        };

        let pipeline_idx = if matches!(
            target,
            StripPassRenderTarget::Root(RootRenderTarget::AtlasLayer)
                | StripPassRenderTarget::FilterLayer(_)
        ) {
            1
        } else {
            0
        };

        let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render to Texture Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        if let Some([x, y, width, height]) = scissor_rect {
            render_pass.set_scissor_rect(x, y, width, height);
        }
        render_pass.set_pipeline(&self.programs.strip_pipelines[pipeline_idx]);
        render_pass.set_bind_group(0, bind_group.as_ref(), &[]);
        render_pass.set_bind_group(1, &self.programs.resources.atlas_bind_group, &[]);
        render_pass.set_bind_group(2, &self.programs.resources.encoded_paints_bind_group, &[]);
        render_pass.set_bind_group(3, &self.programs.resources.gradient_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.programs.resources.strips_buffer.slice(..));
        render_pass.draw(0..4, 0..u32::try_from(strips.len()).unwrap());
    }

    /// Clear specific slots from a slot texture.
    fn do_clear_slots_render_pass(&mut self, ix: usize, slot_indices: &[u32]) {
        if slot_indices.is_empty() {
            return;
        }

        let resources = &mut self.programs.resources;
        let size = size_of_val(slot_indices) as u64;
        // TODO: We currently allocate a new strips buffer for each render pass. A more efficient
        // approach would be to re-use buffers or slices of a larger buffer.
        resources.clear_slot_indices_buffer =
            Programs::create_clear_slot_indices_buffer(self.device, size);
        // TODO: Consider using a staging belt to avoid an extra staging buffer allocation.
        let mut buffer = self
            .queue
            .write_buffer_with(
                &resources.clear_slot_indices_buffer,
                0,
                size.try_into().unwrap(),
            )
            .expect("Capacity handled in creation");
        buffer.copy_from_slice(bytemuck::cast_slice(slot_indices));

        {
            let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Clear Slots Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &resources.slot_texture_views[ix],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Don't clear the entire texture, just specific slots
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
            render_pass.set_bind_group(0, &resources.clear_bind_group, &[]);
            render_pass.set_vertex_buffer(0, resources.clear_slot_indices_buffer.slice(..));
            render_pass.draw(0..4, 0..u32::try_from(slot_indices.len()).unwrap());
        }
    }
}

impl RendererBackend for RendererContext<'_> {
    /// Execute the render pass for clearing slots.
    fn clear_slots(&mut self, texture_index: usize, slots: &[u32]) {
        self.do_clear_slots_render_pass(texture_index, slots);
    }

    /// Execute the render pass for rendering strips.
    fn render_strips(
        &mut self,
        strips: &[GpuStrip],
        target: StripPassRenderTarget,
        load_op: LoadOp,
    ) {
        let wgpu_load_op = match load_op {
            LoadOp::Load => wgpu::LoadOp::Load,
            LoadOp::Clear => wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        };
        self.do_strip_render_pass(strips, target, wgpu_load_op);
    }

    fn apply_filter(&mut self, layer_id: LayerId) {
        let filter_atlas = &self.programs.resources.filter_atlas;
        self.filter_context.build_filter_passes(
            self.filter_pass_state,
            &layer_id,
            self.image_cache,
            |atlas_idx| {
                let size = filter_atlas.textures[atlas_idx as usize].size();
                [size.width, size.height]
            },
            || {
                let size = self.programs.resources.atlas_texture_array.size();
                [size.width, size.height]
            },
        );

        let filter_passes = self.filter_pass_state.filter_passes();
        if filter_passes.is_empty() {
            return;
        }

        let instances = self.filter_pass_state.instances();
        let instance_stride = size_of::<FilterInstanceData>() as u64;
        let total_size = instances.len() as u64 * instance_stride;
        // TODO: Reuse buffer (https://github.com/linebender/vello/pull/1494#discussion_r2937890819)
        self.programs.resources.filter_instance_buffer =
            Programs::create_filter_instance_buffer(self.device, total_size);
        self.queue.write_buffer(
            &self.programs.resources.filter_instance_buffer,
            0,
            bytemuck::cast_slice(instances),
        );

        let programs = &self.programs;
        let encoder = &mut self.encoder;
        let filter_atlas = &programs.resources.filter_atlas;
        for (i, pass) in filter_passes.iter().enumerate() {
            let input_bg = &filter_atlas.input_bind_groups[pass.input_atlas_idx as usize];
            // If this is `None`, it's unused, so we can just pass anything here.
            let original_idx = pass.original_atlas_idx.unwrap_or(pass.input_atlas_idx) as usize;
            let original_bg = &filter_atlas.original_bind_groups[original_idx];

            let (output_view, target_width, target_height) = match &pass.output {
                FilterPassTarget::FilterAtlas(idx) => {
                    let size = filter_atlas.textures[*idx as usize].size();
                    (&filter_atlas.views[*idx as usize], size.width, size.height)
                }
                FilterPassTarget::MainAtlas(idx) => {
                    let size = programs.resources.atlas_texture_array.size();
                    (
                        &create_atlas_layer_view(&programs.resources.atlas_texture_array, *idx),
                        size.width,
                        size.height,
                    )
                }
            };

            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Apply Filter Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
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
            let instance = &instances[i];
            let [x, y, width, height] = instance.scissor_rect([target_width, target_height]);
            render_pass.set_scissor_rect(x, y, width, height);
            render_pass.set_pipeline(&programs.filter_pipeline);
            render_pass.set_bind_group(0, &programs.resources.filter_base_bind_group, &[]);
            render_pass.set_bind_group(1, input_bg, &[]);
            render_pass.set_bind_group(2, original_bg, &[]);
            render_pass.set_vertex_buffer(
                0,
                programs
                    .resources
                    .filter_instance_buffer
                    .slice((i as u64 * instance_stride)..((i as u64 + 1) * instance_stride)),
            );
            render_pass.draw(0..4, 0..1);
        }
    }
}

fn create_filter_input_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    sampler: &Sampler,
    texture_view: &TextureView,
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

fn create_atlas_layer_view(atlas: &Texture, layer: u32) -> TextureView {
    atlas.create_view(&TextureViewDescriptor {
        label: Some("Atlas Layer View"),
        format: None,
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: layer,
        array_layer_count: Some(1),
        usage: None,
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
