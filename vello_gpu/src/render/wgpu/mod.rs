// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `wgpu` backend for Vello GPU.

#![expect(
    clippy::cast_possible_truncation,
    reason = "We temporarily ignore those because the casts\
only break in edge cases, and some of them are also only related to conversions from f64 to f32."
)]

mod atlas;
mod bindings;
mod passes;
#[cfg(feature = "probe")]
mod probe;
mod resources;

pub use atlas::AtlasWriter;
pub use bindings::{TextureBinding, TextureBindings};
#[cfg(feature = "probe")]
pub use probe::{WgpuPendingProbe, WgpuProbeError, WgpuProbeStatus};

use crate::render::common::{IMAGE_PADDING, ScenePreparer};
use crate::{
    ClearSettings, GpuStrip, RenderError, RenderSettings, RenderSize, Resources, TargetInit,
    render::common::DeviceLimits, scene::Scene, schedule::IntermediateTextureAllocations,
    target::RootTarget,
};
use alloc::vec::Vec;
use core::fmt::Debug;
#[cfg(feature = "text")]
use glifo::PendingClearRect;
use hashbrown::HashMap;
use passes::{Programs, RendererContext};
use resources::GpuResources;
use vello_common::geometry::{RectU16, SizeU16};
use vello_common::image_cache::ImageCache;
use vello_common::multi_atlas::{AtlasConfig, AtlasId};
use wgpu::{
    CommandEncoder, Device, Extent3d, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    Texture, TextureView, TextureViewDescriptor,
};

/// Options for the renderer
#[derive(Debug)]
pub struct RenderTargetConfig {
    /// Format of the rendering target
    pub format: wgpu::TextureFormat,
    /// Width of the rendering target
    pub width: u16,
    /// Height of the rendering target
    pub height: u16,
}

/// The target that a scene is rendered into.
#[derive(Debug, Clone, Copy)]
pub struct RenderTarget<'a> {
    /// The color attachment to render into.
    ///
    /// The view must cover mip level 0 of its texture.
    pub view: &'a TextureView,
    /// The optional depth attachment.
    ///
    /// Passing `Some(depth_view)` enables depth buffering to reduce overdraw. The depth view must
    /// use [`TextureFormat::Depth24Plus`](wgpu::TextureFormat::Depth24Plus), have the same
    /// dimensions as `view`, have a sample count of one, and support
    /// [`TextureUsages::RENDER_ATTACHMENT`](wgpu::TextureUsages::RENDER_ATTACHMENT). Pass `None`
    /// to render without a depth buffer.
    ///
    /// The [WebGPU render pass validation rules] require every color and depth attachment view in
    /// a render pass to have matching render extents.
    ///
    /// [WebGPU render pass validation rules]: https://gpuweb.github.io/gpuweb/#abstract-opdef-gpurenderpassdescriptor-valid-usage
    pub depth_view: Option<&'a TextureView>,
}

impl RenderTarget<'_> {
    fn size(&self) -> RenderSize {
        let texture = self.view.texture();
        RenderSize {
            width: texture.width().try_into().unwrap(),
            height: texture.height().try_into().unwrap(),
        }
    }
}

/// Options for [`Renderer::render`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct RenderOptions<'a> {
    /// How existing target contents are handled before drawing.
    pub target_init: TargetInit<'a>,
    /// The position of the scene's origin in the target.
    ///
    /// Clear rectangles are not offset.
    pub offset: (i32, i32),
}

impl Default for RenderOptions<'_> {
    fn default() -> Self {
        Self {
            target_init: TargetInit::Clear(ClearSettings::default()),
            offset: (0, 0),
        }
    }
}

/// Vello GPU's renderer.
#[derive(Debug)]
pub struct Renderer {
    /// The device that all GPU resources of the renderer are created on.
    device: Device,
    /// The queue of `device`.
    queue: Queue,
    /// Pipelines and bind group layouts of the render passes.
    programs: Programs,
    /// GPU resources for rendering.
    resources: GpuResources,
    /// Encodes scenes and schedules their rendering.
    scene_preparer: ScenePreparer,
    /// Format of the render target that the pipelines are created for.
    #[cfg_attr(
        not(feature = "probe"),
        expect(dead_code, reason = "only read when probing")
    )]
    target_format: wgpu::TextureFormat,
    #[cfg(feature = "text")]
    atlas_clear_scratch: Vec<u8>,
}

impl Renderer {
    /// Creates a new renderer and its persistent resources.
    pub fn new(
        device: &Device,
        queue: &Queue,
        render_target_config: &RenderTargetConfig,
    ) -> (Self, Resources) {
        Self::new_with(
            device,
            queue,
            render_target_config,
            RenderSettings::default(),
        )
    }

    /// Creates a new renderer and its persistent resources with specific settings.
    pub fn new_with(
        device: &Device,
        queue: &Queue,
        render_target_config: &RenderTargetConfig,
        settings: RenderSettings,
    ) -> (Self, Resources) {
        super::common::maybe_warn_about_webgl_feature_conflict();

        let mut settings = settings;
        let limits = device.limits();
        let device_limits = DeviceLimits {
            max_texture_dimension_2d: u16::try_from(limits.max_texture_dimension_2d)
                .unwrap_or(u16::MAX),
        };
        let (scene_preparer, resources) = ScenePreparer::new(&mut settings, &device_limits);
        let resource_texture_dimension_2d = device_limits.resource_texture_dimension_2d();

        let programs = Programs::new(
            device,
            render_target_config,
            settings.pipeline_cache.as_ref(),
        );
        let gpu_resources = GpuResources::new(
            device,
            &programs,
            &resources.image_cache,
            render_target_config,
            scene_preparer.layers_config,
            resource_texture_dimension_2d,
        );
        let renderer = Self {
            device: device.clone(),
            queue: queue.clone(),
            programs,
            resources: gpu_resources,
            scene_preparer,
            target_format: render_target_config.format,
            #[cfg(feature = "text")]
            atlas_clear_scratch: Vec::new(),
        };

        (renderer, resources)
    }

    /// Creates a depth texture view compatible with [`render`](Self::render).
    ///
    /// The returned view has the same dimensions as `render_size`, uses
    /// [`TextureFormat::Depth24Plus`](wgpu::TextureFormat::Depth24Plus), and can be used as a
    /// render attachment.
    pub fn create_depth_texture_view(&self, render_size: &RenderSize) -> TextureView {
        self.device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Vello GPU Depth Texture"),
                size: Extent3d {
                    width: u32::from(render_size.width.max(1)),
                    height: u32::from(render_size.height.max(1)),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth24Plus,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&TextureViewDescriptor::default())
    }

    /// Render `scene` into `target`, recording the GPU commands into `encoder`.
    ///
    /// Every [`TextureId`] referenced by the scene must have a binding; this returns
    /// [`RenderError::MissingTextureBinding`] otherwise. See [`TextureBindings::insert`] for the
    /// requirements on the bound texture views.
    ///
    /// To render without any texture bindings, you can pass an empty [`TextureBindings`].
    ///
    /// The commands are recorded into `encoder`, which must be submitted to the renderer's queue.
    /// If this is not done, subsequent calls to `render` might produce invalid rendering.
    ///
    /// [`TextureId`]: vello_common::TextureId
    pub fn render(
        &mut self,
        scene: &Scene,
        resources: &mut Resources,
        encoder: &mut CommandEncoder,
        target: RenderTarget<'_>,
        texture_bindings: &TextureBindings,
        options: RenderOptions<'_>,
    ) -> Result<(), RenderError> {
        #[cfg(feature = "text")]
        {
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
                        encoder,
                        upload.image_id,
                        &upload.pixmap,
                        Some([dst_x, dst_y]),
                    );

                    Ok(())
                },
            )?;
        }

        let result = self.render_scene(
            scene,
            encoder,
            &target.size(),
            target.view,
            target.depth_view,
            &resources.image_cache,
            options.target_init,
            RootTarget::UserSurface,
            texture_bindings,
            options.offset,
        );

        #[cfg(feature = "text")]
        resources.after_render(self, |renderer, rect| {
            clear_atlas_region(renderer, rect);

            Ok::<(), RenderError>(())
        })?;
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
    /// [`render`](Self::render) call, since the two methods share GPU resources.
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
        atlas_id: AtlasId,
        texture_bindings: &TextureBindings,
    ) -> Result<(), RenderError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render to Atlas Encoder"),
            });

        self.resources
            .maybe_create_atlas_textures(&self.device, atlas_count);

        let (atlas_width, atlas_height) = atlas_config.atlas_size;
        let atlas_render_size = RenderSize {
            width: atlas_width,
            height: atlas_height,
        };

        let layer_view = self.resources.atlas_texture_views[atlas_id.as_u32() as usize].clone();

        let dummy_image_cache = self
            .scene_preparer
            .dummy_image_cache
            .take()
            .expect("dummy image cache must exist");
        let result = self.render_scene(
            scene,
            &mut encoder,
            &atlas_render_size,
            &layer_view,
            None,
            &dummy_image_cache,
            TargetInit::SrcOver,
            RootTarget::AtlasLayer,
            texture_bindings,
            (0, 0),
        );
        self.scene_preparer.dummy_image_cache = Some(dummy_image_cache);

        // Submit immediately so the atlas content is committed before subsequent
        // render() calls overwrite the shared alpha/config/paint resources.
        self.queue.submit(Some(encoder.finish()));

        result
    }

    fn current_allocations(&self) -> IntermediateTextureAllocations {
        IntermediateTextureAllocations {
            layer_pages: self.resources.layer_textures.each_ref().map(Vec::len),
            scratch: self.resources.scratch_texture.is_some(),
        }
    }

    /// Shared render pipeline: prepares GPU resources, runs the scheduler against
    /// the provided `view` at `render_size`, and maintains caches.
    fn render_scene(
        &mut self,
        scene: &Scene,
        encoder: &mut CommandEncoder,
        render_size: &RenderSize,
        view: &TextureView,
        depth_view: Option<&TextureView>,
        image_cache: &ImageCache,
        target_init: TargetInit<'_>,
        root_output_target: RootTarget,
        texture_bindings: &TextureBindings,
        scene_offset: (i32, i32),
    ) -> Result<(), RenderError> {
        self.resources.depth_cleared_this_frame = false;
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
            depth_view.is_some(),
            self.resources.texture_size,
            current_allocations,
            |id, image, region| texture_bindings.encode_paint(id, image, region, view.texture()),
        )?;
        self.resources
            .prepare_intermediate_textures(&self.device, &self.programs, &schedule);
        // TODO: For the time being, we upload the entire alpha buffer as one big chunk. As a future
        // refinement, we could have a bounded alpha buffer, and break draws when the alpha
        // buffer fills.
        let draw_buffers = &self.scene_preparer.schedule_storage.buffers.draw_buffers;
        let strip_count = draw_buffers.opaque.strips().len() + draw_buffers.strips.len();
        let strip_bytes = (strip_count * size_of::<GpuStrip>()) as u64;
        self.resources.prepare(
            &self.device,
            &self.programs,
            encoder,
            &mut self.scene_preparer.gradient_cache,
            &self.scene_preparer.encoded_paints,
            &scene.strip_storage.borrow().alphas,
            render_size,
            scene_offset,
            &self.scene_preparer.paint_idxs,
            &self.scene_preparer.schedule_storage.filter_context,
        )?;

        self.resources
            .strips_arena
            .begin_frame(&self.device, strip_bytes);

        let mut ctx = RendererContext {
            programs: &self.programs,
            resources: &mut self.resources,
            device: &self.device,
            encoder,
            view,
            depth_view,
            texture_bindings,
            external_texture_bind_groups: HashMap::new(),
            scratch_buffers: &mut self.scene_preparer.scratch_buffers,
            root_load_op: wgpu::LoadOp::Load,
        };

        ctx.init_root_clear(target_init, root_output_target);

        crate::schedule::execute(
            &mut ctx,
            &mut self.scene_preparer.schedule_storage,
            schedule,
            root_output_target,
        )
        .unwrap_or_else(|error| match error {});

        ctx.finish_root_clear();

        self.resources
            .staging_belt
            .belt
            .finish_and_recall_on_submit(encoder);

        Ok(())
    }

    /// Upload image to cache and atlas in one step. Returns the `ImageId`.
    ///
    /// It's used when an image is not already in the cache.
    ///
    /// This is a convenience method that:
    /// 1. Reserves space in the image cache
    /// 2. Writes the image data directly to the atlas
    /// 3. Returns the `ImageId` for use in rendering
    #[track_caller]
    pub fn upload_image<T: AtlasWriter>(
        &mut self,
        resources: &mut Resources,
        encoder: &mut CommandEncoder,
        writer: &T,
    ) -> vello_common::paint::ImageId {
        self.upload_image_with(&mut resources.image_cache, encoder, writer, IMAGE_PADDING)
    }

    #[track_caller]
    pub(crate) fn upload_image_with<T: AtlasWriter>(
        &mut self,
        image_cache: &mut ImageCache,
        encoder: &mut CommandEncoder,
        writer: &T,
        padding: u16,
    ) -> vello_common::paint::ImageId {
        if !cfg!(feature = "full_image_support") {
            panic!("uploaded images require the `full_image_support` feature");
        }

        let width = writer.width();
        let height = writer.height();
        let image_id = image_cache.allocate(width, height, padding).unwrap();
        self.write_to_atlas(image_cache, encoder, image_id, writer, None);
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
        encoder: &mut CommandEncoder,
        image_id: vello_common::paint::ImageId,
        writer: &T,
        offset_override: Option<[u16; 2]>,
    ) {
        let image_resource = image_cache.get(image_id).expect("Image resource not found");

        self.resources
            .maybe_create_atlas_textures(&self.device, image_cache.atlas_count() as u32);
        let offset = offset_override.unwrap_or(image_resource.offset);
        let atlas_texture =
            &self.resources.atlas_textures[image_resource.atlas_id.as_u32() as usize];
        writer.write_to_atlas(
            &self.device,
            &self.queue,
            encoder,
            atlas_texture,
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
            let padding = image_resource.padding;

            self.clear_atlas_region(
                encoder,
                image_resource.atlas_id,
                [
                    image_resource.offset[0] - padding,
                    image_resource.offset[1] - padding,
                ],
                image_resource.width + padding * 2,
                image_resource.height + padding * 2,
            );
        }
    }

    /// Returns an individual image atlas texture.
    pub fn atlas_texture(&self, atlas_id: AtlasId) -> &Texture {
        self.resources
            .atlas_textures
            .get(atlas_id.as_u32() as usize)
            .unwrap()
    }

    /// Clear a specific region of the atlas texture.
    fn clear_atlas_region(
        &mut self,
        encoder: &mut CommandEncoder,
        atlas_id: AtlasId,
        offset: [u16; 2],
        width: u16,
        height: u16,
    ) {
        let layer_view = self.resources.atlas_texture_views[atlas_id.as_u32() as usize].clone();

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
        render_pass.set_scissor_rect(
            u32::from(offset[0]),
            u32::from(offset[1]),
            u32::from(width),
            u32::from(height),
        );
        // Use atlas clear pipeline to render transparent pixels
        render_pass.set_pipeline(&self.programs.clear.atlas);
        // Draw fullscreen quad
        render_pass.draw(0..4, 0..1);
    }
}

#[cfg(feature = "text")]
fn clear_atlas_region(renderer: &mut Renderer, rect: &PendingClearRect) {
    // TODO: Can we optimize this more?
    let byte_count = rect.width as usize * rect.height as usize * 4;
    renderer.atlas_clear_scratch.resize(byte_count, 0);
    renderer.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: renderer.atlas_texture(AtlasId::new(rect.page_index)),
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: rect.x as u32,
                y: rect.y as u32,
                z: 0,
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
