// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Renderer state and steps shared by all backends.

use super::paint::{
    GpuEncodedPaint, encode_blurred_rounded_rect_paint, encode_gradient_paint, encode_image_paint,
};
use super::{DeviceLimits, ScratchBuffers};
use crate::{
    LayersConfig, RenderError, RenderSettings, Resources,
    gradient_cache::GradientRampCache,
    paint::PaintResolver,
    scene::Scene,
    schedule::{IntermediateTextureAllocations, Schedule, ScheduleStorage},
    target::RootTarget,
};
use alloc::vec::Vec;
use vello_common::TextureId;
use vello_common::encode::{EncodedImage, EncodedPaint, MAX_GRADIENT_LUT_SIZE};
use vello_common::geometry::{RectU16, SizeU16};
use vello_common::image_cache::ImageCache;
use vello_common::paint::{ImageSource, TextureRegion};

/// Renderer state that doesn't depend on the backend.
#[derive(Debug)]
pub(crate) struct Core {
    /// GPU encodings of the scene's encoded paints, in the same order.
    pub(crate) encoded_paints: Vec<GpuEncodedPaint>,
    /// Stores the index (offset) of the encoded paints in the encoded paints texture.
    pub(crate) paint_idxs: Vec<u32>,
    /// Gradient cache for storing gradient ramps.
    pub(crate) gradient_cache: GradientRampCache,
    pub(crate) dummy_image_cache: Option<ImageCache>,
    pub(crate) schedule_storage: ScheduleStorage,
    pub(crate) scratch_buffers: ScratchBuffers,
    pub(crate) layers_config: LayersConfig,
}

impl Core {
    /// Normalizes the memory settings in `settings` to `device_limits`, then creates the renderer
    /// state and the persistent resources.
    pub(crate) fn new(
        settings: &mut RenderSettings,
        device_limits: &DeviceLimits,
    ) -> (Self, Resources) {
        settings.memory_settings.normalize(device_limits);
        let resources = Resources::new(settings.memory_settings.image_atlas_config);
        let resource_texture_dimension_2d = device_limits.resource_texture_dimension_2d();
        // Estimate the maximum number of gradient cache entries based on the resource texture
        // dimension and the maximum gradient LUT size - worst case scenario.
        let max_gradient_cache_size = resource_texture_dimension_2d * resource_texture_dimension_2d
            / u32::try_from(MAX_GRADIENT_LUT_SIZE).unwrap();

        let core = Self {
            encoded_paints: Vec::new(),
            paint_idxs: Vec::new(),
            gradient_cache: GradientRampCache::new(max_gradient_cache_size, settings.level),
            dummy_image_cache: Some(ImageCache::new_dummy()),
            schedule_storage: ScheduleStorage::default(),
            scratch_buffers: ScratchBuffers::default(),
            layers_config: settings.memory_settings.layers_config,
        };

        (core, resources)
    }

    /// Encodes `encoded_paints` for the GPU.
    ///
    /// `encode_external_texture` encodes image paints that sample an external texture, given the
    /// texture, the image and the sampled region of the texture.
    pub(crate) fn prepare_paints(
        &mut self,
        encoded_paints: &[EncodedPaint],
        image_cache: &ImageCache,
        mut encode_external_texture: impl FnMut(
            TextureId,
            &EncodedImage,
            TextureRegion,
        ) -> Result<GpuEncodedPaint, RenderError>,
    ) -> Result<(), RenderError> {
        self.encoded_paints
            .resize_with(encoded_paints.len(), || GpuEncodedPaint::PLACEHOLDER);
        self.paint_idxs.resize(encoded_paints.len() + 1, 0);

        let mut current_idx = 0;
        for (encoded_paint_idx, paint) in encoded_paints.iter().enumerate() {
            self.paint_idxs[encoded_paint_idx] = current_idx;
            let gpu_paint = match paint {
                EncodedPaint::Image(img) => match &img.source {
                    ImageSource::OpaqueId { id, .. } => {
                        let image_resource =
                            image_cache.get(*id).ok_or(RenderError::MissingImage(*id))?;
                        encode_image_paint(img, image_resource)
                    }
                    ImageSource::ExternalTexture {
                        id, source_region, ..
                    } => encode_external_texture(*id, img, *source_region)?,
                    ImageSource::Pixmap(_) => {
                        panic!("pixmap image sources are not supported by Vello GPU")
                    }
                },
                EncodedPaint::Gradient(gradient) => {
                    let (gradient_start, gradient_width) =
                        self.gradient_cache.get_or_create_ramp(gradient);
                    encode_gradient_paint(gradient, gradient_width, gradient_start)
                }
                EncodedPaint::BlurredRoundedRect(blurred_rect) => {
                    encode_blurred_rounded_rect_paint(blurred_rect)
                }
            };

            current_idx += gpu_paint.size_texels();
            self.encoded_paints[encoded_paint_idx] = gpu_paint;
        }
        self.paint_idxs[encoded_paints.len()] = current_idx;
        Ok(())
    }

    /// Schedules the passes that render `scene`.
    ///
    /// Must be called after [`Self::prepare_paints`] for the same scene.
    pub(crate) fn schedule(
        &mut self,
        scene: &Scene,
        image_cache: &ImageCache,
        root_output_target: RootTarget,
        visible_scene_area: Option<RectU16>,
        use_depth_buffer: bool,
        current_texture_size: SizeU16,
        current_allocations: IntermediateTextureAllocations,
    ) -> Result<Schedule, RenderError> {
        let required_texture_size = self
            .layers_config
            .required_intermediate_texture_size(&scene.recorder)?;
        // We currently only grow and never shrink textures, so max it with whatever
        // we had in the previous run.
        let texture_size = current_texture_size.max(required_texture_size);
        let paint_resolver = PaintResolver::new(
            &scene.encoded_paints,
            &self.paint_idxs,
            &self.encoded_paints,
        )
        .with_image_cache(image_cache);

        Schedule::try_new(
            &mut self.schedule_storage,
            scene,
            root_output_target,
            visible_scene_area,
            use_depth_buffer,
            paint_resolver,
            texture_size,
            current_allocations,
            self.layers_config.max_textures,
        )
    }
}
