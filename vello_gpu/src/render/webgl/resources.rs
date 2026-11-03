// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Creation and growth of the renderer's GPU resources.

use crate::render::webgl::gl::resource::{Buffer, Framebuffer, Sampler, Texture, VertexArray};
use crate::render::webgl::passes::WebGlPrograms;
use crate::render::webgl::target::{WebGlTextureTarget, create_framebuffer_for_texture};
use crate::render::webgl::{
    WebGlDataTransferOperation, WebGlError, WebGlOperation, WebGlResultExt,
};
use crate::{
    LayersConfig, RenderSize,
    filter::FilterContext,
    gradient_cache::GradientRampCache,
    paint::{ExternalSampler, SamplerAddress, SamplerFilter},
    render::{
        Config,
        common::{GpuEncodedPaint, ScratchTexture},
    },
    schedule::Schedule,
    target::{DrawPassTarget, LayerTextureId, RootTarget},
};
use alloc::vec::Vec;
use hashbrown::HashMap;
use vello_common::image_cache::ImageCache;
use vello_common::multi_atlas::AtlasConfig;
use vello_common::{geometry::SizeU16, tile::Tile};
use web_sys::{WebGl2RenderingContext, WebGlFramebuffer, WebGlTexture};

/// Contains all WebGL resources needed for rendering.
#[derive(Debug)]
pub(crate) struct WebGlResources {
    /// Shared indices for four-vertex triangle-strip draws.
    pub(crate) quad_index_buffer: Buffer,
    /// VAO for strip rendering.
    pub(crate) strip_vao: VertexArray,
    /// Buffer for [`GpuStrip`](crate::GpuStrip) data.
    pub(crate) strips_buffer: Buffer,
    /// Texture for alpha values.
    pub(crate) alphas_texture: Texture,
    /// Height of alpha texture.
    alpha_texture_height: u32,
    /// One 2D texture per image atlas.
    pub(crate) atlas_textures: Vec<Texture>,
    /// The render target of each texture in `atlas_textures`.
    pub(crate) atlas_targets: Vec<WebGlTextureTarget>,
    /// Configured atlas dimensions.
    pub(crate) atlas_size: (u16, u16),
    /// Encoded paints texture for image metadata.
    pub(crate) encoded_paints_texture: Texture,
    /// Height of encoded paints texture.
    encoded_paints_texture_height: u32,
    /// Gradient texture for gradient ramp data.
    pub(crate) gradient_texture: Texture,
    /// Placeholder texture bound to unoccupied external texture slots.
    pub(crate) placeholder_external_texture: Texture,
    /// Native image samplers shared by external texture slots.
    pub(crate) external_samplers: HashMap<ExternalSampler, Sampler>,
    /// Height of gradient texture.
    gradient_texture_height: u32,

    /// Config buffer for rendering strips into the root target.
    pub(crate) view_config_buffer: Buffer,
    /// Whether the depth buffer of the drawing buffer is enabled and may be used.
    pub(crate) drawing_buffer_uses_depth_buffer: bool,
    /// The target that is rendered into, or `None` for the drawing buffer.
    pub(crate) root_target: Option<WebGlTextureTarget>,
    /// Whether the depth buffer has been cleared this frame.
    pub(crate) depth_cleared_this_frame: bool,
    /// Pre-allocated depth attachment list for the default framebuffer.
    pub(crate) default_depth_attachment_array: js_sys::Array,
    /// Pre-allocated depth attachment list for offscreen framebuffers.
    pub(crate) offscreen_depth_attachment_array: js_sys::Array,

    /// Maximum width or height of packed resource textures.
    pub(crate) resource_texture_dimension_2d: u32,

    /// Dimensions of the intermediate layer and scratch textures.
    pub(crate) texture_size: SizeU16,

    /// RGBA32UI texture storing filter parameters.
    pub(crate) filter_data_texture: Texture,
    /// Current height of filter data texture.
    pub(crate) filter_data_texture_height: u32,
    /// Per-instance vertex data buffer for filter draws.
    pub(crate) filter_instance_buffer: Buffer,
    /// VAO for filter rendering.
    pub(crate) filter_vao: VertexArray,
    /// Per-instance vertex data buffer for blend draws.
    pub(crate) blend_instance_buffer: Buffer,
    /// VAO for blend rendering.
    pub(crate) blend_vao: VertexArray,
    /// Per-instance vertex data buffer for copy draws.
    pub(crate) copy_instance_buffer: Buffer,
    /// VAO for copy rendering.
    pub(crate) copy_vao: VertexArray,
    /// Scratch texture.
    pub(crate) scratch_texture: Option<ScratchTexture<WebGlIntermediateTexture>>,
    /// Config buffer for rendering strips into a layer texture.
    pub(crate) layer_config_buffer: Buffer,
    /// Layer texture pages grouped by parity.
    pub(crate) layer_textures: [Vec<WebGlIntermediateTexture>; 2],
}

impl WebGlResources {
    pub(crate) fn root_framebuffer(&self) -> Option<&WebGlFramebuffer> {
        self.root_target
            .as_ref()
            .map(|target| &**target.framebuffer)
    }

    pub(crate) fn root_uses_depth_buffer(&self) -> bool {
        self.root_target
            .as_ref()
            .map_or(self.drawing_buffer_uses_depth_buffer, |target| {
                target.use_depth_buffer
            })
    }

    pub(crate) fn layer_texture(&self, id: LayerTextureId) -> &Texture {
        self.layer_textures[id.texture_parity.get_parity()][usize::from(id.page_index)]
            .binding_texture()
    }

    pub(crate) fn layer_framebuffer(&self, id: LayerTextureId) -> &Framebuffer {
        self.layer_textures[id.texture_parity.get_parity()][usize::from(id.page_index)]
            .framebuffer()
    }

    pub(crate) fn scratch_binding_texture(&self) -> &Texture {
        self.scratch_texture
            .as_ref()
            .map_or(&self.placeholder_external_texture, |texture| {
                texture.get().binding_texture()
            })
    }

    pub(crate) fn scratch_framebuffer(&self) -> &Framebuffer {
        self.scratch_texture.as_ref().unwrap().get().framebuffer()
    }
}

#[derive(Debug)]
pub(crate) struct WebGlIntermediateTexture {
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

impl WebGlPrograms {
    /// Prepare resources for rendering.
    pub(crate) fn prepare(
        &mut self,
        gl: &WebGl2RenderingContext,
        gradient_cache: &mut GradientRampCache,
        encoded_paints: &[GpuEncodedPaint],
        alphas: &mut Vec<u8>,
        render_size: &RenderSize,
        scene_offset: (i32, i32),
        paint_idxs: &[u32],
        filter_context: &FilterContext,
        root_target: RootTarget,
    ) -> Result<(), WebGlError> {
        let resource_texture_dimension_2d = self.resources.resource_texture_dimension_2d;

        self.maybe_resize_alphas_tex(gl, resource_texture_dimension_2d, alphas.len())?;
        self.maybe_resize_encoded_paints_tex(gl, resource_texture_dimension_2d, paint_idxs)?;
        self.maybe_resize_filter_data_tex(gl, filter_context)?;
        let negate_ndc = DrawPassTarget::Root(root_target).negate_ndc();
        self.maybe_update_config_buffer(
            gl,
            resource_texture_dimension_2d,
            render_size,
            scene_offset,
            negate_ndc,
        );

        self.upload_alpha_texture(gl, alphas)?;
        self.upload_encoded_paints_texture(gl, encoded_paints, paint_idxs)?;
        self.upload_filter_data_texture(gl, filter_context)?;

        if gradient_cache.has_changed() {
            self.maybe_resize_gradient_tex(gl, resource_texture_dimension_2d, gradient_cache)?;
            self.upload_gradient_texture(gl, gradient_cache)?;
            gradient_cache.mark_synced();
        }

        Ok(())
    }

    pub(crate) fn prepare_intermediate_textures(
        &mut self,
        gl: &WebGl2RenderingContext,
        schedule: &Schedule,
    ) -> Result<(), WebGlError> {
        let current_size = self.resources.texture_size;
        let requirements = schedule.intermediate_texture_requirements();
        let texture_size = requirements.size;
        let allocations = requirements.allocations;
        let layer_pages = allocations.layer_pages;
        let scratch_required = allocations.scratch;
        let size_changed = current_size != texture_size;

        if size_changed {
            upload_layer_config_buffer(
                gl,
                &self.resources.layer_config_buffer,
                texture_size,
                self.resources.resource_texture_dimension_2d,
            );
        }

        for (index, textures) in self.resources.layer_textures.iter_mut().enumerate() {
            let required_page_count = layer_pages[index];

            // When resizing textures, we explicitly first destroy the old textures before creating
            // the new ones. The hope is that this will allow the WebGL driver to recognize
            // that the memory of the old texture can be deallocated before allocating the
            // new one, reducing peak memory consumption. It has not been validated whether
            // this actually makes any difference in practice, but it shouldn't hurt either.

            // Note: Currently, `texture_size` only grows across frames, so if this condition
            // is true it means that the new required size is larger than what we currently have.
            if size_changed {
                let page_count = textures.len().max(required_page_count);
                textures.clear();
                for _ in 0..page_count {
                    textures.push(create_intermediate_texture(gl, texture_size)?);
                }
            } else {
                for _ in textures.len()..required_page_count {
                    textures.push(create_intermediate_texture(gl, texture_size)?);
                }
            }
        }

        let recreate_scratch = (self.resources.scratch_texture.is_some() && size_changed)
            || (self.resources.scratch_texture.is_none() && scratch_required);

        if recreate_scratch {
            // Make sure the old texture is destroyed first.
            let _ = self.resources.scratch_texture.take();

            self.resources.scratch_texture = Some(ScratchTexture::new(
                create_intermediate_texture(gl, texture_size)?,
            ));
        }

        self.resources.texture_size = texture_size;

        Ok(())
    }

    /// Create any newly allocated atlas textures.
    pub(crate) fn maybe_create_atlas_textures(
        &mut self,
        gl: &WebGl2RenderingContext,
        required_atlas_count: u32,
    ) -> Result<(), WebGlError> {
        let (width, height) = self.resources.atlas_size;
        while self.resources.atlas_textures.len() < required_atlas_count as usize {
            let texture = create_atlas_texture(gl, width, height)?;
            self.resources
                .atlas_targets
                .push(WebGlTextureTarget::for_atlas(gl, &texture, width, height)?);
            self.resources.atlas_textures.push(texture);
        }

        Ok(())
    }

    /// Grow the alpha texture if needed.
    fn maybe_resize_alphas_tex(
        &mut self,
        gl: &WebGl2RenderingContext,
        resource_texture_dimension_2d: u32,
        alphas_len: usize,
    ) -> Result<(), WebGlError> {
        let required_alpha_height = (alphas_len as u32)
            // There are 16 1-byte alpha values per texel.
            .div_ceil(resource_texture_dimension_2d << 4);

        let current_alpha_height = self.resources.alpha_texture_height;
        if required_alpha_height > current_alpha_height {
            validate_resource_texture_size(
                resource_texture_dimension_2d,
                required_alpha_height,
                resource_texture_dimension_2d,
            )?;

            // We need to resize the alpha texture to fit the new alpha data.
            self.resources.alphas_texture = create_data_texture_storage(
                gl,
                WebGl2RenderingContext::RGBA32UI,
                resource_texture_dimension_2d,
                required_alpha_height,
            )?;
            self.resources.alpha_texture_height = required_alpha_height;
        }

        Ok(())
    }

    /// Grow the encoded paints texture if needed.
    fn maybe_resize_encoded_paints_tex(
        &mut self,
        gl: &WebGl2RenderingContext,
        resource_texture_dimension_2d: u32,
        paint_idxs: &[u32],
    ) -> Result<(), WebGlError> {
        let required_texels = paint_idxs.last().unwrap();
        let required_encoded_paints_height =
            required_texels.div_ceil(resource_texture_dimension_2d);
        let current_encoded_paints_height = self.resources.encoded_paints_texture_height;
        if required_encoded_paints_height > current_encoded_paints_height {
            validate_resource_texture_size(
                resource_texture_dimension_2d,
                required_encoded_paints_height,
                resource_texture_dimension_2d,
            )?;

            let required_encoded_paints_size =
                (resource_texture_dimension_2d * required_encoded_paints_height) << 4;
            self.encoded_paints_data
                .resize(required_encoded_paints_size as usize, 0);
            self.resources.encoded_paints_texture = create_data_texture_storage(
                gl,
                WebGl2RenderingContext::RGBA32UI,
                resource_texture_dimension_2d,
                required_encoded_paints_height,
            )?;
            self.resources.encoded_paints_texture_height = required_encoded_paints_height;
        }

        Ok(())
    }

    /// Grow the gradient texture if needed.
    fn maybe_resize_gradient_tex(
        &mut self,
        gl: &WebGl2RenderingContext,
        resource_texture_dimension_2d: u32,
        gradient_cache: &GradientRampCache,
    ) -> Result<(), WebGlError> {
        if gradient_cache.is_empty() {
            return Ok(());
        }

        let gradient_data_size = gradient_cache.luts_size();
        // Each texel is RGBA8, so 4 bytes per texel
        let required_gradient_height =
            (gradient_data_size as u32).div_ceil(resource_texture_dimension_2d * 4);

        let current_gradient_height = self.resources.gradient_texture_height;
        if required_gradient_height > current_gradient_height {
            validate_resource_texture_size(
                resource_texture_dimension_2d,
                required_gradient_height,
                resource_texture_dimension_2d,
            )?;

            self.resources.gradient_texture = create_data_texture_storage(
                gl,
                WebGl2RenderingContext::RGBA8,
                resource_texture_dimension_2d,
                required_gradient_height,
            )?;
            self.resources.gradient_texture_height = required_gradient_height;
        }

        Ok(())
    }

    /// Update config buffer if dimensions, the scene offset or NDC negation changed.
    fn maybe_update_config_buffer(
        &mut self,
        gl: &WebGl2RenderingContext,
        resource_texture_dimension_2d: u32,
        new_render_size: &RenderSize,
        scene_offset: (i32, i32),
        negate_ndc: bool,
    ) {
        // TODO: Collect all attributes that influence the config buffer into a
        // single struct and compare that, such that we cannot forget to update the
        // condition in case we add new fields in the future.
        if self.render_size != *new_render_size
            || self.scene_offset != scene_offset
            || self.negate_ndc != negate_ndc
        {
            let config = Config {
                width: u32::from(new_render_size.width),
                height: u32::from(new_render_size.height),
                strip_height: u32::from(Tile::HEIGHT),
                alphas_tex_width_bits: resource_texture_dimension_2d.trailing_zeros(),
                encoded_paints_tex_width_bits: resource_texture_dimension_2d.trailing_zeros(),
                strip_offset_x: scene_offset.0,
                strip_offset_y: scene_offset.1,
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

            self.render_size = *new_render_size;
            self.scene_offset = scene_offset;
            self.negate_ndc = negate_ndc;
        }
    }

    /// Upload alpha data to the texture.
    fn upload_alpha_texture(
        &mut self,
        gl: &WebGl2RenderingContext,
        alphas: &mut Vec<u8>,
    ) -> Result<(), WebGlError> {
        if alphas.is_empty() {
            return Ok(());
        }

        let alpha_texture_width = self.resources.resource_texture_dimension_2d;
        // There are 16 1-byte alpha values per texel.
        let used_height = (alphas.len() as u32).div_ceil(alpha_texture_width << 4);
        let used_size = (alpha_texture_width as usize) * (used_height as usize) * 16;

        let original_len = alphas.len();

        // Temporarily pad the length of the alphas to the end of the last used row.
        alphas.resize(used_size, 0);

        let result = upload_rgba32ui_rows(
            gl,
            &self.resources.alphas_texture,
            bytemuck::cast_slice::<u8, u32>(alphas),
            alpha_texture_width,
            used_height,
        );

        // Truncate back to the original size.
        alphas.truncate(original_len);

        result
    }

    /// Upload encoded paints to the texture.
    fn upload_encoded_paints_texture(
        &mut self,
        gl: &WebGl2RenderingContext,
        encoded_paints: &[GpuEncodedPaint],
        paint_idxs: &[u32],
    ) -> Result<(), WebGlError> {
        if !encoded_paints.is_empty() {
            let encoded_paints_texture_width = self.resources.resource_texture_dimension_2d;
            let used_texels = *paint_idxs.last().unwrap();
            let used_height = used_texels.div_ceil(encoded_paints_texture_width);
            let used_size = ((encoded_paints_texture_width * used_height) << 4) as usize;

            GpuEncodedPaint::serialize_to_buffer(encoded_paints, &mut self.encoded_paints_data);

            upload_rgba32ui_rows(
                gl,
                &self.resources.encoded_paints_texture,
                bytemuck::cast_slice::<u8, u32>(&self.encoded_paints_data[..used_size]),
                encoded_paints_texture_width,
                used_height,
            )?;
        }

        Ok(())
    }

    /// Upload gradient data to the texture.
    fn upload_gradient_texture(
        &mut self,
        gl: &WebGl2RenderingContext,
        gradient_cache: &mut GradientRampCache,
    ) -> Result<(), WebGlError> {
        if gradient_cache.is_empty() {
            return Ok(());
        }

        let gradient_texture_width = self.resources.resource_texture_dimension_2d;
        // Each texel is RGBA8, so 4 bytes per texel. Only the rows covering the LUT cache are
        // uploaded; see `upload_alpha_texture`.
        let used_height = (gradient_cache.luts_size() as u32).div_ceil(gradient_texture_width * 4);
        let used_size = (gradient_texture_width * used_height * 4) as usize;

        // Borrow the luts and pad to the end of the last used row.
        let luts = gradient_cache.luts_mut();
        let old_luts_len = luts.len();
        luts.resize(used_size, 0);

        gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        gl.bind_texture(
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&self.resources.gradient_texture),
        );

        let result = gl
            .tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
                WebGl2RenderingContext::TEXTURE_2D,
                0,
                0,
                0,
                gradient_texture_width as i32,
                used_height as i32,
                WebGl2RenderingContext::RGBA,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                Some(luts.as_slice()),
            )
            .map_js_error(WebGlOperation::DataTransfer(
                WebGlDataTransferOperation::TextureUpload,
            ));

        luts.truncate(old_luts_len);
        result
    }
}
/// Create a texture with the requested filtering and clamp-to-edge wrapping.
pub(crate) fn create_texture(
    gl: &WebGl2RenderingContext,
    min_filter: u32,
    mag_filter: u32,
) -> Result<Texture, WebGlError> {
    create_texture_inner(
        gl,
        WebGl2RenderingContext::TEXTURE_2D,
        min_filter,
        mag_filter,
    )
}

pub(crate) fn create_texture_inner(
    gl: &WebGl2RenderingContext,
    target: u32,
    min_filter: u32,
    mag_filter: u32,
) -> Result<Texture, WebGlError> {
    let texture = Texture::new(gl)?;
    gl.active_texture(WebGl2RenderingContext::TEXTURE0);
    gl.bind_texture(target, Some(&texture));
    gl.tex_parameteri(
        target,
        WebGl2RenderingContext::TEXTURE_MIN_FILTER,
        min_filter as i32,
    );
    gl.tex_parameteri(
        target,
        WebGl2RenderingContext::TEXTURE_MAG_FILTER,
        mag_filter as i32,
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

    // Strictly speaking this shouldn't be necessary since all our textures are created
    // as immutable texture, but it seems like there exist drivers that have a bug here
    // (https://stackoverflow.com/questions/13859061/does-an-immutable-texture-need-a-gl-texture-max-level).
    // Also keep in mind the following bug: https://issues.chromium.org/issues/355605685
    // We must ensure to never use mutable storage when changing this parameter.
    gl.tex_parameteri(target, WebGl2RenderingContext::TEXTURE_MAX_LEVEL, 0);

    Ok(texture)
}

/// Create a texture backed by immutable storage (`texStorage2D`).
pub(crate) fn create_texture_storage(
    gl: &WebGl2RenderingContext,
    internal_format: u32,
    width: u32,
    height: u32,
    min_filter: u32,
    mag_filter: u32,
) -> Result<Texture, WebGlError> {
    let texture = create_texture(gl, min_filter, mag_filter)?;
    // `create_texture` leaves the new texture bound on the active texture unit.
    // Prefer `texStorage2D` over `texImage2D` for potentially better performance; see
    // <https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices#use_texstorage_to_create_textures>.
    gl.tex_storage_2d(
        WebGl2RenderingContext::TEXTURE_2D,
        1,
        internal_format,
        width as i32,
        height as i32,
    );
    Ok(texture)
}

/// Validate that a resource texture fits within its configured dimension limit.
pub(crate) fn validate_resource_texture_size(
    width: u32,
    height: u32,
    max_dimension: u32,
) -> Result<(), WebGlError> {
    if width > max_dimension || height > max_dimension {
        return Err(WebGlError::ResourceTextureTooLarge {
            width,
            height,
            max_dimension,
        });
    }

    Ok(())
}

/// Create a zero-initialized data texture backed by immutable storage (`texStorage2D`).
pub(crate) fn create_data_texture_storage(
    gl: &WebGl2RenderingContext,
    internal_format: u32,
    width: u32,
    height: u32,
) -> Result<Texture, WebGlError> {
    create_texture_storage(
        gl,
        internal_format,
        width,
        height,
        WebGl2RenderingContext::NEAREST,
        WebGl2RenderingContext::NEAREST,
    )
}

/// Create a 1x1 RGBA32UI placeholder texture.
pub(crate) fn create_placeholder_rgba32ui_texture(
    gl: &WebGl2RenderingContext,
) -> Result<Texture, WebGlError> {
    // Allocated storage keeps the placeholder texture complete when no data is available.
    create_data_texture_storage(gl, WebGl2RenderingContext::RGBA32UI, 1, 1)
}

/// Create a 1x1 RGBA8 placeholder texture.
///
/// A texture with no allocated level is incomplete, and binding an incomplete texture to a sampler
/// makes the draw call non-renderable: the driver substitutes a black texture for every draw and
/// browsers log a render warning for each one. Data textures are only uploaded when a scene has
/// data for them, so allocate a minimal level 0 up front to keep them complete for scenes that
/// don't.
pub(crate) fn create_placeholder_rgba8_texture(
    gl: &WebGl2RenderingContext,
) -> Result<Texture, WebGlError> {
    create_data_texture_storage(gl, WebGl2RenderingContext::RGBA8, 1, 1)
}

pub(crate) fn upload_layer_config_buffer(
    gl: &WebGl2RenderingContext,
    buffer: &Buffer,
    size: SizeU16,
    resource_texture_dimension_2d: u32,
) {
    let config = Config {
        width: u32::from(size.width()),
        height: u32::from(size.height()),
        strip_height: u32::from(Tile::HEIGHT),
        alphas_tex_width_bits: resource_texture_dimension_2d.trailing_zeros(),
        encoded_paints_tex_width_bits: resource_texture_dimension_2d.trailing_zeros(),
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
pub(crate) fn create_external_sampler(
    gl: &WebGl2RenderingContext,
    key: ExternalSampler,
) -> Result<Sampler, WebGlError> {
    let sampler = Sampler::new(gl)?;

    let filter = match key.filter {
        SamplerFilter::Nearest => WebGl2RenderingContext::NEAREST,
        SamplerFilter::Linear => WebGl2RenderingContext::LINEAR,
    } as i32;

    let address = |mode| match mode {
        SamplerAddress::Pad => WebGl2RenderingContext::CLAMP_TO_EDGE,
        SamplerAddress::Repeat => WebGl2RenderingContext::REPEAT,
        SamplerAddress::Reflect => WebGl2RenderingContext::MIRRORED_REPEAT,
    } as i32;

    gl.sampler_parameteri(&sampler, WebGl2RenderingContext::TEXTURE_MIN_FILTER, filter);
    gl.sampler_parameteri(&sampler, WebGl2RenderingContext::TEXTURE_MAG_FILTER, filter);
    gl.sampler_parameteri(
        &sampler,
        WebGl2RenderingContext::TEXTURE_WRAP_S,
        address(key.address_x),
    );
    gl.sampler_parameteri(
        &sampler,
        WebGl2RenderingContext::TEXTURE_WRAP_T,
        address(key.address_y),
    );

    Ok(sampler)
}

pub(crate) fn create_webgl_resources(
    gl: &WebGl2RenderingContext,
    image_cache: &ImageCache,
    layer_config: LayersConfig,
    resource_texture_dimension_2d: u32,
    use_depth_buffer: bool,
) -> Result<WebGlResources, WebGlError> {
    let quad_index_buffer = Buffer::new(gl)?;
    let strip_vao = VertexArray::new(gl)?;
    // We use u16 for the indices instead of u8 due to better compatibility, to avoid overhead
    // from any potential emulation that the driver might have to do for correctness reasons.
    // D3D11 only supports u16/u32: https://learn.microsoft.com/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-iasetindexbuffer
    // Same for Metal: https://developer.apple.com/documentation/metal/mtlindextype
    // In Vulkan u8 only was added in later versions: https://docs.vulkan.org/refpages/latest/refpages/source/VkIndexType.html
    // ANGLE also seems to special-case on Metal, as additional evidence: https://chromium.googlesource.com/angle/angle/+/bfc764c553fa0613f858315bc6c0cc1ecee469a1/src/libANGLE/renderer/metal/VertexArrayMtl.mm#63
    let quad_indices = js_sys::Uint16Array::from(&[0_u16, 1, 2, 3][..]);

    // Just to avoid potentially modifying a VAO that might
    // have been bound by a caller before calling Vello, since
    // binding `ELEMENT_ARRAY_BUFFER` also updates VAO state.
    gl.bind_vertex_array(Some(&strip_vao));
    gl.bind_buffer(
        WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
        Some(&quad_index_buffer),
    );
    gl.buffer_data_with_array_buffer_view(
        WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
        &quad_indices,
        WebGl2RenderingContext::STATIC_DRAW,
    );
    gl.bind_vertex_array(None);

    let filter_vao = VertexArray::new(gl)?;
    let blend_vao = VertexArray::new(gl)?;
    let copy_vao = VertexArray::new(gl)?;
    let filter_instance_buffer = Buffer::new(gl)?;
    let blend_instance_buffer = Buffer::new(gl)?;
    let copy_instance_buffer = Buffer::new(gl)?;

    let strips_buffer = Buffer::new(gl)?;
    let view_config_buffer = Buffer::new(gl)?;
    let texture_size = layer_config.min_texture_size;
    let layer_config_buffer = Buffer::new(gl)?;
    upload_layer_config_buffer(
        gl,
        &layer_config_buffer,
        texture_size,
        resource_texture_dimension_2d,
    );

    // Create and configure alpha texture.
    let alphas_texture = create_placeholder_rgba32ui_texture(gl)?;

    let AtlasConfig {
        atlas_size: (atlas_width, atlas_height),
        initial_atlas_count,
        ..
    } = *image_cache.atlas_manager().config();
    let atlas_size = (atlas_width, atlas_height);
    let atlas_textures = (0..initial_atlas_count)
        .map(|_| create_atlas_texture(gl, atlas_width, atlas_height))
        .collect::<Result<Vec<_>, _>>()?;
    let atlas_targets = atlas_textures
        .iter()
        .map(|texture| WebGlTextureTarget::for_atlas(gl, texture, atlas_width, atlas_height))
        .collect::<Result<Vec<_>, _>>()?;

    // Create and configure encoded paints texture.
    let encoded_paints_texture = create_placeholder_rgba32ui_texture(gl)?;

    // Create and configure gradient texture.
    let gradient_texture = create_placeholder_rgba8_texture(gl)?;
    let placeholder_external_texture = create_placeholder_rgba8_texture(gl)?;
    let external_samplers = HashMap::new();

    let layer_textures: [Vec<WebGlIntermediateTexture>; 2] = core::array::from_fn(|_| Vec::new());
    let scratch_texture = None;
    let filter_data_texture = create_placeholder_rgba32ui_texture(gl)?;

    Ok(WebGlResources {
        quad_index_buffer,
        strip_vao,
        strips_buffer,
        alphas_texture,
        alpha_texture_height: 0,
        atlas_textures,
        atlas_targets,
        atlas_size,
        encoded_paints_texture,
        encoded_paints_texture_height: 0,
        gradient_texture,
        placeholder_external_texture,
        external_samplers,
        gradient_texture_height: 0,
        view_config_buffer,
        drawing_buffer_uses_depth_buffer: use_depth_buffer,
        root_target: None,
        depth_cleared_this_frame: false,
        default_depth_attachment_array: js_sys::Array::of1(&WebGl2RenderingContext::DEPTH.into()),
        offscreen_depth_attachment_array: js_sys::Array::of1(
            &WebGl2RenderingContext::DEPTH_ATTACHMENT.into(),
        ),
        resource_texture_dimension_2d,
        texture_size,
        filter_data_texture,
        filter_data_texture_height: 0,
        filter_instance_buffer,
        filter_vao,
        blend_instance_buffer,
        blend_vao,
        copy_instance_buffer,
        copy_vao,
        scratch_texture,
        layer_config_buffer,
        layer_textures,
    })
}

pub(crate) fn create_intermediate_texture(
    gl: &WebGl2RenderingContext,
    size: SizeU16,
) -> Result<WebGlIntermediateTexture, WebGlError> {
    let texture = create_texture_storage(
        gl,
        WebGl2RenderingContext::RGBA8,
        u32::from(size.width()),
        u32::from(size.height()),
        WebGl2RenderingContext::LINEAR,
        WebGl2RenderingContext::LINEAR,
    )?;
    let framebuffer = create_framebuffer_for_texture(gl, &texture)?;
    Ok(WebGlIntermediateTexture {
        texture,
        framebuffer,
    })
}

/// Create an image atlas texture.
pub(crate) fn create_atlas_texture(
    gl: &WebGl2RenderingContext,
    width: u16,
    height: u16,
) -> Result<Texture, WebGlError> {
    create_texture_storage(
        gl,
        WebGl2RenderingContext::RGBA8,
        u32::from(width),
        u32::from(height),
        WebGl2RenderingContext::NEAREST,
        WebGl2RenderingContext::NEAREST,
    )
}

/// Upload full rows of RGBA32UI data into immutable texture storage.
///
/// `data` must contain exactly `width * rows * 4` `u32` values.
/// Texels outside the uploaded rows remain unchanged.
pub(crate) fn upload_rgba32ui_rows(
    gl: &WebGl2RenderingContext,
    texture: &WebGlTexture,
    data: &[u32],
    width: u32,
    rows: u32,
) -> Result<(), WebGlError> {
    debug_assert_eq!(
        data.len(),
        (width * rows * 4) as usize,
        "row upload data must cover exactly the uploaded region"
    );
    gl.active_texture(WebGl2RenderingContext::TEXTURE0);
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(texture));

    // Safety: This calling `Uint32Array::view` is unsafe because it provides a view into
    // WASM linear memory, and any additional allocations might invalidate that view.
    // In our case, this is not an issue because we only use this view once for uploading
    // data to the GPU below, and no allocations happen between that.
    // The `tex_sub_image_2d` method is synchronous in the sense that once it returns, it is
    // guaranteed that all necessary data has already been read, so any allocations that happen
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

    gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_array_buffer_view(
        WebGl2RenderingContext::TEXTURE_2D,
        0,
        0,
        0,
        width as i32,
        rows as i32,
        WebGl2RenderingContext::RGBA_INTEGER,
        WebGl2RenderingContext::UNSIGNED_INT,
        Some(&packed_array),
    )
    .map_js_error(WebGlOperation::DataTransfer(
        WebGlDataTransferOperation::TextureUpload,
    ))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_texture_size_must_fit_dimension_limit() {
        assert!(validate_resource_texture_size(4096, 4096, 4096).is_ok());
        assert!(matches!(
            validate_resource_texture_size(4096, 4097, 4096),
            Err(WebGlError::ResourceTextureTooLarge {
                width: 4096,
                height: 4097,
                max_dimension: 4096,
            })
        ));
    }
}
