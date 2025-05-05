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

use alloc::vec;
use alloc::vec::Vec;
use core::{fmt::Debug, mem, num::NonZeroU64};

use bytemuck::{Pod, Zeroable};
use vello_common::{coarse::WideTile, tile::Tile};
use wgpu::{
    BindGroup, BindGroupLayout, BlendState, Buffer, ColorTargetState, ColorWrites, CommandEncoder,
    Device, PipelineCompilationOptions, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, Texture, TextureView, util::DeviceExt,
};

use crate::{RenderError, scene::Scene, schedule::Scheduler};

/// Dimensions of the rendering target
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RenderSize {
    /// Width of the rendering target
    pub width: u32,
    /// Height of the rendering target
    pub height: u32,
}

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
    programs: Programs,
    scheduler: Scheduler,
}

impl Renderer {
    /// Creates a new renderer.
    pub fn new(device: &Device, render_target_config: &RenderTargetConfig) -> Self {
        let slot_count = (device.limits().max_texture_dimension_2d / Tile::HEIGHT as u32) as usize;

        Self {
            programs: Programs::new(device, render_target_config, slot_count),
            scheduler: Scheduler::new(slot_count),
        }
    }

    /// Render `scene` into the provided command encoder.
    ///
    /// This method creates GPU resources as needed and schedules potentially multiple
    /// render passes.
    pub fn render(
        &mut self,
        scene: &Scene,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        render_size: &RenderSize,
        view: &TextureView,
    ) -> Result<(), RenderError> {
        // For the time being, we upload the entire alpha buffer as one big chunk. As a future
        // refinement, we could have a bounded alpha buffer, and break draws when the alpha
        // buffer fills.
        self.programs
            .prepare(device, queue, &scene.alphas, render_size);
        let mut junk = RendererJunk {
            programs: &mut self.programs,
            device,
            queue,
            encoder,
            view,
        };

        self.scheduler.do_scene(&mut junk, scene)
    }
}

/// Defines the GPU resources and pipelines for rendering.
#[derive(Debug)]
struct Programs {
    /// Pipeline for rendering wide tile commands.
    strip_pipeline: RenderPipeline,
    /// Bind group layout for strip draws
    strip_bind_group_layout: BindGroupLayout,

    /// Pipeline for clearing slots in slot textures.
    clear_pipeline: RenderPipeline,

    /// GPU resources for rendering (created during prepare)
    resources: GpuResources,
    /// Dimensions of the rendering target
    render_size: RenderSize,
    /// Scratch buffer for staging alpha texture data.
    alpha_data: Vec<u8>,
}

/// Contains all GPU resources needed for rendering
#[derive(Debug)]
struct GpuResources {
    /// Buffer for [`GpuStrip`] data
    strips_buffer: Buffer,
    /// Texture for alpha values (used by both view and slot rendering)
    alphas_texture: Texture,

    /// Config buffer for rendering wide tile commands into the view texture.
    view_config_buffer: Buffer,
    /// Config buffer for rendering wide tile commands into a slot texture.
    slot_config_buffer: Buffer,

    /// Buffer for slot indices used in `clear_slots`
    clear_slot_indices_buffer: Buffer,
    // Bind groups for rendering with clip buffers
    slot_bind_groups: [BindGroup; 3],
    /// Slot texture views
    slot_texture_views: [TextureView; 2],

    /// Bind group for clear slots operation
    clear_bind_group: BindGroup,
}

/// Configuration for the GPU renderer
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct Config {
    /// Width of the rendering target
    pub width: u32,
    /// Height of the rendering target
    pub height: u32,
    /// Height of a strip in the rendering
    pub strip_height: u32,
    /// Number of trailing zeros in `alphas_tex_width` (log2 of width).
    /// Pre-calculated on CPU since downlevel targets do not support `firstTrailingBit`.
    pub alphas_tex_width_bits: u32,
}

const SIZE_OF_CONFIG: NonZeroU64 = NonZeroU64::new(size_of::<Config>() as u64).unwrap();

/// Represents a GPU strip for rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct GpuStrip {
    /// X coordinate of the strip
    pub x: u16,
    /// Y coordinate of the strip
    pub y: u16,
    /// Width of the strip
    pub width: u16,
    /// Width of the portion where alpha blending should be applied.
    pub dense_width: u16,
    /// Column-index into the alpha texture where this strip's alpha values begin.
    ///
    /// There are [`Config::strip_height`] alpha values per column.
    pub col: u32,
    /// RGBA color value
    pub rgba: u32,
}

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
    pub fn vertex_attributes() -> [wgpu::VertexAttribute; 4] {
        wgpu::vertex_attr_array![
            0 => Uint32,
            1 => Uint32,
            2 => Uint32,
            3 => Uint32,
        ]
    }
}

impl Programs {
    fn new(device: &Device, render_target_config: &RenderTargetConfig, slot_count: usize) -> Self {
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/render_strips.wgsl").into()),
        });

        let clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Clear Slots Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/clear_slots.wgsl").into()),
        });

        let strip_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Strip Pipeline Layout"),
                bind_group_layouts: &[&strip_bind_group_layout],
                push_constant_ranges: &[],
            });

        let clear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Clear Slots Pipeline Layout"),
                bind_group_layouts: &[&clear_bind_group_layout],
                push_constant_ranges: &[],
            });

        let strip_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    format: render_target_config.format,
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
            multiview: None,
            cache: None,
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
            multiview: None,
            cache: None,
        });

        let slot_texture_views: [TextureView; 2] = core::array::from_fn(|_| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("slot temp texture"),
                    size: wgpu::Extent3d {
                        width: WideTile::WIDTH as u32,
                        height: Tile::HEIGHT as u32 * slot_count as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    // TODO: Is this correct or need it be RGBA8Unorm?
                    format: render_target_config.format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        });

        let clear_config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Clear Slots Config Buffer"),
            contents: bytemuck::bytes_of(&ClearSlotsConfig {
                slot_width: WideTile::WIDTH as u32,
                slot_height: Tile::HEIGHT as u32,
                texture_height: Tile::HEIGHT as u32 * slot_count as u32,
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
        let clear_slot_indices_buffer = Self::make_clear_slot_indices_buffer(
            device,
            slot_count as u64 * size_of::<u32>() as u64,
        );

        let slot_config_buffer = Self::make_config_buffer(
            device,
            &RenderSize {
                width: WideTile::WIDTH as u32,
                height: Tile::HEIGHT as u32 * slot_count as u32,
            },
            device.limits().max_texture_dimension_2d,
        );

        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        let alpha_texture_height = 2;
        let alphas_texture =
            Self::make_alphas_texture(device, max_texture_dimension_2d, alpha_texture_height);
        let alpha_data = vec![0; (max_texture_dimension_2d * alpha_texture_height * 16) as usize];
        let view_config_buffer = Self::make_config_buffer(
            device,
            &RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },
            max_texture_dimension_2d,
        );

        let slot_bind_groups = Self::make_strip_bind_groups(
            device,
            &strip_bind_group_layout,
            &alphas_texture,
            &slot_config_buffer,
            &view_config_buffer,
            &slot_texture_views,
        );

        let resources = GpuResources {
            strips_buffer: Self::make_strips_buffer(device, 0),
            clear_slot_indices_buffer,
            slot_texture_views,
            slot_config_buffer,
            slot_bind_groups,
            clear_bind_group,
            alphas_texture,
            view_config_buffer,
        };

        Self {
            strip_pipeline,
            strip_bind_group_layout,
            resources,
            alpha_data,
            render_size: RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },

            clear_pipeline,
        }
    }

    fn make_strips_buffer(device: &Device, required_strips_size: u64) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Strips Buffer"),
            size: required_strips_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_clear_slot_indices_buffer(device: &Device, required_size: u64) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Slot Indices Buffer"),
            size: required_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_config_buffer(
        device: &Device,
        render_size: &RenderSize,
        max_texture_dimension_2d: u32,
    ) -> Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::bytes_of(&Config {
                width: render_size.width,
                height: render_size.height,
                strip_height: Tile::HEIGHT.into(),
                alphas_tex_width_bits: max_texture_dimension_2d.trailing_zeros(),
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn make_alphas_texture(
        device: &Device,
        max_texture_dimension_2d: u32,
        alpha_texture_height: u32,
    ) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Alpha Texture"),
            size: wgpu::Extent3d {
                width: max_texture_dimension_2d,
                height: alpha_texture_height,
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

    fn make_strip_bind_groups(
        device: &Device,
        strip_bind_group_layout: &BindGroupLayout,
        alphas_texture: &Texture,
        strip_config_buffer: &Buffer,
        config_buffer: &Buffer,
        strip_texture_views: &[TextureView],
    ) -> [BindGroup; 3] {
        [
            Self::make_strip_bind_group(
                device,
                strip_bind_group_layout,
                alphas_texture,
                strip_config_buffer,
                &strip_texture_views[1],
            ),
            Self::make_strip_bind_group(
                device,
                strip_bind_group_layout,
                alphas_texture,
                strip_config_buffer,
                &strip_texture_views[0],
            ),
            Self::make_strip_bind_group(
                device,
                strip_bind_group_layout,
                alphas_texture,
                config_buffer,
                &strip_texture_views[1],
            ),
        ]
    }

    fn make_strip_bind_group(
        device: &Device,
        strip_bind_group_layout: &BindGroupLayout,
        alphas_texture: &Texture,
        config_buffer: &Buffer,
        strip_texture_view: &TextureView,
    ) -> BindGroup {
        let alphas_texture_view =
            alphas_texture.create_view(&wgpu::TextureViewDescriptor::default());
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Strip Bind Group"),
            layout: strip_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&alphas_texture_view),
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
        alphas: &[u8],
        new_render_size: &RenderSize,
    ) {
        let alpha_width = device.limits().max_texture_dimension_2d;
        // Update the alpha texture size if needed
        {
            let required_alpha_height = u32::try_from(alphas.len())
                .unwrap()
                // There are 16 1-byte alpha values per texel.
                .div_ceil(alpha_width * 16);
            let required_alpha_size = alpha_width * required_alpha_height * 16;
            let current_alpha_size = {
                let alphas_texture = &self.resources.alphas_texture;
                alphas_texture.width() * alphas_texture.height() * 16
            };
            if required_alpha_size > current_alpha_size {
                // We need to resize the alpha texture to fit the new alpha data.
                assert!(
                    required_alpha_height <= alpha_width,
                    "Alpha texture height exceeds max texture dimensions"
                );

                // Resize the alpha texture staging buffer.
                self.alpha_data.resize(required_alpha_size as usize, 0);
                // The alpha texture encodes 16 1-byte alpha values per texel, with 4 alpha values packed in each channel
                let alphas_texture =
                    Self::make_alphas_texture(device, alpha_width, required_alpha_height);
                self.resources.alphas_texture = alphas_texture;

                // Since the alpha texture has changed, we need to update the clip bind groups.
                self.resources.slot_bind_groups = Self::make_strip_bind_groups(
                    device,
                    &self.strip_bind_group_layout,
                    &self.resources.alphas_texture,
                    &self.resources.slot_config_buffer,
                    &self.resources.view_config_buffer,
                    &self.resources.slot_texture_views,
                );
            }
        }

        // Update config buffer if dimensions changed.
        if self.render_size != *new_render_size {
            let config = Config {
                width: new_render_size.width,
                height: new_render_size.height,
                strip_height: Tile::HEIGHT.into(),
                alphas_tex_width_bits: alpha_width.trailing_zeros(),
            };
            let mut buffer = queue
                .write_buffer_with(&self.resources.view_config_buffer, 0, SIZE_OF_CONFIG)
                .expect("Buffer only ever holds `Config`");
            buffer.copy_from_slice(bytemuck::bytes_of(&config));

            self.render_size = new_render_size.clone();
        }

        // Prepare alpha data for the texture with 16 1-byte alpha values per texel (4 per channel)
        let texture_width = self.resources.alphas_texture.width();
        let texture_height = self.resources.alphas_texture.height();
        debug_assert!(
            alphas.len() <= (texture_width * texture_height * 16) as usize,
            "Alpha texture dimensions are too small to fit the alpha data"
        );

        // After this copy to `self.alpha_data`, there may be stale trailing alpha values. These
        // are not sampled, so can be left as-is.
        self.alpha_data[0..alphas.len()].copy_from_slice(alphas);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.resources.alphas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.alpha_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // 16 bytes per RGBA32Uint texel (4 u32s × 4 bytes each)
                bytes_per_row: Some(texture_width * 16),
                rows_per_image: Some(texture_height),
            },
            wgpu::Extent3d {
                width: texture_width,
                height: texture_height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Upload the strip data by appending to `self.resources.strips_buffer`.
    fn upload_strips(&mut self, device: &Device, queue: &Queue, strips: &[GpuStrip]) {
        if strips.is_empty() {
            return;
        }

        let required_strips_size = size_of_val(strips) as u64;
        self.resources.strips_buffer = Self::make_strips_buffer(device, required_strips_size);
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
pub(crate) struct RendererJunk<'a> {
    programs: &'a mut Programs,
    device: &'a Device,
    queue: &'a Queue,
    encoder: &'a mut CommandEncoder,
    view: &'a TextureView,
}

impl RendererJunk<'_> {
    /// Render the strips to either the view or a slot texture (depending on `ix`).
    pub(crate) fn do_strip_render_pass(
        &mut self,
        strips: &[GpuStrip],
        ix: usize,
        load: wgpu::LoadOp<wgpu::Color>,
    ) {
        // TODO: We currently allocate a new strips buffer for each render pass. A more efficient
        // approach would be to re-use buffers or slices of a larger buffer.
        self.programs.upload_strips(self.device, self.queue, strips);

        let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render to Texture Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: if ix == 2 {
                    self.view
                } else {
                    &self.programs.resources.slot_texture_views[ix]
                },
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        render_pass.set_pipeline(&self.programs.strip_pipeline);
        render_pass.set_bind_group(0, &self.programs.resources.slot_bind_groups[ix], &[]);
        render_pass.set_vertex_buffer(0, self.programs.resources.strips_buffer.slice(..));

        let strips_to_draw = strips.len();
        render_pass.draw(0..4, 0..u32::try_from(strips_to_draw).unwrap());
    }

    /// Clear specific slots from a slot texture.
    pub(crate) fn do_clear_slots_render_pass(&mut self, ix: usize, slot_indices: &[u32]) {
        if slot_indices.is_empty() {
            return;
        }

        let resources = &mut self.programs.resources;
        let size = mem::size_of_val(slot_indices) as u64;
        // TODO: We currently allocate a new strips buffer for each render pass. A more efficient
        // approach would be to re-use buffers or slices of a larger buffer.
        resources.clear_slot_indices_buffer =
            Programs::make_clear_slot_indices_buffer(self.device, size);
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
            });

            render_pass.set_pipeline(&self.programs.clear_pipeline);
            render_pass.set_bind_group(0, &resources.clear_bind_group, &[]);
            render_pass.set_vertex_buffer(0, resources.clear_slot_indices_buffer.slice(..));
            render_pass.draw(0..4, 0..slot_indices.len() as u32);
        }
    }
}
