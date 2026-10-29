// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Pipeline creation and execution of the scheduled render passes.

pub(crate) mod blend;
pub(crate) mod clear;
pub(crate) mod copy;
pub(crate) mod filter;
pub(crate) mod strip;

use crate::draw::{EXTERNAL_TEXTURE_SLOT_COUNT, ExternalTextureBindings, ExternalTextureRun};
use crate::render::wgpu::RenderTargetConfig;
use crate::render::wgpu::bindings::TextureBindings;
use crate::render::wgpu::passes::clear::clear_color;
use crate::render::wgpu::passes::filter::{
    FilterPairBindGroups, create_filter_original_texture_bind_group,
};
use crate::render::wgpu::passes::strip::StripTargetKind;
use crate::render::wgpu::resources::{GpuResources, StripBufferArena, UploadStagingBelt};
use crate::util::RangedSlice;
use crate::{
    ClearSettings, GpuStrip, LayersConfig, RenderSize, TargetInit,
    blend::{BlendStrip, GpuBlendInstance},
    copy::GpuCopyInstance,
    filter::{FilterInstanceData, FilterPassPlan},
    paint::ExternalSampler,
    render::common::{GpuClearInstance, ScratchBuffers},
    schedule::{Backend, round::BlendOp},
    target::{
        BlendPassBindings, DrawPassBindings, DrawPassTarget, FilterPassBindings, LayerTextureId,
        RootTarget,
    },
};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use hashbrown::HashMap;
use vello_common::color::AlphaColor;
use vello_common::geometry::{RectU16, SizeU16};
use vello_common::image_cache::ImageCache;
use vello_common::multi_atlas::AtlasConfig;
use wgpu::{
    BindGroup, BindGroupLayout, BlendState, ColorTargetState, ColorWrites, CommandEncoder, Device,
    PipelineCompilationOptions, RenderPipeline, Sampler, TextureView, TextureViewDescriptor,
};

/// Defines the GPU resources and pipelines for rendering.
#[derive(Debug)]
pub(crate) struct Programs {
    /// Intermediate strip pipeline.
    intermediate_strip_pipeline: RenderPipeline,
    /// Root alpha-strip pipeline.
    alpha_strip_pipeline: RenderPipeline,
    /// Root alpha-strip pipeline with depth testing.
    depth_alpha_strip_pipeline: RenderPipeline,
    /// Root opaque-strip pipeline.
    opaque_strip_pipeline: RenderPipeline,
    /// Whether the depth buffer has been cleared this frame.
    pub(crate) depth_cleared_this_frame: bool,
    /// Bind group layout for strip draws
    strip_bind_group_layout: BindGroupLayout,
    /// Bind group layout for encoded paints
    pub(crate) encoded_paints_bind_group_layout: BindGroupLayout,
    /// Bind group layout for gradient texture
    pub(crate) gradient_bind_group_layout: BindGroupLayout,
    /// Bind group layout for external textures.
    external_texture_bind_group_layout: BindGroupLayout,
    /// Bind group layout for filter data texture.
    filter_bind_group_layout: BindGroupLayout,
    /// Filter input layouts.
    pub(crate) filter_input_bind_group_layouts: [BindGroupLayout; 2],
    /// Filter sampler.
    filter_sampler: Sampler,
    /// Native image samplers shared by external texture slots.
    external_samplers: HashMap<ExternalSampler, Sampler>,
    /// Blend bind group layout.
    blend_layer_bind_group_layout: BindGroupLayout,
    /// Copy bind group layout.
    pub(crate) copy_bind_group_layout: BindGroupLayout,
    /// Strip bind group cache.
    pub(crate) strip_layer_bind_groups:
        HashMap<(StripTargetKind, Option<LayerTextureId>), BindGroup>,
    /// Blend bind group cache.
    pub(crate) blend_layer_bind_groups: HashMap<BlendPassBindings, BindGroup>,
    /// Filter bind group cache.
    pub(crate) filter_pair_bind_groups: HashMap<FilterPassBindings, FilterPairBindGroups>,
    /// Pipeline for applying filter effects.
    filter_pipeline: RenderPipeline,
    /// Layer-clear pipeline.
    clear_pipeline: RenderPipeline,
    /// User-target rectangle-clear pipeline.
    root_clear_pipeline: RenderPipeline,
    /// Pipeline for clearing atlas regions.
    pub(crate) atlas_clear_pipeline: RenderPipeline,
    /// Blend pipeline.
    blend_pipeline: RenderPipeline,
    /// Copy pipeline.
    copy_pipeline: RenderPipeline,
    /// GPU resources for rendering (created during prepare)
    pub(crate) resources: GpuResources,
    /// Arena holding all [`GpuStrip`] data.
    pub(crate) strips_arena: StripBufferArena,
    /// Whether the external texture bind group has a `texture_external` binding.
    wgpu_external_textures: bool,
    /// Shared staging memory for buffer and texture uploads.
    pub(crate) staging_belt: UploadStagingBelt,
    /// Dimensions of the rendering target
    pub(crate) render_size: RenderSize,
    /// Offset of the scene within the rendering target.
    pub(crate) scene_offset: (i32, i32),
    /// Scratch buffer for staging encoded paints texture data.
    pub(crate) encoded_paints_data: Vec<u8>,
    /// Scratch buffer for staging filter data texture data.
    filter_data: Vec<u8>,
}

impl Programs {
    pub(crate) fn new(
        device: &Device,
        image_cache: &ImageCache,
        render_target_config: &RenderTargetConfig,
        layer_config: LayersConfig,
        resource_texture_dimension_2d: u32,
        pipeline_cache: Option<&wgpu::PipelineCache>,
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

        #[cfg(feature = "wgpu_external_texture")]
        let wgpu_external_textures = device.features().contains(wgpu::Features::EXTERNAL_TEXTURE);
        #[cfg(not(feature = "wgpu_external_texture"))]
        let wgpu_external_textures = false;
        let external_texture_layout_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let external_sampler_layout_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let mut external_texture_layout_entries = vec![
            external_texture_layout_entry(0),
            external_sampler_layout_entry(1),
        ];
        if wgpu_external_textures {
            external_texture_layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::ExternalTexture,
                count: None,
            });
        }
        let external_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("External Texture Bind Group Layout"),
                entries: &external_texture_layout_entries,
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

        #[cfg(feature = "wgpu_external_texture")]
        let strip_shader_source = if wgpu_external_textures {
            vello_gpu_shaders::wgsl::RENDER_TEXTURE_EXTERNAL
        } else {
            vello_gpu_shaders::wgsl::RENDER
        };
        #[cfg(not(feature = "wgpu_external_texture"))]
        let strip_shader_source = vello_gpu_shaders::wgsl::RENDER;
        let strip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Strip Shader"),
            source: wgpu::ShaderSource::Wgsl(strip_shader_source.into()),
        });

        let clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Clear Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_gpu_shaders::wgsl::CLEAR.into()),
        });

        let strip_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Strip Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&strip_bind_group_layout),
                    Some(&external_texture_bind_group_layout),
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

        let strip_vertex_state = wgpu::VertexBufferLayout {
            array_stride: size_of::<GpuStrip>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &GpuStrip::vertex_attributes(),
        };

        let create_strip_pipeline =
            |label, format, blend, depth_stencil: Option<wgpu::DepthStencilState>| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&strip_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &strip_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[Some(strip_vertex_state.clone())],
                        compilation_options: PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &strip_shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(ColorTargetState {
                            format,
                            blend,
                            write_mask: ColorWrites::ALL,
                        })],
                        compilation_options: PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        ..Default::default()
                    },
                    depth_stencil,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: pipeline_cache,
                })
            };

        let depth_stencil = |depth_write_enabled| wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: Some(depth_write_enabled),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        let intermediate_strip_pipeline = create_strip_pipeline(
            "Strip Intermediate Pipeline",
            wgpu::TextureFormat::Rgba8Unorm,
            Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            None,
        );

        let alpha_strip_pipeline = create_strip_pipeline(
            "Strip Alpha Pipeline",
            render_target_config.format,
            Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            None,
        );

        let depth_alpha_strip_pipeline = create_strip_pipeline(
            "Strip Depth Alpha Pipeline",
            render_target_config.format,
            Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            Some(depth_stencil(false)),
        );

        let opaque_strip_pipeline = create_strip_pipeline(
            "Strip Opaque Pipeline",
            render_target_config.format,
            None,
            Some(depth_stencil(true)),
        );

        let clear_vertex_state = wgpu::VertexBufferLayout {
            array_stride: size_of::<GpuClearInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![
                0 => Uint32x2,
                1 => Uint32x2,
                2 => Uint32x2,
                3 => Uint32,
            ],
        };
        let create_clear_pipeline = |label, format| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&clear_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &clear_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(clear_vertex_state.clone())],
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &clear_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format,
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
                cache: pipeline_cache,
            })
        };
        let clear_pipeline =
            create_clear_pipeline("Clear Pipeline", wgpu::TextureFormat::Rgba8Unorm);
        let root_clear_pipeline =
            create_clear_pipeline("Root Clear Pipeline", render_target_config.format);

        // Create atlas clear pipeline
        let atlas_clear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Atlas Clear Pipeline Layout"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
        // TODO: Change the atlas clear pipeline to use the existing layer clear mechanism.
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
                entry_point: Some("fs_transparent"),
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
            cache: pipeline_cache,
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
            // The original layer texture.
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Filter Original Texture Bind Group Layout"),
                entries: &[filter_texture_entry],
            }),
        ];

        let filter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Filter Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_gpu_shaders::wgsl::FILTER.into()),
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
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<FilterInstanceData>() as u64,
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
                        8 => Uint32,
                    ],
                })],
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
            cache: pipeline_cache,
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let copy_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Copy Bind Group Layout"),
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
        let blend_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blend Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_gpu_shaders::wgsl::BLEND.into()),
        });
        let copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Copy Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_gpu_shaders::wgsl::COPY.into()),
        });
        let blend_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Blend Pipeline Layout"),
                bind_group_layouts: &[Some(&blend_layer_bind_group_layout)],
                immediate_size: 0,
            });
        let copy_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Copy Pipeline Layout"),
            bind_group_layouts: &[Some(&copy_bind_group_layout)],
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
        let create_texture_op_pipeline =
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
                        buffers: &[Some(vertex_state.clone())],
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
                    cache: pipeline_cache,
                })
            };
        let blend_pipeline = create_texture_op_pipeline(
            "Blend Pipeline",
            &blend_shader,
            &blend_pipeline_layout,
            &blend_vertex_state,
        );
        let copy_pipeline = create_texture_op_pipeline(
            "Copy Pipeline",
            &copy_shader,
            &copy_pipeline_layout,
            &copy_vertex_state,
        );

        let texture_size = layer_config.min_texture_size;
        let filter_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Filter Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // The empty bind group needs a sampler even before any external image is drawn.
        let mut external_samplers = HashMap::new();
        external_samplers.insert(
            ExternalSampler::DEFAULT,
            Self::create_external_sampler(device, ExternalSampler::DEFAULT),
        );

        let layer_textures: [Vec<TextureView>; 2] = core::array::from_fn(|_| Vec::new());
        let scratch_texture = None;
        let placeholder_external_texture_view = Self::create_placeholder_external_texture(device);
        let filter_original_bind_group = create_filter_original_texture_bind_group(
            device,
            &filter_input_bind_group_layouts[1],
            &placeholder_external_texture_view,
        );
        let layer_config_buffer = Self::create_config_buffer(
            device,
            u32::from(texture_size.width()),
            u32::from(texture_size.height()),
            resource_texture_dimension_2d,
        );

        const INITIAL_ALPHA_TEXTURE_HEIGHT: u32 = 1;
        let alphas_texture = Self::create_alphas_texture(
            device,
            resource_texture_dimension_2d,
            INITIAL_ALPHA_TEXTURE_HEIGHT,
        );
        let view_config_buffer = Self::create_config_buffer(
            device,
            u32::from(render_target_config.width),
            u32::from(render_target_config.height),
            resource_texture_dimension_2d,
        );

        let AtlasConfig {
            atlas_size: (atlas_width, atlas_height),
            initial_atlas_count,
            ..
        } = image_cache.atlas_manager().config();
        let atlas_size = (*atlas_width, *atlas_height);
        let atlas_textures: Vec<_> = (0..*initial_atlas_count)
            .map(|_| Self::create_atlas_texture(device, *atlas_width, *atlas_height))
            .collect();
        let atlas_texture_views = atlas_textures
            .iter()
            .map(|texture| texture.create_view(&TextureViewDescriptor::default()))
            .collect();
        let empty_external_texture_bind_group = Self::create_external_texture_bind_group(
            device,
            &external_texture_bind_group_layout,
            [&placeholder_external_texture_view; EXTERNAL_TEXTURE_SLOT_COUNT],
            [&external_samplers[&ExternalSampler::DEFAULT]; EXTERNAL_TEXTURE_SLOT_COUNT],
            wgpu_external_textures.then_some(wgpu::BindingResource::TextureView(
                &placeholder_external_texture_view,
            )),
        );

        const INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT: u32 = 1;
        let encoded_paints_data = vec![
            0;
            ((resource_texture_dimension_2d * INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT) << 4)
                as usize
        ];
        let encoded_paints_texture = Self::create_encoded_paints_texture(
            device,
            resource_texture_dimension_2d,
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
            resource_texture_dimension_2d,
            INITIAL_GRADIENT_TEXTURE_HEIGHT,
        );
        let gradient_bind_group = Self::create_gradient_bind_group(
            device,
            &gradient_bind_group_layout,
            &gradient_texture.create_view(&TextureViewDescriptor::default()),
        );

        // TODO: We really should deduplicate handling of this this with encoded paints texture.
        const INITIAL_FILTER_TEXTURE_HEIGHT: u32 = 1;
        let filter_data = vec![
            0_u8;
            ((resource_texture_dimension_2d * INITIAL_FILTER_TEXTURE_HEIGHT) << 4)
                as usize
        ];
        let filter_data_texture = Self::create_filter_data_texture(
            device,
            resource_texture_dimension_2d,
            INITIAL_FILTER_TEXTURE_HEIGHT,
        );
        let filter_base_bind_group = Self::create_filter_base_bind_group(
            device,
            &filter_bind_group_layout,
            &filter_data_texture.create_view(&TextureViewDescriptor::default()),
        );
        let scratch_copy_bind_group = Self::create_copy_bind_group(
            device,
            &copy_bind_group_layout,
            &placeholder_external_texture_view,
        );

        let resources = GpuResources {
            layer_textures,
            scratch_texture,
            filter_original_bind_group,
            scratch_copy_bind_group,
            layer_config_buffer,
            alphas_texture,
            resource_texture_dimension_2d,
            atlas_textures,
            atlas_texture_views,
            atlas_size,
            placeholder_external_texture_view,
            empty_external_texture_bind_group,
            encoded_paints_texture,
            encoded_paints_bind_group,
            gradient_texture,
            gradient_bind_group,
            filter_data_texture,
            filter_base_bind_group,
            texture_size,
            view_config_buffer,
        };

        Self {
            intermediate_strip_pipeline,
            alpha_strip_pipeline,
            depth_alpha_strip_pipeline,
            opaque_strip_pipeline,
            depth_cleared_this_frame: false,
            strip_bind_group_layout,
            encoded_paints_bind_group_layout,
            gradient_bind_group_layout,
            external_texture_bind_group_layout,
            filter_bind_group_layout,
            filter_pipeline,
            blend_pipeline,
            copy_pipeline,
            resources,
            strips_arena: StripBufferArena::new(device),
            wgpu_external_textures,
            staging_belt: UploadStagingBelt::new(device),
            encoded_paints_data,
            filter_data,
            render_size: RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },
            scene_offset: (0, 0),
            clear_pipeline,
            root_clear_pipeline,
            atlas_clear_pipeline,
            filter_input_bind_group_layouts,
            filter_sampler,
            external_samplers,
            blend_layer_bind_group_layout,
            copy_bind_group_layout,
            strip_layer_bind_groups: HashMap::new(),
            blend_layer_bind_groups: HashMap::new(),
            filter_pair_bind_groups: HashMap::new(),
        }
    }
}

/// A struct containing references to the many objects needed to get work
/// scheduled onto the GPU.
pub(crate) struct RendererContext<'a> {
    pub(crate) programs: &'a mut Programs,
    pub(crate) device: &'a Device,
    pub(crate) encoder: &'a mut CommandEncoder,
    pub(crate) view: &'a TextureView,
    pub(crate) depth_view: Option<&'a TextureView>,
    pub(crate) texture_bindings: &'a TextureBindings,
    pub(crate) external_texture_bind_groups: HashMap<ExternalTextureBindings, BindGroup>,
    pub(crate) scratch_buffers: &'a mut ScratchBuffers,
    pub(crate) root_load_op: wgpu::LoadOp<wgpu::Color>,
}

impl RendererContext<'_> {
    pub(crate) fn init_root_clear(&mut self, target_init: TargetInit<'_>, root_target: RootTarget) {
        self.root_load_op = match target_init {
            TargetInit::SrcOver => wgpu::LoadOp::Load,
            TargetInit::Clear(ClearSettings::Viewport { color }) => {
                wgpu::LoadOp::Clear(clear_color(color))
            }
            TargetInit::Clear(clear @ ClearSettings::Rects { .. }) => {
                self.clear_pass_inner(DrawPassTarget::Root(root_target), clear);

                wgpu::LoadOp::Load
            }
        };
    }

    pub(crate) fn finish_root_clear(&mut self) {
        // If the load mode was set to clear but no actual drawing instructions were
        // emitted, we still need to start a pass to clear the output.
        if let wgpu::LoadOp::Clear(color) = self.take_root_load_op() {
            Self::clear_full_target(self.encoder, self.view, color);
        }
    }

    fn take_root_load_op(&mut self) -> wgpu::LoadOp<wgpu::Color> {
        core::mem::replace(&mut self.root_load_op, wgpu::LoadOp::Load)
    }

    fn texture_size(&self) -> SizeU16 {
        self.programs.resources.texture_size
    }
}

impl Backend for RendererContext<'_> {
    type Error = core::convert::Infallible;

    fn opaque_draw_pass(
        &mut self,
        strips: &[GpuStrip],
        external_texture_runs: &[ExternalTextureRun],
    ) -> Result<(), Self::Error> {
        self.strip_pass_inner(
            strips,
            RangedSlice::empty(),
            external_texture_runs,
            DrawPassTarget::Root(RootTarget::UserSurface),
            None,
        );

        Ok(())
    }

    fn draw_pass(
        &mut self,
        strips: RangedSlice<'_, GpuStrip>,
        external_texture_runs: &[ExternalTextureRun],
        bindings: DrawPassBindings,
    ) -> Result<(), Self::Error> {
        self.strip_pass_inner(
            &[],
            strips,
            external_texture_runs,
            bindings.target,
            bindings.child,
        );

        Ok(())
    }

    fn blend_pass(
        &mut self,
        blends: RangedSlice<'_, BlendOp>,
        blend_strips: &[BlendStrip],
        bindings: BlendPassBindings,
    ) -> Result<(), Self::Error> {
        self.blend_pass_inner(blends, blend_strips, bindings);

        Ok(())
    }

    fn filter_pass(
        &mut self,
        plan: &FilterPassPlan,
        bindings: FilterPassBindings,
    ) -> Result<(), Self::Error> {
        self.filter_pass_inner(plan, bindings);

        Ok(())
    }

    fn clear_pass(&mut self, target: LayerTextureId, rects: &[RectU16]) -> Result<(), Self::Error> {
        self.clear_pass_inner(
            DrawPassTarget::Layer(target),
            ClearSettings::Rects {
                color: AlphaColor::TRANSPARENT,
                rects,
            },
        );

        Ok(())
    }
}
