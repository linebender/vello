//! Rasterize Bintje's wide tile per-tile command lists using wgpu.
//!
//! The limits are low enough to, in principle, run on WebGL2.
//!
//! This currently hardcodes the texture size to 256x256 pixels.

use color::PremulRgba8;
use wgpu::util::DeviceExt;

/// Re-export pollster's `block_on` for convenience.
pub use pollster::block_on;

/// Targetting WebGL2.
const LIMITS: wgpu::Limits = wgpu::Limits::downlevel_webgl2_defaults();

/// Number of tiles per wide tile.
pub(crate) const WIDE_TILE_WIDTH_TILES: u16 = 32;

/// Number of pixels per wide tile.
pub(crate) const WIDE_TILE_WIDTH_PX: u16 = Tile::WIDTH * WIDE_TILE_WIDTH_TILES;

#[derive(Debug)]
pub enum Command {
    /// A fill sampling from an alpha mask.
    Sample(Sample),

    /// A fill between two strips sampling from an alpha mask column.
    SparseSample(SparseSample),
    /// An opaque fill between two strips.
    SparseFill(SparseFill),

    /// TODO(Tom).
    PushClip(()),
    /// TODO(Tom).
    PopClip(()),
}

#[derive(Debug)]
pub struct Sample {
    /// The offset within the wide tile, in tiles.
    pub x: u16,
    /// The width of the area to be filled, in tiles.
    pub width: u16,
    pub color: PremulRgba8,
    /// The index into the global alpha mask, encoding the pixel coverage of the area to be filled.
    pub alpha_idx: u32,
}

#[derive(Debug)]
pub struct SparseSample {
    pub x: u16,
    pub width: u16,
    pub color: PremulRgba8,
    pub alpha_mask: [u8; Tile::HEIGHT as usize],
}

#[derive(Clone, Copy, Debug)]
pub struct Tile {
    /// The tile x-coordinate.
    pub(crate) x: u16,
    /// The index of the line that belongs to this tile into the line buffer.
    pub(crate) line_idx: u32,
}

impl Tile {
    /// Tile width in pixels.
    pub const WIDTH: u16 = 4;

    /// Tile height in pixels.
    pub const HEIGHT: u16 = 4;
}

#[derive(Debug)]
pub struct SparseFill {
    pub x: u16,
    pub width: u16,
    pub color: PremulRgba8,
}

#[derive(Debug)]
pub struct WideTile {
    pub commands: Vec<Command>,
}

impl WideTile {
    /// Number of tiles per wide tile.
    pub const WIDTH_TILES: u16 = WIDE_TILE_WIDTH_TILES;

    /// Number of pixels per wide tile.
    pub const WIDTH_PX: u16 = WIDE_TILE_WIDTH_PX;
}

pub struct RenderContext {
    #[expect(unused, reason = "might come in handy later")]
    instance: wgpu::Instance,
    #[expect(unused, reason = "might come in handy later")]
    adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawConfig {
    width: u32,
    height: u32,
}

impl RenderContext {
    pub async fn create() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
            .await
            .expect("would like to get an adapter");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: LIMITS,
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("failed to find a device");

        RenderContext {
            instance,
            adapter,
            device,
            queue,
        }
    }

    /// Create the actual rasterizer. Currently this only creates the shader required for
    /// rasterizing draw commands (fills with and without alpha masks).
    pub fn rasterizer(&mut self, width: u16, height: u16) -> Rasterizer {
        let draw_shader = self
            .device
            .create_shader_module(wgpu::include_wgsl!("../shaders/draw.wgsl"));

        let target_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: width.into(),
                height: height.into(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let vertex_instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertex instance buffer"),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            // TODO(Tom): how to determine a good size for this buffer?
            size: 1 << 19, // 512 KiB
            mapped_at_creation: false,
        });
        let draw_config_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("draw config buffer"),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    contents: bytemuck::bytes_of(&DrawConfig {
                        width: width.into(),
                        height: height.into(),
                    }),
                });
        let alpha_masks_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("alpha masks buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            // TODO(Tom): how to determine a good size for this buffer?
            size: 1 << 19, // 512 KiB
            mapped_at_creation: false,
        });
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        // Draw configuration uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: Some(
                                    draw_config_buffer.size().try_into().unwrap(),
                                ),
                            },
                            count: None,
                        },
                        // Alpha masks
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: true,
                                min_binding_size: Some(
                                    (LIMITS.max_uniform_buffer_binding_size as u64)
                                        .try_into()
                                        .unwrap(),
                                ),
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &draw_shader,
                    entry_point: Some("vs"),
                    buffers: &[DrawCmdVertexInstance::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &draw_shader,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        // We send non-linear sRGB8 to the shader, but let the shader pretend its
                        // linear sRGB.
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Cw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        Rasterizer {
            device: self.device.clone(),
            queue: self.queue.clone(),
            pipeline,

            width,
            height,

            target_texture,
            texture_copy_buffer: TextureCopyBuffer::new(&self.device, width, height),

            bind_group_layout,
            vertex_instance_buffer,
            draw_config_buffer,
            alpha_masks_buffer,

            fine_time: std::time::Duration::ZERO,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawCmdVertexInstance {
    x: u16,
    y: u16,
    width: u16,
    alpha_idx: u16,
    color: PremulRgba8,
    column_mask: [u8; Tile::HEIGHT as usize],
}

impl DrawCmdVertexInstance {
    fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Uint16,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<u16>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Uint16,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[u16; 2]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Uint16,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[u16; 3]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Uint16,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[u16; 4]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[u16; 6]>() as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Unorm8x4,
                },
            ],
        }
    }
}

pub struct Rasterizer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::RenderPipeline,

    width: u16,
    height: u16,

    target_texture: wgpu::Texture,
    texture_copy_buffer: TextureCopyBuffer,

    bind_group_layout: wgpu::BindGroupLayout,
    vertex_instance_buffer: wgpu::Buffer,
    draw_config_buffer: wgpu::Buffer,
    alpha_masks_buffer: wgpu::Buffer,

    pub fine_time: std::time::Duration,
}

/// A buffer to copy textures into from the GPU.
///
/// This pads internal buffer to adhere to the `bytes_per_row` size requirement of
/// [`wgpu::CommandEncoder::copy_texture_to_buffer`], see [`wgpu::TexelCopyBufferLayout`].
struct TextureCopyBuffer {
    buffer: wgpu::Buffer,
    bytes_per_row: u32,
}

impl TextureCopyBuffer {
    pub fn new(device: &wgpu::Device, width: u16, height: u16) -> Self {
        let bytes_per_row = ((width as u32) * 4).next_multiple_of(256);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texture-out"),
            size: bytes_per_row as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            bytes_per_row,
        }
    }
}

impl Rasterizer {
    fn add_draw_render_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        clear_texture: bool,
        instances: &[DrawCmdVertexInstance],
        instance_offsets: &[u32],
        alpha_mask_buf_step: u32,
    ) {
        self.queue.write_buffer(
            &self.vertex_instance_buffer,
            0,
            bytemuck::cast_slice(instances),
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self
                        .target_texture
                        .create_view(&wgpu::TextureViewDescriptor::default()),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if clear_texture {
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.draw_config_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.alpha_masks_buffer,
                            offset: 0,
                            size: Some(
                                (LIMITS.max_uniform_buffer_binding_size as u64)
                                    .try_into()
                                    .unwrap(),
                            ),
                        }),
                    },
                ],
            });

            render_pass.set_vertex_buffer(
                0,
                self.vertex_instance_buffer
                    .slice(0..(instances.len() * size_of::<DrawCmdVertexInstance>()) as u64),
            );
            render_pass.set_pipeline(&self.pipeline);
            let mut instance_offset = 0;
            for step in 0..=alpha_mask_buf_step {
                let next_instance_offset = instance_offsets[step as usize];
                render_pass.set_bind_group(
                    0,
                    &bind_group,
                    &[step * LIMITS.max_uniform_buffer_binding_size],
                );
                render_pass.draw(0..4, instance_offset..next_instance_offset as u32);
                instance_offset = next_instance_offset;
            }
        }
    }

    /// Rasterize the per-tile command lists and given alpha masks, and copy the resulting GPU
    /// texture to the destination image.
    ///
    /// Note: the texture size is currently hardcoded to 256x256 pixels.
    pub fn rasterize(
        &mut self,
        alpha_masks: &[u8],
        wide_tiles: &[WideTile],
        width: u16,
        dest_img: &mut [u8],
    ) {
        let t_start = std::time::Instant::now();
        let wide_tiles_per_row = width.div_ceil(WideTile::WIDTH_PX);
        let mut submits = 0;

        let mut instances = Vec::with_capacity(
            (self.vertex_instance_buffer.size()
                / std::mem::size_of::<DrawCmdVertexInstance>() as u64) as usize,
        );
        let mut instance_offsets = Vec::with_capacity(
            self.alpha_masks_buffer.size() as usize
                / LIMITS.max_uniform_buffer_binding_size as usize,
        );

        // The uniform-buffer-sized step within the alpha mask buffer.
        let mut alpha_masks_buffer_step = 0;
        let mut alpha_masks_buffer_idx = 0;
        let mut alpha_masks_buffer = self
            .queue
            .write_buffer_with(
                &self.alpha_masks_buffer,
                0,
                self.alpha_masks_buffer.size().try_into().unwrap(),
            )
            .unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut render_target_cleared = false;
        for (idx, wide_tile) in wide_tiles.iter().enumerate() {
            let wide_tile_y = (idx / wide_tiles_per_row as usize) as u16;
            let wide_tile_x = (idx - (wide_tile_y as usize * wide_tiles_per_row as usize)) as u16;

            // TODO(Tom): this doesn't account for overflowing the vertex instance buffer (what are
            // the limits?)
            for command in &wide_tile.commands {
                match command {
                    Command::Sample(sample) => {
                        let alpha_mask_size =
                            sample.width as usize * Tile::WIDTH as usize * Tile::HEIGHT as usize;
                        if alpha_masks_buffer_idx + alpha_mask_size
                            > (alpha_masks_buffer_step + 1) as usize
                                * LIMITS.max_uniform_buffer_binding_size as usize
                        {
                            alpha_masks_buffer_step += 1;
                            instance_offsets.push(instances.len() as u32);

                            if alpha_masks_buffer_step
                                == (self.alpha_masks_buffer.size()
                                    / LIMITS.max_uniform_buffer_binding_size as u64)
                                    as u32
                            {
                                self.add_draw_render_pass(
                                    &mut encoder,
                                    !render_target_cleared,
                                    &instances,
                                    &instance_offsets,
                                    alpha_masks_buffer_step - 1,
                                );
                                render_target_cleared = true;
                                let encoder = std::mem::replace(
                                    &mut encoder,
                                    self.device.create_command_encoder(
                                        &wgpu::CommandEncoderDescriptor { label: None },
                                    ),
                                );
                                // Replace the writable alpha mask view. The old view is dropped
                                // and the data is queued for uploading.
                                let _ = std::mem::replace(
                                    &mut alpha_masks_buffer,
                                    self.queue
                                        .write_buffer_with(
                                            &self.alpha_masks_buffer,
                                            0,
                                            self.alpha_masks_buffer.size().try_into().unwrap(),
                                        )
                                        .unwrap(),
                                );
                                self.queue.submit([encoder.finish()]);
                                submits += 1;
                                instances.clear();
                                instance_offsets.clear();
                                alpha_masks_buffer_step = 0;
                                alpha_masks_buffer_idx = 0;
                            }
                        }
                        let alpha_idx = alpha_masks_buffer_idx
                            % LIMITS.max_uniform_buffer_binding_size as usize;
                        alpha_masks_buffer
                            [alpha_masks_buffer_idx..alpha_masks_buffer_idx + alpha_mask_size]
                            .copy_from_slice(
                                &alpha_masks[sample.alpha_idx as usize
                                    ..sample.alpha_idx as usize + alpha_mask_size],
                            );
                        alpha_masks_buffer_idx += alpha_mask_size;
                        instances.push(DrawCmdVertexInstance {
                            x: (wide_tile_x * WideTile::WIDTH_TILES + sample.x) * Tile::WIDTH,
                            y: wide_tile_y * Tile::HEIGHT,
                            width: sample.width * Tile::WIDTH,
                            color: sample.color,
                            alpha_idx: alpha_idx as u16 / (Tile::WIDTH * Tile::HEIGHT),
                            column_mask: [255; Tile::HEIGHT as usize],
                        });
                    }
                    Command::SparseSample(sparse_sample) => {
                        instances.push(DrawCmdVertexInstance {
                            x: (wide_tile_x * WideTile::WIDTH_TILES + sparse_sample.x)
                                * Tile::WIDTH,
                            y: wide_tile_y * Tile::HEIGHT,
                            width: sparse_sample.width * Tile::WIDTH,
                            color: sparse_sample.color,
                            alpha_idx: u16::MAX,
                            column_mask: sparse_sample.alpha_mask,
                        });
                    }
                    Command::SparseFill(sparse_fill) => {
                        instances.push(DrawCmdVertexInstance {
                            x: (wide_tile_x * WideTile::WIDTH_TILES + sparse_fill.x) * Tile::WIDTH,
                            y: wide_tile_y * Tile::HEIGHT,
                            width: sparse_fill.width * Tile::WIDTH,
                            color: sparse_fill.color,
                            alpha_idx: u16::MAX,
                            column_mask: [255; Tile::HEIGHT as usize],
                        });
                    }
                    _ => {}
                }
            }
        }
        if !instances.is_empty() {
            // Drop the writable alpha mask view, the data is queued for uploading.
            drop(alpha_masks_buffer);
            instance_offsets.push(instances.len() as u32);
            self.add_draw_render_pass(
                &mut encoder,
                !render_target_cleared,
                &instances,
                &instance_offsets,
                alpha_masks_buffer_step,
            );
            self.queue.submit([encoder.finish()]);
            submits += 1;
        }
        dbg!(submits);

        // Do not account for copying the buffer out to the texture. That wouldn't happen when
        // rendering to the surface.
        self.fine_time += t_start.elapsed();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.texture_copy_buffer.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    // Must be a multiple of 256 bytes.
                    bytes_per_row: Some(self.texture_copy_buffer.bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.width.into(),
                height: self.height.into(),
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        self.texture_copy_buffer
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_err() {
                    panic!("failed to map texture for reading")
                }
            });

        self.device.poll(wgpu::Maintain::Wait);
        let mut img_idx = 0;
        for row in (self.texture_copy_buffer.buffer.slice(..).get_mapped_range())
            .chunks_exact(self.texture_copy_buffer.bytes_per_row as usize)
        {
            dest_img[img_idx..img_idx + self.width as usize * 4]
                .copy_from_slice(&row[0..self.width as usize * 4]);
            img_idx += self.width as usize * 4;
        }
        self.texture_copy_buffer.buffer.unmap();
    }
}
