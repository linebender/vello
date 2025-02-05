use vello_encoding::{
    BinHeader, BufferSize, BumpAllocators, Clip, ClipBbox, ClipBic, ClipElement, DrawBbox,
    DrawMonoid, Encoding, IndirectCount, Layout, LineSoup, Path, PathBbox, PathMonoid, PathSegment,
    RenderConfig, Resolver, SegmentCount, Tile,
};
use vello_shaders::{
    cpu::{
        backdrop_main, bbox_clear_main, binning_main, clip_leaf_main, clip_reduce_main,
        coarse_main, draw_leaf_main, draw_reduce_main, flatten_main, path_count_main,
        path_count_setup_main, path_tiling_main, path_tiling_setup_main, pathtag_reduce_main,
        pathtag_scan_main, tile_alloc_main,
    },
    SHADERS,
};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipelineDescriptor, Device, PipelineCompilationOptions, Queue,
    TexelCopyTextureInfo, TextureAspect, TextureDescriptor, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor,
};

use crate::RenderParams;

#[derive(Default)]
pub struct Buffer<T: bytemuck::Zeroable + bytemuck::NoUninit> {
    inner: Vec<T>,
}

impl<T: bytemuck::Zeroable + bytemuck::NoUninit> Buffer<T> {
    fn to_fit(&mut self, size: BufferSize<T>) -> &mut [T] {
        self.inner
            .resize_with(size.len().try_into().expect("32 bit platform"), || {
                T::zeroed()
            });
        &mut self.inner
    }

    fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.inner)
    }
}

#[derive(Default)]
pub struct CoarseBuffers {
    packed: Vec<u8>,
    path_reduced: Buffer<PathMonoid>,
    path_monoids: Buffer<PathMonoid>,
    path_bboxes: Buffer<PathBbox>,
    draw_reduced: Buffer<DrawMonoid>,
    draw_monoids: Buffer<DrawMonoid>,
    info: Buffer<u32>,
    clip_inps: Buffer<Clip>,
    clip_els: Buffer<ClipElement>,
    clip_bics: Buffer<ClipBic>,
    clip_bboxes: Buffer<ClipBbox>,
    draw_bboxes: Buffer<DrawBbox>,
    bump_alloc: BumpAllocators,
    bin_headers: Buffer<BinHeader>,
    paths: Buffer<Path>,
    // Bump allocated buffers
    lines: Buffer<LineSoup>,
    bin_data: Buffer<u32>,
    tiles: Buffer<Tile>,
    seg_counts: Buffer<SegmentCount>,
    segments: Buffer<PathSegment>,
    ptcl: Buffer<u32>,
}

pub fn run_coarse_cpu(
    params: &RenderParams,
    buffers: &mut CoarseBuffers,
    cpu_config: &RenderConfig,
) {
    let packed = &mut buffers.packed;

    // HACK: The coarse workgroup counts is the number of active bins.
    if (cpu_config.workgroup_counts.coarse.0
        * cpu_config.workgroup_counts.coarse.1
        * cpu_config.workgroup_counts.coarse.2)
        > 256
    {
        log::warn!(
            "Trying to paint too large image. {}x{}.\n\
                See https://github.com/linebender/vello/issues/680 for details",
            params.width,
            params.height
        );
    }
    let buffer_sizes = &cpu_config.buffer_sizes;
    let wg_counts = &cpu_config.workgroup_counts;

    // TODO: This is an alignment hazard, which just happens to work on mainstream platforms
    // Maybe don't merge as-is?
    let scene_buf = bytemuck::cast_slice(packed);
    let config_buf = cpu_config.gpu;
    let info_bin_data_buf = buffers.bin_data.to_fit(buffer_sizes.bin_data);
    let tile_buf = buffers.tiles.to_fit(buffer_sizes.tiles);
    let segments_buf = buffers.segments.to_fit(buffer_sizes.segments);

    let ptcl_buf = buffers.ptcl.to_fit(buffer_sizes.ptcl);
    let reduced_buf = buffers.path_reduced.to_fit(buffer_sizes.path_reduced);

    pathtag_reduce_main(wg_counts.path_reduce.0, &config_buf, scene_buf, reduced_buf);

    let tagmonoid_buf = buffers.path_monoids.to_fit(buffer_sizes.path_monoids);

    pathtag_scan_main(
        wg_counts.path_scan.0,
        &config_buf,
        scene_buf,
        reduced_buf,
        tagmonoid_buf,
    );

    // Could re-use `reduced_buf` from this point

    let path_bbox_buf = buffers.path_bboxes.to_fit(buffer_sizes.path_bboxes);

    bbox_clear_main(&config_buf, path_bbox_buf);
    let bump_buf = &mut buffers.bump_alloc;
    let lines_buf = buffers.lines.to_fit(buffer_sizes.lines);
    flatten_main(
        wg_counts.flatten.0,
        &config_buf,
        scene_buf,
        tagmonoid_buf,
        path_bbox_buf,
        bump_buf,
        lines_buf,
    );

    let draw_reduced_buf = buffers.draw_reduced.to_fit(buffer_sizes.draw_reduced);

    draw_reduce_main(
        wg_counts.draw_reduce.0,
        &config_buf,
        scene_buf,
        draw_reduced_buf,
    );

    let draw_monoid_buf = buffers.draw_monoids.to_fit(buffer_sizes.draw_monoids);
    let clip_inp_buf = buffers.clip_inps.to_fit(buffer_sizes.clip_inps);
    draw_leaf_main(
        wg_counts.draw_leaf.0,
        &config_buf,
        scene_buf,
        draw_reduced_buf,
        path_bbox_buf,
        draw_monoid_buf,
        info_bin_data_buf,
        clip_inp_buf,
    );

    // Could re-use `draw_reduced_buf` from this point

    let clip_el_buf = buffers.clip_els.to_fit(buffer_sizes.clip_els);

    let clip_bic_buf = buffers.clip_bics.to_fit(buffer_sizes.clip_bics);

    if wg_counts.clip_reduce.0 > 0 {
        clip_reduce_main(
            wg_counts.clip_reduce.0,
            clip_inp_buf,
            path_bbox_buf,
            clip_bic_buf,
            clip_el_buf,
        );
    }
    let clip_bbox_buf = buffers.clip_bboxes.to_fit(buffer_sizes.clip_bboxes);

    if wg_counts.clip_leaf.0 > 0 {
        clip_leaf_main(
            &config_buf,
            clip_inp_buf,
            path_bbox_buf,
            draw_monoid_buf,
            clip_bbox_buf,
        );
    }

    // Could re-use `clip_inp_buf`, `clip_bic_buf`, and `clip_el_buf` from this point

    let draw_bbox_buf = buffers.draw_bboxes.to_fit(buffer_sizes.draw_bboxes);

    let bin_header_buf = buffers.bin_headers.to_fit(buffer_sizes.bin_headers);

    binning_main(
        wg_counts.binning.0,
        &config_buf,
        draw_monoid_buf,
        path_bbox_buf,
        clip_bbox_buf,
        draw_bbox_buf,
        bump_buf,
        info_bin_data_buf,
        bin_header_buf,
    );

    // Could re-use `draw_monoid_buf` and `clip_bbox_buf` from this point

    // TODO: What does this comment mean?
    // Note: this only needs to be rounded up because of the workaround to store the tile_offset
    // in storage rather than workgroup memory.
    let path_buf = buffers.paths.to_fit(buffer_sizes.paths);
    tile_alloc_main(
        &config_buf,
        scene_buf,
        draw_bbox_buf,
        bump_buf,
        path_buf,
        tile_buf,
    );

    // Could re-use `draw_bbox_buf` and `tagmonoid_buf` from this point

    let mut indirect_count_buf = IndirectCount::default();

    path_count_setup_main(bump_buf, &mut indirect_count_buf);

    let seg_counts_buf = buffers.seg_counts.to_fit(buffer_sizes.seg_counts);
    path_count_main(bump_buf, lines_buf, path_buf, tile_buf, seg_counts_buf);

    backdrop_main(&config_buf, bump_buf, path_buf, tile_buf);

    coarse_main(
        &config_buf,
        scene_buf,
        draw_monoid_buf,
        bin_header_buf,
        info_bin_data_buf,
        path_buf,
        tile_buf,
        bump_buf,
        ptcl_buf,
    );

    path_tiling_setup_main(
        bump_buf,
        &mut indirect_count_buf, /* ptcl_buf (for forwarding errors to fine)*/
    );

    path_tiling_main(
        bump_buf,
        seg_counts_buf,
        lines_buf,
        path_buf,
        tile_buf,
        segments_buf,
    );
}

pub fn render_to_texture(
    encoding: &Encoding,
    resolver: &mut Resolver,
    buffers: &mut CoarseBuffers,
    device: &Device,
    queue: &Queue,
    texture: &TextureView,
    params: &RenderParams,
) -> wgpu::CommandBuffer {
    let (layout, ramps, images) = resolver.resolve(encoding, &mut buffers.packed);
    let cpu_config = RenderConfig::new(&layout, params.width, params.height, &params.base_color);
    run_coarse_cpu(params, buffers, &cpu_config);
    // Yes, this needs to be retained. This is very intentionally done before API design to retain things properly.
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("TEMP FINE"),
        source: wgpu::ShaderSource::Wgsl(SHADERS.fine_area.wgsl.code),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: None,
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);

    let config_buf = device.create_buffer_init(&BufferInitDescriptor {
        contents: bytemuck::bytes_of(&cpu_config.gpu),
        label: None,
        usage: BufferUsages::UNIFORM,
    });
    let segments_buf = device.create_buffer_init(&BufferInitDescriptor {
        contents: buffers.segments.bytes(),
        label: None,
        usage: BufferUsages::STORAGE,
    });
    let ptcl_buf = device.create_buffer_init(&BufferInitDescriptor {
        contents: buffers.ptcl.bytes(),
        label: None,
        usage: BufferUsages::STORAGE,
    });
    let info_bin_data_buf = device.create_buffer_init(&BufferInitDescriptor {
        contents: buffers.bin_data.bytes(),
        label: None,
        usage: BufferUsages::STORAGE,
    });
    let blend_spill_buf = device.create_buffer(&BufferDescriptor {
        label: None,
        size: cpu_config.buffer_sizes.blend_spill.size_in_bytes().into(),
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let gradient_image = if ramps.height == 0 {
        device.create_texture(&TextureDescriptor {
            label: None,
            size: wgpu::Extent3d::default(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    } else {
        device.create_texture_with_data(
            queue,
            &TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: ramps.width,
                    height: ramps.height,
                    ..Default::default()
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::default(),
            bytemuck::cast_slice(ramps.data),
        )
    };
    let gradient_view = gradient_image.create_view(&TextureViewDescriptor::default());

    let (image_w, image_h) = if images.images.is_empty() {
        (1, 1)
    } else {
        (images.width, images.height)
    };
    let format = TextureFormat::Rgba8Unorm;
    let image_proxy = device.create_texture(&TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: image_w,
            height: image_h,
            ..Default::default()
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let block_size = format
        .block_copy_size(None)
        .expect("ImageFormat must have a valid block size");
    for (image, x, y) in images.images {
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &image_proxy,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: *x,
                    y: *y,
                    ..Default::default()
                },
                aspect: TextureAspect::All,
            },
            image.data.data(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(image.width * block_size),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: image.width,
                height: image.height,
                depth_or_array_layers: 1,
            },
        );
    }
    let image_view = image_proxy.create_view(&TextureViewDescriptor::default());

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &config_buf,
                    offset: 0,
                    size: None,
                }),
            },
            BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &segments_buf,
                    offset: 0,
                    size: None,
                }),
            },
            BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &ptcl_buf,
                    offset: 0,
                    size: None,
                }),
            },
            BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &info_bin_data_buf,
                    offset: 0,
                    size: None,
                }),
            },
            BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &blend_spill_buf,
                    offset: 0,
                    size: None,
                }),
            },
            BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(texture),
            },
            BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&gradient_view),
            },
            BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&image_view),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) = cpu_config.workgroup_counts.fine;
        cpass.dispatch_workgroups(x, y, z);
    }
    encoder.finish()
}
