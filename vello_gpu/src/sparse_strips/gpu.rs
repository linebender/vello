// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! wgpu driver of the GPU sparse strip passes.

use super::{ACC_WORDS, GpuPathInfo, GpuStripDesc, MERGE_WG, StripInput, TILE_PIXELS};
use alloc::vec::Vec;
use vello_common::strip::Strip;
use wgpu::util::DeviceExt;

/// Byte size of one [`super::model::Agg`] in a storage buffer (two `vec2<u32>` plus five
/// scalars, rounded up to the 8-byte alignment).
const AGG_SIZE: u64 = 40;
/// Words in the indirect buffer: resolve dispatch arguments, strip total, block total.
const INDIRECT_WORDS: u64 = 5;

/// The compute pipelines of the sparse strip passes.
#[derive(Debug)]
pub struct SparseStripPipelines {
    input_layout: wgpu::BindGroupLayout,
    alloc_layout: wgpu::BindGroupLayout,
    merge_layout: wgpu::BindGroupLayout,
    resolve_layout: wgpu::BindGroupLayout,
    reduce: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    alloc: wgpu::ComputePipeline,
    merge: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
}

/// The buffers and bind groups of one input.
#[derive(Debug)]
pub struct SparseStripBuffers {
    /// Per-path info; the strip and block ranges are written by the GPU.
    pub paths: wgpu::Buffer,
    /// Strips as `(x | y << 16, alpha_idx | fill_gap << 31)`, `indirect[3]` of them.
    pub strips: wgpu::Buffer,
    /// Alphas, `indirect[4]` blocks of 16 bytes.
    pub alphas: wgpu::Buffer,
    /// Resolve dispatch arguments and the strip and block totals.
    pub indirect: wgpu::Buffer,
    /// Capacity of `strips`, in strips.
    pub max_strips: u32,
    /// Capacity of `alphas`, in blocks.
    pub max_blocks: u32,
    n_paths: u32,
    n_entries: u32,
    n_wg: u32,
    acc: wgpu::Buffer,
    input_group: wgpu::BindGroup,
    alloc_group: wgpu::BindGroup,
    merge_group: wgpu::BindGroup,
    resolve_group: wgpu::BindGroup,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl SparseStripPipelines {
    /// Create the pipelines.
    pub fn new(device: &wgpu::Device) -> Self {
        let input_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sparse strips input"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
            ],
        });
        let alloc_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sparse strips alloc"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
            ],
        });
        let merge_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sparse strips merge"),
            entries: &[storage_entry(0, true), storage_entry(1, false)],
        });
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sparse strips resolve"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
            ],
        });

        let make = |label: &str, source: &str, layout: &wgpu::BindGroupLayout, entry: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&input_layout), Some(layout)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let alloc_src = vello_gpu_shaders::wgsl::SPARSE_STRIPS_ALLOC;
        let reduce = make("sparse strips reduce", alloc_src, &alloc_layout, "reduce");
        let scan = make("sparse strips scan", alloc_src, &alloc_layout, "scan");
        let alloc = make("sparse strips alloc", alloc_src, &alloc_layout, "alloc");
        let merge = make(
            "sparse strips merge",
            vello_gpu_shaders::wgsl::SPARSE_STRIPS_MERGE,
            &merge_layout,
            "merge",
        );
        let resolve = make(
            "sparse strips resolve",
            vello_gpu_shaders::wgsl::SPARSE_STRIPS_RESOLVE,
            &resolve_layout,
            "resolve",
        );
        Self {
            input_layout,
            alloc_layout,
            merge_layout,
            resolve_layout,
            reduce,
            scan,
            alloc,
            merge,
            resolve,
        }
    }

    /// Upload an input and allocate the output buffers.
    pub fn create_buffers(&self, device: &wgpu::Device, input: &StripInput) -> SparseStripBuffers {
        let config = input.config();
        let upload = |label: &str, data: &[u8], extra: wgpu::BufferUsages| {
            // Buffers cannot be empty.
            let data = if data.is_empty() { &[0_u8; 16] } else { data };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE | extra,
            })
        };
        let scratch = |label: &str, size: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };
        let config_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sparse strips config"),
            contents: bytemuck::bytes_of(&config),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let paths = upload(
            "sparse strips paths",
            bytemuck::cast_slice(&input.paths),
            wgpu::BufferUsages::COPY_SRC,
        );
        let aux = upload(
            "sparse strips aux",
            bytemuck::cast_slice(&input.aux),
            wgpu::BufferUsages::empty(),
        );
        let entries = upload(
            "sparse strips entries",
            bytemuck::cast_slice(&input.entries),
            wgpu::BufferUsages::empty(),
        );
        let lines = upload(
            "sparse strips lines",
            bytemuck::cast_slice(&input.lines),
            wgpu::BufferUsages::empty(),
        );
        let wg_aggs = scratch(
            "sparse strips wg aggs",
            u64::from(config.n_wg) * AGG_SIZE,
            wgpu::BufferUsages::empty(),
        );
        let entry_block = scratch(
            "sparse strips entry block",
            u64::from(config.n_entries) * 4,
            wgpu::BufferUsages::empty(),
        );
        let strip_descs = scratch(
            "sparse strips descs",
            u64::from(config.max_strips) * size_of::<GpuStripDesc>() as u64,
            wgpu::BufferUsages::empty(),
        );
        let indirect = scratch(
            "sparse strips indirect",
            INDIRECT_WORDS * 4,
            wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
        );
        let acc = scratch(
            "sparse strips acc",
            u64::from(config.max_blocks) * u64::from(ACC_WORDS) * 4,
            wgpu::BufferUsages::COPY_DST,
        );
        let strips = scratch(
            "sparse strips strips",
            u64::from(config.max_strips) * 8,
            wgpu::BufferUsages::COPY_SRC,
        );
        let alphas = scratch(
            "sparse strips alphas",
            u64::from(config.max_blocks) * u64::from(TILE_PIXELS),
            wgpu::BufferUsages::COPY_SRC,
        );

        let group = |label: &str, layout: &wgpu::BindGroupLayout, buffers: &[&wgpu::Buffer]| {
            let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &entries,
            })
        };
        let input_group = group(
            "sparse strips input",
            &self.input_layout,
            &[&config_buf, &paths, &aux, &entries, &lines],
        );
        let alloc_group = group(
            "sparse strips alloc",
            &self.alloc_layout,
            &[&wg_aggs, &entry_block, &strip_descs, &indirect],
        );
        let merge_group = group(
            "sparse strips merge",
            &self.merge_layout,
            &[&entry_block, &acc],
        );
        let resolve_group = group(
            "sparse strips resolve",
            &self.resolve_layout,
            &[&strip_descs, &acc, &strips, &alphas],
        );
        SparseStripBuffers {
            paths,
            strips,
            alphas,
            indirect,
            max_strips: config.max_strips,
            max_blocks: config.max_blocks,
            n_paths: config.n_paths,
            n_entries: config.n_entries,
            n_wg: config.n_wg,
            acc,
            input_group,
            alloc_group,
            merge_group,
            resolve_group,
        }
    }

    /// Record the passes. `buffers` must come from [`Self::create_buffers`] with a non-empty
    /// input.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, buffers: &SparseStripBuffers) {
        encoder.clear_buffer(&buffers.acc, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sparse strips"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &buffers.input_group, &[]);
        pass.set_bind_group(1, &buffers.alloc_group, &[]);
        pass.set_pipeline(&self.reduce);
        pass.dispatch_workgroups(buffers.n_wg, 1, 1);
        pass.set_pipeline(&self.scan);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.alloc);
        pass.dispatch_workgroups(buffers.n_wg, 1, 1);
        pass.set_bind_group(1, &buffers.merge_group, &[]);
        pass.set_pipeline(&self.merge);
        pass.dispatch_workgroups(buffers.n_entries.div_ceil(MERGE_WG), 1, 1);
        pass.set_bind_group(1, &buffers.resolve_group, &[]);
        pass.set_pipeline(&self.resolve);
        pass.dispatch_workgroups_indirect(&buffers.indirect, 0);
    }
}

/// GPU output read back to the CPU. For validation only.
#[derive(Debug, Default)]
pub struct Readback {
    /// Strips of all paths, in path order.
    pub strips: Vec<Strip>,
    /// Alphas of all paths.
    pub alphas: Vec<u8>,
    /// Per-path info with the strip and block ranges filled in.
    pub paths: Vec<GpuPathInfo>,
}

/// Copy the outputs into staging buffers, submit, and wait. Native only.
pub fn read_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    buffers: &SparseStripBuffers,
) -> Readback {
    let staging = |label: &str, size: u64| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let strips_size = u64::from(buffers.max_strips) * 8;
    let alphas_size = u64::from(buffers.max_blocks) * u64::from(TILE_PIXELS);
    let paths_size = u64::from(buffers.n_paths) * size_of::<GpuPathInfo>() as u64;
    let st_strips = staging("readback strips", strips_size);
    let st_alphas = staging("readback alphas", alphas_size);
    let st_paths = staging("readback paths", paths_size);
    let st_indirect = staging("readback indirect", INDIRECT_WORDS * 4);
    encoder.copy_buffer_to_buffer(&buffers.strips, 0, &st_strips, 0, strips_size.max(16));
    encoder.copy_buffer_to_buffer(&buffers.alphas, 0, &st_alphas, 0, alphas_size.max(16));
    encoder.copy_buffer_to_buffer(&buffers.paths, 0, &st_paths, 0, paths_size.max(16));
    encoder.copy_buffer_to_buffer(&buffers.indirect, 0, &st_indirect, 0, INDIRECT_WORDS * 4);
    queue.submit([encoder.finish()]);
    for b in [&st_strips, &st_alphas, &st_paths, &st_indirect] {
        b.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    }
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");

    let mapped = |b: &wgpu::Buffer| b.slice(..).get_mapped_range().expect("mapped");
    let indirect: Vec<u32> = bytemuck::cast_slice(&mapped(&st_indirect)).to_vec();
    let n_strips = indirect[3] as usize;
    let n_blocks = indirect[4] as usize;
    assert!(
        n_strips <= buffers.max_strips as usize,
        "strip bound exceeded"
    );
    assert!(
        n_blocks <= buffers.max_blocks as usize,
        "block bound exceeded"
    );
    let raw_strips: Vec<[u32; 2]> = bytemuck::cast_slice(&mapped(&st_strips))[..n_strips].to_vec();
    let strips = raw_strips
        .iter()
        .map(|s| {
            Strip::new(
                s[0] as u16,
                (s[0] >> 16) as u16,
                s[1] & 0x7fff_ffff,
                s[1] >> 31 != 0,
            )
        })
        .collect();
    let alphas = mapped(&st_alphas)[..n_blocks * TILE_PIXELS as usize].to_vec();
    let paths = bytemuck::cast_slice(&mapped(&st_paths))[..buffers.n_paths as usize].to_vec();
    Readback {
        strips,
        alphas,
        paths,
    }
}

/// Run the passes on `input` and read the result back. For validation and tools.
pub fn render_blocking(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipelines: &SparseStripPipelines,
    input: &StripInput,
) -> Readback {
    if input.entries.is_empty() {
        return Readback::default();
    }
    let buffers = pipelines.create_buffers(device, input);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sparse strips"),
    });
    pipelines.encode(&mut encoder, &buffers);
    read_back(device, queue, encoder, &buffers)
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::sparse_strips::model;
    use crate::sparse_strips::test_util::{Scene, compare, tiger_scene};
    use alloc::format;
    use vello_common::kurbo::{Affine, Circle, Point, Shape};
    use vello_common::peniko::Fill;

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("sparse strips test"),
            required_features: wgpu::Features::empty(),
            ..Default::default()
        }))
        .ok()
    }

    fn check_gpu(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipelines: &SparseStripPipelines,
        scene: &Scene,
        what: &str,
    ) {
        let (input, cpu_strips, cpu_alphas) = scene.build();
        std::eprintln!(
            "{what}: {} paths, {} entries, {} cpu strips",
            input.paths.len(),
            input.entries.len(),
            cpu_strips.len()
        );
        let out = render_blocking(device, queue, pipelines, &input);
        compare(
            &format!("{what} (gpu vs cpu)"),
            (&out.strips, &out.alphas),
            (&cpu_strips, &cpu_alphas),
            1,
        );
        let (model_strips, model_alphas) = model::render(&input);
        compare(
            &format!("{what} (gpu vs model)"),
            (&out.strips, &out.alphas),
            (&model_strips, &model_alphas),
            1,
        );
        // The GPU-written path ranges must cover the output.
        let mut strip_cursor = 0;
        let mut block_cursor = 0;
        for p in &out.paths {
            assert_eq!(p.strip_start, strip_cursor, "{what}: path strip start");
            assert_eq!(p.block_start, block_cursor, "{what}: path block start");
            strip_cursor = p.strip_end;
            block_cursor = p.block_end;
        }
        assert_eq!(
            strip_cursor as usize,
            out.strips.len(),
            "{what}: total strips"
        );
        assert_eq!(
            block_cursor as usize * TILE_PIXELS as usize,
            out.alphas.len(),
            "{what}: total alphas"
        );
    }

    #[test]
    fn gpu_matches_cpu() {
        let Some((device, queue)) = device() else {
            std::eprintln!("no GPU adapter, skipping");
            return;
        };
        let pipelines = SparseStripPipelines::new(&device);

        let mut scene = Scene::new(100, 80);
        scene.fill(
            Circle::new((50.0, 40.0), 30.0).to_path(0.1),
            Affine::IDENTITY,
            Fill::NonZero,
            None,
        );
        check_gpu(&device, &queue, &pipelines, &scene, "circle");

        let mut scene = Scene::new(100, 80);
        scene.fill(
            Scene::star(Point::new(50.0, 40.0), 5, 15.0, 38.0),
            Affine::IDENTITY,
            Fill::EvenOdd,
            Some(128),
        );
        scene.fill(
            Circle::new((10.0, 40.0), 35.0).to_path(0.1),
            Affine::IDENTITY,
            Fill::NonZero,
            None,
        );
        scene.fill(
            vello_common::kurbo::Rect::new(-20.0, 20.5, 130.0, 60.5).to_path(0.1),
            Affine::IDENTITY,
            Fill::NonZero,
            None,
        );
        scene.fill(
            Circle::new((-30.0, 40.0), 20.0).to_path(0.1),
            Affine::IDENTITY,
            Fill::NonZero,
            None,
        );
        check_gpu(&device, &queue, &pipelines, &scene, "mixed culled");

        let mut scene = Scene::new(100, 80);
        for seed in 0..40_u64 {
            let fill = if seed % 2 == 0 {
                Fill::NonZero
            } else {
                Fill::EvenOdd
            };
            scene.fill(
                Scene::polygon(seed, 3 + (seed % 9) as usize, 140.0, 110.0),
                Affine::translate((-20.0, -15.0)),
                fill,
                None,
            );
        }
        check_gpu(&device, &queue, &pipelines, &scene, "polygons");
    }

    #[test]
    fn gpu_tiger() {
        let Some((device, queue)) = device() else {
            std::eprintln!("no GPU adapter, skipping");
            return;
        };
        let pipelines = SparseStripPipelines::new(&device);
        let scene = tiger_scene(800, 600, Affine::scale(1.5));
        check_gpu(&device, &queue, &pipelines, &scene, "tiger");
        // Shifted so that geometry is culled on the left and top, and scaled so that the
        // entry count spans many workgroups.
        let scene = tiger_scene(
            800,
            600,
            Affine::translate((-300.0, -200.0)) * Affine::scale(3.0),
        );
        assert!(scene.build().0.entries.len() > 20 * crate::sparse_strips::SCAN_WG as usize);
        check_gpu(&device, &queue, &pipelines, &scene, "tiger culled");
    }
}
