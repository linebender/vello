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
/// Words in the indirect buffer: strip dispatch arguments, strip total, block total, block
/// dispatch arguments.
const INDIRECT_WORDS: u64 = 8;

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
    resolve_strips: wgpu::ComputePipeline,
    resolve_blocks: wgpu::ComputePipeline,
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
        let resolve_strips = make(
            "sparse strips resolve strips",
            vello_gpu_shaders::wgsl::SPARSE_STRIPS_RESOLVE,
            &resolve_layout,
            "resolve_strips",
        );
        let resolve_blocks = make(
            "sparse strips resolve blocks",
            vello_gpu_shaders::wgsl::SPARSE_STRIPS_RESOLVE,
            &resolve_layout,
            "resolve_blocks",
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
            resolve_strips,
            resolve_blocks,
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
        self.encode_timed(encoder, buffers, None);
    }

    /// Record the passes, each in its own compute pass, writing a start and end timestamp per
    /// pass into `timestamps` (query `2 * i` and `2 * i + 1` for pass `i` in [`PASS_NAMES`]
    /// order) when given.
    pub fn encode_timed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        buffers: &SparseStripBuffers,
        timestamps: Option<&wgpu::QuerySet>,
    ) {
        encoder.clear_buffer(&buffers.acc, 0, None);
        // Direct workgroup counts, or the byte offset of indirect arguments.
        let passes: [(
            &str,
            &wgpu::ComputePipeline,
            &wgpu::BindGroup,
            Result<u32, u64>,
        ); PASS_COUNT] = [
            (
                "reduce",
                &self.reduce,
                &buffers.alloc_group,
                Ok(buffers.n_wg),
            ),
            ("scan", &self.scan, &buffers.alloc_group, Ok(1)),
            ("alloc", &self.alloc, &buffers.alloc_group, Ok(buffers.n_wg)),
            (
                "merge",
                &self.merge,
                &buffers.merge_group,
                Ok(buffers.n_entries.div_ceil(MERGE_WG)),
            ),
            (
                "resolve_strips",
                &self.resolve_strips,
                &buffers.resolve_group,
                Err(0),
            ),
            (
                "resolve_blocks",
                &self.resolve_blocks,
                &buffers.resolve_group,
                Err(20),
            ),
        ];
        for (i, (label, pipeline, group, workgroups)) in passes.into_iter().enumerate() {
            let timestamp_writes = timestamps.map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(2 * i as u32),
                end_of_pass_write_index: Some(2 * i as u32 + 1),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes,
            });
            pass.set_bind_group(0, &buffers.input_group, &[]);
            pass.set_bind_group(1, group, &[]);
            pass.set_pipeline(pipeline);
            match workgroups {
                Ok(n) => pass.dispatch_workgroups(n, 1, 1),
                Err(offset) => pass.dispatch_workgroups_indirect(&buffers.indirect, offset),
            }
        }
    }
}

/// Number of compute passes recorded by [`SparseStripPipelines::encode_timed`].
pub const PASS_COUNT: usize = 6;
/// Names of the passes, in dispatch order.
pub const PASS_NAMES: [&str; PASS_COUNT] = [
    "reduce",
    "scan",
    "alloc",
    "merge",
    "resolve_strips",
    "resolve_blocks",
];

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

    /// Record the same drawing twice, once with GPU strips, and compare the scenes draw by
    /// draw after the GPU strips are resolved: strip geometry identical, alphas within one.
    #[test]
    fn scene_matches_cpu_scene() {
        use crate::Scene;
        use crate::scene::RecordedDraw;
        use crate::sparse_strips::test_util::Scene as TestScene;
        use vello_common::kurbo::{Rect, Stroke};
        use vello_common::peniko::color::palette::css;
        use vello_common::record::Drawable;
        use vello_common::strip::Strip;

        let Some((device, queue)) = device() else {
            std::eprintln!("no GPU adapter, skipping");
            return;
        };
        let pipelines = SparseStripPipelines::new(&device);

        fn draw(scene: &mut Scene) {
            scene.set_paint(css::REBECCA_PURPLE);
            scene.fill_path(&Circle::new((50.0, 40.0), 30.0).to_path(0.1));
            scene.set_fill_rule(Fill::EvenOdd);
            scene.fill_path(&TestScene::star(Point::new(50.0, 40.0), 5, 15.0, 38.0));
            scene.set_fill_rule(Fill::NonZero);
            // Culled on the left, and entirely off to the left (CPU fallback).
            scene.fill_path(&Circle::new((10.0, 40.0), 35.0).to_path(0.1));
            scene.fill_path(&Circle::new((-30.0, 40.0), 20.0).to_path(0.1));
            // Stroke.
            scene.set_stroke(Stroke::new(3.0));
            scene.stroke_path(&TestScene::polygon(7, 6, 90.0, 70.0));
            // Aliased.
            scene.set_aliasing_threshold(Some(128));
            scene.fill_path(&Circle::new((70.0, 60.0), 15.0).to_path(0.1));
            scene.set_aliasing_threshold(None);
            // Fast rectangle (CPU) and a rotated rectangle (path).
            scene.fill_rect(&Rect::new(10.0, 10.0, 30.0, 25.0));
            scene.set_transform(Affine::rotate(0.3));
            scene.fill_rect(&Rect::new(40.0, 10.0, 70.0, 25.0));
            scene.reset_transform();
            // Under a clip path: intersected after readback.
            scene.push_clip_path(&Circle::new((40.0, 40.0), 25.0).to_path(0.1));
            scene.fill_path(&TestScene::polygon(3, 5, 100.0, 80.0));
            scene.fill_path(&Circle::new((90.0, 40.0), 30.0).to_path(0.1));
            scene.push_clip_rect(&Rect::new(20.0, 20.0, 60.0, 70.0));
            scene.fill_path(&TestScene::polygon(4, 7, 100.0, 80.0));
            scene.pop_clip();
            scene.pop_clip();
            // Clip layer (CPU-generated clip strips) with GPU draws inside.
            scene.push_clip_layer(&TestScene::star(Point::new(60.0, 50.0), 7, 10.0, 35.0));
            scene.fill_path(&Circle::new((60.0, 50.0), 30.0).to_path(0.1));
            scene.pop_layer();
        }

        let mut cpu = Scene::new(100, 80);
        draw(&mut cpu);
        let mut gpu = Scene::new(100, 80);
        gpu.set_gpu_strips(true);
        draw(&mut gpu);
        gpu.resolve_gpu_strips(&device, &queue, &pipelines);

        let cpu_storage = cpu.strip_storage.borrow();
        let gpu_storage = gpu.strip_storage.borrow();
        let gpu_state = gpu.gpu_strips.as_ref().unwrap().borrow();
        assert_eq!(
            cpu.recorder.draws.len(),
            gpu.recorder.draws.len(),
            "draw count"
        );
        let mut gpu_draws = 0;
        for (i, (c, g)) in cpu
            .recorder
            .draws
            .iter()
            .zip(&gpu.recorder.draws)
            .enumerate()
        {
            let (c_strips, c_bbox): (&[Strip], _) = match c {
                RecordedDraw::Path(p) => (
                    &cpu_storage.strips[p.strips.clone()],
                    c.bbox(&cpu_storage.strips[p.strips.clone()]),
                ),
                RecordedDraw::Rect(_) => (&[], None),
                RecordedDraw::GpuPath(_) => unreachable!(),
            };
            let (g_strips, g_bbox): (&[Strip], _) = match g {
                RecordedDraw::Path(p) => (
                    &gpu_storage.strips[p.strips.clone()],
                    g.bbox(&gpu_storage.strips[p.strips.clone()]),
                ),
                RecordedDraw::Rect(_) => (&[], None),
                RecordedDraw::GpuPath(p) => {
                    gpu_draws += 1;
                    (
                        &gpu_storage.strips[gpu_state.ranges[p.path as usize].clone()],
                        Some(p.bbox),
                    )
                }
            };
            assert_eq!(c_strips.len(), g_strips.len(), "draw {i}: strip count");
            if let (Some(cb), Some(gb)) = (c_bbox, g_bbox) {
                // Exact except for clipped paths, where the recorded bbox is conservative.
                assert_eq!(
                    gb.intersect(cb),
                    cb,
                    "draw {i}: bbox {gb:?} does not contain {cb:?}"
                );
            }
            let (Some(c0), Some(g0)) = (c_strips.first(), g_strips.first()) else {
                continue;
            };
            for (k, (cs, gs)) in c_strips.iter().zip(g_strips).enumerate() {
                assert_eq!(
                    (cs.x, cs.y, cs.fill_gap()),
                    (gs.x, gs.y, gs.fill_gap()),
                    "draw {i} strip {k}"
                );
                assert_eq!(
                    cs.alpha_idx() - c0.alpha_idx(),
                    gs.alpha_idx() - g0.alpha_idx(),
                    "draw {i} strip {k}: alpha offset"
                );
            }
            let n_alphas = (c_strips.last().unwrap().alpha_idx() - c0.alpha_idx()) as usize;
            let ca = &cpu_storage.alphas[c0.alpha_idx() as usize..][..n_alphas];
            let ga = &gpu_storage.alphas[g0.alpha_idx() as usize..][..n_alphas];
            for (k, (a, b)) in ca.iter().zip(ga).enumerate() {
                assert!(
                    (i32::from(*a) - i32::from(*b)).abs() <= 1,
                    "draw {i} alpha {k}: cpu {a} gpu {b}"
                );
            }
        }
        assert!(
            gpu_draws >= 8,
            "expected most draws on the GPU, got {gpu_draws}"
        );
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
