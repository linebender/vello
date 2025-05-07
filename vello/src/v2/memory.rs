// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use bytemuck::Zeroable;
use vello_encoding::{
    BinHeader, BufferSize, BufferSizes, BumpAllocators, Clip, ClipBbox, ClipBic, ClipElement,
    ConfigUniform, DrawBbox, DrawMonoid, LineSoup, Path, PathBbox, PathMonoid, PathSegment,
    SegmentCount, Tile,
};

pub(crate) struct Buffers {
    // bump_alloc and config are special, because there's
    // only one of them (they are also never bound write-only, so
    // they don't need to have the possible "write directly into staging" behaviour)
    pub(crate) bump_alloc: BumpAllocators,
    pub(crate) config: ConfigUniform,

    pub(crate) scene: Buffer<u32>,
    pub(crate) path_reduced: Buffer<PathMonoid>,
    pub(crate) path_monoids: Buffer<PathMonoid>,
    pub(crate) path_bboxes: Buffer<PathBbox>,
    pub(crate) draw_reduced: Buffer<DrawMonoid>,
    pub(crate) draw_monoids: Buffer<DrawMonoid>,
    pub(crate) info: Buffer<u32>,
    pub(crate) clip_inps: Buffer<Clip>,
    pub(crate) clip_els: Buffer<ClipElement>,
    pub(crate) clip_bics: Buffer<ClipBic>,
    pub(crate) clip_bboxes: Buffer<ClipBbox>,
    pub(crate) draw_bboxes: Buffer<DrawBbox>,

    pub(crate) bin_headers: Buffer<BinHeader>,
    pub(crate) paths: Buffer<Path>,

    // Bump allocated buffers
    pub(crate) lines: Buffer<LineSoup>,
    pub(crate) bin_data: Buffer<u32>,
    pub(crate) tiles: Buffer<Tile>,
    pub(crate) seg_counts: Buffer<SegmentCount>,
    pub(crate) segments: Buffer<PathSegment>,
    pub(crate) ptcl: Buffer<u32>,
}

impl Buffers {
    pub(crate) fn visit(&mut self, mut visitor: impl BufferVisitor) {
        visitor.visit(&mut self.scene, "scene");
        visitor.visit(&mut self.path_reduced, "path_reduced");
        visitor.visit(&mut self.path_monoids, "path_monoids");
        visitor.visit(&mut self.path_bboxes, "path_bboxes");
        visitor.visit(&mut self.draw_reduced, "draw_reduced");
        visitor.visit(&mut self.draw_monoids, "draw_monoids");
        visitor.visit(&mut self.info, "info");
        visitor.visit(&mut self.clip_inps, "clip_inps");
        visitor.visit(&mut self.clip_els, "clip_els");
        visitor.visit(&mut self.clip_bics, "clip_bics");
        visitor.visit(&mut self.clip_bboxes, "clip_bboxes");
        visitor.visit(&mut self.draw_bboxes, "draw_bboxes");

        visitor.visit(&mut self.bin_headers, "bin_headers");
        visitor.visit(&mut self.paths, "paths");

        // Bump allocated buffers
        visitor.visit(&mut self.lines, "lines");
        visitor.visit(&mut self.bin_data, "bin_data");
        visitor.visit(&mut self.tiles, "tiles");
        visitor.visit(&mut self.seg_counts, "seg_counts");
        visitor.visit(&mut self.segments, "segments");
        visitor.visit(&mut self.ptcl, "ptcl");
    }

    pub(crate) fn resize(&mut self, sizes: BufferSizes) {
        self.path_reduced.size = sizes.path_reduced;

        self.path_monoids.size = sizes.path_monoids;
        self.path_bboxes.size = sizes.path_bboxes;
        self.draw_reduced.size = sizes.draw_reduced;
        self.draw_monoids.size = sizes.draw_monoids;
        self.info.size = sizes.info;
        self.clip_inps.size = sizes.clip_inps;
        self.clip_els.size = sizes.clip_els;
        self.clip_bics.size = sizes.clip_bics;
        self.clip_bboxes.size = sizes.clip_bboxes;
        self.draw_bboxes.size = sizes.draw_bboxes;
        self.bin_headers.size = sizes.bin_headers;
        self.paths.size = sizes.paths;
        self.lines.size = sizes.lines;
        self.bin_data.size = sizes.bin_data;
        self.tiles.size = sizes.tiles;
        self.seg_counts.size = sizes.seg_counts;
        self.segments.size = sizes.segments;
        self.ptcl.size = sizes.ptcl;

        // The CPU pipeline doesn't need a two-pass reduction (indeed, it doesn't really need a reduction at all...)
        // self.path_reduced2.size = sizes.path_reduced2;
        // self.path_reduced_scan.size = sizes.path_reduced_scan;

        // I haven't quite worked out buffers for single values yet
        // self.bump_alloc.size = sizes.bump_alloc;

        // Our pipeline stages fully subsume the indirect stages
        // self.indirect_count.size = sizes.indirect_count;
        // We don't yet support `fine` in the CPU shaders
        // self.blend_spill.size = sizes.blend_spill;
    }
}

pub(crate) trait BufferVisitor {
    fn visit<T: Zeroable>(&mut self, buffer: &mut Buffer<T>, name: &'static str);
}

pub(crate) struct FirstPass;
impl BufferVisitor for FirstPass {
    fn visit<T>(&mut self, buffer: &mut Buffer<T>, name: &'static str) {
        buffer.run = false;
        buffer.cpu_write_count = 0;
    }
}

pub(crate) struct SecondPass;
impl BufferVisitor for SecondPass {
    fn visit<T: Zeroable>(&mut self, buffer: &mut Buffer<T>, name: &'static str) {
        buffer.run = true;
        buffer.remaining_writes_cpu = buffer.cpu_write_count;
        buffer.staging_written = false;
        if buffer.last_operation_writes_all && buffer.cpu_write_count > 0 {
            // buffer.staging_buffer = self.get_suitable_staging_buffer(self.staging_buffer.take(), ...);
        }
        buffer
            .cpu_content
            .resize_with(buffer.size.len() as usize, bytemuck::zeroed);
    }
}

pub(crate) struct ValidationPass;
impl BufferVisitor for ValidationPass {
    fn visit<T>(&mut self, buffer: &mut Buffer<T>, name: &'static str) {
        debug_assert_eq!(
            buffer.remaining_writes_cpu, 0,
            "Should have done all the writes expected"
        );
        debug_assert_eq!(
            buffer.last_operation_writes_all, buffer.staging_written,
            "Should have written to the buffer as the last operation in both passes"
        );
    }
}

pub(crate) struct Buffer<T> {
    // The total size of the buffer
    size: BufferSize<T>,

    // State used by the CPU pass
    run: bool,
    cpu_write_count: u16,
    remaining_writes_cpu: u16,
    last_operation_writes_all: bool,

    /// The contents of this buffer on the CPU
    /// Only initialised if `cpu_write_count > 0`
    cpu_content: Vec<T>,

    /// The buffer used to stage content into the GPU.
    /// The buffer is mapped for writing.
    ///
    /// Will be `None` if the buffer is never used on the CPU.
    staging_buffer: Option<wgpu::Buffer>,
    /// Whether the content has been written into the staging buffer.
    /// If `cpu_write_count` is 0, will be ignored.
    staging_written: bool,

    /// TODO: Does this need to be always initialised?
    /// Should this be multiple structs for the CPU or GPU pass?
    gpu_buffer: Option<wgpu::Buffer>,
    /// Whether content has been copied from `staging_buffer`
    gpu_written: bool,
}

impl<T> Buffer<T> {
    pub(crate) fn set_size(&mut self, size: BufferSize<T>) {
        self.size = size;
    }
    pub(crate) fn read(&mut self) -> &[T] {
        if self.run {
            &self.cpu_content
        } else {
            self.last_operation_writes_all = false;
            &[]
        }
    }
    /// This kernel will overwrite *all* existing content in the buffer.
    pub(crate) fn write_all(&mut self) -> &mut [T] {
        if self.run {
            self.remaining_writes_cpu -= 1;
            // If the buffer is being written to, but will never be used again on the CPU,
            // it must be needed on the GPU (otherwise, why would we write to it?).
            // Therefore, we can write directly into the staging buffer.
            // Technically, this breaks down if we run a GPU pipeline only partially (e.g. for debugging)
            // but that case is rare enough that we don't optimise it (it never happens for an end-user)
            // Note that the doc comments on this method rule out this being a pass which only
            // overwrites *some* content in the buffer
            if self.last_operation_writes_all && self.remaining_writes_cpu == 0 {
                // self.staging_written = true;
                // return self
                //     .staging_buffer
                //     .slice(..)
                //     .get_mapped_range_mut()
                //     .deref_mut();
            }
            &mut self.cpu_content
        } else {
            self.last_operation_writes_all = true;
            self.cpu_write_count += 1;
            &mut []
        }
    }

    pub(crate) fn read_write(&mut self) -> &mut [T] {
        if self.run {
            self.remaining_writes_cpu -= 1;
            &mut self.cpu_content
        } else {
            self.last_operation_writes_all = false;
            self.cpu_write_count += 1;
            &mut []
        }
    }
}
