// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::ops::ControlFlow;

use vello_encoding::ConfigUniform;
use vello_shaders::cpu::{
    backdrop_main, bbox_clear_main, binning_main, clip_leaf_main, clip_reduce_main, coarse_main,
    draw_leaf_main, draw_reduce_main, flatten_main, path_count_main, path_tiling_main,
    pathtag_reduce_main, pathtag_scan_main, tile_alloc_main,
};

use super::{memory::Buffers, CpuSteps, PipelineStep};

fn pathtag_reduce(stages: &mut CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    let meta = stages.start_stage(PipelineStep::PathTagReduce)?;
    // TODO: Scene
    let scene = &[];
    // TODO: Config
    let config = ConfigUniform::default();
    // TODO: workgroup count
    let n_wg = 0;
    let path_reduced = buffers.path_reduced.read_write();
    if meta.run {
        pathtag_reduce_main(n_wg, &config, scene, path_reduced);
    }
    ControlFlow::Continue(())
}

fn pathtag_scan(stages: &mut CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    let meta = stages.start_stage(PipelineStep::PathTagScan)?;
    // TODO: Scene
    let scene = &[];
    // TODO: Config
    let config = ConfigUniform::default();
    // TODO: workgroup count
    let n_wg = 0;
    let path_reduced = buffers.path_reduced.read();
    let path_monoids = buffers.path_monoids.read_write();
    if meta.run {
        pathtag_scan_main(n_wg, &config, scene, path_reduced, path_monoids);
    }
    ControlFlow::Continue(())
}

fn bbox_clear(stages: &mut CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    let meta = stages.start_stage(PipelineStep::BboxClear)?;
    // TODO: Config
    let config = ConfigUniform::default();
    let path_bboxes = buffers.path_bboxes.read_write();
    if meta.run {
        bbox_clear_main(&config, path_bboxes);
    }
    ControlFlow::Continue(())
}

fn flatten(stages: &mut CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    let meta = stages.start_stage(PipelineStep::Flatten)?;
    // TODO: Scene
    let scene = &[];
    // TODO: Config
    let config = ConfigUniform::default();
    // TODO: workgroup count
    let n_wg = 0;
    let path_bboxes = buffers.path_bboxes.read_write();
    let bump = &mut buffers.bump_alloc;
    let path_monoids = buffers.path_monoids.read();
    let lines = buffers.lines.read_write();
    if meta.run {
        flatten_main(n_wg, &config, scene, path_monoids, path_bboxes, bump, lines);
    }
    ControlFlow::Continue(())
}

fn draw_reduce(stages: &mut CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    let meta = stages.start_stage(PipelineStep::DrawReduce)?;
    // TODO: Scene
    let scene = &[];
    // TODO: Config
    let config = ConfigUniform::default();
    // TODO: workgroup count
    let n_wg = 0;
    let draw_reduced = buffers.draw_reduced.read_write();
    if meta.run {
        draw_reduce_main(n_wg, &config, scene, draw_reduced);
    }
    ControlFlow::Continue(())
}

fn draw_leaf(stages: &mut CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    let meta = stages.start_stage(PipelineStep::DrawLeaf)?;
    // TODO: Scene
    let scene = &[];
    // TODO: Config
    let config = ConfigUniform::default();
    // TODO: workgroup count
    let n_wg = 0;
    let draw_monoid = buffers.draw_monoids.read_write();
    let path_bbox = buffers.path_bboxes.read();
    let info_bin_data = buffers.bin_data.read_write();
    let clip_inp = buffers.clip_inps.read_write();
    let draw_reduced = buffers.draw_reduced.read();
    if meta.run {
        draw_leaf_main(
            n_wg,
            &config,
            scene,
            &*draw_reduced,
            &*path_bbox,
            draw_monoid,
            info_bin_data,
            clip_inp,
        );
    }
    ControlFlow::Continue(())
}

fn clip_reduce(stages: &mut CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    let meta = stages.start_stage(PipelineStep::ClipReduce)?;
    // TODO: workgroup count
    let n_wg = 0;
    let path_bbox = buffers.path_bboxes.read();
    let clip_inp = buffers.clip_inps.read();
    let clip_bics = buffers.clip_bics.read_write();
    let clip_els = buffers.clip_els.read_write();
    if meta.run {
        clip_reduce_main(n_wg, clip_inp, path_bbox, clip_bics, clip_els);
    }
    ControlFlow::Continue(())
}

fn clip_leaf(stages: &mut CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    let meta = stages.start_stage(PipelineStep::ClipLeaf)?;
    // TODO: Config
    let config = ConfigUniform::default();
    let draw_monoid = buffers.draw_monoids.read_write();
    let path_bbox = buffers.path_bboxes.read();
    let clip_bbox = buffers.clip_bboxes.read_write();
    let clip_inp = buffers.clip_inps.read();
    if meta.run {
        clip_leaf_main(&config, clip_inp, path_bbox, draw_monoid, clip_bbox);
    }
    ControlFlow::Continue(())
}

fn binning(stages: &mut CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    let meta = stages.start_stage(PipelineStep::Binning)?;
    // TODO: workgroup count
    let n_wg = 0;
    // TODO: Config
    let config = ConfigUniform::default();
    let draw_monoid = buffers.draw_monoids.read();
    let path_bbox = buffers.path_bboxes.read();
    let clip_bbox = buffers.clip_bboxes.read();
    let draw_bbox = buffers.draw_bboxes.read_write();
    let bump = &mut buffers.bump_alloc;
    let info_bin_data = buffers.bin_data.read_write();
    let bin_header = buffers.bin_headers.read_write();
    if meta.run {
        binning_main(
            n_wg,
            &config,
            draw_monoid,
            path_bbox,
            clip_bbox,
            draw_bbox,
            bump,
            info_bin_data,
            bin_header,
        );
    }
    ControlFlow::Continue(())
}

fn tile_alloc(stages: &mut CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    let meta = stages.start_stage(PipelineStep::TileAlloc)?;
    // TODO: Config
    let config = ConfigUniform::default();
    // TODO: Scene
    let scene = &[];
    let draw_bbox = buffers.draw_bboxes.read();
    let bump = &mut buffers.bump_alloc;
    let paths = buffers.paths.read_write();
    let tiles = buffers.tiles.read_write();
    if meta.run {
        tile_alloc_main(&config, scene, &*draw_bbox, bump, paths, tiles);
    }
    ControlFlow::Continue(())
}

fn path_count(stages: &mut CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    let meta = stages.start_stage(PipelineStep::PathCount)?;
    let bump = &mut buffers.bump_alloc;
    let lines = buffers.lines.read();
    let paths = buffers.paths.read();
    let tiles = buffers.tiles.read_write();
    let seg_counts = buffers.seg_counts.read_write();
    if meta.run {
        path_count_main(bump, &*lines, &*paths, tiles, seg_counts);
    }
    ControlFlow::Continue(())
}

fn backdrop(stages: &mut CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    let meta = stages.start_stage(PipelineStep::Backdrop)?;
    // TODO: Config
    let config = ConfigUniform::default();
    let bump = &buffers.bump_alloc;
    let paths = buffers.paths.read();
    let tiles = buffers.tiles.read_write();
    if meta.run {
        backdrop_main(&config, &*bump, &*paths, tiles);
    }
    ControlFlow::Continue(())
}

fn coarse(stages: &mut CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    let meta = stages.start_stage(PipelineStep::Coarse)?;
    // TODO: Config
    let config = ConfigUniform::default();
    // TODO: Scene
    let scene = &[];
    let draw_monoids = buffers.draw_monoids.read();
    let bin_header = buffers.bin_headers.read();
    let info_bin_data = buffers.bin_data.read();
    let paths = buffers.paths.read();
    let tiles = buffers.tiles.write();
    let bump = &mut buffers.bump_alloc;
    let ptcl = buffers.ptcl.write();
    if meta.run {
        coarse_main(
            &config,
            scene,
            draw_monoids,
            &*bin_header,
            info_bin_data,
            paths,
            tiles,
            bump,
            ptcl,
        );
    }
    ControlFlow::Continue(())
}

fn path_tiling(stages: &mut CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    let meta = stages.start_stage(PipelineStep::PathTiling)?;
    let bump = &mut buffers.bump_alloc;

    let seg_counts = buffers.seg_counts.read();
    let lines = buffers.lines.read();
    let paths = buffers.paths.read();
    let tiles = buffers.tiles.read();

    // TODO: Is it read/write or just write?
    let segments = buffers.segments.read_write();
    if meta.run {
        path_tiling_main(bump, seg_counts, lines, paths, tiles, segments);
    }
    ControlFlow::Continue(())
}
