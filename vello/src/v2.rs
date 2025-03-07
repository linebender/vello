// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::ops::ControlFlow;

use memory::{Buffers, FirstPass, SecondPass, ValidationPass};

mod kernels;
mod memory;

// Use case: Runs on demand?
// 1) CPU then GPU

// Use case: Debug stages
// 1) Run single stage with fixed input
// 2) Maybe generate inputs on CPU first
// 3) Download results

/// Perform a (potentially partial) run of the CPU pipeline.
///
/// `buffers.packed` and `buffers.config` must be initialised with content from the scene.
pub(crate) fn cpu_pipeline(buffers: &mut Buffers, end_after: Option<PipelineStep>) {
    let end_after = end_after.unwrap_or(PipelineStep::LAST);
    // We run in two passes, to collect which buffers are written to, then to .
    buffers.visit(FirstPass);
    let _: ControlFlow<()> = cpu_pipeline_raw(
        CpuSteps {
            end_after,
            run: false,
        },
        buffers,
    );

    buffers.visit(SecondPass);
    // This step actually runs the pipeline.
    let _: ControlFlow<()> = cpu_pipeline_raw(
        CpuSteps {
            end_after,
            run: false,
        },
        buffers,
    );
    buffers.visit(ValidationPass);
}

fn cpu_pipeline_raw(mut stages: CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    kernels::pathtag_reduce(&mut stages, buffers)?;
    kernels::pathtag_scan(&mut stages, buffers)?;
    kernels::bbox_clear(&mut stages, buffers)?;
    kernels::flatten(&mut stages, buffers)?;
    kernels::draw_reduce(&mut stages, buffers)?;
    kernels::draw_leaf(&mut stages, buffers)?;
    kernels::clip_reduce(&mut stages, buffers)?;
    kernels::clip_leaf(&mut stages, buffers)?;
    kernels::binning(&mut stages, buffers)?;
    kernels::tile_alloc(&mut stages, buffers)?;
    kernels::path_count(&mut stages, buffers)?;
    kernels::backdrop(&mut stages, buffers)?;
    kernels::coarse(&mut stages, buffers)?;
    kernels::path_tiling(&mut stages, buffers)?;

    ControlFlow::Continue(())
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub(crate) enum PipelineStep {
    PathTagReduce,
    PathTagScan,
    BboxClear,
    Flatten,
    DrawReduce,
    DrawLeaf,
    ClipReduce,
    ClipLeaf,
    Binning,
    TileAlloc,
    PathCount,
    Backdrop,
    Coarse,
    PathTiling,
    // If another step is added, update `Self::LAST`
}

impl PipelineStep {
    const LAST: Self = Self::PathTiling;
}

struct CpuSteps {
    end_after: PipelineStep,
    run: bool,
}

#[derive(Clone, Copy)]
struct StepMeta {
    run: bool,
}

impl CpuSteps {
    fn start_stage(&mut self, step: PipelineStep) -> ControlFlow<(), StepMeta> {
        // If we're a later step than the final CPU step
        if step > self.end_after {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(StepMeta { run: self.run })
    }
}
