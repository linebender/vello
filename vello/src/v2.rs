// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::ops::ControlFlow;

use memory::Buffers;

mod infra;
mod kernels;
mod memory;

// Use case: Runs on demand?
// 1) CPU then GPU

// Use case: Debug stages
// 1) Run single stage with fixed input
// 2) Maybe generate inputs on CPU first
// 3) Download results

pub(crate) fn tiny_pipeline_model(mut stages: CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    cpu_stage_1(&mut stages, buffers)?;
    cpu_stage_1(&mut stages, buffers)?;
    cpu_stage_1(&mut stages, buffers)
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
enum PipelineStep {
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
}

pub struct CpuSteps {
    end_cpu_after: PipelineStep,
    run: bool,
}

#[derive(Clone, Copy)]
struct StepMeta {
    run: bool,
}

impl CpuSteps {
    fn start_stage(&mut self, step: PipelineStep) -> ControlFlow<(), StepMeta> {
        // If we're a later step than the final CPU step
        if step > self.end_cpu_after {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(StepMeta { run: self.run })
    }
}
