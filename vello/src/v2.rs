// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Thinking about it: what do we need?

// Use case: Runs on demand?
// 1) CPU then GPU

// Use case: Debug stages
// 1) Run single stage with fixed input
// 2) Maybe generate inputs on CPU first
// 3) Download results

use std::ops::ControlFlow;

pub struct CpuSteps {
    end_cpu_after: PipelineStep,
    run: bool,
}

#[derive(Clone, Copy)]
struct StepMeta {
    run: bool,
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
enum PipelineStep {
    One,
    Two,
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
struct Buffer<T> {
    cpu_write_count: u16,
    cpu_read_count: u16,
    remaining_writes_cpu: u16,
    remaining_reads_cpu: u16,
    cpu_content: Vec<T>,
    staging_buffer: wgpu::Buffer,
    staging_written: bool,

    gpu_written: bool,
    gpu_buffer: wgpu::Buffer,
    staging_queue: Vec<wgpu::Buffer>,
}
impl<T> Buffer<T> {
    fn read(&mut self, stage: StepMeta) -> &[T] {
        if stage.run {
            self.remaining_reads_cpu -= 1;
            &self.cpu_content
        } else {
            self.cpu_read_count += 1;
            &[]
        }
    }
    fn write(&mut self, stage: StepMeta) -> &mut [T] {
        if stage.run {
            self.remaining_writes_cpu -= 1;
            if self.remaining_reads_cpu == 0 && self.remaining_writes_cpu == 0 {
                // self.staging_written = true;
                // return self
                //     .staging_buffer
                //     .slice(..)
                //     .get_mapped_range_mut()
                //     .deref_mut();
            }
            &mut self.cpu_content
        } else {
            self.cpu_write_count += 1;
            &mut []
        }
    }
    fn read_write(&mut self, stage: StepMeta) -> &mut [T] {
        if stage.run {
            self.remaining_reads_cpu -= 1;
            self.remaining_writes_cpu -= 1;
            &mut self.cpu_content
        } else {
            self.cpu_write_count += 1;
            self.cpu_read_count += 1;
            &mut []
        }
    }
}

struct Buffers {
    a: Buffer<u8>,
    b: Buffer<u16>,
    c: Buffer<u16>,
}

pub fn tiny_pipeline_model(mut stages: CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    cpu_stage_1(&mut stages, buffers)?;
    cpu_stage_1(&mut stages, buffers)?;
    cpu_stage_1(&mut stages, buffers)
}

fn cpu_stage_1(stages: &mut CpuSteps, buffers: &mut Buffers) -> ControlFlow<()> {
    let meta = stages.start_stage(PipelineStep::One)?;
    let a = buffers.a.read(meta);
    let b = buffers.b.write(meta);
    let c = buffers.c.read_write(meta);
    if meta.run {
        stage_1::stage_1(a, &*b, c);
    }
    ControlFlow::Continue(())
}

mod stage_1 {
    pub fn stage_1(a: &[u8], b: &[u16], c: &mut [u16]) {
        // ..
    }
}
