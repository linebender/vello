// Copyright 2021 The piet-gpu authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

use piet_gpu_hal::{include_shader, BufferUsage, DescriptorSet};
use piet_gpu_hal::{Buffer, Pipeline};

use crate::runner::{Commands, Runner};

const WG_SIZE: u64 = 512;
const N_ROWS: u64 = 8;
const ELEMENTS_PER_WG: u64 = WG_SIZE * N_ROWS;

struct PrefixTreeCode {
    reduce_pipeline: Pipeline,
    scan_pipeline: Pipeline,
    root_pipeline: Pipeline,
}

struct PrefixTreeStage {
    sizes: Vec<u64>,
    tmp_bufs: Vec<Buffer>,
}

struct PrefixTreeBinding {
    // All but the first and last can be moved to stage.
    descriptor_sets: Vec<DescriptorSet>,
}

pub unsafe fn run_prefix_test(runner: &mut Runner) {
    // This will be configurable. Note though that the current code is
    // prone to reading and writing past the end of buffers if this is
    // not a power of the number of elements processed in a workgroup.
    let n_elements: u64 = 1 << 24;
    let data: Vec<u32> = (0..n_elements as u32).collect();
    let data_buf = runner
        .session
        .create_buffer_init(&data, BufferUsage::STORAGE)
        .unwrap();
    let out_buf = runner.buf_down(data_buf.size());
    let code = PrefixTreeCode::new(runner);
    let stage = PrefixTreeStage::new(runner, n_elements);
    let binding = stage.bind(runner, &code, &out_buf.dev_buf);
    // Also will be configurable of course.
    let n_iter = 1000;
    let mut total_elapsed = 0.0;
    for i in 0..n_iter {
        let mut commands = runner.commands();
        commands.cmd_buf.copy_buffer(&data_buf, &out_buf.dev_buf);
        commands.cmd_buf.memory_barrier();
        commands.write_timestamp(0);
        stage.record(&mut commands, &code, &binding);
        commands.write_timestamp(1);
        if i == 0 {
            commands.cmd_buf.memory_barrier();
            commands.download(&out_buf);
        }
        total_elapsed += runner.submit(commands);
        if i == 0 {
            let mut dst: Vec<u32> = Default::default();
            out_buf.read(&mut dst);
            println!("failures: {:?}", verify(&dst));
        }
    }
    let throughput = (n_elements * n_iter) as f64 / total_elapsed;
    println!(
        "total {:?}ms, throughput = {}G el/s",
        total_elapsed * 1e3,
        throughput * 1e-9
    );
}

impl PrefixTreeCode {
    unsafe fn new(runner: &mut Runner) -> PrefixTreeCode {
        let reduce_code = include_shader!(&runner.session, "../shader/gen/prefix_reduce");
        let reduce_pipeline = runner
            .session
            .create_simple_compute_pipeline(reduce_code, 2)
            .unwrap();
        let scan_code = include_shader!(&runner.session, "../shader/gen/prefix_scan");
        let scan_pipeline = runner
            .session
            .create_simple_compute_pipeline(scan_code, 2)
            .unwrap();
        let root_code = include_shader!(&runner.session, "../shader/gen/prefix_root");
        let root_pipeline = runner
            .session
            .create_simple_compute_pipeline(root_code, 1)
            .unwrap();
        PrefixTreeCode {
            reduce_pipeline,
            scan_pipeline,
            root_pipeline,
        }
    }
}

impl PrefixTreeStage {
    unsafe fn new(runner: &mut Runner, n_elements: u64) -> PrefixTreeStage {
        let mut size = n_elements;
        let mut sizes = vec![size];
        let mut tmp_bufs = Vec::new();
        while size > ELEMENTS_PER_WG {
            size = (size + ELEMENTS_PER_WG - 1) / ELEMENTS_PER_WG;
            sizes.push(size);
            let buf = runner
                .session
                .create_buffer(4 * size, BufferUsage::STORAGE)
                .unwrap();
            tmp_bufs.push(buf);
        }
        PrefixTreeStage { sizes, tmp_bufs }
    }

    unsafe fn bind(
        &self,
        runner: &mut Runner,
        code: &PrefixTreeCode,
        data_buf: &Buffer,
    ) -> PrefixTreeBinding {
        let mut descriptor_sets = Vec::with_capacity(2 * self.tmp_bufs.len() + 1);
        for i in 0..self.tmp_bufs.len() {
            let buf0 = if i == 0 {
                data_buf
            } else {
                &self.tmp_bufs[i - 1]
            };
            let buf1 = &self.tmp_bufs[i];
            let descriptor_set = runner
                .session
                .create_simple_descriptor_set(&code.reduce_pipeline, &[buf0, buf1])
                .unwrap();
            descriptor_sets.push(descriptor_set);
        }
        let buf0 = self.tmp_bufs.last().unwrap_or(data_buf);
        let descriptor_set = runner
            .session
            .create_simple_descriptor_set(&code.root_pipeline, &[buf0])
            .unwrap();
        descriptor_sets.push(descriptor_set);
        for i in (0..self.tmp_bufs.len()).rev() {
            let buf0 = if i == 0 {
                data_buf
            } else {
                &self.tmp_bufs[i - 1]
            };
            let buf1 = &self.tmp_bufs[i];
            let descriptor_set = runner
                .session
                .create_simple_descriptor_set(&code.scan_pipeline, &[buf0, buf1])
                .unwrap();
            descriptor_sets.push(descriptor_set);
        }
        PrefixTreeBinding { descriptor_sets }
    }

    unsafe fn record(
        &self,
        commands: &mut Commands,
        code: &PrefixTreeCode,
        bindings: &PrefixTreeBinding,
    ) {
        let n = self.tmp_bufs.len();
        for i in 0..n {
            let n_workgroups = self.sizes[i + 1];
            commands.cmd_buf.dispatch(
                &code.reduce_pipeline,
                &bindings.descriptor_sets[i],
                (n_workgroups as u32, 1, 1),
                (WG_SIZE as u32, 1, 1),
            );
            commands.cmd_buf.memory_barrier();
        }
        commands.cmd_buf.dispatch(
            &code.root_pipeline,
            &bindings.descriptor_sets[n],
            (1, 1, 1),
            (WG_SIZE as u32, 1, 1),
        );
        for i in (0..n).rev() {
            commands.cmd_buf.memory_barrier();
            let n_workgroups = self.sizes[i + 1];
            commands.cmd_buf.dispatch(
                &code.scan_pipeline,
                &bindings.descriptor_sets[2 * n - i],
                (n_workgroups as u32, 1, 1),
                (WG_SIZE as u32, 1, 1),
            );
        }
    }
}

// Verify that the data is OEIS A000217
fn verify(data: &[u32]) -> Option<usize> {
    data.iter()
        .enumerate()
        .position(|(i, val)| ((i * (i + 1)) / 2) as u32 != *val)
}
