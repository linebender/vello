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
const N_ROWS: u64 = 16;
const ELEMENTS_PER_WG: u64 = WG_SIZE * N_ROWS;

/// The shader code for the prefix sum example.
///
/// A code struct can be created once and reused any number of times.
struct PrefixCode {
    pipeline: Pipeline,
}

/// The stage resources for the prefix sum example.
///
/// A stage resources struct is specific to a particular problem size
/// and queue.
struct PrefixStage {
    // This is the actual problem size but perhaps it would be better to
    // treat it as a capacity.
    n_elements: u64,
    state_buf: Buffer,
}

/// The binding for the prefix sum example.
struct PrefixBinding {
    descriptor_set: DescriptorSet,
}

pub unsafe fn run_prefix_test(runner: &mut Runner) {
    // This will be configurable.
    let n_elements: u64 = 1 << 23;
    let data: Vec<u32> = (0..n_elements as u32).collect();
    let data_buf = runner
        .session
        .create_buffer_init(&data, BufferUsage::STORAGE)
        .unwrap();
    let out_buf = runner.buf_down(data_buf.size());
    let code = PrefixCode::new(runner);
    let stage = PrefixStage::new(runner, n_elements);
    let binding = stage.bind(runner, &code, &data_buf, &out_buf.dev_buf);
    // Also will be configurable of course.
    let n_iter = 5000;
    let mut total_elapsed = 0.0;
    for i in 0..n_iter {
        let mut commands = runner.commands();
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

impl PrefixCode {
    unsafe fn new(runner: &mut Runner) -> PrefixCode {
        let code = include_shader!(&runner.session, "../shader/gen/prefix");
        let pipeline = runner
            .session
            .create_simple_compute_pipeline(code, 3)
            .unwrap();
        PrefixCode { pipeline }
    }
}

impl PrefixStage {
    unsafe fn new(runner: &mut Runner, n_elements: u64) -> PrefixStage {
        let n_workgroups = (n_elements + ELEMENTS_PER_WG - 1) / ELEMENTS_PER_WG;
        let state_buf_size = 4 + 12 * n_workgroups;
        let state_buf = runner
            .session
            .create_buffer(state_buf_size, BufferUsage::STORAGE | BufferUsage::COPY_DST)
            .unwrap();
        PrefixStage {
            n_elements,
            state_buf,
        }
    }

    unsafe fn bind(&self, runner: &mut Runner, code: &PrefixCode, in_buf: &Buffer, out_buf: &Buffer) -> PrefixBinding {
        let descriptor_set = runner
            .session
            .create_simple_descriptor_set(&code.pipeline, &[in_buf, out_buf, &self.state_buf])
            .unwrap();
        PrefixBinding { descriptor_set }
    }

    unsafe fn record(&self, commands: &mut Commands, code: &PrefixCode, bindings: &PrefixBinding) {
        let n_workgroups = (self.n_elements + ELEMENTS_PER_WG - 1) / ELEMENTS_PER_WG;
        commands.cmd_buf.clear_buffer(&self.state_buf, None);
        commands.cmd_buf.memory_barrier();
        commands.cmd_buf.dispatch(
            &code.pipeline,
            &bindings.descriptor_set,
            (n_workgroups as u32, 1, 1),
            (WG_SIZE as u32, 1, 1),
        );
        // One thing that's missing here is registering the buffers so
        // they can be safely dropped by Rust code before the execution
        // of the command buffer completes.
    }
}

// Verify that the data is OEIS A000217
fn verify(data: &[u32]) -> Option<usize> {
    data.iter()
        .enumerate()
        .position(|(i, val)| ((i * (i + 1)) / 2) as u32 != *val)
}
