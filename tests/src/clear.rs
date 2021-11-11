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

//! Utilities (and a benchmark) for clearing buffers with compute shaders.

use piet_gpu_hal::{include_shader, BindType, BufferUsage, DescriptorSet};
use piet_gpu_hal::{Buffer, Pipeline};

use crate::config::Config;
use crate::runner::{Commands, Runner};
use crate::test_result::TestResult;

const WG_SIZE: u64 = 256;

/// The shader code for clearing buffers.
pub struct ClearCode {
    pipeline: Pipeline,
}

/// The stage resources for clearing buffers.
pub struct ClearStage {
    n_elements: u64,
    config_buf: Buffer,
}

/// The binding for clearing buffers.
pub struct ClearBinding {
    descriptor_set: DescriptorSet,
}

pub unsafe fn run_clear_test(runner: &mut Runner, config: &Config) -> TestResult {
    let mut result = TestResult::new("clear buffers");
    let n_elements: u64 = config.size.choose(1 << 12, 1 << 20, 1 << 24);
    let out_buf = runner.buf_down(n_elements * 4);
    let code = ClearCode::new(runner);
    let stage = ClearStage::new_with_value(runner, n_elements, 0x42);
    let binding = stage.bind(runner, &code, &out_buf.dev_buf);
    let n_iter = config.n_iter;
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
            if let Some(failure) = verify(&dst) {
                result.fail(format!("failure at {}", failure));
            }
        }
    }
    result.timing(total_elapsed, n_elements * n_iter);
    result
}

impl ClearCode {
    pub unsafe fn new(runner: &mut Runner) -> ClearCode {
        let code = include_shader!(&runner.session, "../shader/gen/Clear");
        let pipeline = runner
            .session
            .create_compute_pipeline(code, &[BindType::BufReadOnly, BindType::Buffer])
            .unwrap();
        ClearCode { pipeline }
    }
}

impl ClearStage {
    pub unsafe fn new(runner: &mut Runner, n_elements: u64) -> ClearStage {
        Self::new_with_value(runner, n_elements, 0)
    }

    pub unsafe fn new_with_value(runner: &mut Runner, n_elements: u64, value: u32) -> ClearStage {
        let config = [n_elements as u32, value];
        let config_buf = runner
            .session
            .create_buffer_init(&config, BufferUsage::STORAGE)
            .unwrap();
        ClearStage {
            n_elements,
            config_buf,
        }
    }

    pub unsafe fn bind(
        &self,
        runner: &mut Runner,
        code: &ClearCode,
        out_buf: &Buffer,
    ) -> ClearBinding {
        let descriptor_set = runner
            .session
            .create_simple_descriptor_set(&code.pipeline, &[&self.config_buf, out_buf])
            .unwrap();
        ClearBinding { descriptor_set }
    }

    pub unsafe fn record(
        &self,
        commands: &mut Commands,
        code: &ClearCode,
        bindings: &ClearBinding,
    ) {
        let n_workgroups = (self.n_elements + WG_SIZE - 1) / WG_SIZE;
        // An issue: for clearing large buffers (>16M), we need to check the
        // number of workgroups against the (dynamically detected) limit, and
        // potentially issue multiple dispatches.
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

// Verify that the data is cleared.
fn verify(data: &[u32]) -> Option<usize> {
    data.iter().position(|val| *val != 0x42)
}
