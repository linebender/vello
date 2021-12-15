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

use piet_gpu_hal::{include_shader, BindType, BufferUsage, DescriptorSet};
use piet_gpu_hal::{Buffer, Pipeline};

use crate::config::Config;
use crate::runner::{Commands, Runner};
use crate::test_result::TestResult;

const WG_SIZE: u64 = 512;
const N_ROWS: u64 = 1;
const ELEMENTS_PER_WG: u64 = WG_SIZE * N_ROWS;

struct StackCode {
    reduce_pipeline: Pipeline,
    leaf_pipeline: Pipeline,
}

struct StackStage {
    bic_buf: Buffer,
    stack_buf: Buffer,
}

struct StackBinding {
    reduce_ds: DescriptorSet,
    leaf_ds: DescriptorSet,
}

struct StackData {
    dyck: Vec<u32>,
}

pub unsafe fn run_stack_test(runner: &mut Runner, config: &Config) -> TestResult {
    let mut result = TestResult::new("stack monoid");
    let n_elements: u64 = config.size.choose(1 << 9, 1 << 12, 1 << 18);
    let data = StackData::new(n_elements);
    let data_buf = runner
        .session
        .create_buffer_init(&data.dyck, BufferUsage::STORAGE)
        .unwrap();
    let out_buf = runner.buf_down(data_buf.size(), BufferUsage::empty());

    let code = StackCode::new(runner);
    let stage = StackStage::new(runner, n_elements);
    let binding = stage.bind(runner, &code, &data_buf, &out_buf.dev_buf);

    let mut total_elapsed = 0.0;
    let n_iter = config.n_iter;
    for i in 0..n_iter {
        let mut commands = runner.commands();
        commands.write_timestamp(0);
        stage.record(&mut commands, &code, &binding, n_elements);
        commands.write_timestamp(1);
        if i == 0 || config.verify_all {
            commands.cmd_buf.memory_barrier();
            commands.download(&out_buf);
        }
        total_elapsed += runner.submit(commands);
        if i == 0 || config.verify_all {
            let dst = out_buf.map_read(..);
            if let Some(failure) = data.verify(dst.cast_slice()) {
                result.fail(failure);
            }
        }
    }

    result.timing(total_elapsed, n_elements * n_iter);
    result
}

impl StackCode {
    unsafe fn new(runner: &mut Runner) -> StackCode {
        let reduce_code =
            piet_gpu_hal::ShaderCode::Spv(include_bytes!("../shader/gen/stack_reduce.spv"));
        let reduce_pipeline = runner
            .session
            .create_compute_pipeline(
                reduce_code,
                &[BindType::BufReadOnly, BindType::Buffer, BindType::Buffer],
            )
            .unwrap();
        let leaf_code =
            piet_gpu_hal::ShaderCode::Spv(include_bytes!("../shader/gen/stack_leaf.spv"));
        let leaf_pipeline = runner
            .session
            .create_compute_pipeline(
                leaf_code,
                &[
                    BindType::BufReadOnly,
                    BindType::BufReadOnly,
                    BindType::BufReadOnly,
                    BindType::Buffer,
                ],
            )
            .unwrap();
        StackCode {
            reduce_pipeline,
            leaf_pipeline,
        }
    }
}

impl StackStage {
    unsafe fn new(runner: &mut Runner, n_elements: u64) -> StackStage {
        assert!(n_elements <= ELEMENTS_PER_WG.pow(2));
        let stack_buf = runner
            .session
            .create_buffer(4 * n_elements, BufferUsage::STORAGE)
            .unwrap();
        let bic_size = ELEMENTS_PER_WG * 8;
        let bic_buf = runner
            .session
            .create_buffer(bic_size, BufferUsage::STORAGE)
            .unwrap();
        StackStage { bic_buf, stack_buf }
    }

    unsafe fn bind(
        &self,
        runner: &mut Runner,
        code: &StackCode,
        in_buf: &Buffer,
        out_buf: &Buffer,
    ) -> StackBinding {
        let reduce_ds = runner
            .session
            .create_simple_descriptor_set(
                &code.reduce_pipeline,
                &[in_buf, &self.bic_buf, &self.stack_buf],
            )
            .unwrap();
        let leaf_ds = runner
            .session
            .create_simple_descriptor_set(
                &code.leaf_pipeline,
                &[in_buf, &self.bic_buf, &self.stack_buf, out_buf],
            )
            .unwrap();
        StackBinding { reduce_ds, leaf_ds }
    }

    unsafe fn record(
        &self,
        commands: &mut Commands,
        code: &StackCode,
        binding: &StackBinding,
        size: u64,
    ) {
        let n_workgroups = (size + ELEMENTS_PER_WG - 1) / ELEMENTS_PER_WG;
        commands.cmd_buf.dispatch(
            &code.reduce_pipeline,
            &binding.reduce_ds,
            (n_workgroups as u32, 1, 1),
            (WG_SIZE as u32, 1, 1),
        );
        commands.cmd_buf.memory_barrier();
        commands.cmd_buf.dispatch(
            &code.leaf_pipeline,
            &binding.leaf_ds,
            (n_workgroups as u32, 1, 1),
            (WG_SIZE as u32, 1, 1),
        );
    }
}

impl StackData {
    /// Generate a random Dyck sequence.
    ///
    /// Here the encoding is: 1 is push, 0 is pop.
    fn new(n: u64) -> StackData {
        // Simple LCG random generator, so we don't need to import rand
        let mut z = 20170705u64;
        let mut depth = 0;
        let dyck = (0..n)
            .map(|_| {
                let is_push = if depth < 2 {
                    1
                } else {
                    z = z.wrapping_mul(742938285) % ((1 << 31) - 1);
                    (z % 2) as u32
                };
                if is_push == 1 {
                    depth += 1;
                } else {
                    depth -= 1;
                }
                is_push
            })
            .collect();
        StackData { dyck }
    }

    fn verify(&self, data: &[u32]) -> Option<String> {
        let mut stack = Vec::new();
        for (i, (inp, outp)) in self.dyck.iter().zip(data).enumerate() {
            if let Some(tos) = stack.last() {
                if tos != outp {
                    return Some(format!("mismatch at {}: {} != {}", i, tos, outp));
                }
            }
            if *inp == 0 {
                stack.pop();
            } else {
                stack.push(i as u32);
            }
        }
        None
    }
}
