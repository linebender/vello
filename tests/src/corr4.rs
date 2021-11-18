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

use piet_gpu_hal::{include_shader, BackendType, BindType, BufferUsage, DescriptorSet};
use piet_gpu_hal::{Buffer, Pipeline};

use crate::clear::{ClearBinding, ClearCode, ClearStage};
use crate::config::Config;
use crate::runner::{Commands, Runner};
use crate::test_result::TestResult;

const N_ELEMENTS: u64 = 65536;

/// The shader code for corr4 example.
struct Corr4Code {
    pipeline: Pipeline,
    clear_code: Option<ClearCode>,
}

/// The stage resources for corr4 example.
struct Corr4Stage {
    data_buf: Buffer,
    clear_stages: Option<(ClearStage, ClearBinding, ClearStage)>,
}

/// The binding for corr4 example.
struct Corr4Binding {
    descriptor_set: DescriptorSet,
    clear_binding: Option<ClearBinding>,
}

pub unsafe fn run_corr4_test(
    runner: &mut Runner,
    config: &Config,
) -> TestResult {
    let mut result = TestResult::new(format!("CoRR4 litmus"));
    let out_buf = runner.buf_down(16 * N_ELEMENTS);
    let code = Corr4Code::new(runner);
    let stage = Corr4Stage::new(runner, &code);
    let binding = stage.bind(runner, &code, &out_buf.dev_buf);
    let n_iter = config.n_iter;
    let mut total_elapsed = 0.0;
    let mut failures = 0;
    for _ in 0..n_iter {
        let mut commands = runner.commands();
        commands.write_timestamp(0);
        stage.record(&mut commands, &code, &binding, &out_buf.dev_buf);
        commands.write_timestamp(1);
        commands.cmd_buf.memory_barrier();
        commands.download(&out_buf);
        total_elapsed += runner.submit(commands);
        let mut dst: Vec<u32> = Default::default();
        out_buf.read(&mut dst);
        failures += analyze(&dst);
    }
    if failures > 0 {
        result.fail(format!("{} failures", failures));
    }
    result.timing(total_elapsed, N_ELEMENTS * n_iter);
    result
}

impl Corr4Code {
    unsafe fn new(runner: &mut Runner) -> Corr4Code {
        let code = include_shader!(&runner.session, "../shader/gen/corr4");
        let pipeline = runner
            .session
            .create_compute_pipeline(code, &[BindType::Buffer, BindType::Buffer])
            .unwrap();
        // Currently, DX12 and Metal backends don't support buffer clearing, so use a
        // compute shader as a workaround.
        let clear_code = if runner.backend_type() != BackendType::Vulkan {
            Some(ClearCode::new(runner))
        } else {
            None
        };
        Corr4Code {
            pipeline,
            clear_code,
        }
    }
}

impl Corr4Stage {
    unsafe fn new(runner: &mut Runner, code: &Corr4Code) -> Corr4Stage {
        let data_buf_size = 4 * N_ELEMENTS;
        let data_buf = runner
            .session
            .create_buffer(data_buf_size, BufferUsage::STORAGE | BufferUsage::COPY_DST)
            .unwrap();
        let clear_stages = if let Some(clear_code) = &code.clear_code {
            let stage0 = ClearStage::new(runner, N_ELEMENTS * 2);
            let binding0 = stage0.bind(runner, clear_code, &data_buf);
            let stage1 = ClearStage::new(runner, 1);
            Some((stage0, binding0, stage1))
        } else {
            None
        };
        Corr4Stage {
            data_buf,
            clear_stages,
        }
    }

    unsafe fn bind(
        &self,
        runner: &mut Runner,
        code: &Corr4Code,
        out_buf: &Buffer,
    ) -> Corr4Binding {
        let descriptor_set = runner
            .session
            .create_simple_descriptor_set(&code.pipeline, &[&self.data_buf, out_buf])
            .unwrap();
        let clear_binding = if let Some(clear_code) = &code.clear_code {
            Some(
                self.clear_stages
                    .as_ref()
                    .unwrap()
                    .2
                    .bind(runner, clear_code, out_buf),
            )
        } else {
            None
        };
        Corr4Binding {
            descriptor_set,
            clear_binding,
        }
    }

    unsafe fn record(
        &self,
        commands: &mut Commands,
        code: &Corr4Code,
        bindings: &Corr4Binding,
        out_buf: &Buffer,
    ) {
        if let Some((stage0, binding0, stage1)) = &self.clear_stages {
            let code = code.clear_code.as_ref().unwrap();
            stage0.record(commands, code, binding0);
            stage1.record(commands, code, bindings.clear_binding.as_ref().unwrap());
        } else {
            commands.cmd_buf.clear_buffer(&self.data_buf, None);
            commands.cmd_buf.clear_buffer(out_buf, None);
        }
        commands.cmd_buf.memory_barrier();
        commands.cmd_buf.dispatch(
            &code.pipeline,
            &bindings.descriptor_set,
            (256, 1, 1),
            (256, 1, 1),
        );
    }
}

fn analyze(data: &[u32]) -> u64 {
    let mut failures = 0;
    for i in 0..N_ELEMENTS as usize {
        let r0 = data[i * 4 + 0];
        let r1 = data[i * 4 + 1];
        let r2 = data[i * 4 + 2];
        let r3 = data[i * 4 + 3];
        if (r0 == 1 && r1 == 2 && r2 == 2 && r3 == 1) || (r0 == 2 && r1 == 1 && r2 == 1 && r3 == 2) || (r0 != 0 && r1 == 0) || (r2 != 0 && r3 == 0) {
            failures += 1;
        }
    }
    failures
}
