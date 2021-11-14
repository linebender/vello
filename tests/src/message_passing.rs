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

use piet_gpu_hal::{include_shader, BackendType, BindType, BufferUsage, DescriptorSet, ShaderCode};
use piet_gpu_hal::{Buffer, Pipeline};

use crate::clear::{ClearBinding, ClearCode, ClearStage};
use crate::config::Config;
use crate::runner::{Commands, Runner};
use crate::test_result::TestResult;

const N_ELEMENTS: u64 = 65536;

/// The shader code forMessagePassing sum example.
struct MessagePassingCode {
    pipeline: Pipeline,
    clear_code: Option<ClearCode>,
}

/// The stage resources for the prefix sum example.
struct MessagePassingStage {
    data_buf: Buffer,
    clear_stages: Option<(ClearStage, ClearBinding, ClearStage)>,
}

/// The binding for the prefix sum example.
struct MessagePassingBinding {
    descriptor_set: DescriptorSet,
    clear_binding: Option<ClearBinding>,
}

#[derive(Debug)]
pub enum Variant {
    Atomic,
    Vkmm,
}

pub unsafe fn run_message_passing_test(
    runner: &mut Runner,
    config: &Config,
    variant: Variant,
) -> TestResult {
    let mut result = TestResult::new(format!("message passing litmus, {:?}", variant));
    let out_buf = runner.buf_down(4);
    let code = MessagePassingCode::new(runner, variant);
    let stage = MessagePassingStage::new(runner, &code);
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
        failures += dst[0];
    }
    if failures > 0 {
        result.fail(format!("{} failures", failures));
    }
    result.timing(total_elapsed, N_ELEMENTS * n_iter);
    result
}

impl MessagePassingCode {
    unsafe fn new(runner: &mut Runner, variant: Variant) -> MessagePassingCode {
        let code = match variant {
            Variant::Atomic => include_shader!(&runner.session, "../shader/gen/message_passing"),
            Variant::Vkmm => {
                ShaderCode::Spv(include_bytes!("../shader/gen/message_passing_vkmm.spv"))
            }
        };
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
        MessagePassingCode {
            pipeline,
            clear_code,
        }
    }
}

impl MessagePassingStage {
    unsafe fn new(runner: &mut Runner, code: &MessagePassingCode) -> MessagePassingStage {
        let data_buf_size = 8 * N_ELEMENTS;
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
        MessagePassingStage {
            data_buf,
            clear_stages,
        }
    }

    unsafe fn bind(
        &self,
        runner: &mut Runner,
        code: &MessagePassingCode,
        out_buf: &Buffer,
    ) -> MessagePassingBinding {
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
        MessagePassingBinding {
            descriptor_set,
            clear_binding,
        }
    }

    unsafe fn record(
        &self,
        commands: &mut Commands,
        code: &MessagePassingCode,
        bindings: &MessagePassingBinding,
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
