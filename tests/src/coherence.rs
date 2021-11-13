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
use crate::runner::{Commands, Runner};
use crate::test_result::TestResult;
use crate::Config;

const WG_SIZE: u64 = 256;
const N_BUCKETS: u64 = 65536;
const OUT_BUF_SIZE: u64 = 256;

struct CoherenceCode {
    pipeline: Pipeline,
    clear_code: Option<ClearCode>,
}

struct CoherenceStage {
    clear_stage: Option<ClearStage>,
}

struct CoherenceBinding {
    descriptor_set: DescriptorSet,
    clear_binding: Option<ClearBinding>,
}

#[derive(Debug)]
pub enum Variant {
    Load,
    Rmw,
}

pub unsafe fn run_coherence_test(
    runner: &mut Runner,
    _config: &Config,
    variant: Variant,
) -> TestResult {
    let mut result = TestResult::new(format!("coherence, {:?}", variant));
    let data_buf = runner
        .session
        .create_buffer(4 * (N_BUCKETS + 1), BufferUsage::STORAGE)
        .unwrap();
    let out_buf = runner.buf_down(4 * OUT_BUF_SIZE * N_BUCKETS);
    let code = CoherenceCode::new(runner, variant);
    let stage = CoherenceStage::new(runner, &code, N_BUCKETS);
    let binding = stage.bind(runner, &code, &data_buf, &out_buf.dev_buf);
    // This runs long, and we're not really benchmarking, so just do 1 iter.
    let n_iter = 1;
    let mut total_elapsed = 0.0;
    for i in 0..n_iter {
        let mut commands = runner.commands();
        commands.write_timestamp(0);
        stage.record(&mut commands, &code, &binding, &data_buf);
        commands.write_timestamp(1);
        if i == 0 {
            commands.cmd_buf.memory_barrier();
            commands.download(&out_buf);
        }
        total_elapsed += runner.submit(commands);
        if i == 0 {
            let mut dst: Vec<u32> = Default::default();
            out_buf.read(&mut dst);
            result.info(analyze(total_elapsed, &dst));
        }
    }
    result.timing(total_elapsed, 0);
    result
}

impl CoherenceCode {
    unsafe fn new(runner: &mut Runner, variant: Variant) -> CoherenceCode {
        let code = match variant {
            Variant::Load => include_shader!(&runner.session, "../shader/gen/coherence"),
            Variant::Rmw => include_shader!(&runner.session, "../shader/gen/coherence_rmw"),
        };
        let pipeline = runner
            .session
            .create_compute_pipeline(code, &[BindType::Buffer, BindType::Buffer])
            .unwrap();
        let clear_code = if runner.backend_type() != BackendType::Vulkan {
            Some(ClearCode::new(runner))
        } else {
            None
        };
        CoherenceCode {
            pipeline,
            clear_code,
        }
    }
}

impl CoherenceStage {
    unsafe fn new(runner: &mut Runner, code: &CoherenceCode, n_buckets: u64) -> CoherenceStage {
        let clear_stage = if code.clear_code.is_some() {
            Some(ClearStage::new(runner, n_buckets + 1))
        } else {
            None
        };
        CoherenceStage { clear_stage }
    }

    unsafe fn bind(
        &self,
        runner: &mut Runner,
        code: &CoherenceCode,
        data_buf: &Buffer,
        out_buf: &Buffer,
    ) -> CoherenceBinding {
        let descriptor_set = runner
            .session
            .create_simple_descriptor_set(&code.pipeline, &[data_buf, out_buf])
            .unwrap();
        let clear_binding = if let Some(stage) = &self.clear_stage {
            Some(stage.bind(runner, &code.clear_code.as_ref().unwrap(), data_buf))
        } else {
            None
        };
        CoherenceBinding {
            descriptor_set,
            clear_binding,
        }
    }

    unsafe fn record(
        &self,
        commands: &mut Commands,
        code: &CoherenceCode,
        bindings: &CoherenceBinding,
        data_buf: &Buffer,
    ) {
        if let Some(stage) = &self.clear_stage {
            stage.record(
                commands,
                code.clear_code.as_ref().unwrap(),
                bindings.clear_binding.as_ref().unwrap(),
            );
        } else {
            commands.cmd_buf.clear_buffer(data_buf, None);
        }
        commands.cmd_buf.memory_barrier();
        let n_workgroups = N_BUCKETS / WG_SIZE;
        commands.cmd_buf.dispatch(
            &code.pipeline,
            &bindings.descriptor_set,
            (n_workgroups as u32, 1, 1),
            (WG_SIZE as u32, 1, 1),
        );
    }
}

fn analyze(elapsed: f64, results: &[u32]) -> String {
    let mut max_ts = 0;
    let mut sum_ticks = 0.0;
    let mut n_samples = 0;
    for i in 0..N_BUCKETS {
        let start_ix = i * OUT_BUF_SIZE;
        for j in 1..OUT_BUF_SIZE {
            if j == OUT_BUF_SIZE - 1 || results[(start_ix + j + 1) as usize] == !0 {
                let end_ts = results[(start_ix + j) as usize];
                max_ts = max_ts.max(end_ts);
                break;
            }
            sum_ticks += results[(start_ix + j) as usize] as f64;
            n_samples += 1;
        }
    }
    let clock_res = elapsed / max_ts as f64;
    let mean_latency = clock_res * sum_ticks / n_samples as f64;
    format!(
        "clock resolution {}s, mean latency {}s",
        crate::test_result::format_nice(clock_res, 1),
        crate::test_result::format_nice(mean_latency, 1)
    )
}
