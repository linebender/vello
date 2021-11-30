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

use crate::runner::{Commands, Runner};
use crate::test_result::TestResult;
use crate::Config;

const WG_SIZE: u64 = 256;
const N_BUCKETS: u64 = 65536;

struct LinkedListCode {
    pipeline: Pipeline,
}

struct LinkedListStage;

struct LinkedListBinding {
    descriptor_set: DescriptorSet,
}

pub unsafe fn run_linkedlist_test(runner: &mut Runner, config: &Config) -> TestResult {
    let mut result = TestResult::new("linked list");
    let mem_buf = runner.buf_down(1024 * N_BUCKETS, BufferUsage::CLEAR);
    let code = LinkedListCode::new(runner);
    let stage = LinkedListStage::new(runner, &code, N_BUCKETS);
    let binding = stage.bind(runner, &code, &mem_buf.dev_buf);
    let n_iter = config.n_iter;
    let mut total_elapsed = 0.0;
    for i in 0..n_iter {
        let mut commands = runner.commands();
        // Might clear only buckets to save time.
        commands.write_timestamp(0);
        stage.record(&mut commands, &code, &binding, &mem_buf.dev_buf);
        commands.write_timestamp(1);
        if i == 0 || config.verify_all {
            commands.cmd_buf.memory_barrier();
            commands.download(&mem_buf);
        }
        total_elapsed += runner.submit(commands);
        if i == 0 || config.verify_all {
            let mut dst: Vec<u32> = Default::default();
            mem_buf.read(&mut dst);
            if !verify(&dst) {
                result.fail("incorrect data");
            }
        }
    }
    result.timing(total_elapsed, N_BUCKETS * 100 * n_iter);
    result
}

impl LinkedListCode {
    unsafe fn new(runner: &mut Runner) -> LinkedListCode {
        let code = include_shader!(&runner.session, "../shader/gen/linkedlist");
        let pipeline = runner
            .session
            .create_compute_pipeline(code, &[BindType::Buffer])
            .unwrap();
        LinkedListCode { pipeline }
    }
}

impl LinkedListStage {
    unsafe fn new(
        _runner: &mut Runner,
        _code: &LinkedListCode,
        _n_buckets: u64,
    ) -> LinkedListStage {
        LinkedListStage
    }

    unsafe fn bind(
        &self,
        runner: &mut Runner,
        code: &LinkedListCode,
        mem_buf: &Buffer,
    ) -> LinkedListBinding {
        let descriptor_set = runner
            .session
            .create_simple_descriptor_set(&code.pipeline, &[mem_buf])
            .unwrap();
        LinkedListBinding { descriptor_set }
    }

    unsafe fn record(
        &self,
        commands: &mut Commands,
        code: &LinkedListCode,
        bindings: &LinkedListBinding,
        out_buf: &Buffer,
    ) {
        commands.cmd_buf.clear_buffer(out_buf, None);
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

fn verify(data: &[u32]) -> bool {
    let mut expected = (0..N_BUCKETS).map(|_| Vec::new()).collect::<Vec<_>>();
    for ix in 0..N_BUCKETS {
        let mut rng = ix as u32 + 1;
        for _ in 0..100 {
            // xorshift32
            rng ^= rng.wrapping_shl(13);
            rng ^= rng.wrapping_shr(17);
            rng ^= rng.wrapping_shl(5);
            let bucket = rng % N_BUCKETS as u32;
            if bucket != 0 {
                expected[bucket as usize].push(ix as u32);
            }
        }
    }
    let mut actual = Vec::new();
    for (i, expected) in expected.iter_mut().enumerate().skip(1) {
        actual.clear();
        let mut ptr = i;
        loop {
            let next = data[ptr] as usize;
            if next == 0 {
                break;
            }
            let val = data[next + 1];
            actual.push(val);
            ptr = next;
        }
        actual.sort();
        expected.sort();
        if actual != *expected {
            return false;
        }
    }
    true
}
