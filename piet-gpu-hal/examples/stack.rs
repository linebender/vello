// Copyright 2021 Google LLC
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

use piet_gpu_hal::hub;
use piet_gpu_hal::vulkan::VkInstance;
use piet_gpu_hal::{CmdBuf, MemFlags};

/// Generate a random Dyck sequence.
///
/// Here the encoding is: 1 is push, 0 is pop.
fn generate_dyck(n: usize) -> Vec<u32> {
    // Simple LCG random generator, so we don't need to import rand
    let mut z = 20170705u64;
    let mut depth = 0;
    (0..n)
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
        .collect()
}

fn verify_stack(input: &[u32], output: &[u32]) {
    let mut stack = Vec::new();
    for (i, (inp, outp)) in input.iter().zip(output).enumerate() {
        if let Some(tos) = stack.last() {
            assert_eq!(tos, outp, "i = {}", i);
        }
        if *inp == 0 {
            stack.pop();
        } else {
            stack.push(i as u32);
        }
    }
}

const N: usize = 512;

const WG_SIZE: usize = 16;

/// Size in bytes of each element of the state buffer.
const STATE_SIZE: u64 = 24;

fn main() {
    let (instance, _) = VkInstance::new(None).unwrap();
    unsafe {
        let device = instance.device(None).unwrap();
        let session = hub::Session::new(device);
        let mem_flags = MemFlags::host_coherent();
        let src = generate_dyck(N);
        println!("{:?}", src);
        let mut buffer = session
            .create_buffer(std::mem::size_of_val(&src[..]) as u64, mem_flags)
            .unwrap();
        buffer.write(&src).unwrap();
        let n_workgroup = (N / WG_SIZE) as u32;
        let state_buf = session
            .create_buffer(n_workgroup as u64 * STATE_SIZE, mem_flags)
            .unwrap();
        let stack_buf = session
            .create_buffer(std::mem::size_of_val(&src[..]) as u64, mem_flags)
            .unwrap();
        let code = include_bytes!("./shader/stack.spv");
        let pipeline = session.create_simple_compute_pipeline(code, 3).unwrap();
        let descriptor_set = session
            .create_simple_descriptor_set(&pipeline, &[&buffer, &state_buf, &stack_buf])
            .unwrap();
        let query_pool = session.create_query_pool(2).unwrap();
        let mut cmd_buf = session.cmd_buf().unwrap();
        cmd_buf.begin();
        cmd_buf.clear_buffer(&state_buf.vk_buffer(), None);
        cmd_buf.reset_query_pool(&query_pool);
        cmd_buf.write_timestamp(&query_pool, 0);
        cmd_buf.dispatch(&pipeline, &descriptor_set, (n_workgroup, 1, 1));
        cmd_buf.write_timestamp(&query_pool, 1);
        cmd_buf.host_barrier();
        cmd_buf.finish();
        let submitted = session.run_cmd_buf(cmd_buf, &[], &[]).unwrap();
        submitted.wait().unwrap();
        let timestamps = session.fetch_query_pool(&query_pool);
        let mut dst: Vec<u32> = Default::default();
        buffer.read(&mut dst).unwrap();
        for i in 0..n_workgroup as usize {
            println!(" out {}: {:?}", i, &dst[WG_SIZE * i..WG_SIZE * (i + 1)]);
        }
        stack_buf.read(&mut dst).unwrap();
        for i in 0..n_workgroup as usize {
            println!("stack {}: {:?}", i, &dst[WG_SIZE * i..WG_SIZE * (i + 1)]);
        }
        state_buf.read(&mut dst).unwrap();
        for i in 0..n_workgroup as usize {
            println!("state {}: {:?}", i, &dst[6 * i..6 * (i + 1)]);
        }
        println!("{:?}", timestamps);
        buffer.read(&mut dst).unwrap();
        verify_stack(&src, &dst);
    }
}
