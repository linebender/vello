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

//! The compute shader and stage for clearing buffers.

use metal::{ComputePipelineState, Device};

const CLEAR_MSL: &str = r#"
using namespace metal;

struct ConfigBuf
{
    uint size;
    uint value;
};

kernel void main0(const device ConfigBuf& config [[buffer(0)]], device uint *data [[buffer(1)]], uint3 gid [[thread_position_in_grid]])
{
    uint ix = gid.x;
    if (ix < config.size)
    {
        data[ix] = config.value;
    }
}
"#;

pub fn make_clear_pipeline(device: &Device) -> ComputePipelineState {
    let options = metal::CompileOptions::new();
    let library = device.new_library_with_source(CLEAR_MSL, &options).unwrap();
    let function = library.get_function("main0", None).unwrap();
    device
        .new_compute_pipeline_state_with_function(&function).unwrap()

}

pub fn encode_clear(encoder: &metal::ComputeCommandEncoderRef, clear_pipeline: &ComputePipelineState, buffer: &metal::Buffer, size: u64) {
    // TODO: should be more careful with overflow
    let size_in_u32s = (size / 4) as u32;
    encoder.set_compute_pipeline_state(&clear_pipeline);
    let config = [size_in_u32s, 0];
    encoder.set_bytes(0, std::mem::size_of_val(&config) as u64, config.as_ptr() as *const _);
    encoder.set_buffer(1, Some(buffer), 0);
    let n_wg = (size_in_u32s + 255) / 256;
    let workgroup_count = metal::MTLSize {
        width: n_wg as u64,
        height: 1,
        depth: 1,
    };
    let workgroup_size = metal::MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(workgroup_count, workgroup_size);
}
