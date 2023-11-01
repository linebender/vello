// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Writes an array to a texture.

#import config

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> source: array<u32>;

@group(0) @binding(2)
var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let row = global_id.y * config.target_width;
    let pixel = source[row + global_id.x];
    textureStore(output, vec2<i32>(global_id.xy), unpack4x8unorm(pixel));
}
