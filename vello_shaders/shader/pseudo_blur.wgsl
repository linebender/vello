// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// A simple blur-like filter, which is intentionally wrong.


@group(0) @binding(0)
var input: texture_2d<f32>;

@group(0) @binding(1)
var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(64, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim = textureDimensions(input);
    if global_id.y >= dim.y{
        return;
    }
    if global_id.x >= dim.x {
        return;
    }
    let prefixer_idx = max(global_id.x, 2u) - 2u;
    let prefixer_value = textureLoad(input, vec2(prefixer_idx, global_id.y), 0);
    let prefix_idx = max(global_id.x, 1u) - 1u;
    let prefix_value = textureLoad(input, vec2(prefix_idx, global_id.y), 0);
    let central_value = textureLoad(input, vec2(global_id.x, global_id.y), 0);
    let suffix_idx = min(global_id.x+1u, dim.x- 1u);
    let suffix_value = textureLoad(input, vec2(suffix_idx, global_id.y), 0);
    let suffixer_idx = min(global_id.x+2u, dim.x- 2u);
    let suffixer_value = textureLoad(input, vec2(suffixer_idx, global_id.y), 0);
    let value = vec4((prefixer_value + prefix_value + central_value + suffix_value+ suffixer_value) / 5.);
    textureStore(output, vec2<i32>(global_id.xy), value);
}
