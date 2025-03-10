// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// A WGSL shader for rendering sparse strips with alpha blending.
//
// Each strip instance represents a horizontal slice of the rendered output and consists of:
// 1. A variable-width region of alpha values for semi-transparent rendering
// 2. A solid color region for fully opaque areas
//
// The alpha values are stored in a texture and sampled during fragment shading.
// This approach optimizes memory usage by only storing alpha data where needed.

struct Config {
    width: u32,
    height: u32,
    strip_height: u32,
    alpha_texture_width: u32,
}

struct StripInstance {
    // [x, y] packed as u16's
    @location(0) xy: u32,
    // [width, dense_width] packed as u16's
    @location(1) widths: u32,
    @location(2) col: u32,
    @location(3) rgba: u32,
}

struct VertexOutput {
    @location(0) tex_coord: vec2<f32>,
    @location(1) @interpolate(flat) dense_end: u32,
    @location(2) @interpolate(flat) color: u32,
    @builtin(position) position: vec4<f32>,
};

@group(0) @binding(1)
var<uniform> config: Config;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    instance: StripInstance,
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(in_vertex_index & 1u);
    let y = f32(in_vertex_index >> 1u);
    let x0 = instance.xy & 0xffffu;
    let y0 = instance.xy >> 16u;
    let width = instance.widths & 0xffffu;
    let dense_width = instance.widths >> 16u;
    out.dense_end = instance.col + dense_width;
    let pix_x = f32(x0) + f32(width) * x;
    let pix_y = f32(y0) + y * f32(config.strip_height);
    let gl_x = (pix_x + 0.5) * 2.0 / f32(config.width) - 1.0;
    let gl_y = 1.0 - (pix_y + 0.5) * 2.0 / f32(config.height);
    out.position = vec4<f32>(gl_x, gl_y, 0.0, 1.0);
    out.tex_coord = vec2<f32>(f32(instance.col) + x * f32(width), y * f32(config.strip_height));
    out.color = instance.rgba;
    return out;
}

@group(0) @binding(0)
var alphas_texture: texture_2d<u32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let x = u32(floor(in.tex_coord.x));
    var alpha = 1.0;
    // TODO: This is a branch, but we can make it branchless by using a select
    // would it be faster to do a texture lookup for every pixel?
    if x < in.dense_end {
        let y = u32(floor(in.tex_coord.y));
        // Read alpha value from texture
        // Calculate texture coordinates based on x for the u32 value
        let tex_x = x % config.alpha_texture_width;
        let tex_y = x / config.alpha_texture_width;
        let a = textureLoad(alphas_texture, vec2<u32>(tex_x, tex_y), 0).x;
        alpha = f32((a >> (y * 8u)) & 0xffu) * (1.0 / 255.0);
    }
    return alpha * unpack4x8unorm(in.color);
}
