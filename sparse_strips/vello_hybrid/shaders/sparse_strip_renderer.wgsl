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
    // Width of the rendering target    
    width: u32,
    // Height of the rendering target
    height: u32,
    // Height of a strip in the rendering
    strip_height: u32,
}

struct StripInstance {
    // [x, y] packed as u16's
    @location(0) xy: u32,
    // [width, dense_width] packed as u16's
    @location(1) widths: u32,
    // Alpha texture column index where this strip's alpha values begin
    @location(2) col: u32,
    // [r, g, b, a] packed as u8's
    @location(3) rgba: u32,
}

struct VertexOutput {
    // Texture coordinates for the current fragment 
    @location(0) tex_coord: vec2<f32>,
    // Ending x-position of the dense (alpha) region
    @location(1) @interpolate(flat) dense_end: u32,
    // RGBA color value
    @location(2) @interpolate(flat) color: u32,
    // Normalized device coordinates (NDC) for the current vertex
    @builtin(position) position: vec4<f32>,
};

// TODO: Measure performance of moving to a separate group
@group(0) @binding(1)
var<uniform> config: Config;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    instance: StripInstance,
) -> VertexOutput {
    var out: VertexOutput;
    // Map vertex_index (0-3) to quad corners:
    // 0 → (0,0), 1 → (1,0), 2 → (0,1), 3 → (1,1)
    let x = f32(in_vertex_index & 1u);
    let y = f32(in_vertex_index >> 1u);
    // Unpack the x and y coordinates from the packed u32 instance.xy
    let x0 = instance.xy & 0xffffu;
    let y0 = instance.xy >> 16u;
    // Unpack the total width and dense (alpha) width from the packed u32 instance.widths
    let width = instance.widths & 0xffffu;
    let dense_width = instance.widths >> 16u;
    // Calculate the ending x-position of the dense (alpha) region
    // This boundary is used in the fragment shader to determine if alpha sampling is needed
    out.dense_end = instance.col + dense_width;
    // Calculate the pixel coordinates of the current vertex within the strip
    let pix_x = f32(x0) + f32(width) * x;
    let pix_y = f32(y0) + y * f32(config.strip_height);
    // Convert pixel coordinates to normalized device coordinates (NDC)
    // NDC ranges from -1 to 1, with (0,0) at the center of the viewport
    let ndc_x = (pix_x + 0.5) * 2.0 / f32(config.width) - 1.0;
    let ndc_y = 1.0 - (pix_y + 0.5) * 2.0 / f32(config.height);

    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
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
    // Determine if the current fragment is within the dense (alpha) region
    // If so, sample the alpha value from the texture; otherwise, alpha remains fully opaque (1.0)
    // TODO: This is a branch, but we can make it branchless by using a select
    // would it be faster to do a texture lookup for every pixel?
    if x < in.dense_end {
        let y = u32(floor(in.tex_coord.y));
        // Retrieve alpha value from the texture
        // Calculate texture coordinates based on the fragment's x-position
        let tex_dimensions = textureDimensions(alphas_texture);
        let tex_width = tex_dimensions.x;
        let tex_x = x % tex_width;
        let tex_y = x / tex_width;
        let a = textureLoad(alphas_texture, vec2<u32>(tex_x, tex_y), 0).x;
        // Extract the alpha value for the current y-position from the packed u32 texture data
        alpha = f32((a >> (y * 8u)) & 0xffu) * (1.0 / 255.0);
    }
    // Apply the alpha value to the unpacked RGBA color
    return alpha * unpack4x8unorm(in.color);
}
