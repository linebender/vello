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
//
// The `StripInstance`'s `rgba_or_slot` field can either encode a color or a slot index.
// If the alpha value is non-zero, the fragment shader samples the alpha texture.
// Otherwise, the fragment shader samples the source clip texture using the given slot index.

struct Config {
    // Width of the rendering target
    width: u32,
    // Height of the rendering target
    height: u32,
    // Height of a strip in the rendering
    // CAUTION: When changing this value, you must also update the fragment shader's
    // logic to handle the new strip height.
    strip_height: u32,
    // Number of trailing zeros in alphas_tex_width (log2 of width).
    // Pre-calculated on CPU since WebGL2 doesn't support `firstTrailingBit`.
    alphas_tex_width_bits: u32,
}

struct StripInstance {
    // [x, y] packed as u16's
    @location(0) xy: u32,
    // [width, dense_width] packed as u16's
    @location(1) widths: u32,
    // Alpha texture column index where this strip's alpha values begin
    @location(2) col: u32,
    // [r, g, b, a] packed as u8's or a slot index when alpha is 0
    @location(3) rgba_or_slot: u32,
}

struct VertexOutput {
    // Texture coordinates for the current fragment
    @location(0) tex_coord: vec2<f32>,
    // Ending x-position of the dense (alpha) region
    @location(1) @interpolate(flat) dense_end: u32,
    // Color value or slot index when alpha is 0
    @location(2) @interpolate(flat) rgba_or_slot: u32,
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
    let ndc_x = pix_x * 2.0 / f32(config.width) - 1.0;
    let ndc_y = 1.0 - pix_y * 2.0 / f32(config.height);

    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.tex_coord = vec2<f32>(f32(instance.col) + x * f32(width), y * f32(config.strip_height));
    out.rgba_or_slot = instance.rgba_or_slot;
    return out;
}

@group(0) @binding(0)
var alphas_texture: texture_2d<u32>;

@group(0) @binding(2)
var clip_input_texture: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let x = u32(floor(in.tex_coord.x));
    var alpha = 1.0;
    // Determine if the current fragment is within the dense (alpha) region
    // If so, sample the alpha value from the texture; otherwise, alpha remains fully opaque (1.0)
    if x < in.dense_end {
        let y = u32(floor(in.tex_coord.y));
        // Retrieve alpha value from the texture. We store 16 1-byte alpha
        // values per texel, with each color channel packing 4 alpha values.
        // The code here assumes the strip height is 4, i.e., each color
        // channel encodes the alpha values for a single column within a strip.
        // Divide x by 4 to get the texel position.
        let alphas_index = x;
        let tex_dimensions = textureDimensions(alphas_texture);
        let alphas_tex_width = tex_dimensions.x;
        // Which texel contains the alpha values for this column
        let texel_index = alphas_index / 4u;
        // Which channel (R,G,B,A) in the texel contains the alpha values for this column
        let channel_index = alphas_index % 4u;
        // Calculate texel coordinates
        let tex_x = texel_index & (alphas_tex_width - 1u);
        let tex_y = texel_index >> config.alphas_tex_width_bits;

        // Load all 4 channels from the texture
        let rgba_values = textureLoad(alphas_texture, vec2<u32>(tex_x, tex_y), 0);

        // Get the column's alphas from the appropriate RGBA channel based on the index
        let alphas_u32 = unpack_alphas_from_channel(rgba_values, channel_index);
        // Extract the alpha value for the current y-position from the packed u32 data
        alpha = f32((alphas_u32 >> (y * 8u)) & 0xffu) * (1.0 / 255.0);
    }
    // Apply the alpha value to the unpacked RGBA color or slot index
    let alpha_byte = in.rgba_or_slot >> 24u;
    if alpha_byte != 0 {
        // in.rgba_or_slot encodes a color
        return alpha * unpack4x8unorm(in.rgba_or_slot);
    } else {
        // in.rgba_or_slot encodes a slot in the source clip texture
        let clip_x = u32(in.position.x) & 0xFFu;
        let clip_y = (u32(in.position.y) & 3) + in.rgba_or_slot * config.strip_height;
        let clip_in_color = textureLoad(clip_input_texture, vec2(clip_x, clip_y), 0);
        return alpha * clip_in_color;
    }
}

fn unpack_alphas_from_channel(rgba: vec4<u32>, channel_index: u32) -> u32 {
    switch channel_index {
        case 0u: { return rgba.x; }
        case 1u: { return rgba.y; }
        case 2u: { return rgba.z; }
        case 3u: { return rgba.w; }
        // Fallback, should never happen
        default: { return rgba.x; }
    }
}

// Polyfills `unpack4x8unorm`.
//
// Downlevel targets do not support native WGSL `unpack4x8unorm`.
// TODO: Remove once we upgrade to WGPU 25.
fn unpack4x8unorm(rgba_packed: u32) -> vec4<f32> {
    // Extract each byte and convert to float in range [0,1]
    return vec4<f32>(
        f32((rgba_packed >> 0u) & 0xFFu) / 255.0,  // r
        f32((rgba_packed >> 8u) & 0xFFu) / 255.0,  // g
        f32((rgba_packed >> 16u) & 0xFFu) / 255.0, // b
        f32((rgba_packed >> 24u) & 0xFFu) / 255.0  // a
    );
}
