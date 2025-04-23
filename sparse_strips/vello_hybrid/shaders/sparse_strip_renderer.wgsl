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

// Paint types to determine how to process a strip
const PAINT_TYPE_SOLID: u32 = 0u;  
const PAINT_TYPE_ALPHA: u32 = 1u;  
const PAINT_TYPE_IMAGE: u32 = 2u;  

// Configuration for the renderer
struct Config {
    // Width of the rendering target    
    width: u32,
    // Height of the rendering target
    height: u32,
    // Height of a strip in the rendering
    strip_height: u32,
    // Number of trailing zeros in alphas_tex_width (log2 of width).
    // Pre-calculated on CPU since WebGL2 doesn't support `firstTrailingBit`.
    alphas_tex_width_bits: u32,
}


// If paint_type is 0 or 1:
// - paint_data is the packed rgba values
// If paint_type is 2:
// - uv is the packed (u0, v0)
// - paint_data is the packed (extend_x, extend_y)
struct StripInstance {
    // [x, y] packed as u16's
    @location(0) xy: u32,
    // [width, dense_width] packed as u16's
    @location(1) widths: u32,
    // Column index
    @location(2) col_idx: u32,
    // Paint type
    @location(3) paint_type: u32,
    // Paint data
    @location(4) paint_data: u32,
    // Paint index
    @location(5) uv: u32,
    // Paint x_advance
    @location(6) x_advance: vec2<f32>,
    // Paint y_advance
    @location(7) y_advance: vec2<f32>,
}

struct VertexOutput {
    // Render type for the strip
    @location(0) @interpolate(flat) paint_type: u32,
    // Texture coordinates for the current fragment 
    @location(1) tex_coord: vec2<f32>,
    // Texture coordinates for the current fragment 
    @location(2) tex_coord2: vec2<f32>,
    // Ending x-position of the dense (alpha) region
    @location(3) @interpolate(flat) dense_end: u32,
    // RGBA color value
    @location(4) @interpolate(flat) color: u32,
    // Extends
    @location(5) @interpolate(flat) extends_xy: vec2<u32>,
    // Normalized device coordinates (NDC) for the current vertex
    @builtin(position) position: vec4<f32>,
};

// TODO: Measure performance of moving to a separate group
@group(0) @binding(1)
var<uniform> config: Config;

// Image texture
@group(1) @binding(0)
var image_texture: texture_2d<f32>;
@group(1) @binding(1)
var image_sampler: sampler;

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
    // Calculate the pixel coordinates of the current vertex within the strip
    let pix_x = f32(x0) + x * f32(width);
    let pix_y = f32(y0) + y * f32(config.strip_height);
    // Convert pixel coordinates to normalized device coordinates (NDC)
    // NDC ranges from -1 to 1, with (0,0) at the center of the viewport
    let ndc_x = pix_x * 2.0 / f32(config.width) - 1.0;
    let ndc_y = 1.0 - pix_y * 2.0 / f32(config.height);

    // Calculate the ending x-position of the dense (alpha) region
    // This boundary is used in the fragment shader to determine if alpha sampling is needed
    out.dense_end = instance.col_idx + dense_width;
    out.color = instance.paint_data;
    out.paint_type = instance.paint_type;
    // Regular texture coordinates for other render types
    out.tex_coord = vec2<f32>(f32(instance.col_idx) + x * f32(width), f32(y0) + y * f32(config.strip_height));
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);

    if instance.paint_type == PAINT_TYPE_IMAGE {
        // Get texture dimensions 
        let tex_dimensions = textureDimensions(image_texture);
        // Unpack the u0 and v0 from the packed u32 instance.u0v0
        let u0 = instance.uv & 0xffffu;
        let v0 = instance.uv >> 16u;
        // Unpack the x_extend and y_extend from the packed u32 instance.extends_xy
        let extend_x = instance.paint_data & 0xffffu;
        let extend_y = instance.paint_data >> 16u;
        
        // Vertex position within the texture
        let sample_x = f32(u0) + x * f32(width) * instance.x_advance.x + y * f32(config.strip_height) * instance.y_advance.x;
        let sample_y = f32(v0) + x * f32(width) * instance.x_advance.y + y * f32(config.strip_height) * instance.y_advance.y;

        let u = sample_x / f32(tex_dimensions.x);
        let v = sample_y / f32(tex_dimensions.y);

        out.extends_xy = vec2<u32>(extend_x, extend_y);
        out.tex_coord2 = vec2<f32>(u, v);
    }
    
    return out;
}

@group(0) @binding(0)
var alphas_texture: texture_2d<u32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var final_color = unpack4x8unorm(in.color);

    let x = u32(floor(in.tex_coord.x));
    var alpha = 1.0;
    // Determine if the current fragment is within the dense (alpha) region
    // If so, sample the alpha value from the texture; otherwise, alpha remains fully opaque (1.0)
    // TODO: This is a branch, but we can make it branchless by using a select
    // would it be faster to do a texture lookup for every pixel?
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
    final_color = alpha * unpack4x8unorm(in.color);

    if in.paint_type == PAINT_TYPE_IMAGE {
        let u = extend(in.tex_coord2.x, in.extends_xy.x);
        let v = extend(in.tex_coord2.y, in.extends_xy.y);
        final_color = alpha * textureSample(image_texture, image_sampler, vec2<f32>(u, v));
    }
    
    return final_color;
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
fn unpack4x8unorm(rgba_packed: u32) -> vec4<f32> {
    // Extract each byte and convert to float in range [0,1]
    return vec4<f32>(
        f32((rgba_packed >> 0u) & 0xFFu) / 255.0,  // r
        f32((rgba_packed >> 8u) & 0xFFu) / 255.0,  // g
        f32((rgba_packed >> 16u) & 0xFFu) / 255.0, // b
        f32((rgba_packed >> 24u) & 0xFFu) / 255.0  // a
    );
}

const EXTEND_PAD: u32 = 0u;
const EXTEND_REPEAT: u32 = 1u;
const EXTEND_REFLECT: u32 = 2u;
fn extend(t: f32, mode: u32) -> f32 {
    switch mode {
        case EXTEND_PAD: {
            return clamp(t, 0.0, 1.0);
        }
        case EXTEND_REPEAT: {
            return fract(t);
        }
        case EXTEND_REFLECT, default: {
            return abs(t - 2.0 * round(0.5 * t));
        }
    }
}
