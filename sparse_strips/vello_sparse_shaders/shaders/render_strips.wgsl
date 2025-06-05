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

// Paint types to determine how to process a strip
const PAINT_TYPE_SOLID: u32 = 0u;  
const PAINT_TYPE_ALPHA: u32 = 1u;  
const PAINT_TYPE_IMAGE: u32 = 2u;  

// Image quality
const IMAGE_QUALITY_LOW = 0u;
const IMAGE_QUALITY_MEDIUM = 1u;
const IMAGE_QUALITY_HIGH = 2u;

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
    // Alpha texture column index where this strip's alpha values begin
    @location(2) col_idx: u32,
    // Paint type
    @location(3) paint_type: u32,
    // Paint data
    @location(4) paint_data: u32,
    // Paint index
    @location(5) uv: vec2<f32>,
    // Paint x_advance
    @location(6) x_advance: vec2<f32>,
    // Paint y_advance
    @location(7) y_advance: vec2<f32>,
    // Image size
    @location(8) image_size: vec2<u32>,
    // Image offset
    @location(9) image_offset: vec2<u32>,
    // Image quality
    @location(10) image_quality: u32,

    // [r, g, b, a] packed as u8's or a slot index when alpha is 0
    // @location(3) rgba_or_slot: u32,
}

struct VertexOutput {
    // Render type for the strip
    @location(0) @interpolate(flat) paint_type: u32,
    // Texture coordinates for the current fragment
    @location(1) tex_coord: vec2<f32>,
    // Texture coordinates for the current fragment 
    @location(2) uv: vec2<f32>,
    // Ending x-position of the dense (alpha) region
    @location(3) @interpolate(flat) dense_end: u32,
    // Color value or slot index when alpha is 0
    @location(4) @interpolate(flat) paint_data: u32,
    // Extends
    @location(5) @interpolate(flat) extends_xy: vec2<u32>,
    // Image size
    @location(6) @interpolate(flat) image_size: vec2<u32>,
    // Image offset
    @location(7) @interpolate(flat) image_offset: vec2<u32>,
    // Image quality
    @location(8) @interpolate(flat) image_quality: u32,
    // Normalized device coordinates (NDC) for the current vertex
    @builtin(position) position: vec4<f32>,
};

// TODO: Measure performance of moving to a separate group
@group(0) @binding(1)
var<uniform> config: Config;

@group(1) @binding(0)
var atlas_texture: texture_2d<f32>;
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
    out.paint_data = instance.paint_data;
    out.paint_type = instance.paint_type;
    out.image_size = instance.image_size;
    out.image_offset = instance.image_offset;
    out.image_quality = instance.image_quality;
    // Regular texture coordinates for other render types
    out.tex_coord = vec2<f32>(f32(instance.col_idx) + x * f32(width), f32(y0) + y * f32(config.strip_height));
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);

    if instance.paint_type == PAINT_TYPE_IMAGE {
        // Get texture dimensions 
        let tex_dimensions = textureDimensions(atlas_texture);
        // Unpack the u0 and v0 from the packed u32 instance.u0v0
        // let u0 = instance.uv & 0xffffu;
        // let v0 = instance.uv >> 16u;
        let u0 = instance.uv.x;
        let v0 = instance.uv.y;
        // Unpack the x_extend and y_extend from the packed u32 instance.extends_xy
        let extend_x = instance.paint_data & 0xffffu;
        let extend_y = instance.paint_data >> 16u;

        // Apply image offset to the base texture coordinates
        let offset_u0 = u0 + f32(instance.image_offset.x);
        let offset_v0 = v0 + f32(instance.image_offset.y);

        // Vertex position within the texture
        // Since u0, v0 now represent the texture coordinate for the center of the top-left screen pixel,
        // we need to adjust for the fact that quad corners are offset from pixel centers
        let sample_x = offset_u0 + (x * f32(width)) * instance.x_advance.x + (y * f32(config.strip_height)) * instance.y_advance.x;
        let sample_y = offset_v0 + (x * f32(width)) * instance.x_advance.y + (y * f32(config.strip_height)) * instance.y_advance.y;

        let u = sample_x / f32(tex_dimensions.x);
        let v = sample_y / f32(tex_dimensions.y);

        out.extends_xy = vec2<u32>(extend_x, extend_y);
        out.uv = vec2<f32>(u, v);
    }

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
    let alpha_byte = in.paint_data >> 24u;
    var final_color = unpack4x8unorm(in.paint_data);

    if in.paint_type == PAINT_TYPE_IMAGE {
        let tex_dimensions = textureDimensions(atlas_texture);
        let offset_u = (f32(in.image_offset.x)) / f32(tex_dimensions.x);
        let offset_v = (f32(in.image_offset.y)) / f32(tex_dimensions.y);
        let size_u = f32(in.image_size.x) / f32(tex_dimensions.x);
        let size_v = f32(in.image_size.y) / f32(tex_dimensions.y);
        
        // Apply extend mode to the UV coordinates relative to the image bounds
        let local_u = (in.uv.x - offset_u) / size_u;
        let local_v = (in.uv.y - offset_v) / size_v;
        let extended_u = extend_mode_normalized(local_u, in.extends_xy.x);
        let extended_v = extend_mode_normalized(local_v, in.extends_xy.y);
        
        // Add a small offset to avoid sampling outside the image bounds
        let half_pixel_u = 0.5 / f32(tex_dimensions.x);
        let half_pixel_v = 0.5 / f32(tex_dimensions.y);
        
        // Clamp the extended coordinates to stay within bounds
        let clamped_extended_u = clamp(extended_u, 0.0, 1.0 - half_pixel_u / size_u);
        let clamped_extended_v = clamp(extended_v, 0.0, 1.0 - half_pixel_v / size_v);
        
        // Map back to atlas coordinates
        let final_u = offset_u + clamped_extended_u * size_u;
        let final_v = offset_v + clamped_extended_v * size_v;
        
        // Convert normalized coordinates to integer pixel coordinates for textureLoad
        let pixel_u = u32(final_u * f32(tex_dimensions.x));
        let pixel_v = u32(final_v * f32(tex_dimensions.y));
        
        if in.image_quality == IMAGE_QUALITY_LOW {
            final_color = alpha * textureLoad(atlas_texture, vec2<u32>(pixel_u, pixel_v), 0);
        } else if in.image_quality == IMAGE_QUALITY_MEDIUM {
            final_color = alpha * bilinear_sample(atlas_texture, vec2<f32>(final_u, final_v));
        } else {
            final_color = alpha * bicubic_sample(atlas_texture, vec2<f32>(final_u, final_v));
        }
    } else {
        if alpha_byte != 0 {
            // in.paint_data encodes a color    
            final_color = alpha * final_color;
        } else {
            // in.paint_data encodes a slot in the source clip texture
            let clip_x = u32(in.position.x) & 0xFFu;
            let clip_y = (u32(in.position.y) & 3) + in.paint_data * config.strip_height;
            let clip_in_color = textureLoad(clip_input_texture, vec2(clip_x, clip_y), 0);
            final_color = alpha * clip_in_color;
        }
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

const EXTEND_PAD: u32 = 0u;
const EXTEND_REPEAT: u32 = 1u;
const EXTEND_REFLECT: u32 = 2u;
fn extend_mode_normalized(t: f32, mode: u32) -> f32 {
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

// Bilinear filtering
fn bilinear_sample(tex: texture_2d<f32>, coords: vec2<f32>) -> vec4<f32> {
    let tex_dimensions = vec2<u32>(textureDimensions(tex));
    let tex_size = vec2<f32>(tex_dimensions);

    // Convert normalized coordinates to pixel space and subtract 0.5 for proper centering
    let pixel_coords = coords * tex_size - 0.5;
    let base_coords = floor(pixel_coords);
    let frac_coords = pixel_coords - base_coords;
    
    let base_i = vec2<i32>(base_coords);
    let p00 = textureLoad(tex, clamp(base_i + vec2<i32>(0, 0), vec2<i32>(0), vec2<i32>(tex_dimensions) - vec2<i32>(1)), 0);
    let p10 = textureLoad(tex, clamp(base_i + vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(tex_dimensions) - vec2<i32>(1)), 0);
    let p01 = textureLoad(tex, clamp(base_i + vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(tex_dimensions) - vec2<i32>(1)), 0);
    let p11 = textureLoad(tex, clamp(base_i + vec2<i32>(1, 1), vec2<i32>(0), vec2<i32>(tex_dimensions) - vec2<i32>(1)), 0);
    
    // Bilinear interpolation
    let top = mix(p00, p10, frac_coords.x);
    let bottom = mix(p01, p11, frac_coords.x);
    return mix(top, bottom, frac_coords.y);
}

// Bicubic filtering using Mitchell filter with B=1/3, C=1/3 (same as CPU implementation)
fn bicubic_sample(tex: texture_2d<f32>, coords: vec2<f32>) -> vec4<f32> {
    let tex_dimensions = vec2<u32>(textureDimensions(tex));
    let tex_size = vec2<f32>(tex_dimensions);
    let pixel_coords = coords * tex_size - 0.5;
    let base_coords = floor(pixel_coords);
    let frac_coords = pixel_coords - base_coords;
    
    let base_i = vec2<i32>(base_coords);
    
    // Get cubic weights for x and y directions using matrix-based approach (same as CPU)
    let weights_x = cubic_weights(frac_coords.x);
    let weights_y = cubic_weights(frac_coords.y);
    
    
    var result = vec4<f32>(0.0);
    
    // Sample 4x4 grid and apply weights
    for (var j = 0; j < 4; j++) {
        for (var i = 0; i < 4; i++) {
            let sample_pos = base_i + vec2<i32>(i - 1, j - 1);
            let clamped_pos = clamp(sample_pos, vec2<i32>(0), vec2<i32>(tex_dimensions) - vec2<i32>(1));
            let sample_color = textureLoad(tex, clamped_pos, 0);
            let weight = weights_x[i] * weights_y[j];
            result += sample_color * weight;
        }
    }
    
    // Clamp each component to [0,1] and ensure color components don't exceed alpha
    result.r = min(clamp(result.r, 0.0, 1.0), result.a);
    result.g = min(clamp(result.g, 0.0, 1.0), result.a);
    result.b = min(clamp(result.b, 0.0, 1.0), result.a);
    result.a = clamp(result.a, 0.0, 1.0);
    
    return result;
}

// Calculate the weights for a single fractional value (same as CPU weights function)
fn cubic_weights(fract: f32) -> vec4<f32> {
    let MF = mf_resampler();
    
    return vec4<f32>(
        single_weight(fract, MF[0][0], MF[0][1], MF[0][2], MF[0][3]),
        single_weight(fract, MF[1][0], MF[1][1], MF[1][2], MF[1][3]),
        single_weight(fract, MF[2][0], MF[2][1], MF[2][2], MF[2][3]),
        single_weight(fract, MF[3][0], MF[3][1], MF[3][2], MF[3][3])
    );
}

// Mitchell filter with the variables B = 1/3 and C = 1/3 (same as CPU mf_resampler)
fn mf_resampler() -> array<vec4<f32>, 4> {
    return cubic_resampler(1.0 / 3.0, 1.0 / 3.0);
}

// Cubic resampler logic borrowed from Skia (same as CPU cubic_resampler function)
// This allows us to define a resampler kernel based on two variables B and C
fn cubic_resampler(b: f32, c: f32) -> array<vec4<f32>, 4> {
    return array<vec4<f32>, 4>(
        vec4<f32>(
            (1.0 / 6.0) * b,
            -(3.0 / 6.0) * b - c,
            (3.0 / 6.0) * b + 2.0 * c,
            -(1.0 / 6.0) * b - c
        ),
        vec4<f32>(
            1.0 - (2.0 / 6.0) * b,
            0.0,
            -3.0 + (12.0 / 6.0) * b + c,
            2.0 - (9.0 / 6.0) * b - c
        ),
        vec4<f32>(
            (1.0 / 6.0) * b,
            (3.0 / 6.0) * b + c,
            3.0 - (15.0 / 6.0) * b - 2.0 * c,
            -2.0 + (9.0 / 6.0) * b + c
        ),
        vec4<f32>(
            0.0,
            0.0,
            -c,
            (1.0 / 6.0) * b + c
        )
    );
}

// Calculate a weight based on the fractional value t and the cubic coefficients
// This matches the CPU implementation exactly
fn single_weight(t: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return t * (t * (t * d + c) + b) + a;
}
