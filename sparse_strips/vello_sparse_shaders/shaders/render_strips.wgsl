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
//
// `StripInstance::paint` field encodes a color source, a paint type and a paint texture id
// Color source determines where the fragment shader gets color data from
// Paint type determines how the fragment shader uses the color data
// Paint texture id locates the encoded image data `EncodedImage` in `encoded_paints_texture`
// More details in the `StripInstance` documentation below.
//
// `StripInstance::payload` field can either encode a color, [x, y] for image sampling or a slot index
// - If color source is payload and the paint type is solid, the fragment shader uses the color directly.
// - If color source is payload and the paint type is image, the fragment shader samples the image.
// - Otherwise, the fragment shader samples the source clip texture using the given slot index.
// More details in the `StripInstance` documentation below.


// Color source modes - where the fragment shader gets color data from.
// Use payload (color or image coordinates)
const COLOR_SOURCE_PAYLOAD: u32 = 0u;
// Sample from clip texture slot
const COLOR_SOURCE_SLOT: u32 = 1u;
const COLOR_SOURCE_BLEND: u32 = 2u;

// Paint types
const PAINT_TYPE_SOLID: u32 = 0u;  
const PAINT_TYPE_IMAGE: u32 = 1u;
const PAINT_TYPE_LINEAR_GRADIENT: u32 = 2u;
const PAINT_TYPE_RADIAL_GRADIENT: u32 = 3u;
const PAINT_TYPE_SWEEP_GRADIENT: u32 = 4u;

// Paint texture index mask (extracts lower 27 bits from paint field).
const PAINT_TEXTURE_INDEX_MASK: u32 = 0x07FFFFFFu; 

// Image quality
const IMAGE_QUALITY_LOW = 0u;
const IMAGE_QUALITY_MEDIUM = 1u;
const IMAGE_QUALITY_HIGH = 2u;

// Gradient types.
const GRADIENT_TYPE_LINEAR: u32 = 0u;
const GRADIENT_TYPE_RADIAL: u32 = 1u;
const GRADIENT_TYPE_SWEEP: u32 = 2u;

// Radial gradient types.
const RADIAL_GRADIENT_TYPE_STANDARD: u32 = 0u;
const RADIAL_GRADIENT_TYPE_STRIP: u32 = 1u;
const RADIAL_GRADIENT_TYPE_FOCAL: u32 = 2u;

// Mathematical constants.
const PI: f32 = 3.1415926535897932384626433832795028;
const TWO_PI: f32 = 2.0 * PI;
// Tolerance for nearly zero comparisons (matching vello_cpu implementation).
// Note: This must match SCALAR_NEARLY_ZERO in vello_common/src/math.rs
// @see {@link https://github.com/linebender/vello/blob/748ba4c7a8973f642f778591b09658d8ee6e1132/sparse_strips/vello_common/src/math.rs#L21}
const NEARLY_ZERO_TOLERANCE: f32 = 1.0 / 4096.0;

// Composite modes.
const COMPOSE_CLEAR: u32 = 0u;
const COMPOSE_COPY: u32 = 1u;
const COMPOSE_DEST: u32 = 2u;
const COMPOSE_SRC_OVER: u32 = 3u;
const COMPOSE_DEST_OVER: u32 = 4u;
const COMPOSE_SRC_IN: u32 = 5u;
const COMPOSE_DEST_IN: u32 = 6u;
const COMPOSE_SRC_OUT: u32 = 7u;
const COMPOSE_DEST_OUT: u32 = 8u;
const COMPOSE_SRC_ATOP: u32 = 9u;
const COMPOSE_DEST_ATOP: u32 = 10u;
const COMPOSE_XOR: u32 = 11u;
const COMPOSE_PLUS: u32 = 12u;
const COMPOSE_PLUS_LIGHTER: u32 = 13u;

// Mix modes
const MIX_NORMAL = 0u;
const MIX_MULTIPLY = 1u;
const MIX_SCREEN = 2u;
const MIX_OVERLAY = 3u;
const MIX_DARKEN = 4u;
const MIX_LIGHTEN = 5u;
const MIX_COLOR_DODGE = 6u;
const MIX_COLOR_BURN = 7u;
const MIX_HARD_LIGHT = 8u;
const MIX_SOFT_LIGHT = 9u;
const MIX_DIFFERENCE = 10u;
const MIX_EXCLUSION = 11u;
const MIX_HUE = 12u;
const MIX_SATURATION = 13u;
const MIX_COLOR = 14u;
const MIX_LUMINOSITY = 15u;

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

// `paint` bit layout:
//   - Bits 30-31: `color_source`      0 = use payload, 1 = use slot texture, 2 = blend mode
//   - Bits 0-29:  Usage depends on color_source:
//
//     When color_source = 0 (COLOR_SOURCE_PAYLOAD):
//       - Bits 27-29: `paint_type` (0 = solid, 1 = image, 2 = linear_gradient, 3 = radial_gradient, 4 = sweep_gradient)
//       - Bits 0-26: 
//         - If paint_type = 0: unused
//         - If paint_type >= 1: `paint_texture_idx`
//
//     When color_source = 1 (COLOR_SOURCE_SLOT):
//       - Bits 0-7: opacity (0-255)
//       - Bits 8-29: unused
//
//     When color_source = 2 (COLOR_SOURCE_BLEND):
//       - Bits 16-29: `dest_slot` (14 bits)
//       - Bits 8-15: `mix_mode` (8 bits)
//       - Bits 0-7: `compose_mode` (8 bits)
//
// Decision tree for paint/payload interpretation:
//
// color_source = 0 (COLOR_SOURCE_PAYLOAD) - Use payload data directly
// ├── paint_type = 0 (PAINT_TYPE_SOLID) - Solid color rendering
// │   └── payload = [r, g, b, a] RGBA (packed as u8s)
// │
// ├── paint_type = 1 (PAINT_TYPE_IMAGE) - Image rendering
// │   └── payload = packed image parameters
// │
// ├── paint_type = 2 (PAINT_TYPE_LINEAR_GRADIENT) - Linear gradient rendering
// ├── paint_type = 3 (PAINT_TYPE_RADIAL_GRADIENT) - Radial gradient (with kind discriminator)
// └── paint_type = 4 (PAINT_TYPE_SWEEP_GRADIENT) - Sweep gradient rendering
//     ├── payload = [x, y] scene coordinates (packed as u16s)
//     └── bits 0-27 = paint_texture_idx
//
// color_source = 1 (COLOR_SOURCE_SLOT) - Use slot texture
// ├── payload = slot_index (u32)
// └── bits 0-7 = opacity (0-255, where 255 = fully opaque)
//
// color_source = 2 (COLOR_SOURCE_BLEND) - Blend two slots
// ├── payload = [src_slot, dest_slot] slot indices (packed as u16s)
// │   ├── bits 0-15 = src_slot (source slot to blend)
// │   └── bits 16-31 = dest_slot (destination slot to blend with)
// └── paint bits 0-23:
//     ├── bits 16-23 = opacity (0-255, applied to blend result)
//     ├── bits 8-15 = mix_mode (blend mixing mode)
//     └── bits 0-7 = compose_mode (compositing operation)
struct StripInstance {
    // [x, y] packed as u16's
    // x, y — coordinates of the strip
    @location(0) xy: u32,
    // [width, dense_width] packed as u16's
    // width — width of the strip
    // dense_width — width of the portion where alpha blending should be applied
    // Note that currently, if the strip instance represents an actual strip (i.e. an anti-aliased region),
    // width = dense_width. If the StripInstance represents a sparse fill region, then dense_width = 0.
    // TODO: In the future, this could be optimized such that `width` always represents the width and a simple
    // 1-bit flag is used to distinguish between sparse fill region and strip. This frees up 15 other bits.
    // Otherwise, it might also be possible to merge a strip and sparse fill command into a single strip instance.
    @location(1) widths: u32,
    // Alpha texture column index where this strip's alpha values begin
    // There are [`Config::strip_height`] alpha values per column.
    @location(2) col_idx: u32,
    // See StripInstance documentation above.
    @location(3) payload: u32,
    // See StripInstance documentation above.
    @location(4) paint: u32,
}

struct VertexOutput {
    // Render type for the strip
    @location(0) @interpolate(flat) paint: u32,
    // Texture coordinates for the current fragment
    @location(1) tex_coord: vec2<f32>,
    // UV coordinates for the current fragment, used for image sampling
    @location(2) sample_xy: vec2<f32>,
    // Ending x-position of the dense (alpha) region
    @location(3) @interpolate(flat) dense_end: u32,
    // Color value or slot index when alpha is 0
    @location(4) @interpolate(flat) payload: u32,
    // Normalized device coordinates (NDC) for the current vertex
    @builtin(position) position: vec4<f32>,
};

// TODO: Measure performance of moving to a separate group
@group(0) @binding(1)
var<uniform> config: Config;

@group(1) @binding(0)
var atlas_texture_array: texture_2d_array<f32>;

@group(2) @binding(0)
var encoded_paints_texture: texture_2d<u32>;

@group(3) @binding(0)
var gradient_texture: texture_2d<f32>;

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
    out.dense_end = instance.col_idx + dense_width;
    // Calculate the pixel coordinates of the current vertex within the strip
    let pix_x = f32(x0) + x * f32(width);
    let pix_y = f32(y0) + y * f32(config.strip_height);
    // Convert pixel coordinates to normalized device coordinates (NDC)
    // NDC ranges from -1 to 1, with (0,0) at the center of the viewport
    let ndc_x = pix_x * 2.0 / f32(config.width) - 1.0;
    let ndc_y = 1.0 - pix_y * 2.0 / f32(config.height);

    let color_source = (instance.paint >> 30u) & 0x3u;
    if color_source == COLOR_SOURCE_PAYLOAD {
        let paint_type = (instance.paint >> 27u) & 0x7u;
        // Unpack view coordinates for image sampling and gradient calculations
        let scene_strip_x = instance.payload & 0xffffu;
        let scene_strip_y = instance.payload >> 16u;

        if paint_type == PAINT_TYPE_IMAGE {
            let paint_tex_idx = instance.paint & PAINT_TEXTURE_INDEX_MASK;
            let encoded_image = unpack_encoded_image(paint_tex_idx);
            // Use view coordinates for image sampling (always in global view space)
            out.sample_xy = encoded_image.translate 
                + encoded_image.image_offset
                + encoded_image.transform.xy * f32(scene_strip_x) 
                + encoded_image.transform.zw * f32(scene_strip_y)
                + encoded_image.transform.xy * x * f32(width)
                + encoded_image.transform.zw * y * f32(config.strip_height);
        } else if paint_type == PAINT_TYPE_LINEAR_GRADIENT || paint_type == PAINT_TYPE_RADIAL_GRADIENT || paint_type == PAINT_TYPE_SWEEP_GRADIENT {
            // Use view coordinates for gradient transform (always in global view space)
            out.sample_xy = vec2<f32>(
                f32(scene_strip_x) + x * f32(width),
                f32(scene_strip_y) + y * f32(config.strip_height)
            );
        }
    }

    // Regular texture coordinates for other render types
    out.tex_coord = vec2<f32>(f32(instance.col_idx) + x * f32(width), y * f32(config.strip_height));
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.payload = instance.payload;
    out.paint = instance.paint;

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
    // This if condition essentially checks whether the current pixel lies within a strip or a sparse
    // fill region. In the former case, `dense_end` will be bigger than 0 since `dense_width` != 0. In the latter
    // case, `dense_end` will always be zero since for sparse regions `col_idx` and `dense_width` are both set to
    // zero.
    if in.dense_end != 0 {
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
    let color_source = (in.paint >> 30u) & 0x3u;
    var final_color: vec4<f32>;

    if color_source == COLOR_SOURCE_PAYLOAD {
        let paint_type = (in.paint >> 27u) & 0x7u;

        // in.payload encodes a color for PAINT_TYPE_SOLID or sample_xy for PAINT_TYPE_IMAGE
        if paint_type == PAINT_TYPE_SOLID {
            final_color = alpha * unpack4x8unorm(in.payload);
        } else if paint_type == PAINT_TYPE_IMAGE {
            let paint_tex_idx = in.paint & PAINT_TEXTURE_INDEX_MASK;
            let encoded_image = unpack_encoded_image(paint_tex_idx);
            let image_offset = encoded_image.image_offset;
            let image_size = encoded_image.image_size;
            let local_xy = in.sample_xy - image_offset;
            // This offset doesn't exist in vello_cpu, and we use it because 45 degree skewing seems to cause
            // artifacts on the GPU. We have something similar in place for gradients. It might be worth revisiting
            // this to see whether a better approach is possible.
            let offset = 0.00001;
            let extended_xy = vec2<f32>(
                extend_mode(local_xy.x + offset, encoded_image.extend_modes.x, image_size.x),
                extend_mode(local_xy.y + offset, encoded_image.extend_modes.y, image_size.y)
            );
            
            if encoded_image.quality == IMAGE_QUALITY_HIGH {
                let final_xy = image_offset + extended_xy;
                let sample_color = bicubic_sample(
                    atlas_texture_array,
                    final_xy,
                    i32(encoded_image.atlas_index),
                    image_offset,
                    image_size,
                    encoded_image.extend_modes,
                );
                final_color = alpha * sample_color;
            } else if encoded_image.quality == IMAGE_QUALITY_MEDIUM {
                let final_xy = image_offset + extended_xy - vec2(0.5);
                let sample_color = bilinear_sample(
                    atlas_texture_array,
                    final_xy,
                    i32(encoded_image.atlas_index),
                    image_offset,
                    image_size,
                    encoded_image.extend_modes,
                );
                final_color = alpha * sample_color;
            } else if encoded_image.quality == IMAGE_QUALITY_LOW {
                let final_xy = image_offset + extended_xy;
                let sample_color = textureLoad(
                    atlas_texture_array,
                    vec2<u32>(final_xy),
                    i32(encoded_image.atlas_index),
                    0,
                );
                final_color = alpha * sample_color;
            }
        } else if paint_type == PAINT_TYPE_LINEAR_GRADIENT {
            let paint_tex_idx = in.paint & PAINT_TEXTURE_INDEX_MASK;
            let linear_gradient = unpack_linear_gradient(paint_tex_idx);
            
            // Calculate fragment position and apply transform
            let fragment_pos = in.sample_xy;
            let grad_pos = vec2<f32>(
                linear_gradient.transform[0] * fragment_pos.x + 
                linear_gradient.transform[2] * fragment_pos.y +
                linear_gradient.transform[4],
                linear_gradient.transform[1] * fragment_pos.x +
                linear_gradient.transform[3] * fragment_pos.y + 
                linear_gradient.transform[5]
            );
            
            // For linear gradient, t-value is just the x coordinate in gradient space
            let t_value = grad_pos.x + 0.00001;
            let gradient_color = sample_gradient_lut(
                t_value,
                linear_gradient.extend_mode,
                linear_gradient.gradient_start,
                linear_gradient.texture_width,
                true
            );
            final_color = alpha * gradient_color;
        } else if paint_type == PAINT_TYPE_RADIAL_GRADIENT {
            let paint_tex_idx = in.paint & PAINT_TEXTURE_INDEX_MASK;
            let radial_gradient = unpack_radial_gradient(paint_tex_idx);
            
            // Calculate fragment position and apply transform
            let fragment_pos = in.sample_xy;
            let grad_pos = vec2<f32>(
                radial_gradient.transform[0] * fragment_pos.x + 
                radial_gradient.transform[2] * fragment_pos.y + 
                radial_gradient.transform[4],
                radial_gradient.transform[1] * fragment_pos.x + 
                radial_gradient.transform[3] * fragment_pos.y + 
                radial_gradient.transform[5]
            );
            
            // For radial gradient, calculate distance from center
            let gradient_result = calculate_radial_gradient(grad_pos, radial_gradient);
            let gradient_color = sample_gradient_lut(
                gradient_result.t_value, 
                radial_gradient.extend_mode, 
                radial_gradient.gradient_start, 
                radial_gradient.texture_width,
                gradient_result.is_valid
            );
            final_color = alpha * gradient_color;
        } else if paint_type == PAINT_TYPE_SWEEP_GRADIENT {
            let paint_tex_idx = in.paint & PAINT_TEXTURE_INDEX_MASK;
            let sweep_gradient = unpack_sweep_gradient(paint_tex_idx);
            
            // Calculate fragment position and apply transform
            let fragment_pos = in.sample_xy;
            var grad_pos = vec2<f32>(
                sweep_gradient.transform[0] * fragment_pos.x + 
                sweep_gradient.transform[2] * fragment_pos.y + 
                sweep_gradient.transform[4],
                sweep_gradient.transform[1] * fragment_pos.x + 
                sweep_gradient.transform[3] * fragment_pos.y + 
                sweep_gradient.transform[5]
            );

            // Before passing the position to the angle calculation, we bias
            // very small coordinates to 0. Otherwise the sweep gradient's seam
            // may flicker, because the angle calculation uses the coordinates'
            // signs to select a quadrant. For coordinates around 0, slight
            // noise in the coordinate calculation can then land the
            // calculation in different quadrants. That flickering is quite
            // noticeable as the seam is not anti-aliased. The flickering may
            // vary across machines. See
            // <https://github.com/linebender/vello/pull/1352>.
            grad_pos = select(grad_pos, vec2(0.0), abs(grad_pos) < vec2(NEARLY_ZERO_TOLERANCE));
            
            // For sweep gradient, calculate angle from center using fast polynomial approximation
            let unit_angle = xy_to_unit_angle(grad_pos.x, grad_pos.y);
            // Convert unit angle [0, 1) to radians [0, 2π)
            let angle = unit_angle * TWO_PI;
            let t_value = (angle - sweep_gradient.start_angle) * sweep_gradient.inv_angle_delta;
            let gradient_color = sample_gradient_lut(
                t_value,
                sweep_gradient.extend_mode,
                sweep_gradient.gradient_start,
                sweep_gradient.texture_width,
                true
            );
            final_color = alpha * gradient_color;
        }
    } else if color_source == COLOR_SOURCE_SLOT {
        // in.payload encodes a slot in the source clip texture
        let clip_x = u32(in.position.x) & 0xFFu;
        let clip_y = (u32(in.position.y) & 3) + in.payload * config.strip_height;
        let clip_in_color = textureLoad(clip_input_texture, vec2(clip_x, clip_y), 0);

        // Extract opacity from first 8 bits (quantized from [0, 255])
        let opacity = f32(in.paint & 0xFFu) * (1.0 / 255.0);

        final_color = alpha * opacity * clip_in_color;
    } else if color_source == COLOR_SOURCE_BLEND {
        let opacity = f32((in.paint >> 16u) & 0xFFu) * (1.0 / 255.0);
        let mix_mode = (in.paint >> 8u) & 0xFFu;
        let compose_mode = in.paint & 0xFFu;
        
        // Read source color from slot
        let src_slot = in.payload & 0xFFFFu;
        let dest_slot = (in.payload >> 16u) & 0xFFFFu;
        let clip_x = u32(in.position.x) & 0xFFu;
        let src_y = (u32(in.position.y) & 3u) + src_slot * config.strip_height;
        let src_color = textureLoad(clip_input_texture, vec2(clip_x, src_y), 0);
        
        // Read destination color from slot
        let dest_y = (u32(in.position.y) & 3u) + dest_slot * config.strip_height;
        let dest_color = textureLoad(clip_input_texture, vec2(clip_x, dest_y), 0);

        final_color = blend_mix_compose(dest_color, src_color * opacity * alpha, compose_mode, mix_mode);
    }
    return final_color;
}

// Apply color mixing and composition. Both input and output colors are premultiplied RGB.
// Referenced from:
//   <https://github.com/linebender/vello/blob/b0e2e598ac62c7b3d04d8660e7b1b7659b596970/vello_shaders/shader/shared/blend.wgsl#L288-L310>
fn blend_mix_compose(backdrop: vec4<f32>, src: vec4<f32>, compose_mode: u32, mix_mode: u32) -> vec4<f32> {
    // Fast path for src_over
    let BLEND_DEFAULT = ((MIX_NORMAL << 8u) | COMPOSE_SRC_OVER);
    let mode = ((mix_mode << 8u) | compose_mode);
    if mode == BLEND_DEFAULT {
        return backdrop * (1.0 - src.a) + src;
    }
    
    let EPSILON = 1e-15;
    let inv_src_a = 1.0 / max(src.a, EPSILON);
    var cs = src.rgb * inv_src_a;
    let inv_backdrop_a = 1.0 / max(backdrop.a, EPSILON);
    let cb = backdrop.rgb * inv_backdrop_a;
    let mixed = blend_mix(cb, cs, mix_mode);
    cs = mix(cs, mixed, backdrop.a);

    if compose_mode == COMPOSE_SRC_OVER {
        let co = mix(backdrop.rgb, cs, src.a);
        return vec4(co, src.a + backdrop.a * (1.0 - src.a));
    } else {
        return blend_compose_unpremul(cb, cs, backdrop.a, src.a, compose_mode);
    }
}

// Apply general compositing operation. Inputs are separated colors and alpha, output is
// premultiplied. Referenced from:
//   <https://github.com/linebender/vello/blob/b0e2e598ac62c7b3d04d8660e7b1b7659b596970/vello_shaders/shader/shared/blend.wgsl#L215>
fn blend_compose_unpremul(
    cb: vec3<f32>,
    cs: vec3<f32>,
    ab: f32,
    as_: f32,
    mode: u32
) -> vec4<f32> {
    var fa = 0.0;
    var fb = 0.0;
    switch mode {
        case COMPOSE_COPY: {
            fa = 1.0;
            fb = 0.0;
        }
        case COMPOSE_DEST: {
            fa = 0.0;
            fb = 1.0;
        }
        case COMPOSE_SRC_OVER: {
            fa = 1.0;
            fb = 1.0 - as_;
        }
        case COMPOSE_DEST_OVER: {
            fa = 1.0 - ab;
            fb = 1.0;
        }
        case COMPOSE_SRC_IN: {
            fa = ab;
            fb = 0.0;
        }
        case COMPOSE_DEST_IN: {
            fa = 0.0;
            fb = as_;
        }
        case COMPOSE_SRC_OUT: {
            fa = 1.0 - ab;
            fb = 0.0;
        }
        case COMPOSE_DEST_OUT: {
            fa = 0.0;
            fb = 1.0 - as_;
        }
        case COMPOSE_SRC_ATOP: {
            fa = ab;
            fb = 1.0 - as_;
        }
        case COMPOSE_DEST_ATOP: {
            fa = 1.0 - ab;
            fb = as_;
        }
        case COMPOSE_XOR: {
            fa = 1.0 - ab;
            fb = 1.0 - as_;
        }
        case COMPOSE_PLUS: {
            fa = 1.0;
            fb = 1.0;
        }
        case COMPOSE_PLUS_LIGHTER: {
            return min(vec4(1.0), vec4(as_ * cs + ab * cb, as_ + ab));
        }
        default: {}
    }
    let as_fa = as_ * fa;
    let ab_fb = ab * fb;
    let co = as_fa * cs + ab_fb * cb;
    // Modes like COMPOSE_PLUS can generate alpha > 1.0, so clamp.
    return vec4(co, min(as_fa + ab_fb, 1.0));
}

fn screen(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    return cb + cs - (cb * cs);
}

fn color_dodge(cb: f32, cs: f32) -> f32 {
    if cb == 0.0 {
        return 0.0;
    } else if cs == 1.0 {
        return 1.0;
    } else {
        return min(1.0, cb / (1.0 - cs));
    }
}

fn color_burn(cb: f32, cs: f32) -> f32 {
    if cb == 1.0 {
        return 1.0;
    } else if cs == 0.0 {
        return 0.0;
    } else {
        return 1.0 - min(1.0, (1.0 - cb) / cs);
    }
}

fn hard_light(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    return select(
        screen(cb, 2.0 * cs - 1.0),
        cb * 2.0 * cs,
        cs <= vec3(0.5)
    );
}

fn soft_light(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    let d = select(
        sqrt(cb),
        ((16.0 * cb - 12.0) * cb + 4.0) * cb,
        cb <= vec3(0.25)
    );
    return select(
        cb + (2.0 * cs - 1.0) * (d - cb),
        cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb),
        cs <= vec3(0.5)
    );
}

fn sat(c: vec3<f32>) -> f32 {
    return max(c.x, max(c.y, c.z)) - min(c.x, min(c.y, c.z));
}

fn lum(c: vec3<f32>) -> f32 {
    let f = vec3(0.3, 0.59, 0.11);
    return dot(c, f);
}

fn clip_color(c_in: vec3<f32>) -> vec3<f32> {
    var c = c_in;
    let l = lum(c);
    let n = min(c.x, min(c.y, c.z));
    let x = max(c.x, max(c.y, c.z));
    if n < 0.0 {
        c = l + (((c - l) * l) / (l - n));
    }
    if x > 1.0 {
        c = l + (((c - l) * (1.0 - l)) / (x - l));
    }
    return c;
}

fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    return clip_color(c + (l - lum(c)));
}

fn set_sat_inner(
    cmin: ptr<function, f32>,
    cmid: ptr<function, f32>,
    cmax: ptr<function, f32>,
    s: f32
) {
    if *cmax > *cmin {
        *cmid = ((*cmid - *cmin) * s) / (*cmax - *cmin);
        *cmax = s;
    } else {
        *cmid = 0.0;
        *cmax = 0.0;
    }
    *cmin = 0.0;
}

fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    var r = c.r;
    var g = c.g;
    var b = c.b;
    if r <= g {
        if g <= b {
            set_sat_inner(&r, &g, &b, s);
        } else {
            if r <= b {
                set_sat_inner(&r, &b, &g, s);
            } else {
                set_sat_inner(&b, &r, &g, s);
            }
        }
    } else {
        if r <= b {
            set_sat_inner(&g, &r, &b, s);
        } else {
            if g <= b {
                set_sat_inner(&g, &b, &r, s);
            } else {
                set_sat_inner(&b, &g, &r, s);
            }
        }
    }
    return vec3(r, g, b);
}

// Blends two RGB colors together. The colors are assumed to be in sRGB
// color space, and this function does not take alpha into account.
fn blend_mix(cb: vec3<f32>, cs: vec3<f32>, mode: u32) -> vec3<f32> {
    var b = vec3(0.0);
    switch mode {
        case MIX_MULTIPLY: {
            b = cb * cs;
        }
        case MIX_SCREEN: {
            b = screen(cb, cs);
        }
        case MIX_OVERLAY: {
            b = hard_light(cs, cb);
        }
        case MIX_DARKEN: {
            b = min(cb, cs);
        }
        case MIX_LIGHTEN: {
            b = max(cb, cs);
        }
        case MIX_COLOR_DODGE: {
            b = vec3(color_dodge(cb.x, cs.x), color_dodge(cb.y, cs.y), color_dodge(cb.z, cs.z));
        }
        case MIX_COLOR_BURN: {
            b = vec3(color_burn(cb.x, cs.x), color_burn(cb.y, cs.y), color_burn(cb.z, cs.z));
        }
        case MIX_HARD_LIGHT: {
            b = hard_light(cb, cs);
        }
        case MIX_SOFT_LIGHT: {
            b = soft_light(cb, cs);
        }
        case MIX_DIFFERENCE: {
            b = abs(cb - cs);
        }
        case MIX_EXCLUSION: {
            b = cb + cs - 2.0 * cb * cs;
        }
        case MIX_HUE: {
            b = set_lum(set_sat(cs, sat(cb)), lum(cb));
        }
        case MIX_SATURATION: {
            b = set_lum(set_sat(cb, sat(cs)), lum(cb));
        }
        case MIX_COLOR: {
            b = set_lum(cs, lum(cb));
        }
        case MIX_LUMINOSITY: {
            b = set_lum(cb, lum(cs));
        }
        default: {
            b = cs;
        }
    }
    return b;
}


struct EncodedImage {
    /// The rendering quality of the image.
    quality: u32,
    /// The extends in the horizontal and vertical direction.
    extend_modes: vec2<u32>,
    /// The size of the image in pixels.
    image_size: vec2<f32>,
    /// The offset of the image in pixels.
    image_offset: vec2<f32>,
    /// The atlas index containing this image.
    atlas_index: u32,
    /// Linear transformation matrix coefficients for 2D affine transformation.
    /// Contains [a, b, c, d] where the transformation matrix is:
    /// This enables scaling, rotation, and skewing of the image coordinates.
    transform: vec4<f32>,
    /// Translation offset for 2D affine transformation.
    /// Contains [tx, ty] representing the translation component.
    translate: vec2<f32>,
}

// Unpack encoded image from the encoded paints texture.
fn unpack_encoded_image(paint_tex_idx: u32) -> EncodedImage {
    let texel0 = textureLoad(encoded_paints_texture, vec2<u32>(paint_tex_idx, 0), 0);
    let texel1 = textureLoad(encoded_paints_texture, vec2<u32>(paint_tex_idx + 1u, 0), 0);
    let texel2 = textureLoad(encoded_paints_texture, vec2<u32>(paint_tex_idx + 2u, 0), 0);
    
    let quality = texel0.x & 0x3u;
    let extend_x = (texel0.x >> 2u) & 0x3u;
    let extend_y = (texel0.x >> 4u) & 0x3u;
    let atlas_index = (texel0.x >> 6u) & 0xFFu;
    // Unpack image_size from texel0.y (stored as u32, unpack to width/height)
    let image_size = vec2<f32>(f32(texel0.y >> 16u), f32(texel0.y & 0xFFFFu));
    // Unpack image_offset from texel0.z (stored as u32, unpack to x/y)
    let image_offset = vec2<f32>(f32(texel0.z >> 16u), f32(texel0.z & 0xFFFFu));
    let transform = vec4<f32>(
        bitcast<f32>(texel0.w), bitcast<f32>(texel1.x), 
        bitcast<f32>(texel1.y), bitcast<f32>(texel1.z)
    );
    let translate = vec2<f32>(bitcast<f32>(texel1.w), bitcast<f32>(texel2.x));

    return EncodedImage(
        quality, 
        vec2<u32>(extend_x, extend_y),
        image_size,
        image_offset,
        atlas_index,
        transform,
        translate
    );
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

const EXTEND_PAD: u32 = 0u;
const EXTEND_REPEAT: u32 = 1u;
const EXTEND_REFLECT: u32 = 2u;
fn extend_mode(t: f32, mode: u32, max: f32) -> f32 {
    switch mode {
        case EXTEND_PAD: {
            return clamp(t, 0.0, max - 1.0);
        }
        case EXTEND_REPEAT: {
            return extend_mode_normalized(t / max, mode) * max;
        }
        case EXTEND_REFLECT, default: {
            return extend_mode_normalized(t / max, mode) * max;
        }
    }
}

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
//
// Bilinear filtering consists of sampling the 4 surrounding pixels of the target point and
// interpolating them with a bilinear filter.
fn bilinear_sample(
    tex: texture_2d_array<f32>,
    coords: vec2<f32>,
    atlas_idx: i32,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
    extend_modes: vec2<u32>,
) -> vec4<f32> {
    let atlas_max = image_offset + image_size - vec2(1.0);
    let atlas_uv_clamped = clamp(coords, image_offset, atlas_max);
    let uv_quad = vec4(floor(atlas_uv_clamped), ceil(atlas_uv_clamped));
    let uv_frac = fract(coords);
    let a = textureLoad(tex, vec2<i32>(uv_quad.xy), atlas_idx, 0);
    let b = textureLoad(tex, vec2<i32>(uv_quad.xw), atlas_idx, 0);
    let c = textureLoad(tex, vec2<i32>(uv_quad.zy), atlas_idx, 0);
    let d = textureLoad(tex, vec2<i32>(uv_quad.zw), atlas_idx, 0);
    return mix(mix(a, b, uv_frac.y), mix(c, d, uv_frac.y), uv_frac.x);
}

// Bicubic filtering using Mitchell filter with B=1/3, C=1/3
//
// Cubic resampling consists of sampling the 16 surrounding pixels of the target point and
// interpolating them with a cubic filter. The generated matrix is 4x4 and represent the coefficients
// of the cubic function used to  calculate weights based on the `x_fract` and `y_fract` of the
// location we are looking at.
fn bicubic_sample(
    tex: texture_2d_array<f32>,
    coords: vec2<f32>,
    atlas_idx: i32,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
    extend_modes: vec2<u32>,
) -> vec4<f32> {
     let atlas_max = image_offset + image_size - vec2(1.0);
     let frac_coords = fract(coords + 0.5);
     // Get cubic weights for x and y directions
     let cx = cubic_weights(frac_coords.x);
     let cy = cubic_weights(frac_coords.y);
     
     // Sample 4x4 grid around coords
     let s00 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(-1.5, -1.5), image_offset, atlas_max)), atlas_idx, 0);
     let s10 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(-0.5, -1.5), image_offset, atlas_max)), atlas_idx, 0);
     let s20 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(0.5, -1.5), image_offset, atlas_max)), atlas_idx, 0);
     let s30 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(1.5, -1.5), image_offset, atlas_max)), atlas_idx, 0);
     
     let s01 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(-1.5, -0.5), image_offset, atlas_max)), atlas_idx, 0);
     let s11 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(-0.5, -0.5), image_offset, atlas_max)), atlas_idx, 0);
     let s21 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(0.5, -0.5), image_offset, atlas_max)), atlas_idx, 0);
     let s31 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(1.5, -0.5), image_offset, atlas_max)), atlas_idx, 0);
     
     let s02 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(-1.5, 0.5), image_offset, atlas_max)), atlas_idx, 0);
     let s12 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(-0.5, 0.5), image_offset, atlas_max)), atlas_idx, 0);
     let s22 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(0.5, 0.5), image_offset, atlas_max)), atlas_idx, 0);
     let s32 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(1.5, 0.5), image_offset, atlas_max)), atlas_idx, 0);
     
     let s03 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(-1.5, 1.5), image_offset, atlas_max)), atlas_idx, 0);
     let s13 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(-0.5, 1.5), image_offset, atlas_max)), atlas_idx, 0);
     let s23 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(0.5, 1.5), image_offset, atlas_max)), atlas_idx, 0);
     let s33 = textureLoad(tex, vec2<i32>(clamp(coords + vec2(1.5, 1.5), image_offset, atlas_max)), atlas_idx, 0);
    
    // Interpolate in x direction for each row
    let row0 = cx.x * s00 + cx.y * s10 + cx.z * s20 + cx.w * s30;
    let row1 = cx.x * s01 + cx.y * s11 + cx.z * s21 + cx.w * s31;
    let row2 = cx.x * s02 + cx.y * s12 + cx.z * s22 + cx.w * s32;
    let row3 = cx.x * s03 + cx.y * s13 + cx.z * s23 + cx.w * s33;
    // Interpolate in y direction
    let result = cy.x * row0 + cy.y * row1 + cy.z * row2 + cy.w * row3;
    
    // Clamp each component to [0,1] and ensure color components don't exceed alpha
    return vec4<f32>(
        min(clamp(result.r, 0.0, 1.0), result.a),
        min(clamp(result.g, 0.0, 1.0), result.a),
        min(clamp(result.b, 0.0, 1.0), result.a),
        min(clamp(result.a, 0.0, 1.0), result.a)
    );
}

// Cubic resampler logic borrowed from Skia (same as CPU cubic_resampler function)
// Mitchell-Netravali cubic filter coefficients with parameters B=1/3 and C=1/3
const MF: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(
        (1.0 / 6.0) / 3.0,
        -(3.0 / 6.0) / 3.0 - 1.0 / 3.0,
        (3.0 / 6.0) / 3.0 + 2.0 * 1.0 / 3.0,
        -(1.0 / 6.0) / 3.0 - 1.0 / 3.0
    ),
    vec4<f32>(
        1.0 - (2.0 / 6.0) / 3.0,
        0.0,
        -3.0 + (12.0 / 6.0) / 3.0 + 1.0 / 3.0,
        2.0 - (9.0 / 6.0) / 3.0 - 1.0 / 3.0
    ),
    vec4<f32>(
        (1.0 / 6.0) / 3.0,
        (3.0 / 6.0) / 3.0 + 1.0 / 3.0,
        3.0 - (15.0 / 6.0) / 3.0 - 2.0 * 1.0 / 3.0,
        -2.0 + (9.0 / 6.0) / 3.0 + 1.0 / 3.0
    ),
    vec4<f32>(
        0.0,
        0.0,
        -1.0 / 3.0,
        (1.0 / 6.0) / 3.0 + 1.0 / 3.0
    )
);

// Calculate the weights for a single fractional value (same as CPU weights function)
fn cubic_weights(fract: f32) -> vec4<f32> {
    return vec4<f32>(
        single_weight(fract, MF[0][0], MF[0][1], MF[0][2], MF[0][3]),
        single_weight(fract, MF[1][0], MF[1][1], MF[1][2], MF[1][3]),
        single_weight(fract, MF[2][0], MF[2][1], MF[2][2], MF[2][3]),
        single_weight(fract, MF[3][0], MF[3][1], MF[3][2], MF[3][3])
    );
}

// Calculate a weight based on the fractional value t and the cubic coefficients
// This matches the CPU implementation exactly
fn single_weight(t: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return t * (t * (t * d + c) + b) + a;
}

// Fast polynomial approximation for xy_to_unit_angle from Skia
// Returns angle in [0, 1) range representing [0, 2π)
// See: https://github.com/google/skia/blob/30bba741989865c157c7a997a0caebe94921276b/src/opts/SkRasterPipeline_opts.h#L5859
fn xy_to_unit_angle(x: f32, y: f32) -> f32 {
    let xabs = abs(x);
    let yabs = abs(y);
    let slope = min(xabs, yabs) / max(xabs, yabs);
    let s = slope * slope;
    // Use a 7th degree polynomial to approximate atan.
    // This was generated using sollya.gforge.inria.fr.
    // A float optimized polynomial was generated using the following command.
    // P1 = fpminimax((1/(2*Pi))*atan(x),[|1,3,5,7|],[|24...|],[2^(-40),1],relative);
    var phi = slope * (0.15912117063999176025390625 + s * (-5.185396969318389892578125e-2 + s * (2.476101927459239959716796875e-2 + s * (-7.0547382347285747528076171875e-3))));
    // Map from first octant to full circle using quadrant information
    // Handle [0°, 90°] range
    phi = select(phi, 0.25 - phi, xabs < yabs);
    // Handle [90°, 180°] range
    phi = select(phi, 0.5 - phi, x < 0.0);
    // Handle [180°, 360°] range
    phi = select(phi, 1.0 - phi, y < 0.0);
    // Handle NaN cases (using property that NaN != NaN)
    phi = select(phi, 0.0, phi != phi);
    return phi;
}

// Sample from the gradient texture at calculated position.
fn sample_gradient_lut(t_value: f32, extend_mode: u32, gradient_start: u32, texture_width: u32, is_valid: bool) -> vec4<f32> {
    // Apply extend mode to t_value
    let clamped_t = extend_mode_normalized(t_value, extend_mode);
    // Convert t_value to texture coordinate
    let t_offset = select(texture_width, u32(clamped_t * f32(texture_width - 1u)), is_valid);
    // Calculate absolute position in flat gradient texture
    let flat_coord = gradient_start + t_offset;
    // Convert flat coordinate to 2D texture coordinate
    let gradient_tex_width = textureDimensions(gradient_texture).x;
    let tex_x = flat_coord % gradient_tex_width;
    let tex_y = flat_coord / gradient_tex_width;
    // Sample from the gradient texture at calculated position
    let gradient_color = textureLoad(gradient_texture, vec2<u32>(tex_x, tex_y), 0);
    return gradient_color;
}

struct LinearGradient {
    /// The extend mode for the gradient (0=Pad, 1=Repeat).
    extend_mode: u32,
    /// Start coordinate in the flat gradient texture.
    gradient_start: u32,
    /// Width of the gradient texture.
    texture_width: u32,
    /// Transform matrix [a, b, c, d, tx, ty].
    transform: array<f32, 6>,
}

struct RadialGradient {
    /// The extend mode for the gradient (0=Pad, 1=Repeat).
    extend_mode: u32,
    /// Start coordinate in the flat gradient texture.
    gradient_start: u32,
    /// Width of the gradient texture.
    texture_width: u32,
    /// Transform matrix [a, b, c, d, tx, ty].
    transform: array<f32, 6>,
    /// Bias value for radial gradient calculation.
    bias: f32,
    /// Scale factor for radial gradient calculation.
    scale: f32,
    /// Focal point 0 parameter for radial gradient.
    fp0: f32,
    /// Focal point 1 parameter for radial gradient.
    fp1: f32,
    /// Focal radius 1 parameter for radial gradient.
    fr1: f32,
    /// Focal X coordinate for radial gradient.
    f_focal_x: f32,
    /// Whether focal point is swapped for radial gradient (0=false, 1=true).
    f_is_swapped: u32,
    /// Scaled radius 0 squared parameter for radial gradient strip.
    scaled_r0_squared: f32,
    /// Kind of radial gradient (0=Radial, 1=Strip, 2=Focal).
    kind: u32,
}

struct SweepGradient {
    /// The extend mode for the gradient (0=Pad, 1=Repeat).
    extend_mode: u32,
    /// Start coordinate in the flat gradient texture.
    gradient_start: u32,
    /// Width of the gradient texture.
    texture_width: u32,
    /// Transform matrix [a, b, c, d, tx, ty].
    transform: array<f32, 6>,
    /// Starting angle for sweep gradient (in radians).
    start_angle: f32,
    /// Inverse of angle delta for sweep gradient.
    inv_angle_delta: f32,
}

// Unpack linear gradient from the encoded paints texture.
fn unpack_linear_gradient(paint_tex_idx: u32) -> LinearGradient {
    let texel0 = textureLoad(encoded_paints_texture, vec2<u32>(paint_tex_idx, 0), 0);
    let texel1 = textureLoad(encoded_paints_texture, vec2<u32>(paint_tex_idx + 1u, 0), 0);
    
    let texture_width_and_extend_mode = unpack_texture_width_and_extend_mode(texel0.x);
    let texture_width = texture_width_and_extend_mode.x;
    let extend_mode = texture_width_and_extend_mode.y;
    let gradient_start = texel0.y;
    
    let transform = array<f32, 6>(
        bitcast<f32>(texel0.z), bitcast<f32>(texel0.w), bitcast<f32>(texel1.x),
        bitcast<f32>(texel1.y), bitcast<f32>(texel1.z), bitcast<f32>(texel1.w)
    );
    
    return LinearGradient(
        extend_mode, gradient_start, texture_width, transform
    );
}

// Result of calculating a radial gradient.
struct RadialGradientResult {
    t_value: f32,
    is_valid: bool,
}

// Calculate a radial gradient; matches vello_cpu implementation.
fn calculate_radial_gradient(grad_pos: vec2<f32>, radial_gradient: RadialGradient) -> RadialGradientResult {
    let x_pos = grad_pos.x;
    let y_pos = grad_pos.y;
    
    var t_value: f32;
    var is_valid: bool;
    
    switch radial_gradient.kind {
        case RADIAL_GRADIENT_TYPE_STANDARD: {
            // Standard radial gradient: bias + scale * sqrt(x^2 + y^2)
            let radius = sqrt(x_pos * x_pos + y_pos * y_pos);
            t_value = radial_gradient.bias + radial_gradient.scale * radius;
            // Radial gradients are always valid
            is_valid = true;
        }
        case RADIAL_GRADIENT_TYPE_STRIP: {
            // Strip gradient: x + sqrt(scaled_r0_squared - y^2)
            let p1 = radial_gradient.scaled_r0_squared - y_pos * y_pos;
            // Invalid if negative under square root
            is_valid = p1 >= 0.0;
            if is_valid {
                t_value = x_pos + sqrt(p1);
            } else {
                // Value doesn't matter when invalid
                t_value = 0.0;
            }
        }
        case RADIAL_GRADIENT_TYPE_FOCAL, default: {
            // Focal gradient implementation
            var t = 0.0;
            let fp0 = radial_gradient.fp0;
            let fp1 = radial_gradient.fp1;
            let fr1 = radial_gradient.fr1;
            let f_focal_x = radial_gradient.f_focal_x;
            let is_swapped = radial_gradient.f_is_swapped;
            
            // Calculate focal flags directly from field values (matching FocalData implementation)
            let is_focal_on_circle = abs(1.0 - fr1) <= NEARLY_ZERO_TOLERANCE;
            let is_well_behaved = !is_focal_on_circle && fr1 > 1.0;
            let is_natively_focal = abs(f_focal_x) <= NEARLY_ZERO_TOLERANCE;
            
            // Start with valid assumption
            is_valid = true;
            
            if is_focal_on_circle {
                t = x_pos + y_pos * y_pos / x_pos;
                // Check for division by zero and negative t
                is_valid = t >= 0.0 && x_pos != 0.0;
            } else if is_well_behaved {
                t = sqrt(x_pos * x_pos + y_pos * y_pos) - x_pos * fp0;
            } else {
                // For non-well-behaved gradients, check if calculation is valid
                let xx = x_pos * x_pos;
                let yy = y_pos * y_pos;
                let discriminant = xx - yy;
                
                if is_swapped != 0u || (1.0 - f_focal_x < 0.0) {
                    t = -sqrt(discriminant) - x_pos * fp0;
                } else {
                    t = sqrt(discriminant) - x_pos * fp0;
                }
                
                // Invalid if discriminant is negative or t is negative
                is_valid = discriminant >= 0.0 && t >= 0.0;
            }
            
            // Apply additional focal transforms only if still valid
            if is_valid {
                if 1.0 - f_focal_x < 0.0 {
                    t = -t;
                }
                
                if !is_natively_focal {
                    t = t + fp1;
                }
                
                if is_swapped != 0u {
                    t = 1.0 - t;
                }
            }
            
            t_value = t;
        }
    }
    
    return RadialGradientResult(t_value, is_valid);
}

// Unpack radial gradient from the encoded paints texture.
fn unpack_radial_gradient(paint_tex_idx: u32) -> RadialGradient {
    let texel0 = textureLoad(encoded_paints_texture, vec2<u32>(paint_tex_idx, 0), 0);
    let texel1 = textureLoad(encoded_paints_texture, vec2<u32>(paint_tex_idx + 1u, 0), 0);
    let texel2 = textureLoad(encoded_paints_texture, vec2<u32>(paint_tex_idx + 2u, 0), 0);
    let texel3 = textureLoad(encoded_paints_texture, vec2<u32>(paint_tex_idx + 3u, 0), 0);
    
    let texture_width_and_extend_mode = unpack_texture_width_and_extend_mode(texel0.x);
    let texture_width = texture_width_and_extend_mode.x;
    let extend_mode = texture_width_and_extend_mode.y;
    let gradient_start = texel0.y;
    let transform = array<f32, 6>(
        bitcast<f32>(texel0.z), bitcast<f32>(texel0.w), bitcast<f32>(texel1.x),
        bitcast<f32>(texel1.y), bitcast<f32>(texel1.z), bitcast<f32>(texel1.w)
    );
    
    let kind_and_swapped = unpack_radial_kind_and_swapped(texel2.x);
    let kind = kind_and_swapped.x;
    let f_is_swapped = kind_and_swapped.y;
    
    let bias = bitcast<f32>(texel2.y);
    let scale = bitcast<f32>(texel2.z);
    let fp0 = bitcast<f32>(texel2.w);
    let fp1 = bitcast<f32>(texel3.x);
    let fr1 = bitcast<f32>(texel3.y);
    let f_focal_x = bitcast<f32>(texel3.z);
    let scaled_r0_squared = bitcast<f32>(texel3.w);
    
    return RadialGradient(
        extend_mode, gradient_start, texture_width, transform,
        bias, scale, fp0, fp1, fr1, f_focal_x, f_is_swapped, scaled_r0_squared, kind
    );
}

// Unpack sweep gradient from the encoded paints texture.
fn unpack_sweep_gradient(paint_tex_idx: u32) -> SweepGradient {
    let texel0 = textureLoad(encoded_paints_texture, vec2<u32>(paint_tex_idx, 0), 0);
    let texel1 = textureLoad(encoded_paints_texture, vec2<u32>(paint_tex_idx + 1u, 0), 0);
    let texel2 = textureLoad(encoded_paints_texture, vec2<u32>(paint_tex_idx + 2u, 0), 0);
    
    let texture_width_and_extend_mode = unpack_texture_width_and_extend_mode(texel0.x);
    let texture_width = texture_width_and_extend_mode.x;
    let extend_mode = texture_width_and_extend_mode.y;
    let gradient_start = texel0.y;
    let transform = array<f32, 6>(
        bitcast<f32>(texel0.z), bitcast<f32>(texel0.w), bitcast<f32>(texel1.x),
        bitcast<f32>(texel1.y), bitcast<f32>(texel1.z), bitcast<f32>(texel1.w)
    );
    
    let start_angle = bitcast<f32>(texel2.x);
    let inv_angle_delta = bitcast<f32>(texel2.y);

    return SweepGradient(
        extend_mode, gradient_start, texture_width, transform, start_angle, inv_angle_delta
    );
}

// Unpack texture_width and extend_mode from packed field.
// Returns (texture_width, extend_mode).
fn unpack_texture_width_and_extend_mode(packed: u32) -> vec2<u32> {
    let texture_width = packed & 0x0FFFFFFFu;  // Mask out bits 30 & 31
    let extend_mode = (packed >> 30u) & 3u;    // Extract bits 30 & 31
    return vec2<u32>(texture_width, extend_mode);
}

// Unpack radial gradient kind and f_is_swapped from packed field.
// Returns (kind, f_is_swapped).
fn unpack_radial_kind_and_swapped(packed: u32) -> vec2<u32> {
    let kind = packed & 0x3u;           // Extract bits 0-1
    let f_is_swapped = (packed >> 2u) & 1u;  // Extract bit 2
    return vec2<u32>(kind, f_is_swapped);
}
