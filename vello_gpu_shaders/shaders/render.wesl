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
// Paint texture id locates the encoded paint data in `encoded_paints_texture`
// More details in the `StripInstance` documentation below.
//
// `StripInstance::payload` field can either encode a color, [x, y] for image sampling or a layer texture origin
// - If color source is payload and the paint type is solid, the fragment shader uses the color directly.
// - If color source is payload and the paint type is image, the fragment shader samples the image.
// - If color source is layer, the fragment shader samples the layer texture and applies layer opacity.
// More details in the `StripInstance` documentation below.

import package::helpers::alpha::sample_strip_alpha;
@if(blurred_rounded_rect)
import package::helpers::blurred_rounded_rect::{
    get_blurred_rounded_rect_sample_xy,
    sample_blurred_rounded_rect,
};
import package::helpers::external_texture::sample_image_paint;
import package::helpers::gradient::get_gradient_sample_xy;
import package::helpers::gradient::sample_linear_gradient;
import package::helpers::gradient::sample_radial_gradient;
@if(gradient_sweep)
import package::helpers::gradient::sample_sweep_gradient;
import package::helpers::geometry::{pixel_to_ndc, quad_corner, rect_coverage};
import package::helpers::image::{
    get_image_transform,
    get_image_translate
};
import package::helpers::paint::{PAINT_TEXTURE_INDEX_MASK, load_encoded_paint_texel};
import package::helpers::util::unpack_u16_pair;

// Color source modes - where the fragment shader gets color data from.
// Use payload (color or image coordinates)
const COLOR_SOURCE_PAYLOAD: u32 = 0u;
// Sample from a rendered layer texture.
const COLOR_SOURCE_LAYER: u32 = 1u;

// Paint types
const PAINT_TYPE_SOLID: u32 = 0u;  
const PAINT_TYPE_IMAGE: u32 = 1u;
const PAINT_TYPE_LINEAR_GRADIENT: u32 = 2u;
const PAINT_TYPE_RADIAL_GRADIENT: u32 = 3u;
const PAINT_TYPE_SWEEP_GRADIENT: u32 = 4u;
const PAINT_TYPE_BLURRED_ROUNDED_RECT: u32 = 5u;

const EXTERNAL_TEXTURE_SLOT_SHIFT: u32 = 24u;

const RECT_STRIP_FLAG: u32 = 0x80000000u;

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
    // Number of trailing zeros in encoded_paints_tex_width (log2 of width).
    // Pre-calculated on CPU since WebGL2 doesn't support `firstTrailingBit`.
    encoded_paints_tex_width_bits: u32,
    // An offset to apply to the strip.
    //
    // In most cases, these will be zero. However,
    // when rendering filter layers, we need to account for the 1) shift caused by
    // only rendering the tight bounding box of the filter layer and 2) the offset
    // within the atlas where the filter layer will be rendered to.
    strip_offset_x: i32,
    strip_offset_y: i32,
    // Whether to flip the y-component of the NDC coordinates.
    ndc_y_negate: u32,
}

// A `StripInstance` can represent either a **normal strip** (representing a sparse fill or alpha fill of height
// Tile::HEIGHT) or a **rect strip** (an entire rectangle rendered as a single quad, with anti-aliasing support).
// The two modes are distinguished by RECT_STRIP_FLAG (bit 31 of `paint_and_rect_flag`).
//
// Depending on the active mode, the fields are interpreted as follows:
//
//   Field                 | Normal strip                      | Rect strip
//   ----------------------+-----------------------------------+-----------------------------------
//   xy                    | Strip position                    | Rect top-left (snapped outward)
//   widths_or_rect_height | [width, dense_width]              | [width, height] (both snapped)
//   col_idx_or_rect_frac  | Alpha column index                | Packed AA edge fractions (4 × u8)
//   payload               | Color / scene coords / layer xy   | Color / scene coords / layer xy
//   paint_and_rect_flag   | Paint encoding                    | Paint encoding | RECT_STRIP_FLAG
//
//
// `paint_and_rect_flag` bit layout:
//   - Bit  31:    `RECT_STRIP_FLAG`  0 = normal strip, 1 = rect strip
//   - Bits 29-30: `color_source`     0 = use payload, 1 = use layer texture
//   - Bits 0-28:  Usage depends on color_source:
//
//     When color_source = 0 (COLOR_SOURCE_PAYLOAD):
//       - Bits 26-28: `paint_type` (0 = solid, 1 = image, 2 = linear_gradient, 3 = radial_gradient, 4 = sweep_gradient)
//       - Bits 0-25:
//         - If paint_type = 0: unused
//         - Bits 0-23: `paint_texture_idx` for every non-solid paint
//         - Bits 24-25: reserved
//
//     When color_source = 1 (COLOR_SOURCE_LAYER):
//       - Bits 0-7: opacity (0-255)
//       - Bits 8-28: unused
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
// ├── paint_type = 4 (PAINT_TYPE_SWEEP_GRADIENT) - Sweep gradient rendering
//     ├── payload = [x, y] scene coordinates (packed as u16s)
//     └── bits 0-23 = paint_texture_idx
// └── paint_type = 5 (PAINT_TYPE_BLURRED_ROUNDED_RECT) - Analytic blurred rounded rectangle
//     ├── payload = [x, y] scene coordinates (packed as u16s)
//     └── bits 0-23 = paint_texture_idx
//
// color_source = 1 (COLOR_SOURCE_LAYER) - Use rendered layer texture
// ├── payload = [x, y] source layer texture origin (packed as u16s)
// └── bits 0-7 = opacity
struct StripInstance {
    // [x, y] packed as u16's
    // x, y — coordinates of the strip or rect
    @location(0)
    xy: u32,
    // [width, dense_width] packed as u16's
    // width — width of the strip or rect
    // dense_width — width of the portion where alpha blending should be applied
    // Note that currently, if the strip instance represents an actual strip (i.e. an anti-aliased region),
    // width = dense_width. If the StripInstance represents a sparse fill region, then dense_width = 0.
    // For rect strips, dense_width is repurposed to hold the rectangle height.
    // TODO: In the future, this could be optimized such that `width` always represents the width and a simple
    // 1-bit flag is used to distinguish between sparse fill region and strip. This frees up 15 other bits.
    // Otherwise, it might also be possible to merge a strip and sparse fill command into a single strip instance.
    @location(1)
    widths_or_rect_height: u32,
    // For normal strips: alpha texture column index where this strip's alpha values begin.
    // There are [`Config::strip_height`] alpha values per column.
    // For rect strips: packed fractional edge offsets for AA.
    @location(2)
    col_idx_or_rect_frac: u32,
    // See StripInstance documentation above.
    @location(3)
    payload: u32,
    // See StripInstance documentation above.
    @location(4)
    paint_and_rect_flag: u32,
    // Painter's-order index for z-depth computation.
    @location(5)
    depth_index: u32,
}

struct VertexOutput {
    // Render type for the strip
    @location(0) @interpolate(flat)
    paint_and_rect_flag: u32,
    // Texture coordinates for the current fragment
    @location(1)
    tex_coord: vec2<f32>,
    // UV coordinates for the current fragment, used for image sampling
    @location(2)
    sample_xy: vec2<f32>,
    // For normal strips: ending x-position of the dense (alpha) region.
    // For rect strips: packed dimensions (width | height << 16).
    @location(3) @interpolate(flat)
    dense_end_or_rect_size: u32,
    // Packed paint payload or layer sample coordinate.
    @location(4) @interpolate(flat)
    payload: u32,
    // Packed fractional edge offsets for rectangles.
    // Bits 0-7: x0, 8-15: y0, 16-23: x1, 24-31: y1.
    // Zero for normal strips.
    @location(5) @interpolate(flat)
    rect_frac: u32,
    // Normalized device coordinates (NDC) for the current vertex
    @builtin(position)
    position: vec4<f32>,
};

// TODO: Measure performance of moving to a separate group
@group(0) @binding(1)
var<uniform> config: Config;

@group(1) @binding(0)
var external_texture_0: texture_2d<f32>;

@group(2) @binding(0)
var encoded_paints_texture: texture_2d<u32>;

@group(3) @binding(0)
var gradient_texture: texture_2d<f32>;

@vertex
fn vs_main(
    @builtin(vertex_index)in_vertex_index: u32,
    instance: StripInstance,
) -> VertexOutput {
    var out: VertexOutput;
    out.sample_xy = vec2(0.0);
    // Map vertex_index (0-3) to quad corners:
    // 0 → (0,0), 1 → (1,0), 2 → (0,1), 3 → (1,1)
    let corner = quad_corner(in_vertex_index);
    let x = corner.x;
    let y = corner.y;
    // Unpack the x and y coordinates from the packed u32 instance.xy
    let strip_position = unpack_u16_pair(instance.xy);
    let widths = unpack_u16_pair(instance.widths_or_rect_height);
    let x0 = strip_position.x;
    let y0 = strip_position.y;
    let width = widths.x;
    let dense_width = widths.y;

    let is_rect = (instance.paint_and_rect_flag & RECT_STRIP_FLAG) != 0u;
    var height = config.strip_height;
    if is_rect {
        height = dense_width;
        out.dense_end_or_rect_size = width | (dense_width << 16u);
        out.rect_frac = instance.col_idx_or_rect_frac;
    } else {
        out.dense_end_or_rect_size = instance.col_idx_or_rect_frac + dense_width;
        out.rect_frac = 0u;
    }
    // Calculate the pixel coordinates of the current vertex within the strip.
    // Don't forget to apply the strip offset!
    let pixel = vec2<f32>(
        f32(i32(x0) + config.strip_offset_x) + x * f32(width),
        f32(i32(y0) + config.strip_offset_y) + y * f32(height),
    );
    // Convert pixel coordinates to normalized device coordinates (NDC)
    // NDC ranges from -1 to 1, with (0,0) at the center of the viewport
    let ndc = pixel_to_ndc(pixel, vec2<f32>(f32(config.width), f32(config.height)));

    let color_source = (instance.paint_and_rect_flag >> 29u) & 0x3u;
    if color_source == COLOR_SOURCE_PAYLOAD {
        let paint_type = (instance.paint_and_rect_flag >> 26u) & 0x7u;
        // Unpack view coordinates for image sampling and gradient calculations
        let scene_strip = unpack_u16_pair(instance.payload);

        if paint_type == PAINT_TYPE_IMAGE {
            let paint_tex_idx = instance.paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
            let image_texel0 = load_encoded_paint_texel(
                encoded_paints_texture, config.encoded_paints_tex_width_bits, paint_tex_idx, 0u,
            );
            let image_texel1 = load_encoded_paint_texel(
                encoded_paints_texture, config.encoded_paints_tex_width_bits, paint_tex_idx, 1u,
            );
            let image_texel2 = load_encoded_paint_texel(
                encoded_paints_texture, config.encoded_paints_tex_width_bits, paint_tex_idx, 2u,
            );
            // Use view coordinates for image sampling (always in global view space)
            let pos = vec2<f32>(
                f32(scene_strip.x) + x * f32(width),
                f32(scene_strip.y) + y * f32(height),
            );
            out.sample_xy = get_image_translate(image_texel1, image_texel2)
                + get_image_transform(image_texel0, image_texel1) * pos;
        } else if paint_type == PAINT_TYPE_LINEAR_GRADIENT || paint_type == PAINT_TYPE_RADIAL_GRADIENT || paint_type == PAINT_TYPE_SWEEP_GRADIENT {
            // Use view coordinates for gradient transform (always in global view space)
            out.sample_xy = get_gradient_sample_xy(scene_strip, corner, width, height);
        } else if paint_type == PAINT_TYPE_BLURRED_ROUNDED_RECT {
            @if(blurred_rounded_rect)
            out.sample_xy = get_blurred_rounded_rect_sample_xy(scene_strip, corner, width, height);
        }
    } else if color_source == COLOR_SOURCE_LAYER {
        let source = unpack_u16_pair(instance.payload);
        out.sample_xy = vec2<f32>(
            f32(source.x) + x * f32(width),
            f32(source.y) + y * f32(height),
        );
    }

    let col_offset = select(f32(instance.col_idx_or_rect_frac), 0.0, is_rect);
    out.tex_coord = vec2<f32>(col_offset + x * f32(width), y * f32(height));

    // Divide by a power of 2 to ensure exact f32 arithmetic (and divide by the expected depth
    // buffer precision of 24 bits).
    let z = 1.0 - f32(instance.depth_index) / f32(1u << 24u);
    // Flip it based on the flag.
    let final_ndc_y = select(ndc.y, -ndc.y, config.ndc_y_negate != 0u);
    out.position = vec4<f32>(ndc.x, final_ndc_y, z, 1.0);
    out.payload = instance.payload;
    out.paint_and_rect_flag = instance.paint_and_rect_flag;

    return out;
}

@group(0) @binding(0)
var alphas_texture: texture_2d<u32>;

@group(0) @binding(2)
var layer_input_texture: texture_2d<f32>;

@fragment
fn fs_main(
    @location(0) @interpolate(flat)paint_and_rect_flag: u32,
    @location(1) tex_coord: vec2<f32>,
    @location(2) sample_xy: vec2<f32>,
    @location(3) @interpolate(flat) dense_end_or_rect_size: u32,
    @location(4) @interpolate(flat) payload: u32,
    @location(5) @interpolate(flat) rect_frac: u32,
    @builtin(position) position: vec4<f32>,
) -> @location(0) vec4<f32> {
    var alpha = 1.0;
    let is_rect = (paint_and_rect_flag & RECT_STRIP_FLAG) != 0u;
    if is_rect && rect_frac != 0u {
        alpha = rect_coverage(dense_end_or_rect_size, rect_frac, tex_coord);
    } else if !is_rect && dense_end_or_rect_size != 0u {
        alpha = sample_strip_alpha(alphas_texture, config.alphas_tex_width_bits, tex_coord);
    }
    let color_source = (paint_and_rect_flag >> 29u) & 0x3u;
    var final_color: vec4<f32>;

    if color_source == COLOR_SOURCE_PAYLOAD {
        let paint_type = (paint_and_rect_flag >> 26u) & 0x7u;

        // in.payload encodes a color for PAINT_TYPE_SOLID or sample_xy for PAINT_TYPE_IMAGE
        if paint_type == PAINT_TYPE_SOLID {
            final_color = unpack4x8unorm(payload);
        } else if paint_type == PAINT_TYPE_IMAGE {
            final_color = sample_image_paint(
                encoded_paints_texture, config.encoded_paints_tex_width_bits,
                external_texture_0, paint_and_rect_flag, sample_xy,
            );
        } else if paint_type == PAINT_TYPE_LINEAR_GRADIENT {
            final_color = sample_linear_gradient(
                encoded_paints_texture, config.encoded_paints_tex_width_bits,
                gradient_texture, paint_and_rect_flag, sample_xy,
            );
        } else if paint_type == PAINT_TYPE_RADIAL_GRADIENT {
            final_color = sample_radial_gradient(
                encoded_paints_texture, config.encoded_paints_tex_width_bits,
                gradient_texture, paint_and_rect_flag, sample_xy,
            );
        } else if paint_type == PAINT_TYPE_SWEEP_GRADIENT {
            @if(gradient_sweep)
            final_color = sample_sweep_gradient(
                encoded_paints_texture, config.encoded_paints_tex_width_bits,
                gradient_texture, paint_and_rect_flag, sample_xy,
            );
        } else if paint_type == PAINT_TYPE_BLURRED_ROUNDED_RECT {
            @if(blurred_rounded_rect)
            final_color = sample_blurred_rounded_rect(
                encoded_paints_texture, config.encoded_paints_tex_width_bits,
                paint_and_rect_flag, sample_xy,
            );
        }
    } else if color_source == COLOR_SOURCE_LAYER {
        let layer_opacity = f32(paint_and_rect_flag & 0xffu) * (1.0 / 255.0);
        final_color = layer_opacity * textureLoad(layer_input_texture, vec2<i32>(sample_xy), 0);
    } else {
        final_color = vec4<f32>(0.0);
    }

    return alpha * final_color;
}
