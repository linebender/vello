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
// Paint types
const PAINT_TYPE_SOLID: u32 = 0u;  
const PAINT_TYPE_IMAGE: u32 = 1u;

// Paint texture index mask (extracts lower 26 bits from paint field).
const PAINT_TEXTURE_INDEX_MASK: u32 = 0x03FFFFFFu;

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
    // Padding to satisfy WebGL's 16-byte alignment requirement for uniform buffers.
    _padding0: u32,
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
//   payload               | Color / scene coords / slot idx   | Color / scene coords
//   paint_and_rect_flag   | Paint encoding                    | Paint encoding | RECT_STRIP_FLAG
//
//
// `paint_and_rect_flag` bit layout:
//   - Bit  31:    `RECT_STRIP_FLAG`  0 = normal strip, 1 = rect strip
//   - Bits 29-30: `color_source`     0 = use payload, 1 = use slot texture, 2 = blend mode
//   - Bits 0-28:  Usage depends on color_source:
//
//     When color_source = 0 (COLOR_SOURCE_PAYLOAD):
//       - Bits 26-28: `paint_type` (0 = solid, 1 = image, 2 = linear_gradient, 3 = radial_gradient, 4 = sweep_gradient)
//       - Bits 0-25:
//         - If paint_type = 0: unused
//         - If paint_type >= 1: `paint_texture_idx`
//
//     When color_source = 1 (COLOR_SOURCE_SLOT):
//       - Bits 0-7: opacity (0-255)
//       - Bits 8-28: unused
//
//     When color_source = 2 (COLOR_SOURCE_BLEND):
//       - Bits 16-28: `dest_slot` (14 bits)
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
//     └── bits 0-25 = paint_texture_idx
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
    // x, y — coordinates of the strip or rect
    @location(0) xy: u32,
    // [width, dense_width] packed as u16's
    // width — width of the strip or rect
    // dense_width — width of the portion where alpha blending should be applied
    // Note that currently, if the strip instance represents an actual strip (i.e. an anti-aliased region),
    // width = dense_width. If the StripInstance represents a sparse fill region, then dense_width = 0.
    // For rect strips, dense_width is repurposed to hold the rectangle height.
    // TODO: In the future, this could be optimized such that `width` always represents the width and a simple
    // 1-bit flag is used to distinguish between sparse fill region and strip. This frees up 15 other bits.
    // Otherwise, it might also be possible to merge a strip and sparse fill command into a single strip instance.
    @location(1) widths_or_rect_height: u32,
    // For normal strips: alpha texture column index where this strip's alpha values begin.
    // There are [`Config::strip_height`] alpha values per column.
    // For rect strips: packed fractional edge offsets for AA.
    @location(2) col_idx_or_rect_frac: u32,
    // See StripInstance documentation above.
    @location(3) payload: u32,
    // See StripInstance documentation above.
    @location(4) paint_and_rect_flag: u32,
}

struct VertexOutput {
    @location(0) @interpolate(flat) paint_and_rect_flag: u32,
    @location(1) sample_xy: vec2<f32>,
    @location(2) @interpolate(flat) payload: u32,
    @builtin(position) position: vec4<f32>,
};

// TODO: Measure performance of moving to a separate group
@group(0) @binding(1)
var<uniform> config: Config;

@group(1) @binding(0)
var atlas_texture_array: texture_2d_array<f32>;

@group(1) @binding(1)
var atlas_sampler: sampler;

@group(2) @binding(0)
var encoded_paints_texture: texture_2d<u32>;

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
    let width = instance.widths_or_rect_height & 0xffffu;
    let height = instance.widths_or_rect_height >> 16u;

    let pix_x = f32(i32(x0) + config.strip_offset_x) + x * f32(width);
    let pix_y = f32(i32(y0) + config.strip_offset_y) + y * f32(height);
    let ndc_x = pix_x * 2.0 / f32(config.width) - 1.0;
    let ndc_y = 1.0 - pix_y * 2.0 / f32(config.height);

    let color_source = (instance.paint_and_rect_flag >> 29u) & 0x3u;
    if color_source == COLOR_SOURCE_PAYLOAD {
        let paint_type = (instance.paint_and_rect_flag >> 26u) & 0x7u;
        if paint_type == PAINT_TYPE_IMAGE {
            let scene_strip_x = instance.payload & 0xffffu;
            let scene_strip_y = instance.payload >> 16u;
            let paint_tex_idx = instance.paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
            let encoded_image = unpack_encoded_image(paint_tex_idx);
            let pos = vec2<f32>(f32(scene_strip_x) + x * f32(width), f32(scene_strip_y) + y * f32(height));
            let atlas_xy = encoded_image.translate + encoded_image.image_offset + encoded_image.transform * pos;
            out.sample_xy = atlas_xy / vec2<f32>(textureDimensions(atlas_texture_array));
        }
    }

    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.payload = instance.payload;
    out.paint_and_rect_flag = instance.paint_and_rect_flag;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color_source = (in.paint_and_rect_flag >> 29u) & 0x3u;
    var final_color: vec4<f32>;

    if color_source == COLOR_SOURCE_PAYLOAD {
        let paint_type = (in.paint_and_rect_flag >> 26u) & 0x7u;

        if paint_type == PAINT_TYPE_SOLID {
            final_color = unpack4x8unorm(in.payload);
        } else if paint_type == PAINT_TYPE_IMAGE {
            let paint_tex_idx = in.paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
            let encoded_image = unpack_encoded_image(paint_tex_idx);
            final_color = textureSample(
                atlas_texture_array,
                atlas_sampler,
                in.sample_xy,
                i32(encoded_image.atlas_index),
            );
        }
    }
    return final_color;
}



struct EncodedImage {
    image_offset: vec2<f32>,
    atlas_index: u32,
    transform: mat2x2<f32>,
    translate: vec2<f32>,
}

fn encoded_paint_coord(flat_idx: u32) -> vec2<u32> {
    return vec2<u32>(
        flat_idx & ((1u << config.encoded_paints_tex_width_bits) - 1u),
        flat_idx >> config.encoded_paints_tex_width_bits
    );
}

fn unpack_encoded_image(paint_tex_idx: u32) -> EncodedImage {
    let texel0 = textureLoad(encoded_paints_texture, encoded_paint_coord(paint_tex_idx), 0);
    let texel1 = textureLoad(encoded_paints_texture, encoded_paint_coord(paint_tex_idx + 1u), 0);
    let texel2 = textureLoad(encoded_paints_texture, encoded_paint_coord(paint_tex_idx + 2u), 0);

    let atlas_index = (texel0.x >> 6u) & 0xFFu;
    let image_offset = vec2<f32>(f32(texel0.z >> 16u), f32(texel0.z & 0xFFFFu));
    let transform = mat2x2<f32>(
        vec2<f32>(bitcast<f32>(texel0.w), bitcast<f32>(texel1.x)),
        vec2<f32>(bitcast<f32>(texel1.y), bitcast<f32>(texel1.z))
    );
    let translate = vec2<f32>(bitcast<f32>(texel1.w), bitcast<f32>(texel2.x));

    return EncodedImage(image_offset, atlas_index, transform, translate);
}



