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

// Paint texture index mask (extracts lower 26 bits from paint field).
const PAINT_TEXTURE_INDEX_MASK: u32 = 0x03FFFFFFu;

const RECT_STRIP_FLAG: u32 = 0x80000000u;

// Image quality
const IMAGE_QUALITY_LOW = 0u;
const IMAGE_QUALITY_MEDIUM = 1u;
const IMAGE_QUALITY_HIGH = 2u;

const IMAGE_SOURCE_ATLAS = 0u;
const IMAGE_SOURCE_EXTERNAL = 1u;

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
//         - If paint_type >= 1: `paint_texture_idx`
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
//     └── bits 0-25 = paint_texture_idx
// └── paint_type = 5 (PAINT_TYPE_BLURRED_ROUNDED_RECT) - Analytic blurred rounded rectangle
//     ├── payload = [x, y] scene coordinates (packed as u16s)
//     └── bits 0-25 = paint_texture_idx
//
// color_source = 1 (COLOR_SOURCE_LAYER) - Use rendered layer texture
// ├── payload = [x, y] source layer texture origin (packed as u16s)
// └── bits 0-7 = opacity
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
    // Painter's-order index for z-depth computation.
    @location(5) depth_index: u32,
}

struct VertexOutput {
    // Render type for the strip
    @location(0) @interpolate(flat) paint_and_rect_flag: u32,
    // Texture coordinates for the current fragment
    @location(1) tex_coord: vec2<f32>,
    // UV coordinates for the current fragment, used for image sampling
    @location(2) sample_xy: vec2<f32>,
    // For normal strips: ending x-position of the dense (alpha) region.
    // For rect strips: packed dimensions (width | height << 16).
    @location(3) @interpolate(flat) dense_end_or_rect_size: u32,
    // Packed paint payload or layer sample coordinate.
    @location(4) @interpolate(flat) payload: u32,
    // Packed fractional edge offsets for rectangles.
    // Bits 0-7: x0, 8-15: y0, 16-23: x1, 24-31: y1.
    // Zero for normal strips.
    @location(5) @interpolate(flat) rect_frac: u32,
    // Normalized device coordinates (NDC) for the current vertex
    @builtin(position) position: vec4<f32>,
};

// TODO: Measure performance of moving to a separate group
@group(0) @binding(1)
var<uniform> config: Config;

@group(1) @binding(0)
var atlas_texture_array: texture_2d_array<f32>;

@group(1) @binding(1)
var external_texture: texture_2d<f32>;

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
    out.sample_xy = vec2(0.0);
    // Map vertex_index (0-3) to quad corners:
    // 0 → (0,0), 1 → (1,0), 2 → (0,1), 3 → (1,1)
    let x = f32(in_vertex_index & 1u);
    let y = f32(in_vertex_index >> 1u);
    // Unpack the x and y coordinates from the packed u32 instance.xy
    let x0 = instance.xy & 0xffffu;
    let y0 = instance.xy >> 16u;
    let width = instance.widths_or_rect_height & 0xffffu;
    let dense_width = instance.widths_or_rect_height >> 16u;

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
    let pix_x = f32(i32(x0) + config.strip_offset_x) + x * f32(width);
    let pix_y = f32(i32(y0) + config.strip_offset_y) + y * f32(height);
    // Convert pixel coordinates to normalized device coordinates (NDC)
    // NDC ranges from -1 to 1, with (0,0) at the center of the viewport
    let ndc_x = pix_x * 2.0 / f32(config.width) - 1.0;
    let ndc_y = 1.0 - pix_y * 2.0 / f32(config.height);

    let color_source = (instance.paint_and_rect_flag >> 29u) & 0x3u;
    if color_source == COLOR_SOURCE_PAYLOAD {
        let paint_type = (instance.paint_and_rect_flag >> 26u) & 0x7u;
        // Unpack view coordinates for image sampling and gradient calculations
        let scene_strip_x = instance.payload & 0xffffu;
        let scene_strip_y = instance.payload >> 16u;

        if paint_type == PAINT_TYPE_IMAGE {
            let paint_tex_idx = instance.paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
            let image_texel0 = load_encoded_paint_texel(paint_tex_idx, 0u);
            let image_texel1 = load_encoded_paint_texel(paint_tex_idx, 1u);
            let image_texel2 = load_encoded_paint_texel(paint_tex_idx, 2u);
            // Use view coordinates for image sampling (always in global view space)
            let pos = vec2<f32>(f32(scene_strip_x) + x * f32(width), f32(scene_strip_y) + y * f32(height));
            out.sample_xy = get_image_translate(image_texel1, image_texel2)
                + get_image_offset(image_texel0)
                + get_image_transform(image_texel0, image_texel1) * pos;
        } else if paint_type == PAINT_TYPE_LINEAR_GRADIENT || paint_type == PAINT_TYPE_RADIAL_GRADIENT || paint_type == PAINT_TYPE_SWEEP_GRADIENT || paint_type == PAINT_TYPE_BLURRED_ROUNDED_RECT {
            // Use view coordinates for gradient transform (always in global view space)
            out.sample_xy = vec2<f32>(
                f32(scene_strip_x) + x * f32(width),
                f32(scene_strip_y) + y * f32(height)
            );
        }
    } else if color_source == COLOR_SOURCE_LAYER {
        let src_x = instance.payload & 0xffffu;
        let src_y = instance.payload >> 16u;
        out.sample_xy = vec2<f32>(
            f32(src_x) + x * f32(width),
            f32(src_y) + y * f32(height),
        );
    }

    let col_offset = select(f32(instance.col_idx_or_rect_frac), 0.0, is_rect);
    out.tex_coord = vec2<f32>(col_offset + x * f32(width), y * f32(height));

    // Divide by a power of 2 to ensure exact f32 arithmetic (and divide by the expected depth
    // buffer precision of 24 bits).
    let z = 1.0 - f32(instance.depth_index) / f32(1u << 24u);
    // Flip it based on the flag.
    let final_ndc_y = select(ndc_y, -ndc_y, config.ndc_y_negate != 0u);
    out.position = vec4<f32>(ndc_x, final_ndc_y, z, 1.0);
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
    @location(0) @interpolate(flat) paint_and_rect_flag: u32,
    @location(1) tex_coord: vec2<f32>,
    @location(2) sample_xy: vec2<f32>,
    @location(3) @interpolate(flat) dense_end_or_rect_size: u32,
    @location(4) @interpolate(flat) payload: u32,
    @location(5) @interpolate(flat) rect_frac: u32,
    @builtin(position) position: vec4<f32>,
) -> @location(0) vec4<f32> {
    var alpha = 1.0;
    let is_rect = (paint_and_rect_flag & RECT_STRIP_FLAG) != 0u;
    // TODO: Explore doing these calculations only for rectangle parts that actually need anti-aliasing. See
    // https://github.com/linebender/vello/pull/1482#discussion_r2861311034
    if is_rect && rect_frac != 0u {
        let frac = unpack4x8unorm(rect_frac);
        // Calculate how much of the pixel is actually covered by the rect.
        // We do this by simply calculating the fractions in the x and y direction, and
        // then multiplying them.
        // For (maybe?) better performance, we calculate the x and y dimension in a single
        // pass by packing everything into a vec2.
        let rect_size = vec2<f32>(f32(dense_end_or_rect_size & 0xFFFFu), f32(dense_end_or_rect_size >> 16u));
        let tc = tex_coord;
        // + 0.5 and -0.5 since the fragment shader positions the coordinates in the center of the pixel.
        let bottom_and_right = min(tc + 0.5, rect_size - frac.zw);
        let top_and_left = max(tc - 0.5, frac.xy);
        let a = clamp(bottom_and_right - top_and_left, vec2(0.0), vec2(1.0));
        alpha = a.x * a.y;
    } else if !is_rect && dense_end_or_rect_size != 0u {
        let x = u32(floor(tex_coord.x));
        let y = u32(floor(tex_coord.y));
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
    // Apply the alpha value to the unpacked RGBA color or sampled paint.
    let color_source = (paint_and_rect_flag >> 29u) & 0x3u;
    var final_color: vec4<f32>;

    if color_source == COLOR_SOURCE_PAYLOAD {
        let paint_type = (paint_and_rect_flag >> 26u) & 0x7u;

        // in.payload encodes a color for PAINT_TYPE_SOLID or sample_xy for PAINT_TYPE_IMAGE
        if paint_type == PAINT_TYPE_SOLID {
            final_color = alpha * unpack4x8unorm(payload);
        } else if paint_type == PAINT_TYPE_IMAGE {
            let paint_tex_idx = paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
            let image_texel0 = load_encoded_paint_texel(paint_tex_idx, 0u);
            let image_texel1 = load_encoded_paint_texel(paint_tex_idx, 1u);
            let image_texel2 = load_encoded_paint_texel(paint_tex_idx, 2u);
            let image_offset = get_image_offset(image_texel0);
            let image_size = get_image_size(image_texel0);
            let image_extend_modes = get_image_extend_modes(image_texel0);
            let image_atlas_index = get_image_atlas_index(image_texel0);
            let image_quality = get_image_quality(image_texel0);
            let image_source_kind = get_image_source_kind(image_texel0);
            let image_padding = get_image_padding(image_texel2);
            let packed_tint = image_texel2.y;
            let has_tint = packed_tint != 0u;
            // When packed_tint is zero (no tint), use identity color vec4(1.0) with
            // Multiply mode so the math reduces to sample_color * 1.0 = sample_color.
            let image_tint = select(vec4<f32>(1.0), unpack4x8unorm(packed_tint), has_tint);
            let is_multiply = !has_tint || image_texel2.z != TINT_MODE_ALPHA_MASK;
            let local_xy = sample_xy - image_offset;
            // This offset doesn't exist in vello_cpu, and we use it because 45 degree skewing seems to cause
            // artifacts on the GPU. We have something similar in place for gradients. It might be worth revisiting
            // this to see whether a better approach is possible.
            let offset = 0.00001;
            let extended_xy = vec2<f32>(
                extend_mode(local_xy.x + offset, image_extend_modes.x, image_size.x),
                extend_mode(local_xy.y + offset, image_extend_modes.y, image_size.y)
            );

            // TODO: add a fast path for images where we are using bilinear sampling and want transparent pixels,
            // using GPU-native bilinear sampling

            var sample_color: vec4<f32>;
            if image_source_kind == IMAGE_SOURCE_EXTERNAL {
                let final_xy = image_offset + extended_xy;
                sample_color = sample_external_image(
                    image_quality,
                    final_xy,
                    image_offset,
                    image_size,
                );
            } else if image_quality == IMAGE_QUALITY_HIGH {
                let final_xy = image_offset + extended_xy;
                sample_color = bicubic_sample(
                    atlas_texture_array,
                    final_xy,
                    i32(image_atlas_index),
                    image_offset,
                    image_size,
                    image_extend_modes,
                    image_padding,
                );
            } else if image_quality == IMAGE_QUALITY_MEDIUM {
                let final_xy = image_offset + extended_xy - vec2(0.5);
                sample_color = bilinear_sample(
                    atlas_texture_array,
                    final_xy,
                    i32(image_atlas_index),
                    image_offset,
                    image_size,
                    image_extend_modes,
                    image_padding,
                );
            } else {
                let final_xy = image_offset + extended_xy;
                sample_color = textureLoad(
                    atlas_texture_array,
                    vec2<u32>(final_xy),
                    i32(image_atlas_index),
                    0,
                );
            }

            final_color = alpha * select(
                image_tint * sample_color.a,
                sample_color * image_tint,
                is_multiply
            );
        } else if paint_type == PAINT_TYPE_LINEAR_GRADIENT {
            let paint_tex_idx = paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
            let gradient_texel0 = load_encoded_paint_texel(paint_tex_idx, 0u);
            let gradient_texel1 = load_encoded_paint_texel(paint_tex_idx, 1u);
            
            // Calculate fragment position and apply affine transform
            let fragment_pos = sample_xy;
            let grad_pos = apply_gradient_transform(gradient_texel0, gradient_texel1, fragment_pos);
            
            // For linear gradient, t-value is just the x coordinate in gradient space
            let t_value = grad_pos.x + 0.00001;
            let gradient_color = sample_gradient_lut(
                t_value,
                get_gradient_extend_mode(gradient_texel0),
                get_gradient_start(gradient_texel0),
                get_gradient_texture_width(gradient_texel0)
            );
            final_color = alpha * gradient_color;
        } else if paint_type == PAINT_TYPE_RADIAL_GRADIENT {
            let paint_tex_idx = paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
            let gradient_texel0 = load_encoded_paint_texel(paint_tex_idx, 0u);
            let gradient_texel1 = load_encoded_paint_texel(paint_tex_idx, 1u);
            let gradient_texel2 = load_encoded_paint_texel(paint_tex_idx, 2u);
            let gradient_texel3 = load_encoded_paint_texel(paint_tex_idx, 3u);
            
            // Calculate fragment position and apply affine transform
            let fragment_pos = sample_xy;
            let grad_pos = apply_gradient_transform(gradient_texel0, gradient_texel1, fragment_pos);
            
            // For radial gradient, calculate distance from center
            let gradient_result = calculate_radial_gradient(grad_pos, gradient_texel2, gradient_texel3);
            let gradient_color = sample_gradient_lut(
                gradient_result.x,
                get_gradient_extend_mode(gradient_texel0),
                get_gradient_start(gradient_texel0),
                get_gradient_texture_width(gradient_texel0)
            );
            final_color = select(
                vec4<f32>(0.0, 0.0, 0.0, 0.0),
                alpha * gradient_color,
                gradient_result.y != 0.0
            );
        } else if paint_type == PAINT_TYPE_SWEEP_GRADIENT {
            let paint_tex_idx = paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
            let gradient_texel0 = load_encoded_paint_texel(paint_tex_idx, 0u);
            let gradient_texel1 = load_encoded_paint_texel(paint_tex_idx, 1u);
            let gradient_texel2 = load_encoded_paint_texel(paint_tex_idx, 2u);
            
            // Calculate fragment position and apply affine transform
            let fragment_pos = sample_xy;
            var grad_pos = apply_gradient_transform(gradient_texel0, gradient_texel1, fragment_pos);

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
            let t_value = (angle - get_sweep_start_angle(gradient_texel2)) * get_sweep_inv_angle_delta(gradient_texel2);
            let gradient_color = sample_gradient_lut(
                t_value,
                get_gradient_extend_mode(gradient_texel0),
                get_gradient_start(gradient_texel0),
                get_gradient_texture_width(gradient_texel0)
            );
            final_color = alpha * gradient_color;
        } else if paint_type == PAINT_TYPE_BLURRED_ROUNDED_RECT {
            let paint_tex_idx = paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
            let blurred_texel0 = load_encoded_paint_texel(paint_tex_idx, 0u);
            let blurred_texel1 = load_encoded_paint_texel(paint_tex_idx, 1u);
            let blurred_texel2 = load_encoded_paint_texel(paint_tex_idx, 2u);
            let blurred_texel3 = load_encoded_paint_texel(paint_tex_idx, 3u);
            let blurred_texel4 = load_encoded_paint_texel(paint_tex_idx, 4u);
            final_color = alpha * calculate_blurred_rounded_rect(
                sample_xy,
                blurred_texel0,
                blurred_texel1,
                blurred_texel2,
                blurred_texel3,
                blurred_texel4,
            );
    }
    } else if color_source == COLOR_SOURCE_LAYER {
        let layer_opacity = f32(paint_and_rect_flag & 0xffu) * (1.0 / 255.0);
        final_color = alpha * layer_opacity * textureLoad(layer_input_texture, vec2<i32>(sample_xy), 0);
    } else {
        final_color = vec4<f32>(0.0);
    }
    return final_color;
}

/// Tint mode constants.
const TINT_MODE_ALPHA_MASK: u32 = 0u;
const TINT_MODE_MULTIPLY: u32 = 1u;

// Convert a flat texel index to 2D texture coordinates for the encoded paints texture.
fn encoded_paint_coord(flat_idx: u32) -> vec2<u32> {
    return vec2<u32>(
        flat_idx & ((1u << config.encoded_paints_tex_width_bits) - 1u),
        flat_idx >> config.encoded_paints_tex_width_bits
    );
}

fn load_encoded_paint_texel(paint_tex_idx: u32, texel_offset: u32) -> vec4<u32> {
    return textureLoad(
        encoded_paints_texture,
        encoded_paint_coord(paint_tex_idx + texel_offset),
        0,
    );
}

// Encoded image layout. Must match `GpuEncodedImage` in `vello_hybrid/src/render/common.rs`.
//
// texel0.x: image_params
//   bits 0-1: quality
//   bits 2-3: extend_x
//   bits 4-5: extend_y
//   bits 6-13: atlas_index
//   bit 14: source_kind (0=atlas, 1=external texture)
// texel0.y: image_size, packed as [width:16, height:16]
// texel0.z: image_offset, packed as [x:16, y:16]
// texel0.w/texel1.x/texel1.y/texel1.z: transform matrix [a, b, c, d]
// texel1.w/texel2.x: translation [tx, ty]
// texel2.y: premultiplied tint color packed as RGBA8 unorm; 0 means no tint
// texel2.z: tint mode, only meaningful when texel2.y != 0
// texel2.w: transparent padding pixels around the image in the atlas

/// The rendering quality of the image.
fn get_image_quality(texel0: vec4<u32>) -> u32 { return texel0.x & 0x3u; }

/// The extend modes in the horizontal and vertical direction.
fn get_image_extend_modes(texel0: vec4<u32>) -> vec2<u32> {
    return vec2<u32>((texel0.x >> 2u) & 0x3u, (texel0.x >> 4u) & 0x3u);
}

/// The size of the image in pixels.
fn get_image_size(texel0: vec4<u32>) -> vec2<f32> {
    return vec2<f32>(f32(texel0.y >> 16u), f32(texel0.y & 0xFFFFu));
}

/// The offset of the image in pixels.
fn get_image_offset(texel0: vec4<u32>) -> vec2<f32> {
    return vec2<f32>(f32(texel0.z >> 16u), f32(texel0.z & 0xFFFFu));
}

/// The atlas index containing this image.
fn get_image_atlas_index(texel0: vec4<u32>) -> u32 { return (texel0.x >> 6u) & 0xFFu; }

/// Whether the image is sourced from the atlas or the externally bound texture.
fn get_image_source_kind(texel0: vec4<u32>) -> u32 { return (texel0.x >> 14u) & 0x1u; }

/// 2x2 linear part of the affine transform (columns [a,b] and [c,d]).
fn get_image_transform(texel0: vec4<u32>, texel1: vec4<u32>) -> mat2x2<f32> {
    return mat2x2<f32>(
        vec2<f32>(bitcast<f32>(texel0.w), bitcast<f32>(texel1.x)),
        vec2<f32>(bitcast<f32>(texel1.y), bitcast<f32>(texel1.z))
    );
}

/// Translation part of the affine transform [tx, ty].
fn get_image_translate(texel1: vec4<u32>, texel2: vec4<u32>) -> vec2<f32> {
    return vec2<f32>(bitcast<f32>(texel1.w), bitcast<f32>(texel2.x));
}

/// Number of transparent padding pixels around the image in the atlas.
fn get_image_padding(texel2: vec4<u32>) -> f32 { return f32(texel2.w); }

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
    _extend_modes: vec2<u32>,
    _image_padding: f32,
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

// This is the same as `bilinear_sample` above, but for external textures instead of the atlas texture array.
fn external_bilinear_sample(
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
) -> vec4<f32> {
    let atlas_max = image_offset + image_size - vec2(1.0);
    let atlas_uv_clamped = clamp(coords, image_offset, atlas_max);
    let uv_quad = vec4(floor(atlas_uv_clamped), ceil(atlas_uv_clamped));
    let uv_frac = fract(coords);
    let a = textureLoad(external_texture, vec2<i32>(uv_quad.xy), 0);
    let b = textureLoad(external_texture, vec2<i32>(uv_quad.xw), 0);
    let c = textureLoad(external_texture, vec2<i32>(uv_quad.zy), 0);
    let d = textureLoad(external_texture, vec2<i32>(uv_quad.zw), 0);
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
    _extend_modes: vec2<u32>,
    _image_padding: f32,
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
    
    // Clamp alpha first, then clamp premultiplied color channels against it.
    let a = clamp(result.a, 0.0, 1.0);
    return vec4<f32>(clamp(result.rgb, vec3(0.0), vec3(a)), a);
}

// This is the same as `bicubic_sample` above, but for external textures instead of the atlas texture array.
fn external_bicubic_sample(
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
) -> vec4<f32> {
     let atlas_max = image_offset + image_size - vec2(1.0);
     let frac_coords = fract(coords + 0.5);
     // Get cubic weights for x and y directions
     let cx = cubic_weights(frac_coords.x);
     let cy = cubic_weights(frac_coords.y);

     // Sample 4x4 grid around coords
     let s00 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-1.5, -1.5), image_offset, atlas_max)), 0);
     let s10 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-0.5, -1.5), image_offset, atlas_max)), 0);
     let s20 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(0.5, -1.5), image_offset, atlas_max)), 0);
     let s30 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(1.5, -1.5), image_offset, atlas_max)), 0);

     let s01 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-1.5, -0.5), image_offset, atlas_max)), 0);
     let s11 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-0.5, -0.5), image_offset, atlas_max)), 0);
     let s21 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(0.5, -0.5), image_offset, atlas_max)), 0);
     let s31 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(1.5, -0.5), image_offset, atlas_max)), 0);

     let s02 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-1.5, 0.5), image_offset, atlas_max)), 0);
     let s12 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-0.5, 0.5), image_offset, atlas_max)), 0);
     let s22 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(0.5, 0.5), image_offset, atlas_max)), 0);
     let s32 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(1.5, 0.5), image_offset, atlas_max)), 0);

     let s03 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-1.5, 1.5), image_offset, atlas_max)), 0);
     let s13 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-0.5, 1.5), image_offset, atlas_max)), 0);
     let s23 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(0.5, 1.5), image_offset, atlas_max)), 0);
     let s33 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(1.5, 1.5), image_offset, atlas_max)), 0);

    // Interpolate in x direction for each row
    let row0 = cx.x * s00 + cx.y * s10 + cx.z * s20 + cx.w * s30;
    let row1 = cx.x * s01 + cx.y * s11 + cx.z * s21 + cx.w * s31;
    let row2 = cx.x * s02 + cx.y * s12 + cx.z * s22 + cx.w * s32;
    let row3 = cx.x * s03 + cx.y * s13 + cx.z * s23 + cx.w * s33;
    let result = cy.x * row0 + cy.y * row1 + cy.z * row2 + cy.w * row3;

    // Clamp alpha first, then clamp premultiplied color channels against it.
    let a = clamp(result.a, 0.0, 1.0);
    return vec4<f32>(clamp(result.rgb, vec3(0.0), vec3(a)), a);
}

fn sample_external_image(
    quality: u32,
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
) -> vec4<f32> {
    if quality == IMAGE_QUALITY_HIGH {
        return external_bicubic_sample(coords, image_offset, image_size);
    }
    if quality == IMAGE_QUALITY_MEDIUM {
        return external_bilinear_sample(coords - vec2(0.5), image_offset, image_size);
    }
    return textureLoad(external_texture, vec2<u32>(coords), 0);
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
fn sample_gradient_lut(t_value: f32, extend_mode: u32, gradient_start: u32, texture_width: u32) -> vec4<f32> {
    // Apply extend mode to t_value
    let clamped_t = extend_mode_normalized(t_value, extend_mode);
    // Convert t_value to texture coordinate
    let t_offset = u32(clamped_t * f32(texture_width - 1u));
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

/// Width of the gradient texture.
fn get_gradient_texture_width(texel0: vec4<u32>) -> u32 { return texel0.x & 0x0FFFFFFFu; }

/// The extend mode for the gradient.
fn get_gradient_extend_mode(texel0: vec4<u32>) -> u32 { return (texel0.x >> 30u) & 3u; }

/// Start coordinate in the flat gradient texture.
fn get_gradient_start(texel0: vec4<u32>) -> u32 { return texel0.y; }

/// 2x2 linear part of the affine transform (columns [a,b] and [c,d]).
fn get_gradient_transform(texel0: vec4<u32>, texel1: vec4<u32>) -> mat2x2<f32> {
    return mat2x2<f32>(
        vec2<f32>(bitcast<f32>(texel0.z), bitcast<f32>(texel0.w)),
        vec2<f32>(bitcast<f32>(texel1.x), bitcast<f32>(texel1.y))
    );
}

/// Translation part of the affine transform [tx, ty].
fn get_gradient_translate(texel1: vec4<u32>) -> vec2<f32> {
    return vec2<f32>(bitcast<f32>(texel1.z), bitcast<f32>(texel1.w));
}

fn apply_gradient_transform(
    texel0: vec4<u32>,
    texel1: vec4<u32>,
    fragment_pos: vec2<f32>,
) -> vec2<f32> {
    return get_gradient_transform(texel0, texel1) * fragment_pos + get_gradient_translate(texel1);
}

/// Kind of radial gradient (0=Radial, 1=Strip, 2=Focal).
fn get_radial_kind(texel2: vec4<u32>) -> u32 { return texel2.x & 0x3u; }

/// Whether the focal point is swapped for radial gradient (0=false, 1=true).
fn get_radial_f_is_swapped(texel2: vec4<u32>) -> u32 { return (texel2.x >> 2u) & 1u; }

/// Bias value for radial gradient calculation.
fn get_radial_bias(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.y); }

/// Scale factor for radial gradient calculation.
fn get_radial_scale(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.z); }

/// Focal point 0 parameter for radial gradient.
fn get_radial_fp0(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.w); }

/// Focal point 1 parameter for radial gradient.
fn get_radial_fp1(texel3: vec4<u32>) -> f32 { return bitcast<f32>(texel3.x); }

/// Focal radius 1 parameter for radial gradient.
fn get_radial_fr1(texel3: vec4<u32>) -> f32 { return bitcast<f32>(texel3.y); }

/// Focal X coordinate for radial gradient.
fn get_radial_f_focal_x(texel3: vec4<u32>) -> f32 { return bitcast<f32>(texel3.z); }

/// Scaled radius 0 squared parameter for radial gradient strip.
fn get_radial_scaled_r0_squared(texel3: vec4<u32>) -> f32 { return bitcast<f32>(texel3.w); }

/// Starting angle for sweep gradient (in radians).
fn get_sweep_start_angle(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.x); }

/// Inverse of angle delta for sweep gradient.
fn get_sweep_inv_angle_delta(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.y); }

/// 2x2 linear part of the affine transform (columns [a,b] and [c,d]).
fn get_blurred_rounded_rect_transform(texel0: vec4<u32>) -> mat2x2<f32> {
    return mat2x2<f32>(
        vec2<f32>(bitcast<f32>(texel0.x), bitcast<f32>(texel0.y)),
        vec2<f32>(bitcast<f32>(texel0.z), bitcast<f32>(texel0.w))
    );
}

/// Translation part of the affine transform [tx, ty].
fn get_blurred_rounded_rect_translate(texel1: vec4<u32>) -> vec2<f32> {
    return vec2<f32>(bitcast<f32>(texel1.x), bitcast<f32>(texel1.y));
}

/// Premultiplied rectangle color.
fn get_blurred_rounded_rect_color(texel1: vec4<u32>) -> vec4<f32> { return unpack4x8unorm(texel1.z); }

/// Whether to paint the inverse (`1 - alpha`) of the blur coverage.
fn get_blurred_rounded_rect_invert(texel1: vec4<u32>) -> u32 { return texel1.w; }

fn get_blurred_rounded_rect_exponent(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.x); }

fn get_blurred_rounded_rect_recip_exponent(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.y); }

fn get_blurred_rounded_rect_scale(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.z); }

fn get_blurred_rounded_rect_std_dev_inv(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.w); }

fn get_blurred_rounded_rect_min_edge(texel3: vec4<u32>) -> f32 { return bitcast<f32>(texel3.x); }

fn get_blurred_rounded_rect_w(texel3: vec4<u32>) -> f32 { return bitcast<f32>(texel3.y); }

fn get_blurred_rounded_rect_h(texel3: vec4<u32>) -> f32 { return bitcast<f32>(texel3.z); }

fn get_blurred_rounded_rect_r1(texel3: vec4<u32>) -> f32 { return bitcast<f32>(texel3.w); }

fn get_blurred_rounded_rect_width(texel4: vec4<u32>) -> f32 { return bitcast<f32>(texel4.x); }

fn get_blurred_rounded_rect_height(texel4: vec4<u32>) -> f32 { return bitcast<f32>(texel4.y); }

// Calculate a radial gradient; matches vello_cpu implementation.
fn calculate_radial_gradient(
    grad_pos: vec2<f32>,
    texel2: vec4<u32>,
    texel3: vec4<u32>,
) -> vec2<f32> {
    let x_pos = grad_pos.x;
    let y_pos = grad_pos.y;
    
    var t_value: f32;
    var is_valid: bool;
    let kind = get_radial_kind(texel2);
    
    switch kind {
        case RADIAL_GRADIENT_TYPE_STANDARD: {
            // Standard radial gradient: bias + scale * sqrt(x^2 + y^2)
            let radius = sqrt(x_pos * x_pos + y_pos * y_pos);
            t_value = get_radial_bias(texel2) + get_radial_scale(texel2) * radius;
            // Radial gradients are always valid
            is_valid = true;
        }
        case RADIAL_GRADIENT_TYPE_STRIP: {
            // Strip gradient: x + sqrt(scaled_r0_squared - y^2)
            let p1 = get_radial_scaled_r0_squared(texel3) - y_pos * y_pos;
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
            let fp0 = get_radial_fp0(texel2);
            let fp1 = get_radial_fp1(texel3);
            let fr1 = get_radial_fr1(texel3);
            let f_focal_x = get_radial_f_focal_x(texel3);
            let is_swapped = get_radial_f_is_swapped(texel2);
            
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
    
    return vec2<f32>(t_value, select(0.0, 1.0, is_valid));
}

// Approximation to erf used by Vello Classic and vello_cpu.
fn erf7(x: f32) -> f32 {
    let y = clamp(x * 1.1283791671, -100.0, 100.0);
    let yy = y * y;
    let z = y + (0.24295 + (0.03395 + 0.0104 * yy) * yy) * (y * yy);
    return z / sqrt(1.0 + z * z);
}

// Approximation for the convolution of a gaussian filter with a rounded rectangle, modelled
// after vello_cpu's blurred rounded rectangle painter rather than the Vello Classic shader.
fn calculate_blurred_rounded_rect(
    fragment_pos: vec2<f32>,
    texel0: vec4<u32>,
    texel1: vec4<u32>,
    texel2: vec4<u32>,
    texel3: vec4<u32>,
    texel4: vec4<u32>,
) -> vec4<f32> {
    let transform = get_blurred_rounded_rect_transform(texel0);
    let translate = get_blurred_rounded_rect_translate(texel1);
    let color = get_blurred_rounded_rect_color(texel1);
    let invert = get_blurred_rounded_rect_invert(texel1);
    let exponent = get_blurred_rounded_rect_exponent(texel2);
    let recip_exponent = get_blurred_rounded_rect_recip_exponent(texel2);
    let scale = get_blurred_rounded_rect_scale(texel2);
    let std_dev_inv = get_blurred_rounded_rect_std_dev_inv(texel2);
    let min_edge = get_blurred_rounded_rect_min_edge(texel3);
    let w = get_blurred_rounded_rect_w(texel3);
    let h = get_blurred_rounded_rect_h(texel3);
    let r1 = get_blurred_rounded_rect_r1(texel3);
    let width = get_blurred_rounded_rect_width(texel4);
    let height = get_blurred_rounded_rect_height(texel4);

    let local_xy = transform * fragment_pos + translate;
    // The 0.5 and 0.0 constants correspond to vello_cpu's v1 and v0 respectively.
    let y = local_xy.y - 0.5 * height;
    let y0 = r1 + abs(y) - 0.5 * h;
    let y1 = max(y0, 0.0);

    let x = local_xy.x - 0.5 * width;
    let x0 = r1 + abs(x) - 0.5 * w;
    let x1 = max(x0, 0.0);

    let d_pos = pow(
        pow(x1, exponent) + pow(y1, exponent),
        recip_exponent,
    );
    let d_neg = min(max(x0, y0), 0.0);
    let d = d_pos + d_neg - r1;
    let blur_coverage = scale * (
        erf7(std_dev_inv * (min_edge + d)) -
        erf7(std_dev_inv * d)
    );

    // Invert alpha when `invert` flag is set
    let blur_alpha = select(blur_coverage, 1.0 - blur_coverage, invert != 0u);

    return color * blur_alpha;
}
