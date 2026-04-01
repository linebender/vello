// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// This shader samples rectangular regions from a texture. Each instance
// defines a source region and an affine transform that maps it to a
// destination parallelogram. Edges are anti-aliased using a pixel coverage
// calculation.

const EXTEND_PAD: u32 = 0u;
const EXTEND_REPEAT: u32 = 1u;
const EXTEND_REFLECT: u32 = 2u;

struct TextureRectConfig {
    // Destination size in pixels to convert to NDC.
    width: u32,
    height: u32,

    // A global offset that can be useful for, e.g., rendering into
    // intermediate textures at an offset.
    offset_x: i32,
    offset_y: i32,
}

@group(0) @binding(0)
var<uniform> config: TextureRectConfig;

@group(1) @binding(0)
var source_texture: texture_2d<f32>;

@group(1) @binding(1)
var source_sampler: sampler;

struct TextureRectInstance {
    // Top-left corner and width and height of the source rectangle in source
    // pixel coordinates.
    @location(0) src_origin: vec2<f32>,
    @location(1) src_size: vec2<f32>,

    // The affine transform mapping the rectangle to a parallelogram. The first
    // column's terms, second column's terms, and the translation.
    @location(2) transform_ab: vec2<f32>,
    @location(3) transform_cd: vec2<f32>,
    @location(4) transform_txty: vec2<f32>,

    // Extend mode flags.
    @location(5) flags: u32,
    // Clip rectangle origin in destination coordinates.
    @location(6) clip_origin: vec2<f32>,
    // Clip rectangle size.
    @location(7) clip_size: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,

    // Interpolated local coordinates in (0..src_size.x, 0..src_size.y)
    @location(0) local_uv: vec2<f32>,

    @location(1) @interpolate(flat) src_origin: vec2<f32>,
    @location(2) @interpolate(flat) src_size: vec2<f32>,
    @location(3) @interpolate(flat) flags: u32,
    @location(4) @interpolate(flat) clip_origin: vec2<f32>,
    @location(5) @interpolate(flat) clip_size: vec2<f32>,
    @location(6) dest_pos: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: TextureRectInstance,
) -> VertexOutput {
    // Map vertex_index (0-3) to quad corners:
    // 0 -> (0,0), 1 -> (1,0), 2 -> (0,1), 3 -> (1,1)
    let vi = vertex_index % 4u;
    let x = f32(vi & 1u);
    let y = f32(vi >> 1u);

    // Local coordinates in (0..src_size.x, 0..src_size.y).
    let local = vec2<f32>(x * instance.src_size.x, y * instance.src_size.y);

    // Apply affine transform to get destination pixel coordinates.
    let pix = vec2<f32>(
        instance.transform_ab.x * local.x + instance.transform_cd.x * local.y + instance.transform_txty.x,
        instance.transform_ab.y * local.x + instance.transform_cd.y * local.y + instance.transform_txty.y,
    );

    // Apply offset (for rendering into intermediate textures at an atlas offset).
    let adj_x = pix.x + f32(config.offset_x);
    let adj_y = pix.y + f32(config.offset_y);

    // Convert pixel coordinates to NDC.
    let ndc_x = adj_x * 2.0 / f32(config.width) - 1.0;
    let ndc_y = 1.0 - adj_y * 2.0 / f32(config.height);

    var out: VertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.local_uv = local;
    out.src_origin = instance.src_origin;
    out.src_size = instance.src_size;
    out.flags = instance.flags;
    out.clip_origin = instance.clip_origin;
    out.clip_size = instance.clip_size;
    out.dest_pos = pix;
    return out;
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
            // Triangle wave: 0->1->0->1...
            return abs(t - 2.0 * round(0.5 * t));
        }
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if any(in.dest_pos < in.clip_origin) || any(in.dest_pos >= in.clip_origin + in.clip_size) {
        return vec4(0.0);
    }

    let extend_x = (in.flags >> 2u) & 0x3u;
    let extend_y = (in.flags >> 4u) & 0x3u;

    // Normalize local_uv to 0..1 relative to src_size.
    let u_norm = in.local_uv.x / in.src_size.x;
    let v_norm = in.local_uv.y / in.src_size.y;

    // Apply extend modes in normalized space.
    let u_ext = extend_mode_normalized(u_norm, extend_x);
    let v_ext = extend_mode_normalized(v_norm, extend_y);

    // Convert to texture UV coordinates (normalized 0..1 over the full texture).
    let tex_dims = vec2<f32>(textureDimensions(source_texture));
    let uv = vec2<f32>(
        (in.src_origin.x + u_ext * in.src_size.x) / tex_dims.x,
        (in.src_origin.y + v_ext * in.src_size.y) / tex_dims.y,
    );

    // Sample from the texture.
    var color = textureSample(source_texture, source_sampler, uv);

    // The edges of the region we're sampling may cover pixels in the
    // destination only partially, so we calculate pixel coverage and return
    // the premultiplied color.
    //
    // The destination quad is an affinely transformed rectangle, i.e., a
    // parallelogram. Compute pixel coverage at each parallelogram edge using a
    // linear approximation of the exact coverage. This is exact if the pixel
    // is fully covered, fully uncovered, or if the edge crossing the pixel
    // describes a parallelogram. Otherwise, this is a linear approximation to
    // a quadratic. The maximum error occurs when the edge describes a 45
    // degree angle with the pixel, resulting in over-estimating the pixel
    // coverage by at most 0.125.

    // Calculate the size of the "coverage ramp" where the draw goes from 0 to
    // full pixel coverage. These values are in *local* coordinates; i.e., an
    // axis-aligned draw scaled up by 8x in the destination, has ramp sizes of
    // local 1/8.
    let dx_local = dpdx(in.local_uv);
    let dy_local = dpdy(in.local_uv);
    let w_u = abs(dx_local.x) + abs(dy_local.x);
    let w_v = abs(dx_local.y) + abs(dy_local.y);

    // Combine coverage of this pixel for each of the paralellogram's edges.
    let edge_coverage = clamp(in.local_uv.x / w_u + 0.5, 0.0, 1.0)
                * clamp((in.src_size.x - in.local_uv.x) / w_u + 0.5, 0.0, 1.0)
                * clamp(in.local_uv.y / w_v + 0.5, 0.0, 1.0)
                * clamp((in.src_size.y - in.local_uv.y) / w_v + 0.5, 0.0, 1.0);

    return color * edge_coverage;
}
