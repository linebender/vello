// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// This shader copies a wide tile slot from a slot texture to a target location.

struct Config {
    // Width of a wide tile (matching `WideTile::WIDTH`).
    wide_tile_width: u32,
    // Height of a wide tile (matching `WideTile::HEIGHT`).
    wide_tile_height: u32,
    // Height of the slot texture (source).
    slot_texture_height: u32,
    // Width of the target texture (destination).
    target_texture_width: u32,
    // Height of the target texture (destination).
    target_texture_height: u32,
}

struct CopyCommand {
    // [x, y] packed as u16's
    // x, y — coordinates of the top left of the target wide tile
    @location(0) xy_target: u32,
    // Slot index to identify the pixel position to sample from
    @location(1) slot_ix: u32,
}

struct VertexOutput {
    // Normalized device coordinates (NDC) for the current vertex
    @builtin(position) position: vec4<f32>,
    // Slot index passed to the fragment shader
    @location(0) @interpolate(flat) slot_ix: u32,
}

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var slot_texture: texture_2d<f32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    command: CopyCommand,
) -> VertexOutput {
    var out: VertexOutput;
    out.slot_ix = command.slot_ix;

    // Map vertex_index (0-3) to quad corners:
    // 0 → (0,0), 1 → (1,0), 2 → (0,1), 3 → (1,1)
    let x = f32(vertex_index & 1u);
    let y = f32(vertex_index >> 1u);
    
    // Unpack target coordinates
    let target_x0 = command.xy_target & 0xffffu;
    let target_y0 = command.xy_target >> 16u;
    
    // Calculate pixel coordinates of the current vertex within the wide tile
    let pix_x = f32(target_x0) + x * f32(config.wide_tile_width);
    let pix_y = f32(target_y0) + y * f32(config.wide_tile_height);
    
    // Convert to NDC for the target texture
    let ndc_x = pix_x * 2.0 / f32(config.target_texture_width) - 1.0;
    let ndc_y = 1.0 - pix_y * 2.0 / f32(config.target_texture_height);
    
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate the coordinates to sample from the slot texture
    let slot_x = u32(in.position.x) & 0xFFu;
    let slot_y = (u32(in.position.y) & 3u) + in.slot_ix * config.wide_tile_height;
    
    let color = textureLoad(slot_texture, vec2u(slot_x, slot_y), 0);
    
    if color.a == 0.0 && color.r > 0.0 && color.r <= (1.0 / 255.0) {
        discard;
    }
    return color;
}
