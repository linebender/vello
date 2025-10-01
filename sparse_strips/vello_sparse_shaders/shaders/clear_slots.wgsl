// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// This vertex shader clears specific slots in slot textures to transparent pixels.

// Assumes this texture consists of a single column of slots of `config.slot_height`, 
// numbering from 0 to `texture_height / slot_height - 1` from top to bottom.

struct Config {
    // Width of a slot (matching `WideTile::WIDTH` and the width of a slot texture).
    slot_width: u32,
    // Height of a slot (matching `Tile::HEIGHT`)
    slot_height: u32,
    // Total height of the texture (slot_height * number_of_slots)
    texture_height: u32,
    // Padding for 16-byte alignment
    _padding: u32,
}

@group(0) @binding(0)
var<uniform> config: Config;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) index: u32,
) -> @builtin(position) vec4<f32> {
    // Map vertex_index (0-3) to quad corners:
    // 0 → (0,0), 1 → (1,0), 2 → (0,1), 3 → (1,1)
    let x = f32(vertex_index & 1u);
    let y = f32(vertex_index >> 1u);
    
    // Calculate the y-position based on the slot index
    let slot_y_offset = f32(index * config.slot_height);
    
    // Scale to match slot dimensions
    let pix_x = x * f32(config.slot_width);
    let pix_y = slot_y_offset + y * f32(config.slot_height);
    
    // Convert to NDC
    let ndc_x = pix_x * 2.0 / f32(config.slot_width) - 1.0;
    let ndc_y = 1.0 - pix_y * 2.0 / f32(config.texture_height);
    
    return vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
}

// This vertex shader is used for clearing atlas regions.
@vertex
fn vs_main_fullscreen(
    @builtin(vertex_index) vertex_index: u32,
) -> @builtin(position) vec4<f32> {
    // This generates a quad that covers the entire render target (hence "fullscreen"),
    // but the actual clearing region is controlled by the scissor test set on the render pass.
    // This approach is more efficient than generating region-specific geometry because:
    // 1. No vertex buffer needed - geometry generated from vertex index alone
    // 2. Same simple shader works for any region size
    // 3. Hardware scissor test efficiently clips to the desired region
    // 4. GPU rasterizer + scissor test is faster than complex vertex calculations
    //
    // Map vertex_index (0-3) to fullscreen quad corners in NDC:
    // 0 → (-1,-1), 1 → (1,-1), 2 → (-1,1), 3 → (1,1)
    let x = f32((vertex_index & 1u) * 2u) - 1.0; // 0->-1, 1->1
    let y = f32((vertex_index & 2u)) - 1.0;      // 0->-1, 2->1
    
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    // Clear with transparent pixels
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
} 
