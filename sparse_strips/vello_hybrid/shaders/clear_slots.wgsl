// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// This shader clears specific slots in slot textures to transparent pixels.

// Assumes this texture consists of a single column of slots of `config.slot_height`, 
// numbering from 0 to `texture_height / slot_height - 1` from top to bottom.

struct Config {
    // Width of a slot (typically matching `WideTile::WIDTH`)
    slot_width: u32,
    // Height of a slot (typically matching `Tile::HEIGHT`)
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

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    // Clear with transparent pixels
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
} 
