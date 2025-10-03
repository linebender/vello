// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Layout of per-tile command list
// Initial allocation, in u32's.
const PTCL_INITIAL_ALLOC = 64u;
const PTCL_INCREMENT = 256u;

// Amount of space taken by jump
const PTCL_HEADROOM = 2u;

// Tags for PTCL commands
const CMD_END = 0u;
const CMD_FILL = 1u;
const CMD_STROKE = 2u;
const CMD_SOLID = 3u;
const CMD_COLOR = 5u;
const CMD_LIN_GRAD = 6u;
const CMD_RAD_GRAD = 7u;
const CMD_SWEEP_GRAD = 8u;
const CMD_IMAGE = 9u;
const CMD_BEGIN_CLIP = 10u;
const CMD_END_CLIP = 11u;
const CMD_JUMP = 12u;
const CMD_BLUR_RECT = 13u;

// The individual PTCL structs are written here, but read/write is by
// hand in the relevant shaders

struct CmdFill {
    size_and_rule: u32,
    seg_data: u32,
    backdrop: i32,
}

struct CmdStroke {
    tile: u32,
    half_width: f32,
}

struct CmdJump {
    new_ix: u32,
}

struct CmdColor {
    rgba_color: u32,
}

struct CmdBlurRect {
    // Solid fill color.
    rgba_color: u32,

    // 2x2 transformation matrix (inverse).
    matrx: vec4<f32>,
    // 2D translation (inverse)
    xlat: vec2<f32>,

    // Rounded rectangle properties.
    width: f32,
    height: f32,
    radius: f32,

    // Gaussian filter standard deviation
    std_dev: f32,
}

struct CmdLinGrad {
    index: u32,
    extend_mode: u32,
    line_x: f32,
    line_y: f32,
    line_c: f32,
}

struct CmdRadGrad {
    index: u32,
    extend_mode: u32,
    matrx: vec4<f32>,
    xlat: vec2<f32>,
    focal_x: f32,
    radius: f32,
    kind: u32,
    flags: u32,
}

struct CmdSweepGrad {
    index: u32,
    extend_mode: u32,
    matrx: vec4<f32>,
    xlat: vec2<f32>,
    t0: f32,
    t1: f32,
}

struct CmdImage {
    matrx: vec4<f32>,
    xlat: vec2<f32>,
    atlas_offset: vec2<f32>,
    extents: vec2<f32>,
    format: u32,
    x_extend_mode: u32,
    y_extend_mode: u32,
    quality: u32,
    alpha: f32,
    alpha_type: u32,
}

struct CmdEndClip {
    blend: u32,
    alpha: f32,
}
