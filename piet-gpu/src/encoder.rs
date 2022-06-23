// Copyright 2021 The piet-gpu authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

//! Low-level scene encoding.

use crate::{Blend, SceneStats, DRAWTAG_SIZE, TRANSFORM_SIZE};
use bytemuck::{Pod, Zeroable};
use piet_gpu_hal::BufWrite;

use crate::stages::{
    self, PathEncoder, Transform, DRAW_PART_SIZE, PATHSEG_PART_SIZE, TRANSFORM_PART_SIZE,
};

pub struct Encoder {
    transform_stream: Vec<stages::Transform>,
    tag_stream: Vec<u8>,
    pathseg_stream: Vec<u8>,
    linewidth_stream: Vec<f32>,
    drawtag_stream: Vec<u32>,
    drawdata_stream: Vec<u8>,
    n_path: u32,
    n_pathseg: u32,
    n_clip: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct EncodedSceneRef<'a, T: Copy + Pod> {
    pub transform_stream: &'a [T],
    pub tag_stream: &'a [u8],
    pub pathseg_stream: &'a [u8],
    pub linewidth_stream: &'a [f32],
    pub drawtag_stream: &'a [u32],
    pub drawdata_stream: &'a [u8],
    pub n_path: u32,
    pub n_pathseg: u32,
    pub n_clip: u32,
    pub ramp_data: &'a [u32],
}

impl<'a, T: Copy + Pod> EncodedSceneRef<'a, T> {
    pub(crate) fn stats(&self) -> SceneStats {
        SceneStats {
            n_drawobj: self.drawtag_stream.len(),
            drawdata_len: self.drawdata_stream.len(),
            n_transform: self.transform_stream.len(),
            linewidth_len: std::mem::size_of_val(self.linewidth_stream),
            pathseg_len: self.pathseg_stream.len(),
            n_pathtag: self.tag_stream.len(),

            n_path: self.n_path,
            n_pathseg: self.n_pathseg,
            n_clip: self.n_clip,
        }
    }

    pub fn write_scene(&self, buf: &mut BufWrite) {
        buf.extend_slice(&self.drawtag_stream);
        let n_drawobj = self.drawtag_stream.len();
        buf.fill_zero(padding(n_drawobj, DRAW_PART_SIZE as usize) * DRAWTAG_SIZE);
        buf.extend_slice(&self.drawdata_stream);
        buf.extend_slice(&self.transform_stream);
        let n_trans = self.transform_stream.len();
        buf.fill_zero(padding(n_trans, TRANSFORM_PART_SIZE as usize) * TRANSFORM_SIZE);
        buf.extend_slice(&self.linewidth_stream);
        buf.extend_slice(&self.tag_stream);
        let n_pathtag = self.tag_stream.len();
        buf.fill_zero(padding(n_pathtag, PATHSEG_PART_SIZE as usize));
        buf.extend_slice(&self.pathseg_stream);
    }
}

/// A scene fragment encoding a glyph.
///
/// This is a reduced version of the full encoder.
#[derive(Default)]
pub struct GlyphEncoder {
    tag_stream: Vec<u8>,
    pathseg_stream: Vec<u8>,
    drawtag_stream: Vec<u32>,
    drawdata_stream: Vec<u8>,
    n_path: u32,
    n_pathseg: u32,
}

// Tags for draw objects. See shader/drawtag.h for the authoritative source.
const DRAWTAG_FILLCOLOR: u32 = 0x44;
const DRAWTAG_FILLLINGRADIENT: u32 = 0x114;
const DRAWTAG_FILLRADGRADIENT: u32 = 0x2dc;
const DRAWTAG_BEGINCLIP: u32 = 0x05;
const DRAWTAG_ENDCLIP: u32 = 0x25;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct FillColor {
    rgba_color: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct FillLinGradient {
    index: u32,
    p0: [f32; 2],
    p1: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct FillRadGradient {
    index: u32,
    p0: [f32; 2],
    p1: [f32; 2],
    r0: f32,
    r1: f32,
}

#[allow(unused)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct FillImage {
    index: u32,
    // [i16; 2]
    offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Clip {
    blend: u32,
}

impl Encoder {
    pub fn new() -> Encoder {
        Encoder {
            transform_stream: vec![Transform::IDENTITY],
            tag_stream: Vec::new(),
            pathseg_stream: Vec::new(),
            linewidth_stream: vec![-1.0],
            drawtag_stream: Vec::new(),
            drawdata_stream: Vec::new(),
            n_path: 0,
            n_pathseg: 0,
            n_clip: 0,
        }
    }

    pub fn path_encoder(&mut self) -> PathEncoder {
        PathEncoder::new(&mut self.tag_stream, &mut self.pathseg_stream)
    }

    pub fn finish_path(&mut self, n_pathseg: u32) {
        self.n_path += 1;
        self.n_pathseg += n_pathseg;
    }

    pub fn transform(&mut self, transform: Transform) {
        self.tag_stream.push(0x20);
        self.transform_stream.push(transform);
    }

    // Swap the last two tags in the tag stream; used for transformed
    // gradients.
    pub fn swap_last_tags(&mut self) {
        let len = self.tag_stream.len();
        self.tag_stream.swap(len - 1, len - 2);
    }

    // -1.0 means "fill"
    pub fn linewidth(&mut self, linewidth: f32) {
        self.tag_stream.push(0x40);
        self.linewidth_stream.push(linewidth);
    }

    /// Encode a fill color draw object.
    ///
    /// This should be encoded after a path.
    pub fn fill_color(&mut self, rgba_color: u32) {
        self.drawtag_stream.push(DRAWTAG_FILLCOLOR);
        let element = FillColor { rgba_color };
        self.drawdata_stream.extend(bytemuck::bytes_of(&element));
    }

    /// Encode a fill linear gradient draw object.
    ///
    /// This should be encoded after a path.
    pub fn fill_lin_gradient(&mut self, index: u32, p0: [f32; 2], p1: [f32; 2]) {
        self.drawtag_stream.push(DRAWTAG_FILLLINGRADIENT);
        let element = FillLinGradient { index, p0, p1 };
        self.drawdata_stream.extend(bytemuck::bytes_of(&element));
    }

    /// Encode a fill radial gradient draw object.
    ///
    /// This should be encoded after a path.
    pub fn fill_rad_gradient(&mut self, index: u32, p0: [f32; 2], p1: [f32; 2], r0: f32, r1: f32) {
        self.drawtag_stream.push(DRAWTAG_FILLRADGRADIENT);
        let element = FillRadGradient {
            index,
            p0,
            p1,
            r0,
            r1,
        };
        self.drawdata_stream.extend(bytemuck::bytes_of(&element));
    }

    /// Start a clip.
    pub fn begin_clip(&mut self, blend: Option<Blend>) {
        self.drawtag_stream.push(DRAWTAG_BEGINCLIP);
        let element = Clip {
            blend: blend.unwrap_or(Blend::default()).pack(),
        };
        self.drawdata_stream.extend(bytemuck::bytes_of(&element));
        self.n_clip += 1;
    }

    pub fn end_clip(&mut self, blend: Option<Blend>) {
        self.drawtag_stream.push(DRAWTAG_ENDCLIP);
        let element = Clip {
            blend: blend.unwrap_or(Blend::default()).pack(),
        };
        self.drawdata_stream.extend(bytemuck::bytes_of(&element));
        // This is a dummy path, and will go away with the new clip impl.
        self.tag_stream.push(0x10);
        self.n_path += 1;
        self.n_clip += 1;
    }

    pub fn write_scene(&self, buf: &mut BufWrite) {
        buf.extend_slice(&self.drawtag_stream);
        let n_drawobj = self.drawtag_stream.len();
        buf.fill_zero(padding(n_drawobj, DRAW_PART_SIZE as usize) * DRAWTAG_SIZE);
        buf.extend_slice(&self.drawdata_stream);
        buf.extend_slice(&self.transform_stream);
        let n_trans = self.transform_stream.len();
        buf.fill_zero(padding(n_trans, TRANSFORM_PART_SIZE as usize) * TRANSFORM_SIZE);
        buf.extend_slice(&self.linewidth_stream);
        buf.extend_slice(&self.tag_stream);
        let n_pathtag = self.tag_stream.len();
        buf.fill_zero(padding(n_pathtag, PATHSEG_PART_SIZE as usize));
        buf.extend_slice(&self.pathseg_stream);
    }

    pub(crate) fn stats(&self) -> SceneStats {
        SceneStats {
            n_drawobj: self.drawtag_stream.len(),
            drawdata_len: self.drawdata_stream.len(),
            n_transform: self.transform_stream.len(),
            linewidth_len: std::mem::size_of_val(&*self.linewidth_stream),
            n_pathtag: self.tag_stream.len(),
            pathseg_len: self.pathseg_stream.len(),

            n_path: self.n_path,
            n_pathseg: self.n_pathseg,
            n_clip: self.n_clip,
        }
    }

    pub(crate) fn encode_glyph(&mut self, glyph: &GlyphEncoder) {
        self.tag_stream.extend(&glyph.tag_stream);
        self.pathseg_stream.extend(&glyph.pathseg_stream);
        self.drawtag_stream.extend(&glyph.drawtag_stream);
        self.drawdata_stream.extend(&glyph.drawdata_stream);
        self.n_path += glyph.n_path;
        self.n_pathseg += glyph.n_pathseg;
    }
}

fn padding(x: usize, align: usize) -> usize {
    x.wrapping_neg() & (align - 1)
}

impl GlyphEncoder {
    pub(crate) fn path_encoder(&mut self) -> PathEncoder {
        PathEncoder::new(&mut self.tag_stream, &mut self.pathseg_stream)
    }

    pub(crate) fn finish_path(&mut self, n_pathseg: u32) {
        self.n_path += 1;
        self.n_pathseg += n_pathseg;
    }

    /// Encode a fill color draw object.
    ///
    /// This should be encoded after a path.
    pub(crate) fn fill_color(&mut self, rgba_color: u32) {
        self.drawtag_stream.push(DRAWTAG_FILLCOLOR);
        let element = FillColor { rgba_color };
        self.drawdata_stream.extend(bytemuck::bytes_of(&element));
    }

    pub(crate) fn is_color(&self) -> bool {
        !self.drawtag_stream.is_empty()
    }
}
