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

use bytemuck::{Pod, Zeroable};
use piet_gpu_hal::BufWrite;

use crate::stages::{self, Config, PathEncoder, Transform};

pub struct Encoder {
    transform_stream: Vec<stages::Transform>,
    tag_stream: Vec<u8>,
    pathseg_stream: Vec<u8>,
    linewidth_stream: Vec<f32>,
    drawobj_stream: Vec<u8>,
    n_path: u32,
    n_pathseg: u32,
}

// Currently same as Element, but may change - should become packed.
const DRAWOBJ_SIZE: usize = 36;
const TRANSFORM_SIZE: usize = 24;
const LINEWIDTH_SIZE: usize = 4;
const PATHSEG_SIZE: usize = 52;
const BBOX_SIZE: usize = 24;
const DRAWMONOID_SIZE: usize = 8;
const ANNOTATED_SIZE: usize = 40;

// Maybe pull these from the relevant stages? In any case, they may depend
// on runtime query of GPU (supported workgroup size).
const TRANSFORM_PART_SIZE: usize = 4096;
const PATHSEG_PART_SIZE: usize = 2048;
const DRAWOBJ_PART_SIZE: usize = 4096;

// These are bytemuck versions of elements currently defined in the
// Element struct in piet-gpu-types; that's pretty much going away.

const ELEMENT_FILLCOLOR: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct FillColor {
    tag: u32,
    rgba_color: u32,
    padding: [u32; 7],
}

impl Encoder {
    pub fn new() -> Encoder {
        Encoder {
            transform_stream: vec![Transform::IDENTITY],
            tag_stream: Vec::new(),
            pathseg_stream: Vec::new(),
            linewidth_stream: vec![-1.0],
            drawobj_stream: Vec::new(),
            n_path: 0,
            n_pathseg: 0,
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

    // -1.0 means "fill"
    pub fn linewidth(&mut self, linewidth: f32) {
        self.tag_stream.push(0x40);
        self.linewidth_stream.push(linewidth);
    }

    /// Encode a fill color draw object.
    ///
    /// This should be encoded after a path.
    pub fn fill_color(&mut self, rgba_color: u32) {
        let element = FillColor {
            tag: ELEMENT_FILLCOLOR,
            rgba_color,
            ..Default::default()
        };
        self.drawobj_stream.extend(bytemuck::bytes_of(&element));
    }

    /// Return a config for the element processing pipeline.
    ///
    /// This does not include further pipeline processing. Also returns the
    /// beginning of free memory.
    pub fn stage_config(&self) -> (Config, usize) {
        // Layout of scene buffer
        let n_drawobj = self.n_drawobj();
        let n_drawobj_padded = align_up(n_drawobj, DRAWOBJ_PART_SIZE);
        let trans_offset = n_drawobj_padded * DRAWOBJ_SIZE;
        let n_trans = self.transform_stream.len();
        let n_trans_padded = align_up(n_trans, TRANSFORM_PART_SIZE);
        let linewidth_offset = trans_offset + n_trans_padded * TRANSFORM_SIZE;
        let n_linewidth = self.linewidth_stream.len();
        let pathtag_offset = linewidth_offset + n_linewidth * LINEWIDTH_SIZE;
        let n_pathtag = self.tag_stream.len();
        let n_pathtag_padded = align_up(n_pathtag, PATHSEG_PART_SIZE);
        let pathseg_offset = pathtag_offset + n_pathtag_padded;

        // Layout of memory
        let mut alloc = 0;
        let trans_alloc = alloc;
        alloc += trans_alloc + n_trans_padded * TRANSFORM_SIZE;
        let pathseg_alloc = alloc;
        alloc += pathseg_alloc + self.n_pathseg as usize * PATHSEG_SIZE;
        let bbox_alloc = alloc;
        let n_path = self.n_path as usize;
        alloc += bbox_alloc + n_path * BBOX_SIZE;
        let drawmonoid_alloc = alloc;
        alloc += n_drawobj_padded * DRAWMONOID_SIZE;
        let anno_alloc = alloc;
        alloc += n_drawobj * ANNOTATED_SIZE;

        let config = Config {
            n_elements: n_drawobj as u32,
            n_pathseg: self.n_pathseg,
            pathseg_alloc: pathseg_alloc as u32,
            anno_alloc: anno_alloc as u32,
            trans_alloc: trans_alloc as u32,
            bbox_alloc: bbox_alloc as u32,
            drawmonoid_alloc: drawmonoid_alloc as u32,
            n_trans: n_trans as u32,
            n_path: self.n_path,
            trans_offset: trans_offset as u32,
            linewidth_offset: linewidth_offset as u32,
            pathtag_offset: pathtag_offset as u32,
            pathseg_offset: pathseg_offset as u32,
            ..Default::default()
        };
        (config, alloc)
    }

    pub fn write_scene(&self, buf: &mut BufWrite) {
        buf.extend_slice(&self.drawobj_stream);
        let n_drawobj = self.drawobj_stream.len() / DRAWOBJ_SIZE;
        buf.fill_zero(padding(n_drawobj, DRAWOBJ_PART_SIZE) * DRAWOBJ_SIZE);
        buf.extend_slice(&self.transform_stream);
        let n_trans = self.transform_stream.len();
        buf.fill_zero(padding(n_trans, TRANSFORM_PART_SIZE) * TRANSFORM_SIZE);
        buf.extend_slice(&self.linewidth_stream);
        buf.extend_slice(&self.tag_stream);
        let n_pathtag = self.tag_stream.len();
        buf.fill_zero(padding(n_pathtag, PATHSEG_PART_SIZE));
        buf.extend_slice(&self.pathseg_stream);
    }

    /// The number of elements in the draw object stream.
    pub(crate) fn n_drawobj(&self) -> usize {
        self.drawobj_stream.len() / DRAWOBJ_SIZE
    }

    /// The number of paths.
    pub(crate) fn n_path(&self) -> u32 {
        self.n_path
    }

    /// The number of path segments.
    pub(crate) fn n_pathseg(&self) -> u32 {
        self.n_pathseg
    }

    pub(crate) fn n_transform(&self) -> usize {
        self.transform_stream.len()
    }
}

fn align_up(x: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (x + align - 1) & !(align - 1)
}

fn padding(x: usize, align: usize) -> usize {
    x.wrapping_neg() & (align - 1)
}
