// Copyright 2022 Google LLC
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

use bytemuck::{Pod, Zeroable};
use peniko::kurbo::Shape;

use super::Monoid;

/// Path segment.
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct PathSegment {
    pub origin: [f32; 2],
    pub delta: [f32; 2],
    pub y_edge: f32,
    pub next: u32,
}

/// Path segment type.
///
/// The values of the segment types are equivalent to the number of associated
/// points for each segment in the path data stream.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Pod, Zeroable)]
#[repr(C)]
pub struct PathSegmentType(pub u8);

impl PathSegmentType {
    /// Line segment.
    pub const LINE_TO: Self = Self(0x1);

    /// Quadratic segment.
    pub const QUAD_TO: Self = Self(0x2);

    /// Cubic segment.
    pub const CUBIC_TO: Self = Self(0x3);
}

/// Path tag representation.
#[derive(Copy, Clone, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct PathTag(pub u8);

impl PathTag {
    /// 32-bit floating point line segment.
    ///
    /// This is equivalent to (PathSegmentType::LINE_TO | PathTag::F32_BIT).
    pub const LINE_TO_F32: Self = Self(0x9);

    /// 32-bit floating point quadratic segment.
    ///
    /// This is equivalent to (PathSegmentType::QUAD_TO | PathTag::F32_BIT).
    pub const QUAD_TO_F32: Self = Self(0xa);

    /// 32-bit floating point cubic segment.
    ///
    /// This is equivalent to (PathSegmentType::CUBIC_TO | PathTag::F32_BIT).
    pub const CUBIC_TO_F32: Self = Self(0xb);

    /// 16-bit integral line segment.
    pub const LINE_TO_I16: Self = Self(0x1);

    /// 16-bit integral quadratic segment.
    pub const QUAD_TO_I16: Self = Self(0x2);

    /// 16-bit integral cubic segment.
    pub const CUBIC_TO_I16: Self = Self(0x3);

    /// Transform marker.
    pub const TRANSFORM: Self = Self(0x20);

    /// Path marker.
    pub const PATH: Self = Self(0x10);

    /// Line width setting.
    pub const LINEWIDTH: Self = Self(0x40);

    /// Bit for path segments that are represented as f32 values. If unset
    /// they are represented as i16.
    const F32_BIT: u8 = 0x8;

    /// Bit that marks a segment that is the end of a subpath.
    const SUBPATH_END_BIT: u8 = 0x4;

    /// Mask for bottom 3 bits that contain the [PathSegmentType].
    const SEGMENT_MASK: u8 = 0x3;

    /// Returns true if the tag is a segment.
    pub fn is_path_segment(self) -> bool {
        self.path_segment_type().0 != 0
    }

    /// Returns true if this is a 32-bit floating point segment.
    pub fn is_f32(self) -> bool {
        self.0 & Self::F32_BIT != 0
    }

    /// Returns true if this segment ends a subpath.
    pub fn is_subpath_end(self) -> bool {
        self.0 & Self::SUBPATH_END_BIT != 0
    }

    /// Sets the subpath end bit.
    pub fn set_subpath_end(&mut self) {
        self.0 |= Self::SUBPATH_END_BIT;
    }

    /// Returns the segment type.
    pub fn path_segment_type(self) -> PathSegmentType {
        PathSegmentType(self.0 & Self::SEGMENT_MASK)
    }
}

/// Monoid for the path tag stream.
#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct PathMonoid {
    /// Index into transform stream.
    pub trans_ix: u32,
    /// Path segment index.
    pub pathseg_ix: u32,
    /// Offset into path segment stream.
    pub pathseg_offset: u32,
    /// Index into linewidth stream.
    pub linewidth_ix: u32,
    /// Index of containing path.
    pub path_ix: u32,
}

impl Monoid for PathMonoid {
    type SourceValue = u32;

    /// Reduces a packed 32-bit word containing 4 tags.
    fn new(tag_word: u32) -> Self {
        let mut c = Self::default();
        let point_count = tag_word & 0x3030303;
        c.pathseg_ix = ((point_count * 7) & 0x4040404).count_ones();
        c.trans_ix = (tag_word & (PathTag::TRANSFORM.0 as u32 * 0x1010101)).count_ones();
        let n_points = point_count + ((tag_word >> 2) & 0x1010101);
        let mut a = n_points + (n_points & (((tag_word >> 3) & 0x1010101) * 15));
        a += a >> 8;
        a += a >> 16;
        c.pathseg_offset = a & 0xff;
        c.path_ix = (tag_word & (PathTag::PATH.0 as u32 * 0x1010101)).count_ones();
        c.linewidth_ix = (tag_word & (PathTag::LINEWIDTH.0 as u32 * 0x1010101)).count_ones();
        c
    }

    /// Monoid combination.
    fn combine(&self, other: &Self) -> Self {
        Self {
            trans_ix: self.trans_ix + other.trans_ix,
            pathseg_ix: self.pathseg_ix + other.pathseg_ix,
            pathseg_offset: self.pathseg_offset + other.pathseg_offset,
            linewidth_ix: self.linewidth_ix + other.linewidth_ix,
            path_ix: self.path_ix + other.path_ix,
        }
    }
}

/// Path bounding box.
#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct PathBbox {
    /// Minimum x value.
    pub x0: i32,
    /// Minimum y value.
    pub y0: i32,
    /// Maximum x value.
    pub x1: i32,
    /// Maximum y value.
    pub y1: i32,
    /// Line width.
    pub linewidth: f32,
    /// Index into the transform stream.
    pub trans_ix: u32,
}

/// Encoder for path segments.
pub struct PathEncoder<'a> {
    tags: &'a mut Vec<PathTag>,
    data: &'a mut Vec<u8>,
    n_segments: &'a mut u32,
    n_paths: &'a mut u32,
    first_point: [f32; 2],
    state: PathState,
    n_encoded_segments: u32,
    is_fill: bool,
}

#[derive(PartialEq)]
enum PathState {
    Start,
    MoveTo,
    NonemptySubpath,
}

impl<'a> PathEncoder<'a> {
    /// Creates a new path encoder for the specified path tags and data. If `is_fill` is true,
    /// ensures that all subpaths are closed.
    pub fn new(
        tags: &'a mut Vec<PathTag>,
        data: &'a mut Vec<u8>,
        n_segments: &'a mut u32,
        n_paths: &'a mut u32,
        is_fill: bool,
    ) -> Self {
        Self {
            tags,
            data,
            n_segments,
            n_paths,
            first_point: [0.0, 0.0],
            state: PathState::Start,
            n_encoded_segments: 0,
            is_fill,
        }
    }

    /// Encodes a move, starting a new subpath.
    pub fn move_to(&mut self, x: f32, y: f32) {
        if self.is_fill {
            self.close();
        }
        let buf = [x, y];
        let bytes = bytemuck::bytes_of(&buf);
        self.first_point = buf;
        if self.state == PathState::MoveTo {
            let new_len = self.data.len() - 8;
            self.data.truncate(new_len);
        } else if self.state == PathState::NonemptySubpath {
            if let Some(tag) = self.tags.last_mut() {
                tag.set_subpath_end();
            }
        }
        self.data.extend_from_slice(bytes);
        self.state = PathState::MoveTo;
    }

    /// Encodes a line.
    pub fn line_to(&mut self, x: f32, y: f32) {
        if self.state == PathState::Start {
            if self.n_encoded_segments == 0 {
                // This copies the behavior of kurbo which treats an initial line, quad
                // or curve as a move.
                self.move_to(x, y);
                return;
            }
            self.move_to(self.first_point[0], self.first_point[1]);
        }
        let buf = [x, y];
        let bytes = bytemuck::bytes_of(&buf);
        self.data.extend_from_slice(bytes);
        self.tags.push(PathTag::LINE_TO_F32);
        self.state = PathState::NonemptySubpath;
        self.n_encoded_segments += 1;
    }

    /// Encodes a quadratic bezier.
    pub fn quad_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
        if self.state == PathState::Start {
            if self.n_encoded_segments == 0 {
                self.move_to(x2, y2);
                return;
            }
            self.move_to(self.first_point[0], self.first_point[1]);
        }
        let buf = [x1, y1, x2, y2];
        let bytes = bytemuck::bytes_of(&buf);
        self.data.extend_from_slice(bytes);
        self.tags.push(PathTag::QUAD_TO_F32);
        self.state = PathState::NonemptySubpath;
        self.n_encoded_segments += 1;
    }

    /// Encodes a cubic bezier.
    pub fn cubic_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) {
        if self.state == PathState::Start {
            if self.n_encoded_segments == 0 {
                self.move_to(x3, y3);
                return;
            }
            self.move_to(self.first_point[0], self.first_point[1]);
        }
        let buf = [x1, y1, x2, y2, x3, y3];
        let bytes = bytemuck::bytes_of(&buf);
        self.data.extend_from_slice(bytes);
        self.tags.push(PathTag::CUBIC_TO_F32);
        self.state = PathState::NonemptySubpath;
        self.n_encoded_segments += 1;
    }

    /// Closes the current subpath.
    pub fn close(&mut self) {
        match self.state {
            PathState::Start => return,
            PathState::MoveTo => {
                let new_len = self.data.len() - 8;
                self.data.truncate(new_len);
                self.state = PathState::Start;
                return;
            }
            PathState::NonemptySubpath => (),
        }
        let len = self.data.len();
        if len < 8 {
            // can't happen
            return;
        }
        let first_bytes = bytemuck::bytes_of(&self.first_point);
        if &self.data[len - 8..len] != first_bytes {
            self.data.extend_from_slice(first_bytes);
            let mut tag = PathTag::LINE_TO_F32;
            tag.set_subpath_end();
            self.tags.push(tag);
            self.n_encoded_segments += 1;
        } else if let Some(tag) = self.tags.last_mut() {
            tag.set_subpath_end();
        }
        self.state = PathState::Start;
    }

    /// Encodes a shape.
    pub fn shape(&mut self, shape: &impl Shape) {
        use peniko::kurbo::PathEl;
        for el in shape.path_elements(0.1) {
            match el {
                PathEl::MoveTo(p0) => self.move_to(p0.x as f32, p0.y as f32),
                PathEl::LineTo(p0) => self.line_to(p0.x as f32, p0.y as f32),
                PathEl::QuadTo(p0, p1) => {
                    self.quad_to(p0.x as f32, p0.y as f32, p1.x as f32, p1.y as f32)
                }
                PathEl::CurveTo(p0, p1, p2) => self.cubic_to(
                    p0.x as f32,
                    p0.y as f32,
                    p1.x as f32,
                    p1.y as f32,
                    p2.x as f32,
                    p2.y as f32,
                ),
                PathEl::ClosePath => self.close(),
            }
        }
    }

    /// Completes path encoding and returns the actual number of encoded segments.
    ///
    /// If `insert_path_marker` is true, encodes the [PathTag::PATH] tag to signify
    /// the end of a complete path object. Setting this to false allows encoding
    /// multiple paths with differing transforms for a single draw object.
    pub fn finish(mut self, insert_path_marker: bool) -> u32 {
        if self.is_fill {
            self.close();
        }
        if self.state == PathState::MoveTo {
            let new_len = self.data.len() - 8;
            self.data.truncate(new_len);
        }
        if self.n_encoded_segments != 0 {
            if let Some(tag) = self.tags.last_mut() {
                tag.set_subpath_end();
            }
            *self.n_segments += self.n_encoded_segments;
            if insert_path_marker {
                self.tags.push(PathTag::PATH);
                *self.n_paths += 1;
            }
        }
        self.n_encoded_segments
    }
}
