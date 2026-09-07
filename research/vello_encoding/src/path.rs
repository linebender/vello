// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use bytemuck::{Pod, Zeroable};
use peniko::Fill;
use peniko::kurbo::{Cap, Join, Shape, Stroke};

use super::Monoid;

/// Data structure encoding stroke or fill style.
#[derive(Clone, Copy, Debug, Zeroable, Pod, Default, PartialEq)]
#[repr(C)]
pub struct Style {
    /// Encodes the stroke and fill style parameters. This field is split into two 16-bit
    /// parts:
    ///
    /// - `flags: u16` - encodes fill vs stroke, even-odd vs non-zero fill
    ///   mode for fills and cap and join style for strokes. See the
    ///   `FLAGS_*` constants below for more information.
    ///
    ///   ```text
    ///   flags: |style|fill|join|start cap|end cap|reserved|
    ///    bits:  0     1    2-3  4-5       6-7     8-15
    ///   ```
    ///
    /// - `miter_limit: u16` - The miter limit for a stroke, encoded in
    ///   binary16 (half) floating point representation. This field is
    ///   only meaningful for the `Join::Miter` join style. It's ignored
    ///   for other stroke styles and fills.
    pub flags_and_miter_limit: u32,

    /// Encodes the stroke width. This field is ignored for fills.
    pub line_width: f32,
}

impl Style {
    /// 0 for a fill, 1 for a stroke
    pub const FLAGS_STYLE_BIT: u32 = 0x8000_0000;

    /// 0 for non-zero, 1 for even-odd
    pub const FLAGS_FILL_BIT: u32 = 0x4000_0000;

    /// Encodings for join style:
    ///    - 0b00 -> bevel
    ///    - 0b01 -> miter
    ///    - 0b10 -> round
    pub const FLAGS_JOIN_BITS_BEVEL: u32 = 0;
    pub const FLAGS_JOIN_BITS_MITER: u32 = 0x1000_0000;
    pub const FLAGS_JOIN_BITS_ROUND: u32 = 0x2000_0000;
    pub const FLAGS_JOIN_MASK: u32 = 0x3000_0000;

    /// Encodings for cap style:
    ///    - 0b00 -> butt
    ///    - 0b01 -> square
    ///    - 0b10 -> round
    pub const FLAGS_CAP_BITS_BUTT: u32 = 0;
    pub const FLAGS_CAP_BITS_SQUARE: u32 = 0x0100_0000;
    pub const FLAGS_CAP_BITS_ROUND: u32 = 0x0200_0000;

    pub const FLAGS_START_CAP_BITS_BUTT: u32 = Self::FLAGS_CAP_BITS_BUTT << 2;
    pub const FLAGS_START_CAP_BITS_SQUARE: u32 = Self::FLAGS_CAP_BITS_SQUARE << 2;
    pub const FLAGS_START_CAP_BITS_ROUND: u32 = Self::FLAGS_CAP_BITS_ROUND << 2;
    pub const FLAGS_END_CAP_BITS_BUTT: u32 = Self::FLAGS_CAP_BITS_BUTT;
    pub const FLAGS_END_CAP_BITS_SQUARE: u32 = Self::FLAGS_CAP_BITS_SQUARE;
    pub const FLAGS_END_CAP_BITS_ROUND: u32 = Self::FLAGS_CAP_BITS_ROUND;

    pub const FLAGS_START_CAP_MASK: u32 = 0x0C00_0000;
    pub const FLAGS_END_CAP_MASK: u32 = 0x0300_0000;
    pub const MITER_LIMIT_MASK: u32 = 0xFFFF;

    pub fn from_fill(fill: Fill) -> Self {
        let fill_bit = match fill {
            Fill::NonZero => 0,
            Fill::EvenOdd => Self::FLAGS_FILL_BIT,
        };
        Self {
            flags_and_miter_limit: fill_bit,
            line_width: 0.,
        }
    }

    /// Creates a style from a stroke.
    ///
    /// As it isn't meaningful to encode a zero width stroke, returns None if the width is zero.
    pub fn from_stroke(stroke: &Stroke) -> Option<Self> {
        if stroke.width == 0.0 {
            return None;
        }
        let style = Self::FLAGS_STYLE_BIT;
        let join = match stroke.join {
            Join::Bevel => Self::FLAGS_JOIN_BITS_BEVEL,
            Join::Miter => Self::FLAGS_JOIN_BITS_MITER,
            Join::Round => Self::FLAGS_JOIN_BITS_ROUND,
        };
        let start_cap = match stroke.start_cap {
            Cap::Butt => Self::FLAGS_START_CAP_BITS_BUTT,
            Cap::Square => Self::FLAGS_START_CAP_BITS_SQUARE,
            Cap::Round => Self::FLAGS_START_CAP_BITS_ROUND,
        };
        let end_cap = match stroke.end_cap {
            Cap::Butt => Self::FLAGS_END_CAP_BITS_BUTT,
            Cap::Square => Self::FLAGS_END_CAP_BITS_SQUARE,
            Cap::Round => Self::FLAGS_END_CAP_BITS_ROUND,
        };
        let miter_limit = crate::math::f32_to_f16(stroke.miter_limit as f32) as u32;
        Some(Self {
            flags_and_miter_limit: style | join | start_cap | end_cap | miter_limit,
            line_width: stroke.width as f32,
        })
    }

    #[cfg(test)]
    fn fill(self) -> Option<Fill> {
        if self.is_fill() {
            Some(
                if (self.flags_and_miter_limit & Self::FLAGS_FILL_BIT) == 0 {
                    Fill::NonZero
                } else {
                    Fill::EvenOdd
                },
            )
        } else {
            None
        }
    }

    #[cfg(test)]
    fn stroke_width(self) -> Option<f64> {
        if self.is_fill() {
            return None;
        }
        Some(self.line_width.into())
    }

    #[cfg(test)]
    fn stroke_join(self) -> Option<Join> {
        if self.is_fill() {
            return None;
        }
        let join = self.flags_and_miter_limit & Self::FLAGS_JOIN_MASK;
        Some(match join {
            Self::FLAGS_JOIN_BITS_BEVEL => Join::Bevel,
            Self::FLAGS_JOIN_BITS_MITER => Join::Miter,
            Self::FLAGS_JOIN_BITS_ROUND => Join::Round,
            _ => unreachable!("unsupported join encoding"),
        })
    }

    #[cfg(test)]
    fn stroke_start_cap(self) -> Option<Cap> {
        if self.is_fill() {
            return None;
        }
        let cap = self.flags_and_miter_limit & Self::FLAGS_START_CAP_MASK;
        Some(match cap {
            Self::FLAGS_START_CAP_BITS_BUTT => Cap::Butt,
            Self::FLAGS_START_CAP_BITS_SQUARE => Cap::Square,
            Self::FLAGS_START_CAP_BITS_ROUND => Cap::Round,
            _ => unreachable!("unsupported start cap encoding"),
        })
    }

    #[cfg(test)]
    fn stroke_end_cap(self) -> Option<Cap> {
        if self.is_fill() {
            return None;
        }
        let cap = self.flags_and_miter_limit & Self::FLAGS_END_CAP_MASK;
        Some(match cap {
            Self::FLAGS_END_CAP_BITS_BUTT => Cap::Butt,
            Self::FLAGS_END_CAP_BITS_SQUARE => Cap::Square,
            Self::FLAGS_END_CAP_BITS_ROUND => Cap::Round,
            _ => unreachable!("unsupported end cap encoding"),
        })
    }

    #[cfg(test)]
    fn stroke_miter_limit(self) -> Option<u16> {
        if self.is_fill() {
            return None;
        }
        Some((self.flags_and_miter_limit & Self::MITER_LIMIT_MASK) as u16)
    }

    #[cfg(test)]
    fn is_fill(self) -> bool {
        (self.flags_and_miter_limit & Self::FLAGS_STYLE_BIT) == 0
    }
}

/// Line segment (after flattening, before tiling).
#[derive(Clone, Copy, Debug, Zeroable, Pod, Default)]
#[repr(C)]
pub struct LineSoup {
    pub path_ix: u32,
    pub _padding: u32,
    pub p0: [f32; 2],
    pub p1: [f32; 2],
}

/// Line segment (after flattening, before tiling).
#[derive(Clone, Copy, Debug, Zeroable, Pod, Default)]
#[repr(C)]
pub struct SegmentCount {
    pub line_ix: u32,
    // This could more accurately be modeled as:
    //     segment_within_line: u16,
    //     segment_within_slice: u16,
    // However, here we mirror the way it's written in WGSL
    pub counts: u32,
}

/// Path segment.
#[derive(Clone, Copy, Debug, Zeroable, Pod, Default)]
#[repr(C)]
pub struct PathSegment {
    // Points are relative to tile origin
    pub point0: [f32; 2],
    pub point1: [f32; 2],
    pub y_edge: f32,
    pub _padding: u32,
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
    /// This is equivalent to `(PathSegmentType::LINE_TO | PathTag::F32_BIT)`.
    pub const LINE_TO_F32: Self = Self(0x9);

    /// 32-bit floating point quadratic segment.
    ///
    /// This is equivalent to `(PathSegmentType::QUAD_TO | PathTag::F32_BIT)`.
    pub const QUAD_TO_F32: Self = Self(0xa);

    /// 32-bit floating point cubic segment.
    ///
    /// This is equivalent to `(PathSegmentType::CUBIC_TO | PathTag::F32_BIT)`.
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

    /// Style setting.
    pub const STYLE: Self = Self(0x40);

    /// Bit that marks a segment that is the end of a subpath.
    pub const SUBPATH_END_BIT: u8 = 0x4;

    /// Bit for path segments that are represented as f32 values. If unset
    /// they are represented as i16.
    const F32_BIT: u8 = 0x8;

    /// Mask for bottom 3 bits that contain the [`PathSegmentType`].
    const SEGMENT_MASK: u8 = 0x3;

    /// Returns `true` if the tag is a segment.
    pub fn is_path_segment(self) -> bool {
        self.path_segment_type().0 != 0
    }

    /// Returns `true` if this is a 32-bit floating point segment.
    pub fn is_f32(self) -> bool {
        self.0 & Self::F32_BIT != 0
    }

    /// Returns `true` if this segment ends a subpath.
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
    /// Index into style stream.
    pub style_ix: u32,
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
        let style_size = (size_of::<Style>() / size_of::<u32>()) as u32;
        c.style_ix = (tag_word & (PathTag::STYLE.0 as u32 * 0x1010101)).count_ones() * style_size;
        c
    }

    /// Monoid combination.
    fn combine(&self, other: &Self) -> Self {
        Self {
            trans_ix: self.trans_ix + other.trans_ix,
            pathseg_ix: self.pathseg_ix + other.pathseg_ix,
            pathseg_offset: self.pathseg_offset + other.pathseg_offset,
            style_ix: self.style_ix + other.style_ix,
            path_ix: self.path_ix + other.path_ix,
        }
    }
}

/// Cubic path segment.
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct Cubic {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub p2: [f32; 2],
    pub p3: [f32; 2],
    pub stroke: [f32; 2],
    pub path_ix: u32,
    pub flags: u32,
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
    /// Style flags
    pub draw_flags: u32,
    /// Index into the transform stream.
    pub trans_ix: u32,
}

/// Tiled path object.
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
#[expect(
    clippy::partial_pub_fields,
    reason = "Padding is meaningless to manipulate directly"
)]
pub struct Path {
    /// Bounding box in tiles.
    pub bbox: [u32; 4],
    /// Offset (in u32s) to tile rectangle.
    pub tiles: u32,
    _padding: [u32; 3],
}

/// Tile object.
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct Tile {
    /// Accumulated backdrop at the left edge of the tile.
    pub backdrop: i32,
    /// An enum that holds either the count of the number of path
    /// segments in this tile, or an index to the beginning of an
    /// allocated slice of `PathSegment` objects. In the latter case,
    /// the bits are inverted.
    pub segment_count_or_ix: u32,
}

/// Encoder for path segments.
pub struct PathEncoder<'a> {
    tags: &'a mut Vec<PathTag>,
    data: &'a mut Vec<u32>,
    n_segments: &'a mut u32,
    n_paths: &'a mut u32,
    first_point: [f32; 2],
    first_start_tangent_end: [f32; 2],
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
    /// Creates a new path encoder for the specified path tags and data.
    ///
    /// If `is_fill` is true, ensures that all subpaths are closed. Otherwise, the path is treated
    /// as a stroke and an additional "stroke cap marker" segment is inserted at the end of every
    /// subpath.
    ///
    /// Stroke Encoding
    /// ---------------
    /// Every subpath within a stroked path is terminated with a "stroke cap marker" segment. This
    /// segment tells the GPU stroker whether to draw a cap or a join based on the topology of the
    /// path:
    ///
    /// 1. This marker segment is encoded as a `quad-to` (2 additional points) for an open path and
    ///    a `line-to` (1 additional point) for a closed path. An open path gets drawn with a start
    ///    and end cap. A closed path gets drawn with a single join in place of the caps where the
    ///    subpath's start and end control points meet.
    ///
    /// 2. The marker segment tells the GPU flattening stage how to render caps and joins while
    ///    processing each path segment in parallel. All subpaths end with the marker segment which
    ///    is the only segment that has the `SUBPATH_END_BIT` set to 1.
    ///
    ///    The algorithm is as follows:
    ///
    ///    - If a GPU thread is processing a regular segment (i.e. `SUBPATH_END_BIT` is 0), it
    ///      outputs the offset curves for the segment. If the segment is immediately followed by
    ///      the marker segment, then the same thread draws an end cap if the subpath is open
    ///      (i.e. the marker is a quad-to) or a join if the subpath is closed (i.e. the marker is
    ///      a line-to) using the tangent encoded in the marker segment.
    ///      If the segment is immediately followed by another regular segment, then the thread
    ///      draws a join using the start tangent of the neighboring segment.
    ///
    ///    - If a GPU thread is processing the marker segment (i.e. `SUBPATH_END_BIT` is 1), then
    ///      it draws a start cap using the information encoded in the segment IF the subpath is
    ///      open (i.e. the marker is a quad-to). If the subpath is closed (i.e. the marker is a
    ///      line-to), the thread draws nothing.
    pub fn new(
        tags: &'a mut Vec<PathTag>,
        data: &'a mut Vec<u32>,
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
            first_start_tangent_end: [0.0, 0.0],
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
        let bytes = bytemuck::cast_slice(&buf);
        if self.state == PathState::MoveTo {
            let new_len = self.data.len() - 2;
            self.data.truncate(new_len);
        } else if self.state == PathState::NonemptySubpath {
            if !self.is_fill {
                self.insert_stroke_cap_marker_segment(false);
            }
            if let Some(tag) = self.tags.last_mut() {
                tag.set_subpath_end();
            }
        }
        self.first_point = buf;
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
        if self.state == PathState::MoveTo {
            // Ensure that we don't end up with a zero-length start tangent.
            let Some((x, y)) = self.start_tangent_for_line((x, y)) else {
                return;
            };
            self.first_start_tangent_end = [x, y];
        }
        // Drop the segment if its length is zero
        if self.is_zero_length_segment((x, y), None, None) {
            return;
        }
        let buf = [x, y];
        let bytes = bytemuck::cast_slice(&buf);
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
        if self.state == PathState::MoveTo {
            // Ensure that we don't end up with a zero-length start tangent.
            let Some((x, y)) = self.start_tangent_for_quad((x1, y1), (x2, y2)) else {
                return;
            };
            self.first_start_tangent_end = [x, y];
        }
        // Drop the segment if its length is zero
        if self.is_zero_length_segment((x1, y1), Some((x2, y2)), None) {
            return;
        }
        let buf = [x1, y1, x2, y2];
        let bytes = bytemuck::cast_slice(&buf);
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
        if self.state == PathState::MoveTo {
            // Ensure that we don't end up with a zero-length start tangent.
            let Some((x, y)) = self.start_tangent_for_curve((x1, y1), (x2, y2), (x3, y3)) else {
                return;
            };
            self.first_start_tangent_end = [x, y];
        }
        // Drop the segment if its length is zero
        if self.is_zero_length_segment((x1, y1), Some((x2, y2)), Some((x3, y3))) {
            return;
        }
        let buf = [x1, y1, x2, y2, x3, y3];
        let bytes = bytemuck::cast_slice(&buf);
        self.data.extend_from_slice(bytes);
        self.tags.push(PathTag::CUBIC_TO_F32);
        self.state = PathState::NonemptySubpath;
        self.n_encoded_segments += 1;
    }

    /// Encodes an empty path (as placeholder for begin clip).
    pub(crate) fn empty_path(&mut self) {
        let coords = [0.0_f32, 0., 0., 0.];
        let bytes = bytemuck::cast_slice(&coords);
        self.data.extend_from_slice(bytes);
        self.tags.push(PathTag::LINE_TO_F32);
        self.n_encoded_segments += 1;
    }

    /// Closes the current subpath.
    pub fn close(&mut self) {
        match self.state {
            PathState::Start => return,
            PathState::MoveTo => {
                // If we close a new-opened path, delete it.
                let new_len = self.data.len() - 2;
                self.data.truncate(new_len);
                self.state = PathState::Start;
                return;
            }
            PathState::NonemptySubpath => (),
        }
        let len = self.data.len();
        if len < 2 {
            if cfg!(debug_assertions) {
                unreachable!("There is an open path, so there must be data.")
            }
            return;
        }
        let first_bytes = bytemuck::cast_slice(&self.first_point);
        if &self.data[len - 2..len] != first_bytes {
            self.data.extend_from_slice(first_bytes);
            self.tags.push(PathTag::LINE_TO_F32);
            self.n_encoded_segments += 1;
        }
        if !self.is_fill {
            self.insert_stroke_cap_marker_segment(true);
        }
        if let Some(tag) = self.tags.last_mut() {
            tag.set_subpath_end();
        }
        self.state = PathState::Start;
    }

    /// Encodes a shape.
    pub fn shape(&mut self, shape: &impl Shape) {
        self.path_elements(shape.path_elements(0.1));
    }

    /// Encodes a path iterator
    pub fn path_elements(&mut self, path: impl Iterator<Item = peniko::kurbo::PathEl>) {
        use peniko::kurbo::PathEl;
        for el in path {
            match el {
                PathEl::MoveTo(p0) => self.move_to(p0.x as f32, p0.y as f32),
                PathEl::LineTo(p0) => self.line_to(p0.x as f32, p0.y as f32),
                PathEl::QuadTo(p0, p1) => {
                    self.quad_to(p0.x as f32, p0.y as f32, p1.x as f32, p1.y as f32);
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
    /// If `insert_path_marker` is true, encodes the [`PathTag::PATH`] tag to signify
    /// the end of a complete path object. Setting this to false allows encoding
    /// multiple paths with differing transforms for a single draw object.
    pub fn finish(mut self, insert_path_marker: bool) -> u32 {
        if self.is_fill {
            self.close();
        }
        if self.state == PathState::MoveTo {
            let new_len = self.data.len() - 2;
            self.data.truncate(new_len);
        }
        if self.n_encoded_segments != 0 {
            if !self.is_fill && self.state == PathState::NonemptySubpath {
                self.insert_stroke_cap_marker_segment(false);
            }
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

    fn insert_stroke_cap_marker_segment(&mut self, is_closed: bool) {
        assert!(!self.is_fill);
        assert!(self.state == PathState::NonemptySubpath);
        if is_closed {
            // We expect that the most recently encoded pair of coordinates in the path data stream
            // contain the first control point in the path segment (see `PathEncoder::close`).
            // Hence a line-to encoded here should embed the subpath's start tangent.
            self.line_to(
                self.first_start_tangent_end[0],
                self.first_start_tangent_end[1],
            );
        } else {
            self.quad_to(
                self.first_point[0],
                self.first_point[1],
                self.first_start_tangent_end[0],
                self.first_start_tangent_end[1],
            );
        }
    }

    fn last_point(&self) -> Option<(f32, f32)> {
        let len = self.data.len();
        if len < 2 {
            return None;
        }
        Some((
            bytemuck::cast(self.data[len - 2]),
            bytemuck::cast(self.data[len - 1]),
        ))
    }

    fn is_zero_length_segment(
        &self,
        p1: (f32, f32),
        p2: Option<(f32, f32)>,
        p3: Option<(f32, f32)>,
    ) -> bool {
        let p0 = self.last_point().unwrap();
        let p2 = p2.unwrap_or(p1);
        let p3 = p3.unwrap_or(p1);

        let x_min = p0.0.min(p1.0.min(p2.0.min(p3.0)));
        let x_max = p0.0.max(p1.0.max(p2.0.max(p3.0)));
        let y_min = p0.1.min(p1.1.min(p2.1.min(p3.1)));
        let y_max = p0.1.max(p1.1.max(p2.1.max(p3.1)));

        !(x_max - x_min > EPSILON || y_max - y_min > EPSILON)
    }

    // Returns the end point of the start tangent of a curve starting at `(x0, y0)`, or `None` if the
    // curve is degenerate / has zero-length. The inputs are a sequence of control points that
    // represent a cubic Bezier.
    //
    // `self.first_point` is always treated as the first control point of the curve.
    fn start_tangent_for_curve(
        &self,
        p1: (f32, f32),
        p2: (f32, f32),
        p3: (f32, f32),
    ) -> Option<(f32, f32)> {
        let p0 = (self.first_point[0], self.first_point[1]);
        let pt = if (p1.0 - p0.0).abs() > EPSILON || (p1.1 - p0.1).abs() > EPSILON {
            p1
        } else if (p2.0 - p0.0).abs() > EPSILON || (p2.1 - p0.1).abs() > EPSILON {
            p2
        } else if (p3.0 - p0.0).abs() > EPSILON || (p3.1 - p0.1).abs() > EPSILON {
            p3
        } else {
            return None;
        };
        Some(pt)
    }

    /// Similar to [`Self::start_tangent_for_curve`] but for a line.
    fn start_tangent_for_line(&self, p1: (f32, f32)) -> Option<(f32, f32)> {
        let p0 = (self.first_point[0], self.first_point[1]);
        let pt = if (p1.0 - p0.0).abs() > EPSILON || (p1.1 - p0.1).abs() > EPSILON {
            (
                p0.0 + 1. / 3. * (p1.0 - p0.0),
                p0.1 + 1. / 3. * (p1.1 - p0.1),
            )
        } else {
            return None;
        };
        Some(pt)
    }

    // Similar to `start_tangent_for_curve` but for a quadratic Bézier.
    fn start_tangent_for_quad(&self, p1: (f32, f32), p2: (f32, f32)) -> Option<(f32, f32)> {
        let p0 = (self.first_point[0], self.first_point[1]);
        let pt = if (p1.0 - p0.0).abs() > EPSILON || (p1.1 - p0.1).abs() > EPSILON {
            (
                p1.0 + 1. / 3. * (p0.0 - p1.0),
                p1.1 + 1. / 3. * (p0.1 - p1.1),
            )
        } else if (p2.0 - p0.0).abs() > EPSILON || (p2.1 - p0.1).abs() > EPSILON {
            (
                p1.0 + 1. / 3. * (p2.0 - p1.0),
                p1.1 + 1. / 3. * (p2.1 - p1.1),
            )
        } else {
            return None;
        };
        Some(pt)
    }
}

impl skrifa::outline::OutlinePen for PathEncoder<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.move_to(x, y);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.line_to(x, y);
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.quad_to(cx0, cy0, x, y);
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.cubic_to(cx0, cy0, cx1, cy1, x, y);
    }

    fn close(&mut self) {
        self.close();
    }
}

const EPSILON: f32 = 1e-12;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_style() {
        assert_eq!(Some(Fill::NonZero), Style::from_fill(Fill::NonZero).fill());
        assert_eq!(Some(Fill::EvenOdd), Style::from_fill(Fill::EvenOdd).fill());
        assert_eq!(None, Style::from_stroke(&Stroke::default()).unwrap().fill());
    }

    #[test]
    fn test_stroke_style() {
        assert_eq!(None, Style::from_fill(Fill::NonZero).stroke_width());
        assert_eq!(None, Style::from_fill(Fill::EvenOdd).stroke_width());
        let caps = [Cap::Butt, Cap::Square, Cap::Round];
        let joins = [Join::Bevel, Join::Miter, Join::Round];
        for start in caps {
            for end in caps {
                for join in joins {
                    let stroke = Stroke::new(1.0)
                        .with_start_cap(start)
                        .with_end_cap(end)
                        .with_join(join)
                        .with_miter_limit(0.);
                    let encoded = Style::from_stroke(&stroke).unwrap();
                    assert_eq!(Some(stroke.width), encoded.stroke_width());
                    assert_eq!(Some(stroke.join), encoded.stroke_join());
                    assert_eq!(Some(stroke.start_cap), encoded.stroke_start_cap());
                    assert_eq!(Some(stroke.end_cap), encoded.stroke_end_cap());
                    assert_eq!(Some(0), encoded.stroke_miter_limit());
                }
            }
        }
    }
}
