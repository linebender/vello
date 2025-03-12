// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::ops::Range;

use peniko::{
    kurbo::{BezPath, PathEl, Point, Vec2},
    Font, Style,
};
use skrifa::outline::OutlinePen;

use super::{StreamOffsets, Transform};

/// Positioned glyph.
#[derive(Copy, Clone, Default, Debug)]
pub struct Glyph {
    /// Glyph identifier.
    pub id: u32,
    /// X-offset in run, relative to transform.
    pub x: f32,
    /// Y-offset in run, relative to transform.
    pub y: f32,
}

/// Properties for a sequence of glyphs in an encoding.
#[derive(Clone)]
pub struct GlyphRun {
    /// Font for all glyphs in the run.
    pub font: Font,
    /// Global run transform.
    pub transform: Transform,
    /// Per-glyph transform.
    pub glyph_transform: Option<Transform>,
    /// Size of the font in pixels per em.
    pub font_size: f32,
    /// The amount to embolden in em.
    pub embolden: f32,
    /// True if hinting is enabled.
    pub hint: bool,
    /// Range of normalized coordinates in the parent encoding.
    pub normalized_coords: Range<usize>,
    /// Fill or stroke style.
    pub style: Style,
    /// Range of glyphs in the parent encoding.
    pub glyphs: Range<usize>,
    /// Stream offsets where this glyph run should be inserted.
    pub stream_offsets: StreamOffsets,
}

/// Direct implementation of Swash emboldening for glyph outlines
pub struct SwashEmboldenPen {
    /// Original path elements
    original_elements: Vec<PathEl>,
    /// Strength of emboldening
    x_strength: f64,
    y_strength: f64,
    /// Current subpath points for emboldening (reset at each MoveTo/ClosePath)
    current_subpath: Vec<Point>,
    /// Tracks which points are control points
    is_control_point: Vec<bool>,
    /// Tracks indices of path elements corresponding to current subpath
    current_path_elements: Vec<usize>,
}

impl SwashEmboldenPen {
    pub fn new(x_strength: f32, y_strength: f32) -> Self {
        Self {
            original_elements: Vec::new(),
            x_strength: x_strength as f64,
            y_strength: y_strength as f64,
            current_subpath: Vec::new(),
            is_control_point: Vec::new(),
            current_path_elements: Vec::new(),
        }
    }

    fn process_subpath(&mut self) {
        if self.current_subpath.is_empty() {
            return;
        }

        // Extract only endpoints for emboldening
        let mut endpoints = Vec::new();
        let mut endpoint_indices = Vec::new();

        for (i, (&is_control, _)) in self
            .is_control_point
            .iter()
            .zip(&self.current_subpath)
            .enumerate()
        {
            if !is_control {
                endpoints.push(self.current_subpath[i]);
                endpoint_indices.push(i);
            }
        }

        if !endpoints.is_empty() {
            let winding = compute_winding(&endpoints);
            embolden(&mut endpoints, winding, self.x_strength, self.y_strength);

            // Update the emboldened endpoints back to the subpath
            for (i, idx) in endpoint_indices.iter().enumerate() {
                self.current_subpath[*idx] = endpoints[i];
            }

            // Modify the original path elements with emboldened points
            let mut point_idx = 0;
            for &path_idx in &self.current_path_elements {
                match &mut self.original_elements[path_idx] {
                    PathEl::MoveTo(p) => {
                        if point_idx < self.current_subpath.len() {
                            *p = self.current_subpath[point_idx];
                            point_idx += 1;
                        }
                    }
                    PathEl::LineTo(p) => {
                        if point_idx < self.current_subpath.len() {
                            *p = self.current_subpath[point_idx];
                            point_idx += 1;
                        }
                    }
                    PathEl::QuadTo(cp, p) => {
                        if point_idx + 1 < self.current_subpath.len() {
                            *cp = self.current_subpath[point_idx];
                            *p = self.current_subpath[point_idx + 1];
                            point_idx += 2;
                        }
                    }
                    PathEl::CurveTo(cp1, cp2, p) => {
                        if point_idx + 2 < self.current_subpath.len() {
                            *cp1 = self.current_subpath[point_idx];
                            *cp2 = self.current_subpath[point_idx + 1];
                            *p = self.current_subpath[point_idx + 2];
                            point_idx += 3;
                        }
                    }
                    PathEl::ClosePath => {
                        // No point to update
                    }
                }
            }
        }

        self.current_subpath.clear();
        self.is_control_point.clear();
        self.current_path_elements.clear();
    }

    pub fn process(mut self) -> BezPath {
        self.process_subpath();

        BezPath::from_vec(self.original_elements)
    }
}

pub(crate) struct SwashEmboldenEncoderPen<'a, 'b> {
    embolden_pen: SwashEmboldenPen,
    path_encoder: &'a mut super::PathEncoder<'b>,
}

impl<'a, 'b> SwashEmboldenEncoderPen<'a, 'b> {
    pub(crate) fn new(
        path_encoder: &'a mut super::PathEncoder<'b>,
        x_strength: f32,
        y_strength: f32,
    ) -> Self {
        Self {
            embolden_pen: SwashEmboldenPen::new(x_strength, y_strength),
            path_encoder,
        }
    }

    pub(crate) fn process(&mut self) {
        self.embolden_pen.process_subpath();

        for el in &self.embolden_pen.original_elements {
            match el {
                PathEl::MoveTo(p) => {
                    self.path_encoder.move_to(p.x as f32, p.y as f32);
                }
                PathEl::LineTo(p) => {
                    self.path_encoder.line_to(p.x as f32, p.y as f32);
                }
                PathEl::QuadTo(cp, p) => {
                    self.path_encoder
                        .quad_to(cp.x as f32, cp.y as f32, p.x as f32, p.y as f32);
                }
                PathEl::CurveTo(cp1, cp2, p) => {
                    self.path_encoder.curve_to(
                        cp1.x as f32,
                        cp1.y as f32,
                        cp2.x as f32,
                        cp2.y as f32,
                        p.x as f32,
                        p.y as f32,
                    );
                }
                PathEl::ClosePath => {
                    self.path_encoder.close();
                }
            }
        }
    }
}

impl OutlinePen for SwashEmboldenPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.process_subpath();

        let element_idx = self.original_elements.len();
        self.current_path_elements.push(element_idx);

        let point = Point::new(x as f64, y as f64);
        self.original_elements.push(PathEl::MoveTo(point));

        self.current_subpath.push(point);
        self.is_control_point.push(false);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let element_idx = self.original_elements.len();
        self.current_path_elements.push(element_idx);

        let point = Point::new(x as f64, y as f64);
        self.original_elements.push(PathEl::LineTo(point));

        self.current_subpath.push(point);
        self.is_control_point.push(false);
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        let element_idx = self.original_elements.len();
        self.current_path_elements.push(element_idx);

        let cp = Point::new(cx0 as f64, cy0 as f64);
        let point = Point::new(x as f64, y as f64);
        self.original_elements.push(PathEl::QuadTo(cp, point));

        self.current_subpath.push(cp);
        self.is_control_point.push(true);

        self.current_subpath.push(point);
        self.is_control_point.push(false);
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        let element_idx = self.original_elements.len();
        self.current_path_elements.push(element_idx);

        let cp1 = Point::new(cx0 as f64, cy0 as f64);
        let cp2 = Point::new(cx1 as f64, cy1 as f64);
        let point = Point::new(x as f64, y as f64);
        self.original_elements
            .push(PathEl::CurveTo(cp1, cp2, point));

        self.current_subpath.push(cp1);
        self.is_control_point.push(true);

        self.current_subpath.push(cp2);
        self.is_control_point.push(true);

        self.current_subpath.push(point);
        self.is_control_point.push(false);
    }

    fn close(&mut self) {
        let element_idx = self.original_elements.len();
        self.current_path_elements.push(element_idx);

        self.original_elements.push(PathEl::ClosePath);

        self.process_subpath();
    }
}

impl OutlinePen for SwashEmboldenEncoderPen<'_, '_> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.embolden_pen.move_to(x, y);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.embolden_pen.line_to(x, y);
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.embolden_pen.quad_to(cx0, cy0, x, y);
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.embolden_pen.curve_to(cx0, cy0, cx1, cy1, x, y);
    }

    fn close(&mut self) {
        self.embolden_pen.close();
    }
}

fn compute_winding(points: &[Point]) -> u8 {
    if points.is_empty() {
        return 0;
    }
    let mut area = 0.;
    let last = points.len() - 1;
    let mut prev = points[last];
    for cur in points[0..=last].iter() {
        area += (cur.y - prev.y) * (cur.x + prev.x);
        prev = *cur;
    }
    if area > 0. {
        1
    } else {
        0
    }
}

/// Apply emboldening to a slice of points - direct port of Swash algorithm
fn embolden(points: &mut [Point], winding: u8, x_strength: f64, y_strength: f64) {
    if points.is_empty() {
        return;
    }

    let last = points.len() - 1;
    let mut i = last;
    let mut j = 0;
    let mut k = !0;
    let mut out_len;
    let mut in_len = 0.;
    let mut anchor_len = 0.;
    let mut anchor = Vec2::new(0., 0.);
    let mut out;
    let mut in_ = Vec2::new(0., 0.);

    while j != i && i != k {
        if j != k {
            out = points[j] - points[i];
            out_len = out.length();

            if out_len == 0. {
                j = if j < last { j + 1 } else { 0 };
                continue;
            } else {
                let s = 1.0 / out_len;
                out.x *= s;
                out.y *= s;
            }
        } else {
            out = anchor;
            out_len = anchor_len;
        }

        if in_len != 0. {
            if k == !0 {
                k = i;
                anchor = in_;
                anchor_len = in_len;
            }

            let mut d = in_.x * out.x + in_.y * out.y;
            let shift = if d > -0.9396 {
                d += 1.;
                let mut sx = in_.y + out.y;
                let mut sy = in_.x + out.x;

                if winding == 0 {
                    sx = -sx;
                } else {
                    sy = -sy;
                }

                let mut q = (out.x * in_.y) - (out.y * in_.x);
                if winding == 0 {
                    q = -q;
                }

                let l = in_len.min(out_len);
                if x_strength * q <= l * d {
                    sx = sx * x_strength / d;
                } else {
                    sx = sx * l / q;
                }

                if y_strength * q <= l * d {
                    sy = sy * y_strength / d;
                } else {
                    sy = sy * l / q;
                }

                Vec2::new(sx, sy)
            } else {
                Vec2::new(0.0, 0.0)
            };

            while i != j {
                points[i].x += x_strength + shift.x;
                points[i].y += y_strength + shift.y;
                i = if i < last { i + 1 } else { 0 };
            }
        } else {
            i = j;
        }

        in_ = out;
        in_len = out_len;
        j = if j < last { j + 1 } else { 0 };
    }
}
