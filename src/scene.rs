// Copyright 2022 The piet-gpu authors.
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

use peniko::kurbo::{Affine, PathEl, Point, Rect, Shape};
use peniko::{BlendMode, BrushRef, ColorStop, Fill, Stroke};

use bytemuck::{Pod, Zeroable};
use std::ops::Range;

/// Raw data streams describing an encoded scene.
#[derive(Default)]
pub struct SceneData {
    pub transform_stream: Vec<[f32; 6]>,
    pub tag_stream: Vec<u8>,
    pub pathseg_stream: Vec<u8>,
    pub linewidth_stream: Vec<f32>,
    pub drawtag_stream: Vec<u32>,
    pub drawdata_stream: Vec<u8>,
    pub n_path: u32,
    pub n_pathseg: u32,
    pub n_clip: u32,
    pub resources: ResourceBundle,
}

impl SceneData {
    fn is_empty(&self) -> bool {
        self.pathseg_stream.is_empty()
    }

    fn reset(&mut self, is_fragment: bool) {
        self.transform_stream.clear();
        self.tag_stream.clear();
        self.pathseg_stream.clear();
        self.linewidth_stream.clear();
        self.drawtag_stream.clear();
        self.drawdata_stream.clear();
        self.n_path = 0;
        self.n_pathseg = 0;
        self.n_clip = 0;
        self.resources.clear();
        if !is_fragment {
            self.transform_stream.push([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
            self.linewidth_stream.push(-1.0);
        }
    }

    fn append(&mut self, other: &SceneData, transform: &Option<Affine>) {
        let stops_base = self.resources.stops.len();
        let drawdata_base = self.drawdata_stream.len();
        if let Some(transform) = *transform {
            self.transform_stream.extend(
                other
                    .transform_stream
                    .iter()
                    .map(|x| affine_to_f32(&(transform * affine_from_f32(x)))),
            );
        } else {
            self.transform_stream
                .extend_from_slice(&other.transform_stream);
        }
        self.tag_stream.extend_from_slice(&other.tag_stream);
        self.pathseg_stream.extend_from_slice(&other.pathseg_stream);
        self.linewidth_stream
            .extend_from_slice(&other.linewidth_stream);
        self.drawtag_stream.extend_from_slice(&other.drawtag_stream);
        self.drawdata_stream
            .extend_from_slice(&other.drawdata_stream);
        self.n_path += other.n_path;
        self.n_pathseg += other.n_pathseg;
        self.n_clip += other.n_clip;
        self.resources
            .stops
            .extend_from_slice(&other.resources.stops);
        self.resources
            .patches
            .extend(other.resources.patches.iter().map(|patch| match patch {
                ResourcePatch::Ramp { offset, stops } => {
                    let stops = stops.start + stops_base..stops.end + stops_base;
                    ResourcePatch::Ramp {
                        offset: drawdata_base + offset,
                        stops,
                    }
                }
            }));
    }
}

/// Encoded definition of a scene and associated resources.
#[derive(Default)]
pub struct Scene {
    data: SceneData,
}

impl Scene {
    /// Creates a new scene.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the raw encoded scene data streams.
    pub fn data(&self) -> &SceneData {
        &self.data
    }
}

/// Encoded definition of a scene fragment and associated resources.
#[derive(Default)]
pub struct SceneFragment {
    data: SceneData,
}

impl SceneFragment {
    /// Creates a new scene fragment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true if the fragment does not contain any paths.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the the entire sequence of points in the scene fragment.
    pub fn points(&self) -> &[[f32; 2]] {
        if self.is_empty() {
            &[]
        } else {
            bytemuck::cast_slice(&self.data.pathseg_stream)
        }
    }
}

#[derive(Default)]
/// Collection of late bound resources for a scene or scene fragment.
pub struct ResourceBundle {
    /// Sequence of resource patches.
    pub patches: Vec<ResourcePatch>,
    /// Cache of color stops, referenced by range from the patches.
    pub stops: Vec<ColorStop>,
}

impl ResourceBundle {
    /// Clears the resource set.
    pub(crate) fn clear(&mut self) {
        self.patches.clear();
        self.stops.clear();
    }
}

#[derive(Clone)]
/// Description of a late bound resource.
pub enum ResourcePatch {
    /// Gradient ramp resource.
    Ramp {
        /// Byte offset to the ramp id in the draw data stream.
        offset: usize,
        /// Range of the gradient stops in the resource set.
        stops: Range<usize>,
    },
}

/// Builder for constructing a scene or scene fragment.
pub struct SceneBuilder<'a> {
    scene: &'a mut SceneData,
    layer_depth: u32,
}

impl<'a> SceneBuilder<'a> {
    /// Creates a new builder for filling a scene. Any current content in the scene
    /// will be cleared.
    pub fn for_scene(scene: &'a mut Scene) -> Self {
        Self::new(&mut scene.data, false)
    }

    /// Creates a new builder for filling a scene fragment. Any current content in
    /// the fragment will be cleared.    
    pub fn for_fragment(fragment: &'a mut SceneFragment) -> Self {
        Self::new(&mut fragment.data, true)
    }

    /// Creates a new builder for constructing a scene.
    fn new(scene: &'a mut SceneData, is_fragment: bool) -> Self {
        scene.reset(is_fragment);
        Self {
            scene,
            layer_depth: 0,
        }
    }

    /// Pushes a new layer bound by the specifed shape and composed with
    /// previous layers using the specified blend mode.
    pub fn push_layer(
        &mut self,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        shape: &impl Shape,
    ) {
        let blend = blend.into();
        self.maybe_encode_transform(transform);
        self.linewidth(-1.0);
        if !self.encode_path(shape, true) {
            // If the layer shape is invalid, encode a valid empty path. This suppresses
            // all drawing until the layer is popped.
            self.encode_path(&Rect::new(0.0, 0.0, 0.0, 0.0), true);
        }
        self.begin_clip(blend, alpha.clamp(0.0, 1.0));
        self.layer_depth += 1;
    }

    /// Pops the current layer.
    pub fn pop_layer(&mut self) {
        if self.layer_depth > 0 {
            self.end_clip();
            self.layer_depth -= 1;
        }
    }

    /// Fills a shape using the specified style and brush.
    pub fn fill<'b>(
        &mut self,
        _style: Fill,
        transform: Affine,
        brush: impl Into<BrushRef<'b>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.maybe_encode_transform(transform);
        self.linewidth(-1.0);
        if self.encode_path(shape, true) {
            if let Some(brush_transform) = brush_transform {
                self.encode_transform(transform * brush_transform);
                self.swap_last_tags();
                self.encode_brush(brush);
            } else {
                self.encode_brush(brush);
            }
        }
    }

    /// Strokes a shape using the specified style and brush.
    pub fn stroke<'b>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        brush: impl Into<BrushRef<'b>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.maybe_encode_transform(transform);
        self.linewidth(style.width);
        if self.encode_path(shape, false) {
            if let Some(brush_transform) = brush_transform {
                self.encode_transform(transform * brush_transform);
                self.swap_last_tags();
                self.encode_brush(brush);
            } else {
                self.encode_brush(brush);
            }
        }
    }

    /// Appends a fragment to the scene.
    pub fn append(&mut self, fragment: &SceneFragment, transform: Option<Affine>) {
        self.scene.append(&fragment.data, &transform);
    }

    /// Completes construction and finalizes the underlying scene.
    pub fn finish(mut self) {
        for _ in 0..self.layer_depth {
            self.end_clip();
        }
    }
}

impl<'a> SceneBuilder<'a> {
    /// Encodes a path for the specified shape.
    ///
    /// When the `is_fill` parameter is true, closes any open subpaths by inserting
    /// a line to the start point of the subpath with the end segment bit set.
    fn encode_path(&mut self, shape: &impl Shape, is_fill: bool) -> bool {
        let mut b = PathBuilder::new(
            &mut self.scene.tag_stream,
            &mut self.scene.pathseg_stream,
            is_fill,
        );
        for el in shape.path_elements(0.1) {
            match el {
                PathEl::MoveTo(p0) => b.move_to(p0.x as f32, p0.y as f32),
                PathEl::LineTo(p0) => b.line_to(p0.x as f32, p0.y as f32),
                PathEl::QuadTo(p0, p1) => {
                    b.quad_to(p0.x as f32, p0.y as f32, p1.x as f32, p1.y as f32)
                }
                PathEl::CurveTo(p0, p1, p2) => b.cubic_to(
                    p0.x as f32,
                    p0.y as f32,
                    p1.x as f32,
                    p1.y as f32,
                    p2.x as f32,
                    p2.y as f32,
                ),
                PathEl::ClosePath => b.close_path(),
            }
        }
        b.finish();
        if b.n_pathseg != 0 {
            self.scene.n_path += 1;
            self.scene.n_pathseg += b.n_pathseg;
            true
        } else {
            false
        }
    }

    fn maybe_encode_transform(&mut self, transform: Affine) {
        if self.scene.transform_stream.last() != Some(&affine_to_f32(&transform)) {
            self.encode_transform(transform);
        }
    }

    fn encode_transform(&mut self, transform: Affine) {
        self.scene.tag_stream.push(0x20);
        self.scene.transform_stream.push(affine_to_f32(&transform));
    }

    // Swap the last two tags in the tag stream; used for transformed
    // gradients.
    fn swap_last_tags(&mut self) {
        let len = self.scene.tag_stream.len();
        self.scene.tag_stream.swap(len - 1, len - 2);
    }

    // -1.0 means "fill"
    fn linewidth(&mut self, linewidth: f32) {
        if self.scene.linewidth_stream.last() != Some(&linewidth) {
            self.scene.tag_stream.push(0x40);
            self.scene.linewidth_stream.push(linewidth);
        }
    }

    fn encode_brush<'b>(&mut self, brush: impl Into<BrushRef<'b>>) {
        match brush.into() {
            BrushRef::Solid(color) => {
                self.scene.drawtag_stream.push(DRAWTAG_FILLCOLOR);
                let rgba_color = color.to_premul_u32();
                self.scene
                    .drawdata_stream
                    .extend(bytemuck::bytes_of(&FillColor { rgba_color }));
            }
            BrushRef::LinearGradient(gradient) => {
                let index = self.add_ramp(&gradient.stops);
                self.scene.drawtag_stream.push(DRAWTAG_FILLLINGRADIENT);
                self.scene
                    .drawdata_stream
                    .extend(bytemuck::bytes_of(&FillLinGradient {
                        index,
                        p0: point_to_f32(gradient.start),
                        p1: point_to_f32(gradient.end),
                    }));
            }
            BrushRef::RadialGradient(gradient) => {
                let index = self.add_ramp(&gradient.stops);
                self.scene.drawtag_stream.push(DRAWTAG_FILLRADGRADIENT);
                self.scene
                    .drawdata_stream
                    .extend(bytemuck::bytes_of(&FillRadGradient {
                        index,
                        p0: point_to_f32(gradient.start_center),
                        p1: point_to_f32(gradient.end_center),
                        r0: gradient.start_radius,
                        r1: gradient.end_radius,
                    }));
            }
            BrushRef::SweepGradient(_gradient) => todo!("sweep gradients aren't done yet!"),
        }
    }

    fn add_ramp(&mut self, stops: &[ColorStop]) -> u32 {
        let offset = self.scene.drawdata_stream.len();
        let resources = &mut self.scene.resources;
        let stops_start = resources.stops.len();
        resources.stops.extend_from_slice(stops);
        resources.patches.push(ResourcePatch::Ramp {
            offset,
            stops: stops_start..stops_start + stops.len(),
        });
        0
    }

    /// Start a clip.
    fn begin_clip(&mut self, blend: BlendMode, alpha: f32) {
        self.scene.drawtag_stream.push(DRAWTAG_BEGINCLIP);
        let element = Clip {
            blend: encode_blend_mode(blend),
            alpha,
        };
        self.scene
            .drawdata_stream
            .extend(bytemuck::bytes_of(&element));
        self.scene.n_clip += 1;
    }

    fn end_clip(&mut self) {
        self.scene.drawtag_stream.push(DRAWTAG_ENDCLIP);
        // This is a dummy path, and will go away with the new clip impl.
        self.scene.tag_stream.push(0x10);
        self.scene.n_path += 1;
        self.scene.n_clip += 1;
    }
}

fn encode_blend_mode(mode: BlendMode) -> u32 {
    (mode.mix as u32) << 8 | mode.compose as u32
}

// Tags for draw objects. See shader/shared/drawtag.wgsl for the authoritative source.
const DRAWTAG_FILLCOLOR: u32 = 0x44;
const DRAWTAG_FILLLINGRADIENT: u32 = 0x114;
const DRAWTAG_FILLRADGRADIENT: u32 = 0x2dc;
const DRAWTAG_BEGINCLIP: u32 = 0x9;
const DRAWTAG_ENDCLIP: u32 = 0x21;

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
    alpha: f32,
}

struct PathBuilder<'a> {
    tag_stream: &'a mut Vec<u8>,
    // If we're never going to use the i16 encoding, it might be
    // slightly faster to store this as Vec<u32>, we'd get aligned
    // stores on ARM etc.
    pathseg_stream: &'a mut Vec<u8>,
    first_pt: [f32; 2],
    state: PathState,
    n_pathseg: u32,
    is_fill: bool,
}

#[derive(PartialEq)]
enum PathState {
    Start,
    MoveTo,
    NonemptySubpath,
}

impl<'a> PathBuilder<'a> {
    pub fn new(tags: &'a mut Vec<u8>, pathsegs: &'a mut Vec<u8>, is_fill: bool) -> PathBuilder<'a> {
        PathBuilder {
            tag_stream: tags,
            pathseg_stream: pathsegs,
            first_pt: [0.0, 0.0],
            state: PathState::Start,
            n_pathseg: 0,
            is_fill,
        }
    }

    pub fn move_to(&mut self, x: f32, y: f32) {
        if self.is_fill {
            self.close_path();
        }
        let buf = [x, y];
        let bytes = bytemuck::bytes_of(&buf);
        self.first_pt = buf;
        if self.state == PathState::MoveTo {
            let new_len = self.pathseg_stream.len() - 8;
            self.pathseg_stream.truncate(new_len);
        } else if self.state == PathState::NonemptySubpath {
            if let Some(tag) = self.tag_stream.last_mut() {
                *tag |= 4;
            }
        }
        self.pathseg_stream.extend_from_slice(bytes);
        self.state = PathState::MoveTo;
    }

    pub fn line_to(&mut self, x: f32, y: f32) {
        if self.state == PathState::Start {
            if self.n_pathseg == 0 {
                // This copies the behavior of kurbo which treats an initial line, quad
                // or curve as a move.
                self.move_to(x, y);
                return;
            }
            self.move_to(self.first_pt[0], self.first_pt[1]);
        }
        let buf = [x, y];
        let bytes = bytemuck::bytes_of(&buf);
        self.pathseg_stream.extend_from_slice(bytes);
        self.tag_stream.push(9);
        self.state = PathState::NonemptySubpath;
        self.n_pathseg += 1;
    }

    pub fn quad_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
        if self.state == PathState::Start {
            if self.n_pathseg == 0 {
                self.move_to(x2, y2);
                return;
            }
            self.move_to(self.first_pt[0], self.first_pt[1]);
        }
        let buf = [x1, y1, x2, y2];
        let bytes = bytemuck::bytes_of(&buf);
        self.pathseg_stream.extend_from_slice(bytes);
        self.tag_stream.push(10);
        self.state = PathState::NonemptySubpath;
        self.n_pathseg += 1;
    }

    pub fn cubic_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) {
        if self.state == PathState::Start {
            if self.n_pathseg == 0 {
                self.move_to(x3, y3);
                return;
            }
            self.move_to(self.first_pt[0], self.first_pt[1]);
        }
        let buf = [x1, y1, x2, y2, x3, y3];
        let bytes = bytemuck::bytes_of(&buf);
        self.pathseg_stream.extend_from_slice(bytes);
        self.tag_stream.push(11);
        self.state = PathState::NonemptySubpath;
        self.n_pathseg += 1;
    }

    pub fn close_path(&mut self) {
        match self.state {
            PathState::Start => return,
            PathState::MoveTo => {
                let new_len = self.pathseg_stream.len() - 8;
                self.pathseg_stream.truncate(new_len);
                self.state = PathState::Start;
                return;
            }
            PathState::NonemptySubpath => (),
        }
        let len = self.pathseg_stream.len();
        if len < 8 {
            // can't happen
            return;
        }
        let first_bytes = bytemuck::bytes_of(&self.first_pt);
        if &self.pathseg_stream[len - 8..len] != first_bytes {
            self.pathseg_stream.extend_from_slice(first_bytes);
            self.tag_stream.push(13);
            self.n_pathseg += 1;
        } else {
            if let Some(tag) = self.tag_stream.last_mut() {
                *tag |= 4;
            }
        }
        self.state = PathState::Start;
    }

    pub fn finish(&mut self) {
        if self.is_fill {
            self.close_path();
        }
        if self.state == PathState::MoveTo {
            let new_len = self.pathseg_stream.len() - 8;
            self.pathseg_stream.truncate(new_len);
        }
        if self.n_pathseg != 0 {
            if let Some(tag) = self.tag_stream.last_mut() {
                *tag |= 4;
            }
            self.tag_stream.push(0x10);
        }
    }
}

fn affine_to_f32(affine: &Affine) -> [f32; 6] {
    affine.as_coeffs().map(|value| value as f32)
}

fn affine_from_f32(coeffs: &[f32; 6]) -> Affine {
    Affine::new(coeffs.map(|value| value as f64))
}

fn point_to_f32(point: Point) -> [f32; 2] {
    [point.x as f32, point.y as f32]
}
