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

use super::style::{Fill, Stroke};
use super::{
    Affine, BlendMode, FragmentResources, PathElement, ResourcePatch, Scene, SceneData,
    SceneFragment,
};
use crate::brush::*;
use crate::resource::ResourceContext;
use bytemuck::{Pod, Zeroable};
use core::borrow::Borrow;
use smallvec::SmallVec;

/// Builder for constructing a scene or scene fragment.
pub struct SceneBuilder<'a> {
    scene: &'a mut SceneData,
    resources: ResourceData<'a>,
    layers: SmallVec<[BlendMode; 8]>,
}

impl<'a> SceneBuilder<'a> {
    /// Creates a new builder for filling a scene. Any current content in the scene
    /// will be cleared.
    pub fn for_scene(scene: &'a mut Scene, rcx: &'a mut ResourceContext) -> Self {
        Self::new(&mut scene.data, ResourceData::Scene(rcx))
    }

    /// Creates a new builder for filling a scene fragment. Any current content in
    /// the fragment will be cleared.    
    pub fn for_fragment(fragment: &'a mut SceneFragment) -> Self {
        Self::new(
            &mut fragment.data,
            ResourceData::Fragment(&mut fragment.resources),
        )
    }

    /// Creates a new builder for constructing a scene.
    fn new(scene: &'a mut SceneData, mut resources: ResourceData<'a>) -> Self {
        let is_fragment = match resources {
            ResourceData::Fragment(_) => true,
            _ => false,
        };
        scene.reset(is_fragment);
        resources.clear();
        Self {
            scene,
            resources,
            layers: Default::default(),
        }
    }

    /// Sets the current transformation.
    pub fn transform(&mut self, transform: Affine) {
        if self.scene.transform_stream.last() != Some(&transform) {
            self.encode_transform(transform);
        }
    }

    /// Pushes a new layer bound by the specifed shape and composed with
    /// previous layers using the specified blend mode.
    pub fn push_layer<'s, E>(&mut self, blend: BlendMode, elements: E)
    where
        E: IntoIterator,
        E::IntoIter: Clone,
        E::Item: Borrow<PathElement>,
    {
        self.linewidth(-1.0);
        let elements = elements.into_iter();
        self.encode_path(elements, true);
        self.begin_clip(Some(blend));
        self.layers.push(blend);
    }

    /// Pops the current layer.
    pub fn pop_layer(&mut self) {
        if let Some(layer) = self.layers.pop() {
            self.end_clip(Some(layer));
        }
    }

    /// Fills a shape using the specified style and brush.
    pub fn fill<'s, E>(
        &mut self,
        _style: Fill,
        brush: &Brush,
        brush_transform: Option<Affine>,
        elements: E,
    ) where
        E: IntoIterator,
        E::IntoIter: Clone,
        E::Item: Borrow<PathElement>,
    {
        self.linewidth(-1.0);
        let elements = elements.into_iter();
        if self.encode_path(elements, true) {
            if let Some(brush_transform) = brush_transform {
                if let Some(last_transform) = self.scene.transform_stream.last().copied() {
                    self.encode_transform(brush_transform * last_transform);
                    self.swap_last_tags();
                    self.encode_brush(brush);
                    self.encode_transform(last_transform);
                } else {
                    self.encode_transform(brush_transform);
                    self.swap_last_tags();
                    self.encode_brush(brush);
                    self.encode_transform(Affine::IDENTITY);
                }
            } else {
                self.encode_brush(brush);
            }
        }
    }

    /// Strokes a shape using the specified style and brush.
    pub fn stroke<'s, D, E>(
        &mut self,
        style: &Stroke<D>,
        brush: &Brush,
        brush_transform: Option<Affine>,
        elements: E,
    ) where
        D: Borrow<[f32]>,
        E: IntoIterator,
        E::IntoIter: Clone,
        E::Item: Borrow<PathElement>,
    {
        self.linewidth(style.width);
        let elements = elements.into_iter();
        if self.encode_path(elements, false) {
            if let Some(brush_transform) = brush_transform {
                if let Some(last_transform) = self.scene.transform_stream.last().copied() {
                    self.encode_transform(brush_transform * last_transform);
                    self.swap_last_tags();
                    self.encode_brush(brush);
                    self.encode_transform(last_transform);
                } else {
                    self.encode_transform(brush_transform);
                    self.swap_last_tags();
                    self.encode_brush(brush);
                    self.encode_transform(Affine::IDENTITY);
                }
            } else {
                self.encode_brush(brush);
            }
        }
    }

    /// Appends a fragment to the scene.
    pub fn append(&mut self, fragment: &SceneFragment, transform: Option<Affine>) {
        let drawdata_base = self.scene.drawdata_stream.len();
        let mut cur_transform = self.scene.transform_stream.last().copied();
        if let Some(transform) = transform {
            if cur_transform.is_none() {
                cur_transform = Some(Affine::IDENTITY);
            }
            self.transform(transform);
        } else if cur_transform != Some(Affine::IDENTITY) {
            self.encode_transform(Affine::IDENTITY);
        }
        self.scene.append(&fragment.data, &transform);
        match &mut self.resources {
            ResourceData::Scene(res) => {
                for patch in &fragment.resources.patches {
                    match patch {
                        ResourcePatch::Ramp {
                            drawdata_offset,
                            stops,
                        } => {
                            let stops = &fragment.resources.stops[stops.clone()];
                            let ramp_id = res.add_ramp(stops);
                            let patch_base = *drawdata_offset + drawdata_base;
                            (&mut self.scene.drawdata_stream[patch_base..patch_base + 4])
                                .copy_from_slice(bytemuck::bytes_of(&ramp_id));
                        }
                    }
                }
            }
            ResourceData::Fragment(res) => {
                let stops_base = res.stops.len();
                res.stops.extend_from_slice(&fragment.resources.stops);
                res.patches.extend(fragment.resources.patches.iter().map(
                    |pending| match pending {
                        ResourcePatch::Ramp {
                            drawdata_offset,
                            stops,
                        } => ResourcePatch::Ramp {
                            drawdata_offset: drawdata_offset + drawdata_base,
                            stops: stops.start + stops_base..stops.end + stops_base,
                        },
                    },
                ));
            }
        }
        // Prevent fragments from affecting transform state. Should we allow this?
        if let Some(transform) = cur_transform {
            self.transform(transform);
        }
    }

    /// Completes construction and finalizes the underlying scene.
    pub fn finish(mut self) {
        while let Some(layer) = self.layers.pop() {
            self.end_clip(Some(layer));
        }
    }
}

impl<'a> SceneBuilder<'a> {
    fn encode_path<E>(&mut self, elements: E, is_fill: bool) -> bool
    where
        E: Iterator,
        E::Item: Borrow<PathElement>,
    {
        if is_fill {
            self.encode_path_inner(
                elements
                    .map(|el| *el.borrow())
                    .flat_map(|el| {
                        match el {
                            PathElement::MoveTo(..) => Some(PathElement::Close),
                            _ => None,
                        }
                        .into_iter()
                        .chain(Some(el))
                    })
                    .chain(Some(PathElement::Close)),
            )
        } else {
            self.encode_path_inner(elements.map(|el| *el.borrow()))
        }
    }

    fn encode_path_inner(&mut self, elements: impl Iterator<Item = PathElement>) -> bool {
        let mut b = PathBuilder::new(&mut self.scene.tag_stream, &mut self.scene.pathseg_stream);
        let mut has_els = false;
        for el in elements {
            match el {
                PathElement::MoveTo(p0) => b.move_to(p0.x, p0.y),
                PathElement::LineTo(p0) => b.line_to(p0.x, p0.y),
                PathElement::QuadTo(p0, p1) => b.quad_to(p0.x, p0.y, p1.x, p1.y),
                PathElement::CurveTo(p0, p1, p2) => b.cubic_to(p0.x, p0.y, p1.x, p1.y, p2.x, p2.y),
                PathElement::Close => b.close_path(),
            }
            has_els = true;
        }
        if has_els {
            b.path();
            let n_pathseg = b.n_pathseg();
            self.scene.n_path += 1;
            self.scene.n_pathseg += n_pathseg;
        }
        has_els
    }

    fn encode_transform(&mut self, transform: Affine) {
        self.scene.tag_stream.push(0x20);
        self.scene.transform_stream.push(transform);
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

    fn encode_brush(&mut self, brush: &Brush) {
        match brush {
            Brush::Solid(color) => {
                self.scene.drawtag_stream.push(DRAWTAG_FILLCOLOR);
                let rgba_color = color.to_premul_u32();
                self.scene
                    .drawdata_stream
                    .extend(bytemuck::bytes_of(&FillColor { rgba_color }));
            }
            Brush::LinearGradient(gradient) => {
                let index = self.add_ramp(&gradient.stops);
                self.scene.drawtag_stream.push(DRAWTAG_FILLLINGRADIENT);
                self.scene
                    .drawdata_stream
                    .extend(bytemuck::bytes_of(&FillLinGradient {
                        index,
                        p0: [gradient.start.x, gradient.start.y],
                        p1: [gradient.end.x, gradient.end.y],
                    }));
            }
            Brush::RadialGradient(gradient) => {
                let index = self.add_ramp(&gradient.stops);
                self.scene.drawtag_stream.push(DRAWTAG_FILLRADGRADIENT);
                self.scene
                    .drawdata_stream
                    .extend(bytemuck::bytes_of(&FillRadGradient {
                        index,
                        p0: [gradient.center0.x, gradient.center0.y],
                        p1: [gradient.center1.x, gradient.center1.y],
                        r0: gradient.radius0,
                        r1: gradient.radius1,
                    }));
            }
            Brush::SweepGradient(_gradient) => todo!("sweep gradients aren't done yet!"),
            Brush::Image(_image) => todo!("images aren't done yet!"),
        }
    }

    fn add_ramp(&mut self, stops: &[GradientStop]) -> u32 {
        match &mut self.resources {
            ResourceData::Scene(res) => res.add_ramp(stops),
            ResourceData::Fragment(res) => {
                let stops_start = res.stops.len();
                res.stops.extend_from_slice(stops);
                let id = res.patches.len() as u32;
                res.patches.push(ResourcePatch::Ramp {
                    drawdata_offset: self.scene.drawdata_stream.len(),
                    stops: stops_start..stops_start + stops.len(),
                });
                id
            }
        }
    }

    /// Start a clip.
    fn begin_clip(&mut self, blend: Option<BlendMode>) {
        self.scene.drawtag_stream.push(DRAWTAG_BEGINCLIP);
        let element = Clip {
            blend: blend.unwrap_or(BlendMode::default()).pack(),
        };
        self.scene
            .drawdata_stream
            .extend(bytemuck::bytes_of(&element));
        self.scene.n_clip += 1;
    }

    fn end_clip(&mut self, blend: Option<BlendMode>) {
        self.scene.drawtag_stream.push(DRAWTAG_ENDCLIP);
        let element = Clip {
            blend: blend.unwrap_or(BlendMode::default()).pack(),
        };
        self.scene
            .drawdata_stream
            .extend(bytemuck::bytes_of(&element));
        // This is a dummy path, and will go away with the new clip impl.
        self.scene.tag_stream.push(0x10);
        self.scene.n_path += 1;
        self.scene.n_clip += 1;
    }
}
enum ResourceData<'a> {
    Fragment(&'a mut FragmentResources),
    Scene(&'a mut ResourceContext),
}

impl ResourceData<'_> {
    fn clear(&mut self) {
        match self {
            Self::Fragment(res) => {
                res.patches.clear();
                res.stops.clear();
            }
            _ => {}
        }
    }
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

struct PathBuilder<'a> {
    tag_stream: &'a mut Vec<u8>,
    // If we're never going to use the i16 encoding, it might be
    // slightly faster to store this as Vec<u32>, we'd get aligned
    // stores on ARM etc.
    pathseg_stream: &'a mut Vec<u8>,
    first_pt: [f32; 2],
    state: PathState,
    n_pathseg: u32,
}

#[derive(PartialEq)]
enum PathState {
    Start,
    MoveTo,
    NonemptySubpath,
}

impl<'a> PathBuilder<'a> {
    pub fn new(tags: &'a mut Vec<u8>, pathsegs: &'a mut Vec<u8>) -> PathBuilder<'a> {
        PathBuilder {
            tag_stream: tags,
            pathseg_stream: pathsegs,
            first_pt: [0.0, 0.0],
            state: PathState::Start,
            n_pathseg: 0,
        }
    }

    pub fn move_to(&mut self, x: f32, y: f32) {
        let buf = [x, y];
        let bytes = bytemuck::bytes_of(&buf);
        self.first_pt = buf;
        if self.state == PathState::MoveTo {
            let new_len = self.pathseg_stream.len() - 8;
            self.pathseg_stream.truncate(new_len);
        }
        if self.state == PathState::NonemptySubpath {
            if let Some(tag) = self.tag_stream.last_mut() {
                *tag |= 4;
            }
        }
        self.pathseg_stream.extend_from_slice(bytes);
        self.state = PathState::MoveTo;
    }

    pub fn line_to(&mut self, x: f32, y: f32) {
        if self.state == PathState::Start {
            // should warn or error
            return;
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
            return;
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
            return;
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

    fn finish(&mut self) {
        if self.state == PathState::MoveTo {
            let new_len = self.pathseg_stream.len() - 8;
            self.pathseg_stream.truncate(new_len);
        }
        if let Some(tag) = self.tag_stream.last_mut() {
            *tag |= 4;
        }
    }

    /// Finish encoding a path.
    ///
    /// Encode this after encoding path segments.
    pub fn path(&mut self) {
        self.finish();
        // maybe don't encode if path is empty? might throw off sync though
        self.tag_stream.push(0x10);
    }

    /// Get the number of path segments.
    ///
    /// This is the number of path segments that will be written by the
    /// path stage; use this for allocating the output buffer.
    ///
    /// Also note: it takes `self` for lifetime reasons.
    pub fn n_pathseg(self) -> u32 {
        self.n_pathseg
    }
}
