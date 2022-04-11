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

use super::geometry::{Point, Rect};

/// Action of a path element.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Verb {
    MoveTo,
    LineTo,
    QuadTo,
    CurveTo,
    Close,
}

/// Element of a path represented by a verb and its associated points.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Element {
    MoveTo(Point),
    LineTo(Point),
    QuadTo(Point, Point),
    CurveTo(Point, Point, Point),
    Close,
}

impl Element {
    /// Returns the verb that describes the action of the path element.
    pub fn verb(&self) -> Verb {
        match self {
            Self::MoveTo(..) => Verb::MoveTo,
            Self::LineTo(..) => Verb::LineTo,
            Self::QuadTo(..) => Verb::QuadTo,
            Self::CurveTo(..) => Verb::CurveTo,
            Self::Close => Verb::Close,
        }
    }
}

/// Encoded collection of path elements.
#[derive(Clone, Default, Debug)]
pub struct Path {
    tag_stream: Vec<u8>,
    pathseg_stream: Vec<u8>,
    n_path: u32,
    n_pathseg: u32,
}

impl Path {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn elements(&self) -> Elements {
        Elements::new(&self)
    }
}

#[derive(Clone)]
pub struct Elements<'a> {
    tag_stream: &'a [u8],
    points: &'a [[f32; 2]],
    tag_ix: usize,
    point_ix: usize,
    next_element: Option<Element>,
    close: bool,
}

impl<'a> Elements<'a> {
    fn new(path: &'a Path) -> Self {
        let points: &'a [[f32; 2]] = bytemuck::cast_slice(&path.pathseg_stream);
        let (point_ix, next_element) = match points.get(0) {
            Some(&point) => (1, Some(Element::MoveTo(point.into()))),
            None => (0, None),
        };
        Self {
            tag_stream: &path.tag_stream,
            points,
            tag_ix: 0,
            point_ix,
            next_element,
            close: false,
        }
    }
}

impl<'a> Iterator for Elements<'a> {
    type Item = Element;

    fn next(&mut self) -> Option<Self::Item> {
        // println!("n_points: {}", self.points.len());
        // println!("tag_ix: {}, point_ix: {}, el: {:?}, close: {}", self.tag_ix, self.point_ix, self.next_element, self.close);
        if self.close {
            self.close = false;
            return Some(Element::Close);
        }
        if let Some(next_el) = self.next_element.take() {
            return Some(next_el);
        }
        let tag = *self.tag_stream.get(self.tag_ix)?;
        self.tag_ix += 1;
        let end = tag & 4 != 0;
        let el = match tag & 3 {
            1 => {
                let p0 = *self.points.get(self.point_ix)?;
                self.point_ix += 1;
                Element::LineTo(p0.into())
            }
            2 => {
                let p0 = *self.points.get(self.point_ix)?;
                let p1 = *self.points.get(self.point_ix + 1)?;
                self.point_ix += 2;
                Element::QuadTo(p0.into(), p1.into())
            }
            3 => {
                let p0 = *self.points.get(self.point_ix)?;
                let p1 = *self.points.get(self.point_ix + 1)?;
                let p2 = *self.points.get(self.point_ix + 2)?;
                self.point_ix += 3;
                Element::CurveTo(p0.into(), p1.into(), p2.into())
            }
            _ => return None,
        };
        if end {
            // println!("END!");
            if let Some(&p0) = self.points.get(self.point_ix) {
                self.point_ix += 1;
                self.next_element = Some(Element::MoveTo(p0.into()));
            }
            self.close = tag & 0x80 != 0;
        }
        Some(el)
    }
}

pub struct PathBuilder<'a> {
    tag_stream: &'a mut Vec<u8>,
    // If we're never going to use the i16 encoding, it might be
    // slightly faster to store this as Vec<u32>, we'd get aligned
    // stores on ARM etc.
    pathseg_stream: &'a mut Vec<u8>,
    first_pt: [f32; 2],
    state: State,
    n_pathseg: u32,
}

#[derive(PartialEq)]
enum State {
    Start,
    MoveTo,
    NonemptySubpath,
}

impl<'a> PathBuilder<'a> {
    pub fn new(path: &'a mut Path) -> Self {
        Self {
            tag_stream: &mut path.tag_stream,
            pathseg_stream: &mut path.pathseg_stream,
            first_pt: [0.0, 0.0],
            state: State::Start,
            n_pathseg: 0,
        }
    }

    fn new_inner(tags: &'a mut Vec<u8>, pathsegs: &'a mut Vec<u8>) -> PathBuilder<'a> {
        PathBuilder {
            tag_stream: tags,
            pathseg_stream: pathsegs,
            first_pt: [0.0, 0.0],
            state: State::Start,
            n_pathseg: 0,
        }
    }

    pub fn move_to(&mut self, x: f32, y: f32) {
        let buf = [x, y];
        let bytes = bytemuck::bytes_of(&buf);
        self.first_pt = buf;
        if self.state == State::MoveTo {
            let new_len = self.pathseg_stream.len() - 8;
            self.pathseg_stream.truncate(new_len);
        }
        if self.state == State::NonemptySubpath {
            if let Some(tag) = self.tag_stream.last_mut() {
                *tag |= 4;
            }
        }
        self.pathseg_stream.extend_from_slice(bytes);
        self.state = State::MoveTo;
    }

    pub fn line_to(&mut self, x: f32, y: f32) {
        if self.state == State::Start {
            // should warn or error
            return;
        }
        let buf = [x, y];
        let bytes = bytemuck::bytes_of(&buf);
        self.pathseg_stream.extend_from_slice(bytes);
        self.tag_stream.push(9);
        self.state = State::NonemptySubpath;
        self.n_pathseg += 1;
    }

    pub fn quad_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
        if self.state == State::Start {
            return;
        }
        let buf = [x1, y1, x2, y2];
        let bytes = bytemuck::bytes_of(&buf);
        self.pathseg_stream.extend_from_slice(bytes);
        self.tag_stream.push(10);
        self.state = State::NonemptySubpath;
        self.n_pathseg += 1;
    }

    pub fn cubic_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) {
        if self.state == State::Start {
            return;
        }
        let buf = [x1, y1, x2, y2, x3, y3];
        let bytes = bytemuck::bytes_of(&buf);
        self.pathseg_stream.extend_from_slice(bytes);
        self.tag_stream.push(11);
        self.state = State::NonemptySubpath;
        self.n_pathseg += 1;
    }

    pub fn close_path(&mut self) {
        match self.state {
            State::Start => return,
            State::MoveTo => {
                let new_len = self.pathseg_stream.len() - 8;
                self.pathseg_stream.truncate(new_len);
                self.state = State::Start;
                return;
            }
            State::NonemptySubpath => (),
        }
        let len = self.pathseg_stream.len();
        if len < 8 {
            // can't happen
            return;
        }
        let first_bytes = bytemuck::bytes_of(&self.first_pt);
        if &self.pathseg_stream[len - 8..len] != first_bytes {
            self.pathseg_stream.extend_from_slice(first_bytes);
            self.tag_stream.push(0x80 | 13);
            self.n_pathseg += 1;
        } else {
            if let Some(tag) = self.tag_stream.last_mut() {
                *tag |= 0x80 | 4;
            }
        }
        self.state = State::Start;
    }

    fn finish(&mut self) {
        if self.state == State::MoveTo {
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

impl Rect {
    pub fn elements(&self) -> impl Iterator<Item = Element> + Clone {
        let elements = [
            Element::MoveTo((self.min.x, self.min.y).into()),
            Element::LineTo((self.max.x, self.min.y).into()),
            Element::LineTo((self.max.x, self.max.y).into()),
            Element::LineTo((self.min.x, self.max.y).into()),
            Element::Close,
        ];
        (0..5).map(move |i| elements[i])
    }
}
