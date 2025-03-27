// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::path::Path;
use std::sync::OnceLock;
use usvg::tiny_skia_path::PathSegment;
use usvg::{Group, Node};
use vello_common::flatten::Line;
use vello_common::kurbo::{Affine, BezPath, Stroke};
use vello_common::peniko::Fill;
use vello_common::strip::Strip;
use vello_common::tile::Tiles;
use vello_common::{flatten, strip};

static DATA: OnceLock<Vec<DataItem>> = OnceLock::new();

pub fn get_data_items() -> &'static [DataItem] {
    DATA.get_or_init(|| {
        let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let mut data = vec![];

        // Always use ghostscript tiger.
        data.push(DataItem::from_path(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../examples/assets/Ghostscript_Tiger.svg"),
        ));

        for entry in std::fs::read_dir(&data_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) == Some("svg") {
                data.push(DataItem::from_path(&path));
            }
        }

        data
    })
}

#[derive(Clone, Debug)]
pub struct DataItem {
    pub name: String,
    pub fills: Vec<BezPath>,
    pub strokes: Vec<BezPath>,
    pub width: u16,
    pub height: u16,
}

impl DataItem {
    fn from_path(path: &Path) -> Self {
        let file_name = { path.file_stem().unwrap().to_string_lossy().to_string() };

        let data = std::fs::read(path).unwrap();
        let tree = usvg::Tree::from_data(&data, &usvg::Options::default()).unwrap();
        let mut ctx = ConversionContext::new();
        convert(&mut ctx, tree.root());

        Self {
            name: file_name,
            fills: ctx.fills,
            strokes: ctx.strokes,
            #[allow(
                clippy::cast_possible_truncation,
                reason = "It's okay to ignore for benchmarking."
            )]
            width: tree.size().width() as u16,
            #[allow(
                clippy::cast_possible_truncation,
                reason = "It's okay to ignore for benchmarking."
            )]
            height: tree.size().height() as u16,
        }
    }

    /// Get the raw flattened lines of both fills and strokes.
    ///
    /// A stroke width of 2.0 is assumed.
    pub fn lines(&self) -> Vec<Line> {
        let mut line_buf = vec![];
        let mut temp_buf = vec![];

        for path in &self.fills {
            flatten::fill(path, Affine::default(), &mut temp_buf);
            line_buf.extend(&temp_buf);
        }

        let stroke = Stroke {
            // Obviously not all strokes have that width, but it should be good enough
            // for benchmarking.
            width: 2.0,
            ..Default::default()
        };

        for path in &self.strokes {
            flatten::stroke(path, &stroke, Affine::default(), &mut temp_buf);
            line_buf.extend(&temp_buf);
        }

        line_buf
    }

    /// Get the unsorted tiles.
    pub fn unsorted_tiles(&self) -> Tiles {
        let mut tiles = Tiles::new();
        let lines = self.lines();
        tiles.make_tiles(&lines, self.width, self.height);

        tiles
    }

    /// Get the sorted tiles.
    pub fn sorted_tiles(&self) -> Tiles {
        let mut tiles = self.unsorted_tiles();
        tiles.sort_tiles();

        tiles
    }

    /// Get the alpha buffer and rendered strips.
    pub fn strips(&self) -> (Vec<u8>, Vec<Strip>) {
        let mut strip_buf = vec![];
        let mut alpha_buf = vec![];
        let lines = self.lines();
        let tiles = self.sorted_tiles();

        strip::render(
            &tiles,
            &mut strip_buf,
            &mut alpha_buf,
            Fill::NonZero,
            &lines,
        );

        (alpha_buf, strip_buf)
    }
}

fn convert(ctx: &mut ConversionContext, g: &Group) {
    ctx.push(convert_transform(&g.transform()));

    for child in g.children() {
        match child {
            Node::Group(group) => {
                convert(ctx, group);
            }
            Node::Path(p) => {
                let converted = convert_path_data(p);

                if p.fill().is_some() {
                    ctx.add_filled_path(converted.clone());
                }

                if p.stroke().is_some() {
                    ctx.add_stroked_path(converted);
                }
            }
            Node::Image(_) => {}
            Node::Text(_) => {}
        }
    }

    ctx.pop();
}

#[derive(Debug)]
struct ConversionContext {
    stack: Vec<Affine>,
    fills: Vec<BezPath>,
    strokes: Vec<BezPath>,
}

impl ConversionContext {
    fn new() -> Self {
        Self {
            stack: vec![],
            fills: vec![],
            strokes: vec![],
        }
    }

    fn push(&mut self, transform: Affine) {
        let new = *self.stack.last().unwrap_or(&Affine::IDENTITY) * transform;
        self.stack.push(new);
    }

    fn add_filled_path(&mut self, path: BezPath) {
        self.fills.push(self.get() * path);
    }

    fn add_stroked_path(&mut self, path: BezPath) {
        self.strokes.push(self.get() * path);
    }

    fn get(&self) -> Affine {
        *self.stack.last().unwrap_or(&Affine::IDENTITY)
    }

    fn pop(&mut self) {
        self.stack.pop();
    }
}

fn convert_transform(transform: &usvg::Transform) -> Affine {
    Affine::new([
        transform.sx as f64,
        transform.ky as f64,
        transform.kx as f64,
        transform.sy as f64,
        transform.tx as f64,
        transform.ty as f64,
    ])
}

fn convert_path_data(path: &usvg::Path) -> BezPath {
    let mut bez_path = BezPath::new();

    for e in path.data().segments() {
        match e {
            PathSegment::MoveTo(p) => {
                bez_path.move_to((p.x, p.y));
            }
            PathSegment::LineTo(p) => {
                bez_path.line_to((p.x, p.y));
            }
            PathSegment::QuadTo(p1, p2) => {
                bez_path.quad_to((p1.x, p1.y), (p2.x, p2.y));
            }
            PathSegment::CubicTo(p1, p2, p3) => {
                bez_path.curve_to((p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y));
            }
            PathSegment::Close => {
                bez_path.close_path();
            }
        }
    }

    bez_path
}
