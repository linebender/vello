//! Dump the paths of a

use std::env;
use std::path::Path;
use usvg::tiny_skia_path::PathSegment;
use usvg::{Group, Node, Transform};
use vello_bench::DATA_PATH;
use vello_bench::read::PathContainer;
use vello_common::kurbo::{Affine, BezPath};

fn main() {
    let args: Vec<String> = env::args().collect();

    let gs_data = PathContainer::from_data_file("gs");

    let Some(str_path) = args.get(1) else {
        eprintln!("you need to provide the path to the SVG as the first argument.");
        return;
    };

    let path = Path::new(str_path);
    let file_name = {
        let tmp = path.with_extension("");
        tmp.file_name().unwrap().to_string_lossy().to_string()
    };
    let data = std::fs::read(path).expect(&format!("failed to read path {:?}", path));
    let tree = usvg::Tree::from_data(&data, &usvg::Options::default()).unwrap();
    let mut ctx = ConversionContext::new();
    convert(&mut ctx, tree.root());

    let out_path = DATA_PATH.join(format!("{file_name}.txt"));

    let out_string = {
        let mut buf = String::new();

        let fills = ctx.fills.iter().map(|b| b.to_svg()).collect::<Vec<_>>();
        let strokes = ctx.strokes.iter().map(|b| b.to_svg()).collect::<Vec<_>>();

        buf += &fills.join("\n");
        buf += "\nSTROKES\n";
        buf += &strokes.join("\n");

        buf
    };

    std::fs::write(out_path, out_string).unwrap();
}

fn convert(ctx: &mut ConversionContext, g: &Group) {
    ctx.push(convert_transform(&g.transform()));

    for child in g.children() {
        match child {
            Node::Group(g) => {
                convert(ctx, g);
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
