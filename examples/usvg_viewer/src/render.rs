use std::path::PathBuf;
use usvg::NodeExt;
use vello::kurbo::{Affine, BezPath, Rect};
use vello::peniko::{Brush, Color, Fill, Stroke};
use vello::{SceneBuilder, SceneFragment};

pub fn render_svg(
    sb: &mut SceneBuilder,
    scene: &mut Option<SceneFragment>,
    xform: Affine,
    path: &PathBuf,
) {
    let scene_frag = scene.get_or_insert_with(|| {
        let contents = std::fs::read_to_string(path).expect("failed to read svg file");
        let svg = usvg::Tree::from_str(&contents, &usvg::Options::default())
            .expect("failed to parse svg file");
        let mut new_scene = SceneFragment::new();
        let mut builder = SceneBuilder::for_fragment(&mut new_scene);
        render_tree(&mut builder, svg);
        new_scene
    });
    sb.append(&scene_frag, Some(xform));
}

fn render_tree(sb: &mut SceneBuilder, svg: usvg::Tree) {
    for elt in svg.root.descendants() {
        let transform = elt.abs_transform();
        match &*elt.borrow() {
            usvg::NodeKind::Group(_) => {}
            usvg::NodeKind::Path(path) => {
                let mut local_path = BezPath::new();
                for elt in usvg::TransformedPath::new(&path.data, transform) {
                    match elt {
                        usvg::PathSegment::MoveTo { x, y } => local_path.move_to((x, y)),
                        usvg::PathSegment::LineTo { x, y } => local_path.line_to((x, y)),
                        usvg::PathSegment::CurveTo {
                            x1,
                            y1,
                            x2,
                            y2,
                            x,
                            y,
                        } => local_path.curve_to((x1, y1), (x2, y2), (x, y)),
                        usvg::PathSegment::ClosePath => local_path.close_path(),
                    }
                }

                // FIXME: let path.paint_order determine the fill/stroke order.

                if let Some(fill) = &path.fill {
                    if let Some(brush) = paint_to_brush(&fill.paint, fill.opacity) {
                        // FIXME: Set the fill rule
                        sb.fill(Fill::NonZero, Affine::IDENTITY, &brush, None, &local_path);
                    } else {
                        unimplemented(sb, &elt);
                    }
                }
                if let Some(stroke) = &path.stroke {
                    if let Some(brush) = paint_to_brush(&stroke.paint, stroke.opacity) {
                        // FIXME: handle stroke options such as linecap, linejoin, etc.
                        sb.stroke(
                            &Stroke::new(stroke.width.get() as f32),
                            Affine::IDENTITY,
                            &brush,
                            None,
                            &local_path,
                        );
                    } else {
                        unimplemented(sb, &elt);
                    }
                }
            }
            usvg::NodeKind::Image(_) => {
                unimplemented(sb, &elt);
            }
            usvg::NodeKind::Text(_) => {
                unimplemented(sb, &elt);
            }
        }
    }
}

fn unimplemented(sb: &mut SceneBuilder, node: &usvg::Node) {
    if let Some(bb) = node.calculate_bbox() {
        let rect = Rect {
            x0: bb.left(),
            y0: bb.top(),
            x1: bb.right(),
            y1: bb.bottom(),
        };
        sb.fill(Fill::NonZero, Affine::IDENTITY, Color::RED, None, &rect);
    }
}

fn paint_to_brush(paint: &usvg::Paint, opacity: usvg::Opacity) -> Option<Brush> {
    match paint {
        usvg::Paint::Color(color) => Some(Brush::Solid(Color::rgba8(
            color.red,
            color.green,
            color.blue,
            opacity.to_u8(),
        ))),
        usvg::Paint::LinearGradient(_) => None,
        usvg::Paint::RadialGradient(_) => None,
        usvg::Paint::Pattern(_) => None,
    }
}
