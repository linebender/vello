//! Append a [`usvg::Tree`] to a Vello [`SceneBuilder`]
//!
//! This currently lacks support for a [number of important](crate#unsupported-features) SVG features.
//! This is because this integration was developed for examples, which only need to support enough SVG
//! to demonstrate Vello.
//!
//! However, this is also intended to be the preferred integration between Vello and [usvg], so [consider
//! contributing](https://github.com/linebender/vello) if you need a feature which is missing.
//!
//! [`render_tree_with`] is the primary entry point function, which supports choosing the behaviour
//! when [unsupported features](crate#unsupported-features) are detected. In a future release where there are
//! no unsupported features, this may be phased out
//!
//! [`render_tree`] is a convenience wrapper around [`render_tree_with`] which renders an indicator around not
//! yet supported features
//!
//! This crate also re-exports [`usvg`], to make handling dependency versions easier
//!
//! # Unsupported features
//!
//! Missing features include:
//! - embedded images
//! - text
//! - group opacity
//! - mix-blend-modes
//! - clipping
//! - masking
//! - filter effects
//! - group background
//! - path visibility
//! - path paint order
//! - path shape-rendering
//! - patterns

use std::convert::Infallible;
use usvg::NodeExt;
use vello::{
    kurbo::{Affine, BezPath, Point, Rect, Stroke},
    peniko::{Brush, Color, Fill},
    SceneBuilder,
};

/// Re-export vello.
pub use vello;

/// Re-export usvg.
pub use usvg;

/// Append a [`usvg::Tree`] into a Vello [`SceneBuilder`], with default error
/// handling. This will draw a red box over (some) unsupported elements
///
/// Calls [`render_tree_with`] with an error handler implementing the above.
///
/// See the [module level documentation](crate#unsupported-features) for a list
/// of some unsupported svg features
pub fn render_tree(sb: &mut SceneBuilder, svg: &usvg::Tree) {
    render_tree_with(sb, svg, default_error_handler).unwrap_or_else(|e| match e {})
}

/// Append a [`usvg::Tree`] into a Vello [`SceneBuilder`].
///
/// Calls [`render_tree_with`] with [`default_error_handler`].
/// This will draw a red box over unsupported element types.
///
/// See the [module level documentation](crate#unsupported-features) for a list
/// of some unsupported svg features
pub fn render_tree_with<F: FnMut(&mut SceneBuilder, &usvg::Node) -> Result<(), E>, E>(
    sb: &mut SceneBuilder,
    svg: &usvg::Tree,
    mut on_err: F,
) -> Result<(), E> {
    for elt in svg.root.descendants() {
        let transform = {
            let usvg::Transform {
                sx,
                kx,
                ky,
                sy,
                tx,
                ty,
            } = elt.abs_transform();
            Affine::new([sx, kx, ky, sy, tx, ty].map(f64::from))
        };
        match &*elt.borrow() {
            usvg::NodeKind::Group(_) => {}
            usvg::NodeKind::Path(path) => {
                let mut local_path = BezPath::new();
                // The semantics of SVG paths don't line up with `BezPath`; we
                // must manually track initial points
                let mut just_closed = false;
                let mut most_recent_initial = (0., 0.);
                for elt in path.data.segments() {
                    match elt {
                        usvg::tiny_skia_path::PathSegment::MoveTo(p) => {
                            if std::mem::take(&mut just_closed) {
                                local_path.move_to(most_recent_initial);
                            }
                            most_recent_initial = (p.x.into(), p.y.into());
                            local_path.move_to(most_recent_initial)
                        }
                        usvg::tiny_skia_path::PathSegment::LineTo(p) => {
                            if std::mem::take(&mut just_closed) {
                                local_path.move_to(most_recent_initial);
                            }
                            local_path.line_to(Point::new(p.x as f64, p.y as f64))
                        }
                        usvg::tiny_skia_path::PathSegment::QuadTo(p1, p2) => {
                            if std::mem::take(&mut just_closed) {
                                local_path.move_to(most_recent_initial);
                            }
                            local_path.quad_to(
                                Point::new(p1.x as f64, p1.y as f64),
                                Point::new(p2.x as f64, p2.y as f64),
                            )
                        }
                        usvg::tiny_skia_path::PathSegment::CubicTo(p1, p2, p3) => {
                            if std::mem::take(&mut just_closed) {
                                local_path.move_to(most_recent_initial);
                            }
                            local_path.curve_to(
                                Point::new(p1.x as f64, p1.y as f64),
                                Point::new(p2.x as f64, p2.y as f64),
                                Point::new(p3.x as f64, p3.y as f64),
                            )
                        }
                        usvg::tiny_skia_path::PathSegment::Close => {
                            just_closed = true;
                            local_path.close_path()
                        }
                    }
                }

                // FIXME: let path.paint_order determine the fill/stroke order.

                if let Some(fill) = &path.fill {
                    if let Some((brush, brush_transform)) =
                        paint_to_brush(&fill.paint, fill.opacity)
                    {
                        sb.fill(
                            match fill.rule {
                                usvg::FillRule::NonZero => Fill::NonZero,
                                usvg::FillRule::EvenOdd => Fill::EvenOdd,
                            },
                            transform,
                            &brush,
                            Some(brush_transform),
                            &local_path,
                        );
                    } else {
                        on_err(sb, &elt)?;
                    }
                }
                if let Some(stroke) = &path.stroke {
                    if let Some((brush, brush_transform)) =
                        paint_to_brush(&stroke.paint, stroke.opacity)
                    {
                        // FIXME: handle stroke options such as linecap,
                        // linejoin, etc.
                        sb.stroke(
                            &Stroke::new(stroke.width.get() as f64),
                            transform,
                            &brush,
                            Some(brush_transform),
                            &local_path,
                        );
                    } else {
                        on_err(sb, &elt)?;
                    }
                }
            }
            usvg::NodeKind::Image(_) => {
                on_err(sb, &elt)?;
            }
            usvg::NodeKind::Text(_) => {
                on_err(sb, &elt)?;
            }
        }
    }
    Ok(())
}

/// Error handler function for [`render_tree_with`] which draws a transparent
/// red box instead of unsupported SVG features
pub fn default_error_handler(sb: &mut SceneBuilder, node: &usvg::Node) -> Result<(), Infallible> {
    if let Some(bb) = node.calculate_bbox() {
        let rect = Rect {
            x0: bb.left() as f64,
            y0: bb.top() as f64,
            x1: bb.right() as f64,
            y1: bb.bottom() as f64,
        };
        sb.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            Color::RED.with_alpha_factor(0.5),
            None,
            &rect,
        );
    }
    Ok(())
}

fn paint_to_brush(paint: &usvg::Paint, opacity: usvg::Opacity) -> Option<(Brush, Affine)> {
    match paint {
        usvg::Paint::Color(color) => Some((
            Brush::Solid(Color::rgba8(
                color.red,
                color.green,
                color.blue,
                opacity.to_u8(),
            )),
            Affine::IDENTITY,
        )),
        usvg::Paint::LinearGradient(gr) => {
            let stops: Vec<vello::peniko::ColorStop> = gr
                .stops
                .iter()
                .map(|stop| {
                    let mut cstop = vello::peniko::ColorStop::default();
                    cstop.color.r = stop.color.red;
                    cstop.color.g = stop.color.green;
                    cstop.color.b = stop.color.blue;
                    cstop.color.a = (stop.opacity * opacity).to_u8();
                    cstop.offset = stop.offset.get();
                    cstop
                })
                .collect();
            let start = Point::new(gr.x1 as f64, gr.y1 as f64);
            let end = Point::new(gr.x2 as f64, gr.y2 as f64);
            let arr = [
                gr.transform.sx,
                gr.transform.ky,
                gr.transform.kx,
                gr.transform.sy,
                gr.transform.tx,
                gr.transform.ty,
            ]
            .map(f64::from);
            let transform = Affine::new(arr);
            let gradient =
                vello::peniko::Gradient::new_linear(start, end).with_stops(stops.as_slice());
            Some((Brush::Gradient(gradient), transform))
        }
        usvg::Paint::RadialGradient(gr) => {
            let stops: Vec<vello::peniko::ColorStop> = gr
                .stops
                .iter()
                .map(|stop| {
                    let mut cstop = vello::peniko::ColorStop::default();
                    cstop.color.r = stop.color.red;
                    cstop.color.g = stop.color.green;
                    cstop.color.b = stop.color.blue;
                    cstop.color.a = (stop.opacity * opacity).to_u8();
                    cstop.offset = stop.offset.get();
                    cstop
                })
                .collect();

            let start_center = Point::new(gr.cx as f64, gr.cy as f64);
            let end_center = Point::new(gr.fx as f64, gr.fy as f64);
            let start_radius = 0_f32;
            let end_radius = gr.r.get();
            let arr = [
                gr.transform.sx,
                gr.transform.ky,
                gr.transform.kx,
                gr.transform.sy,
                gr.transform.tx,
                gr.transform.ty,
            ]
            .map(f64::from);
            let transform = Affine::new(arr);
            let gradient = vello::peniko::Gradient::new_two_point_radial(
                start_center,
                start_radius,
                end_center,
                end_radius,
            )
            .with_stops(stops.as_slice());
            Some((Brush::Gradient(gradient), transform))
        }
        usvg::Paint::Pattern(_) => None,
    }
}
