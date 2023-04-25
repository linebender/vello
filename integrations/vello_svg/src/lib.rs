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
//! - gradients
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
use vello::kurbo::{Affine, BezPath, Rect};
use vello::peniko::{Brush, Color, Fill, Stroke};
use vello::SceneBuilder;

pub use usvg;

/// Append a [`usvg::Tree`] into a Vello [`SceneBuilder`], with default error handling
/// This will draw a red box over (some) unsupported elements
///
/// Calls [`render_tree_with`] with an error handler implementing the above.
///
/// See the [module level documentation](crate#unsupported-features) for a list of some unsupported svg features
pub fn render_tree(sb: &mut SceneBuilder, svg: &usvg::Tree) {
    render_tree_with(sb, svg, default_error_handler).unwrap_or_else(|e| match e {})
}

/// Append a [`usvg::Tree`] into a Vello [`SceneBuilder`].
///
/// Calls [`render_tree_with`] with [`default_error_handler`].
/// This will draw a red box over unsupported element types.
///
/// See the [module level documentation](crate#unsupported-features) for a list of some unsupported svg features
pub fn render_tree_with<F: FnMut(&mut SceneBuilder, &usvg::Node) -> Result<(), E>, E>(
    sb: &mut SceneBuilder,
    svg: &usvg::Tree,
    mut on_err: F,
) -> Result<(), E> {
    for elt in svg.root.descendants() {
        let transform = {
            let usvg::Transform { a, b, c, d, e, f } = elt.abs_transform();
            Affine::new([a, b, c, d, e, f])
        };
        match &*elt.borrow() {
            usvg::NodeKind::Group(_) => {}
            usvg::NodeKind::Path(path) => {
                let mut local_path = BezPath::new();
                // The semantics of SVG paths don't line up with `BezPath`; we must manually track initial points
                let mut just_closed = false;
                let mut most_recent_initial = (0., 0.);
                for elt in path.data.segments() {
                    match elt {
                        usvg::PathSegment::MoveTo { x, y } => {
                            if std::mem::take(&mut just_closed) {
                                local_path.move_to(most_recent_initial);
                            }
                            most_recent_initial = (x, y);
                            local_path.move_to(most_recent_initial)
                        }
                        usvg::PathSegment::LineTo { x, y } => {
                            if std::mem::take(&mut just_closed) {
                                local_path.move_to(most_recent_initial);
                            }
                            local_path.line_to((x, y))
                        }
                        usvg::PathSegment::CurveTo {
                            x1,
                            y1,
                            x2,
                            y2,
                            x,
                            y,
                        } => {
                            if std::mem::take(&mut just_closed) {
                                local_path.move_to(most_recent_initial);
                            }
                            local_path.curve_to((x1, y1), (x2, y2), (x, y))
                        }
                        usvg::PathSegment::ClosePath => {
                            just_closed = true;
                            local_path.close_path()
                        }
                    }
                }

                // FIXME: let path.paint_order determine the fill/stroke order.

                if let Some(fill) = &path.fill {
                    if let Some(brush) = paint_to_brush(&fill.paint, fill.opacity) {
                        // FIXME: Set the fill rule
                        sb.fill(Fill::NonZero, transform, &brush, None, &local_path);
                    } else {
                        on_err(sb, &elt)?;
                    }
                }
                if let Some(stroke) = &path.stroke {
                    if let Some(brush) = paint_to_brush(&stroke.paint, stroke.opacity) {
                        // FIXME: handle stroke options such as linecap, linejoin, etc.
                        sb.stroke(
                            &Stroke::new(stroke.width.get() as f32),
                            transform,
                            &brush,
                            None,
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

/// Error handler function for [`render_tree_with`] which draws a transparent red box
/// instead of unsupported SVG features
pub fn default_error_handler(sb: &mut SceneBuilder, node: &usvg::Node) -> Result<(), Infallible> {
    if let Some(bb) = node.calculate_bbox() {
        let rect = Rect {
            x0: bb.left(),
            y0: bb.top(),
            x1: bb.right(),
            y1: bb.bottom(),
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
