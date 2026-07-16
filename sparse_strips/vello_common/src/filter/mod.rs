// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Common filter helper functions.
//!
//! Unlike the filters defines in [`crate::filter_effects`], the filters in this module
//! represent a special representation of each filter to be used as the basis for rendering in
//! `vello_hybrid` and `vello_cpu`.

use crate::filter::drop_shadow::{DropShadow, transform_shadow_params};
use crate::filter::flood::Flood;
use crate::filter::gaussian_blur::{GaussianBlur, transform_blur_params};
use crate::filter::offset::Offset;
use crate::filter_effects::{Filter, FilterPrimitive};
use crate::geometry::{PaddingU16, RectU16};
use crate::kurbo::{Affine, Rect, Vec2};
use crate::tile::Tile;
use crate::util::RectExt;

pub mod drop_shadow;
pub mod flood;
pub mod gaussian_blur;
pub mod offset;

/// A filter that has been prepared for rendering.
#[derive(Debug)]
pub enum PreparedFilter {
    /// A flood filter.
    Flood(Flood),
    /// A gaussian blur filter.
    GaussianBlur(GaussianBlur),
    /// An offset filter.
    Offset(Offset),
    /// A drop shadow filter.
    DropShadow(DropShadow),
}

impl PreparedFilter {
    /// Build a new prepared filter for the given transform.
    pub fn new(filter: &Filter, transform: &Affine) -> Self {
        // Multi-primitive filter graphs are not yet implemented.
        if filter.graph.primitives.len() != 1 {
            unimplemented!("Multi-primitive filter graphs are not yet supported");
        }

        match &filter.graph.primitives[0] {
            FilterPrimitive::Flood { color } => {
                let flood = Flood::new(*color);
                Self::Flood(flood)
            }
            FilterPrimitive::GaussianBlur {
                std_deviation,
                edge_mode,
            } => {
                let scaled_std_dev = transform_blur_params(*std_deviation, transform);
                let blur = GaussianBlur::new(scaled_std_dev, *edge_mode);
                Self::GaussianBlur(blur)
            }
            FilterPrimitive::DropShadow {
                dx,
                dy,
                std_deviation,
                color,
                edge_mode,
            } => {
                let (scaled_dx, scaled_dy, scaled_std_dev) =
                    transform_shadow_params(*dx, *dy, *std_deviation, transform);
                let drop_shadow =
                    DropShadow::new(scaled_dx, scaled_dy, scaled_std_dev, *edge_mode, *color);

                Self::DropShadow(drop_shadow)
            }
            FilterPrimitive::Offset { dx, dy } => {
                let (scaled_dx, scaled_dy) = transform_offset_params(*dx, *dy, transform);
                let offset = Offset::new(scaled_dx, scaled_dy);

                Self::Offset(offset)
            }
            _ => {
                // Other primitives like Blend, ColorMatrix, ComponentTransfer, etc.
                // are not yet implemented
                unimplemented!("Other filter primitives not yet implemented");
            }
        }
    }
}

/// Metadata about a filter layer and how it should be composited back into the parent layer.
#[derive(Debug, Clone, Copy)]
pub struct FilterLayerPlacement {
    /// The conceptual bounding box of the pixmap that needs to be allocated to render
    /// a layer correctly, including the area affected by the filter.
    ///
    /// For example, if the filter layer contains a rect spanning (200, 200) to (300, 300)
    /// with a blur that has a radius exceeding the rectangle 40 pixels on each side, the pixmap
    /// bbox will be (160, 160) to (340, 340).
    ///
    /// See the comments in `FilterLayerPlacement::new` for more information.
    pub pixmap_bbox: RectU16,
    /// Rectangle in the parent layer's coordinate space the filtered pixmap is composited into.
    ///
    /// See the comments in `FilterLayerPlacement::new` for more information.
    pub dest_bbox: RectU16,
    /// Source x offset used when sampling from the filter pixmap.
    ///
    /// See the comments in `FilterLayerPlacement::new` for more information.
    pub src_x: u16,
    /// Source y offset used when sampling from the filter pixmap.
    ///
    /// See the comments in `FilterLayerPlacement::new` for more information.
    pub src_y: u16,
}

impl FilterLayerPlacement {
    pub(crate) const EMPTY: Self = Self {
        pixmap_bbox: RectU16::INVERTED,
        dest_bbox: RectU16::INVERTED,
        src_x: 0,
        src_y: 0,
    };

    pub(crate) fn new(bbox: RectU16, filter_plan: &FilterData) -> Self {
        // Some more detailed explanations of what's going on here since this
        // part is a bit confusing.

        // `bbox` is the tight bounding box across all strips in the filter
        // layer. We now need to expand it by the filter padding to know how
        // large of a pixmap we actually need to allocate. Also, as mentioned
        // in [`FilterLayerPlan::new`], we need to ensure the pixmap itself is
        // also a multiple of the tile width / tile height.
        let pixmap_bbox = bbox
            .expand(filter_plan.filter_padding)
            .snap_to_tile_coordinates();

        // Remember that in `RenderContext`, we eagerly shift everything drawn by `source_shift`
        // to conservatively ensure that everything that might be needed for the filter is in the
        // viewport area. Therefore, when compositing the filter layer back, we need to undo that
        // shift.
        let (shift_x, shift_y) = filter_plan.source_shift();
        // For example, if `shift_x` is 20 and `pixmap_bbox.x0` is 4,
        // shifting the pixmap back would place its left edge at -16. Since we
        // start compositing at x=0, we need to skip the first 16 pixels
        // inside the cropped pixmap (`src_x = 20 - 4`). If `pixmap_bbox.x0`
        // is already >= `shift_x`, nothing is clipped and `src_x` is 0.
        let src_x = shift_x.saturating_sub(pixmap_bbox.x0);
        let src_y = shift_y.saturating_sub(pixmap_bbox.y0);
        let dest_bbox = pixmap_bbox.relative_to_origin((shift_x, shift_y));

        Self {
            pixmap_bbox,
            dest_bbox,
            src_x,
            src_y,
        }
    }

    /// Return the source origin of the filter layer.
    pub fn src_origin(self) -> (u16, u16) {
        (self.src_x, self.src_y)
    }
}

/// Precomputed data for a filter layer.
#[derive(Debug, Clone)]
pub struct FilterData {
    /// The underlying filter.
    pub filter: Filter,
    /// The transform that was in place when the filter layer was invoked.
    pub transform: Affine,
    /// Padding that needs to be added for the area where the filter is applied.
    ///
    /// See [`Filter::filter_expansion`].
    pub filter_padding: PaddingU16,
    /// Padding that needs to be added to the source region for correct filter application.
    ///
    /// See [`Filter::source_expansion`].
    pub source_padding: PaddingU16,
}

impl FilterData {
    /// Create precomputed data for a filter and transform.
    pub fn new(filter: Filter, transform: Affine) -> Self {
        fn snapped_padding(expansion: Rect) -> PaddingU16 {
            // TODO: We technically shouldn't need to snap here. `source_padding` is only
            // used to shift the contents when rendering into the render context, and the
            // final pixmap bbox (which is derived from `filter_expansion` will be snapped
            // separately. However, not snapping here causes larger mismatches with Vello Hybrid
            // since the size of the final pixmap determines in which way we decimate for the
            // gaussian blur filter. Therefore, we keep this for compatibility.
            fn snap_up(value: f64, step: u16) -> u16 {
                let step = f64::from(step);
                ((value / step).ceil() * step) as u16
            }

            PaddingU16::new(
                snap_up(-expansion.x0, Tile::WIDTH),
                snap_up(-expansion.y0, Tile::HEIGHT),
                snap_up(expansion.x1, Tile::WIDTH),
                snap_up(expansion.y1, Tile::HEIGHT),
            )
        }

        let source_padding = snapped_padding(filter.source_expansion(&transform));
        let filter_padding = snapped_padding(filter.filter_expansion(&transform));

        Self {
            filter,
            transform,
            filter_padding,
            source_padding,
        }
    }

    /// By how much to shift all rendered contents to ensure that all rendered contents
    /// are visible in the viewport [0, 0, width, height].
    pub fn source_shift(&self) -> (u16, u16) {
        (self.source_padding.left, self.source_padding.top)
    }
}

/// Transform an offset's dx/dy using the affine transformation's linear part.
///
/// # Returns
/// A tuple of (`scaled_dx`, `scaled_dy`) in device space.
fn transform_offset_params(dx: f32, dy: f32, transform: &Affine) -> (f32, f32) {
    let offset = Vec2::new(dx as f64, dy as f64);
    let [a, b, c, d, _, _] = transform.as_coeffs();
    let transformed_offset = Vec2::new(a * offset.x + c * offset.y, b * offset.x + d * offset.y);
    (transformed_offset.x as f32, transformed_offset.y as f32)
}
