//! Common filter helper functions.

use crate::filter::drop_shadow::{DropShadow, transform_shadow_params};
use crate::filter::flood::Flood;
use crate::filter::gaussian_blur::{GaussianBlur, transform_blur_params};
use crate::filter::offset::Offset;
use crate::filter_effects::{Filter, FilterPrimitive};
use crate::kurbo::{Affine, Vec2};

pub mod drop_shadow;
pub mod flood;
pub mod gaussian_blur;
pub mod offset;

/// A filter that has been instantiated for a specific affine transformation.
#[derive(Debug)]
pub enum InstantiatedFilter {
    /// A flood filter.
    Flood(Flood),
    /// A gaussian blur filter.
    GaussianBlur(GaussianBlur),
    /// An offset filter.
    Offset(Offset),
    /// A drop shadow filter.
    DropShadow(DropShadow),
}

impl InstantiatedFilter {
    /// Build a new instantiated filter.
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
