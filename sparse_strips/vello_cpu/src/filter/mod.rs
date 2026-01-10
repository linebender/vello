// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter effects implementation for `vello_cpu`.
//!
//! This module provides CPU-based implementations of SVG filter effects,
//! supporting both low-precision (u8) and high-precision (f32) rendering paths.
//! Filters are applied to layers through the layer manager, which handles
//! intermediate storage.

mod drop_shadow;
mod flood;
mod gaussian_blur;
mod offset;
mod shift;

pub(crate) use drop_shadow::DropShadow;
pub(crate) use flood::Flood;
pub(crate) use gaussian_blur::GaussianBlur;
pub(crate) use offset::Offset;

use crate::layer_manager::LayerManager;
use vello_common::filter_effects::{Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, Vec2};
use vello_common::pixmap::Pixmap;
use vello_common::util::extract_scales;

/// Trait for filter effects that can be applied to layers.
///
/// Each filter implements this trait with both u8 and f32 variants
/// to support different rendering backends and precision requirements.
/// The low-precision path (`execute_lowp`) uses 8-bit color channels for
/// better performance, while the high-precision path (`execute_highp`)
/// uses 32-bit floating-point for higher quality at the cost of speed.
pub(crate) trait FilterEffect {
    /// Apply the low-precision (u8) version of the filter.
    ///
    /// # Arguments
    /// * `pixmap` - The target pixmap containing rendering metadata
    /// * `layer_manager` - Manager for allocating and accessing intermediate layers
    fn execute_lowp(&self, pixmap: &mut Pixmap, layer_manager: &mut LayerManager);

    /// Apply the high-precision (f32) version of the filter.
    ///
    /// # Arguments
    /// * `pixmap` - The target pixmap containing rendering metadata
    /// * `layer_manager` - Manager for allocating and accessing intermediate layers
    fn execute_highp(&self, pixmap: &mut Pixmap, layer_manager: &mut LayerManager);
}

/// Apply the low-precision (u8) version of a filter effect to a layer.
///
/// This function dispatches filter primitives from a filter graph to their
/// corresponding CPU implementations using 8-bit color channels.
///
/// # Arguments
/// * `filter` - The filter containing the graph of primitives to apply
/// * `pixmap` - The target pixmap containing rendering metadata
/// * `layer_manager` - Manager for allocating and accessing intermediate layers
/// * `transform` - The transformation matrix to extract scale from for filter parameters
///
/// # Limitations
/// Currently only supports filter graphs with a single primitive.
/// Multi-primitive filter graphs are not yet implemented.
pub(crate) fn filter_lowp(
    filter: &Filter,
    pixmap: &mut Pixmap,
    layer_manager: &mut LayerManager,
    transform: Affine,
) {
    // Multi-primitive filter graphs are not yet implemented.
    if filter.graph.primitives.len() != 1 {
        unimplemented!("Multi-primitive filter graphs are not yet supported");
    }

    match &filter.graph.primitives[0] {
        FilterPrimitive::Flood { color } => {
            let flood = Flood::new(*color);
            flood.execute_lowp(pixmap, layer_manager);
        }
        FilterPrimitive::GaussianBlur {
            std_deviation,
            edge_mode,
        } => {
            let scaled_std_dev = transform_blur_params(*std_deviation, &transform);
            let blur = GaussianBlur::new(scaled_std_dev, *edge_mode);
            blur.execute_lowp(pixmap, layer_manager);
        }
        FilterPrimitive::DropShadow {
            dx,
            dy,
            std_deviation,
            color,
            edge_mode,
        } => {
            let (scaled_dx, scaled_dy, scaled_std_dev) =
                transform_shadow_params(*dx, *dy, *std_deviation, &transform);
            let drop_shadow =
                DropShadow::new(scaled_dx, scaled_dy, scaled_std_dev, *edge_mode, *color);
            drop_shadow.execute_lowp(pixmap, layer_manager);
        }
        FilterPrimitive::Offset { dx, dy } => {
            let (scaled_dx, scaled_dy) = transform_offset_params(*dx, *dy, &transform);
            Offset::new(scaled_dx, scaled_dy).execute_lowp(pixmap, layer_manager);
        }
        _ => {
            // Other primitives like Blend, ColorMatrix, ComponentTransfer, etc.
            // are not yet implemented
            unimplemented!("Other filter primitives not yet implemented");
        }
    }
}

/// Apply the high-precision (f32) version of a filter effect to a layer.
///
/// This function dispatches filter primitives from a filter graph to their
/// corresponding CPU implementations using 32-bit floating-point color channels.
///
/// # Arguments
/// * `filter` - The filter containing the graph of primitives to apply
/// * `pixmap` - The target pixmap containing rendering metadata
/// * `layer_manager` - Manager for allocating and accessing intermediate layers
/// * `transform` - The transformation matrix to extract scale from for filter parameters
///
/// # Limitations
/// Currently only supports filter graphs with a single primitive.
/// Multi-primitive filter graphs are not yet implemented.
pub(crate) fn filter_highp(
    filter: &Filter,
    pixmap: &mut Pixmap,
    layer_manager: &mut LayerManager,
    transform: Affine,
) {
    // Multi-primitive filter graphs are not yet implemented.
    if filter.graph.primitives.len() != 1 {
        unimplemented!("Multi-primitive filter graphs are not yet supported");
    }

    match &filter.graph.primitives[0] {
        FilterPrimitive::Flood { color } => {
            let flood = Flood::new(*color);
            flood.execute_highp(pixmap, layer_manager);
        }
        FilterPrimitive::GaussianBlur {
            std_deviation,
            edge_mode,
        } => {
            let scaled_std_dev = transform_blur_params(*std_deviation, &transform);
            let blur = GaussianBlur::new(scaled_std_dev, *edge_mode);
            blur.execute_highp(pixmap, layer_manager);
        }
        FilterPrimitive::DropShadow {
            dx,
            dy,
            std_deviation,
            color,
            edge_mode,
        } => {
            let (scaled_dx, scaled_dy, scaled_std_dev) =
                transform_shadow_params(*dx, *dy, *std_deviation, &transform);
            let drop_shadow =
                DropShadow::new(scaled_dx, scaled_dy, scaled_std_dev, *edge_mode, *color);
            drop_shadow.execute_highp(pixmap, layer_manager);
        }
        FilterPrimitive::Offset { dx, dy } => {
            let (scaled_dx, scaled_dy) = transform_offset_params(*dx, *dy, &transform);
            Offset::new(scaled_dx, scaled_dy).execute_highp(pixmap, layer_manager);
        }
        _ => {
            // Other primitives like Blend, ColorMatrix, ComponentTransfer, etc.
            // are not yet implemented
            unimplemented!("Other filter primitives not yet implemented");
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

/// Transform a drop shadow's offset and standard deviation using the affine transformation.
///
/// Applies the full linear transformation (rotation, scale, and shear) to the offset vector,
/// and scales the blur standard deviation uniformly.
///
/// # Arguments
/// * `dx` - Horizontal offset in user space
/// * `dy` - Vertical offset in user space
/// * `std_deviation` - Blur standard deviation in user space
/// * `transform` - The transformation matrix to apply
///
/// # Returns
/// A tuple of (`scaled_dx`, `scaled_dy`, `scaled_std_dev`) in device space
fn transform_shadow_params(
    dx: f32,
    dy: f32,
    std_deviation: f32,
    transform: &Affine,
) -> (f32, f32, f32) {
    // Transform the offset vector by the full transformation matrix
    // to correctly handle rotation, scale, and shear.
    // We use the linear part only (no translation) since this is a vector offset.
    let offset = Vec2::new(dx as f64, dy as f64);
    let [a, b, c, d, _, _] = transform.as_coeffs();
    let transformed_offset = Vec2::new(a * offset.x + c * offset.y, b * offset.x + d * offset.y);
    let scaled_dx = transformed_offset.x as f32;
    let scaled_dy = transformed_offset.y as f32;

    // Scale the blur radius uniformly
    let scaled_std_dev = transform_blur_params(std_deviation, transform);

    (scaled_dx, scaled_dy, scaled_std_dev)
}

/// Scale a blur's standard deviation uniformly based on the transformation.
///
/// Extracts the scale factors from the transformation matrix using SVD and
/// averages them to get a uniform scale factor for the blur radius.
///
/// # Arguments
/// * `std_deviation` - The blur standard deviation in user space
/// * `transform` - The transformation matrix to extract scale from
///
/// # Returns
/// The scaled standard deviation in device space
fn transform_blur_params(std_deviation: f32, transform: &Affine) -> f32 {
    let (scale_x, scale_y) = extract_scales(transform);
    let uniform_scale = (scale_x + scale_y) / 2.0;
    // TODO: Support separate std_deviation for x and y axes (std_deviation_x, std_deviation_y)
    // to properly handle non-uniform scaling. This would eliminate the need for uniform_scale
    // and allow blur to scale independently along each axis.
    std_deviation * uniform_scale
}
