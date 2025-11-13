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

pub(crate) use drop_shadow::DropShadow;
pub(crate) use flood::Flood;
pub(crate) use gaussian_blur::GaussianBlur;

use crate::layer_manager::LayerManager;
use vello_common::filter_effects::{Filter, FilterPrimitive};
use vello_common::pixmap::Pixmap;

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
///
/// # Limitations
/// Currently only supports filter graphs with a single primitive.
/// Multi-primitive filter graphs are not yet implemented.
pub(crate) fn filter_lowp(filter: &Filter, pixmap: &mut Pixmap, layer_manager: &mut LayerManager) {
    // Currently only single-primitive filters are supported
    if filter.graph.primitives.len() == 1 {
        match &filter.graph.primitives[0] {
            FilterPrimitive::Flood { color } => {
                let flood = Flood::new(*color);
                flood.execute_lowp(pixmap, layer_manager);
            }
            FilterPrimitive::GaussianBlur {
                std_deviation,
                edge_mode,
            } => {
                let blur = GaussianBlur::new(*std_deviation, *edge_mode);
                blur.execute_lowp(pixmap, layer_manager);
            }
            FilterPrimitive::DropShadow {
                dx,
                dy,
                std_deviation,
                color,
                edge_mode: _,
            } => {
                let drop_shadow = DropShadow::new(*dx, *dy, *std_deviation, *color);
                drop_shadow.execute_lowp(pixmap, layer_manager);
            }
            _ => {
                // Other primitives like Blend, ColorMatrix, ComponentTransfer, etc.
                // are not yet implemented
                unimplemented!("Other filter primitives not yet implemented");
            }
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
///
/// # Limitations
/// Currently only supports filter graphs with a single primitive.
/// Multi-primitive filter graphs are not yet implemented.
pub(crate) fn filter_highp(filter: &Filter, pixmap: &mut Pixmap, layer_manager: &mut LayerManager) {
    // Currently only single-primitive filters are supported
    if filter.graph.primitives.len() == 1 {
        match &filter.graph.primitives[0] {
            FilterPrimitive::Flood { color } => {
                let flood = Flood::new(*color);
                flood.execute_highp(pixmap, layer_manager);
            }
            FilterPrimitive::GaussianBlur {
                std_deviation,
                edge_mode,
            } => {
                let blur = GaussianBlur::new(*std_deviation, *edge_mode);
                blur.execute_highp(pixmap, layer_manager);
            }
            FilterPrimitive::DropShadow {
                dx,
                dy,
                std_deviation,
                color,
                edge_mode: _,
            } => {
                let drop_shadow = DropShadow::new(*dx, *dy, *std_deviation, *color);
                drop_shadow.execute_highp(pixmap, layer_manager);
            }
            _ => {
                // Other primitives like Blend, ColorMatrix, ComponentTransfer, etc.
                // are not yet implemented
                unimplemented!("Other filter primitives not yet implemented");
            }
        }
    }
}
