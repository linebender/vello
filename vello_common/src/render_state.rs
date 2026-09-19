// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared render state.

use crate::kurbo::{Cap, Join, Stroke};
use crate::paint::{PaintType, Tint};
use crate::peniko::color::palette::css::BLACK;
use crate::peniko::{BlendMode, Compose, Fill, Mix};
use crate::transforms::Transforms;

/// A render state which contains the style properties for path rendering.
#[derive(Debug, Clone)]
pub struct RenderState {
    /// The paint type (solid color, gradient, or image).
    pub paint: PaintType,
    /// Stroke style for path stroking operations.
    pub stroke: Stroke,
    /// Fill rule for path filling operations.
    pub fill_rule: Fill,
    /// Blend mode for compositing.
    pub blend_mode: BlendMode,
    /// The tint for image painting.
    pub tint: Option<Tint>,
    /// State of active transforms.
    pub transforms: Transforms,
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            paint: BLACK.into(),
            stroke: Stroke {
                width: 1.0,
                join: Join::Bevel,
                start_cap: Cap::Butt,
                end_cap: Cap::Butt,
                ..Default::default()
            },
            fill_rule: Fill::NonZero,
            blend_mode: BlendMode::new(Mix::Normal, Compose::SrcOver),
            tint: None,
            transforms: Transforms::default(),
        }
    }
}

impl RenderState {
    /// Reset to default state.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
