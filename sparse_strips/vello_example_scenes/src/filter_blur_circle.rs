// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scene with a single filled circle wrapped in a Gaussian blur filter.
//!
//! This reproduces element `#0` of [`crate::filter_elements::FilterElementsScene`]
//! (seeded with `0xCAFE_BABE`): a `DEEP_SKY_BLUE` circle with a heavy Gaussian blur.

use crate::{ExampleScene, RenderingContext};
use vello_common::color::palette::css;
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, Circle, Shape};

/// Blur standard deviation matching element `#0` of `FilterElementsScene`.
const STD_DEVIATION: f32 = 190.;

/// Scene with a single circle and a Gaussian blur filter.
#[derive(Debug, Default)]
pub struct FilterBlurCircleScene {
    /// Uniform scale of the most recent root transform, shown in the title.
    current_scale: f64,
}

impl FilterBlurCircleScene {
    /// Create a new `FilterBlurCircleScene`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl ExampleScene for FilterBlurCircleScene {
    fn render<T: RenderingContext>(
        &mut self,
        ctx: &mut T,
        _resources: &mut T::Resources,
        root_transform: Affine,
    ) {
        // Track the uniform scale of the root transform so it can be shown in
        // the window title.
        self.current_scale = root_transform.determinant().abs().sqrt();

        let vw = ctx.width() as f64;
        let vh = ctx.height() as f64;

        // Centre the circle and make it large enough that the (large) blur
        // remains clearly visible rather than being spread thin to invisibility.
        let cx = vw / 2.0;
        let cy = vh / 2.0;
        let radius = vw.min(vh) * 0.3;

        ctx.set_transform(root_transform);
        let filter = Filter::from_primitive(FilterPrimitive::GaussianBlur {
            std_deviation: STD_DEVIATION,
            edge_mode: EdgeMode::None,
        });
        ctx.push_filter_layer(filter);

        ctx.set_transform(root_transform);
        ctx.set_paint(css::DEEP_SKY_BLUE);
        let circle = Circle::new((cx, cy), radius).to_path(0.1);
        ctx.fill_path(&circle);

        ctx.pop_layer();
    }

    fn status(&self) -> Option<String> {
        Some(format!("scale {}x", self.current_scale))
    }
}
