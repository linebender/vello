// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The clear pass.

use crate::render::webgl::passes::WebGlRendererContext;
use crate::{ClearSettings, target::DrawPassTarget};
use vello_common::geometry::RectU16;
use web_sys::WebGl2RenderingContext;

impl WebGlRendererContext<'_> {
    pub(crate) fn clear_pass_inner(&self, target: DrawPassTarget, settings: ClearSettings<'_>) {
        let color = settings.clear_color();

        let negate_ndc = target.negate_ndc();
        let (width, height) = match target {
            DrawPassTarget::Root(_) => {
                self.gl.bind_framebuffer(
                    WebGl2RenderingContext::FRAMEBUFFER,
                    self.resources.root_framebuffer(),
                );
                (
                    self.resources.render_size.width,
                    self.resources.render_size.height,
                )
            }
            DrawPassTarget::Layer(target) => {
                let size = self.texture_size();
                let framebuffer = self.resources.layer_framebuffer(target);
                self.gl
                    .bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(framebuffer));
                (size.width(), size.height())
            }
        };

        let [r, g, b, a] = color.premultiply().components;
        self.gl.clear_color(r, g, b, a);

        match settings {
            ClearSettings::Viewport { .. } => {
                self.gl.disable(WebGl2RenderingContext::SCISSOR_TEST);
                self.gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);
            }
            ClearSettings::Rects { rects, .. } => {
                self.gl.enable(WebGl2RenderingContext::SCISSOR_TEST);
                let bounds = RectU16::new(0, 0, width, height);

                // It's unclear whether it's better to scissor + clear repeatedly, or to do
                // one instanced clear pass via a shader. But using this approach should give the
                // GPU more semantic information about what we want to do, so it seems reasonable
                // to assume that this is better.
                for rect in rects
                    .iter()
                    .map(|rect| rect.intersect(bounds))
                    .filter(|rect| !rect.is_empty())
                {
                    let y = if negate_ndc {
                        height - rect.y1
                    } else {
                        rect.y0
                    };

                    self.gl.scissor(
                        i32::from(rect.x0),
                        i32::from(y),
                        i32::from(rect.width()),
                        i32::from(rect.height()),
                    );
                    self.gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);
                }
                self.gl.disable(WebGl2RenderingContext::SCISSOR_TEST);
            }
        }
    }
}
