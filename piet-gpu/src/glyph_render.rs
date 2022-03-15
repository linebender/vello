// Copyright 2022 The piet-gpu authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

//! An experimental API for glyph rendering.

use piet::{kurbo::Affine, RenderContext};
use swash::{scale::ScaleContext, FontDataRef};

use crate::{encoder::GlyphEncoder, PietGpuRenderContext};

pub struct GlyphOutline {
    encoder: GlyphEncoder,
    upem: u16,
}

impl GlyphOutline {
    pub fn bbox(&self, transform: Option<[f32; 6]>) -> [f32; 4] {
        self.encoder.bbox(transform.map(|transform| {
            Affine::new([
                transform[0] as f64,
                transform[1] as f64,
                transform[2] as f64,
                transform[3] as f64,
                transform[4] as f64,
                transform[5] as f64,
            ])
        }))
    }

    pub fn upem(&self) -> u16 {
        self.upem
    }
}

pub struct GlyphProvider(ScaleContext);

impl GlyphProvider {
    pub fn new() -> GlyphProvider {
        GlyphProvider(ScaleContext::new())
    }

    pub unsafe fn make_glyph(
        &mut self,
        font_data: &[u8],
        font_id: u64,
        glyph_id: u16,
        variations: Option<&[(u32, f32)]>,
    ) -> GlyphOutline {
        let mut encoder = GlyphEncoder::default();
        let font_data = FontDataRef::new(font_data).expect("invalid font");
        let mut font_ref = font_data.get(0).expect("invalid font index");
        // This transmute is dodgy because the definition in swash isn't repr(transparent).
        // I think the best solution is to have a from_u64 method, but we'll work that out
        // later.
        // chad: this will be addressed with the new outline loader
        font_ref.key = std::mem::transmute(font_id);
        let mut builder = self.0.builder(font_ref);
        if let Some(variations) = variations {
            builder = builder.variations(variations);
        }
        let mut scaler = builder.build();
        if let Some(outline) = scaler.scale_outline(glyph_id) {
            crate::text::append_outline(&mut encoder, outline.verbs(), outline.points());
        } else {
            println!("failed to scale");
        }
        // The swash scaler already has access to this value and should expose it.
        let upem = font_ref.metrics(&[]).units_per_em;
        GlyphOutline { encoder, upem }
    }
}

pub struct GlyphRenderer {
    pub render_ctx: PietGpuRenderContext,
}

impl GlyphRenderer {
    pub fn new() -> GlyphRenderer {
        let render_ctx = PietGpuRenderContext::new();
        GlyphRenderer { render_ctx }
    }

    pub fn add_glyph(
        &mut self,
        outline: &GlyphOutline,
        font_size: f32,
        x: f32,
        y: f32,
        transform: [f32; 6],
    ) {
        let affine = Affine::translate((x as f64, y as f64))
            * Affine::new([
                transform[0] as f64,
                transform[1] as f64,
                transform[2] as f64,
                transform[3] as f64,
                transform[4] as f64,
                transform[5] as f64,
            ])
            * Affine::scale(font_size.max(1.0) as f64 / outline.upem.max(1) as f64);
        self.render_ctx.transform(affine);
        self.render_ctx.encode_glyph(&outline.encoder);
        // TODO: don't fill glyph if RGBA
        self.render_ctx.fill_glyph(0xff_ff_ff_ff);
        self.render_ctx.transform(affine.inverse());
    }

    pub fn reset(&mut self) {
        self.render_ctx = PietGpuRenderContext::new();
    }
}
