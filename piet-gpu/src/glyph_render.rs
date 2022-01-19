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

use swash::{scale::ScaleContext, CacheKey, FontRef};

use crate::{encoder::GlyphEncoder, PietGpuRenderContext};

pub struct GlyphRenderer {
    pub render_ctx: PietGpuRenderContext,
    scale_context: ScaleContext,
}

#[repr(transparent)]
pub struct FontId(CacheKey);

impl GlyphRenderer {
    pub fn new() -> GlyphRenderer {
        GlyphRenderer {
            render_ctx: PietGpuRenderContext::new(),
            scale_context: ScaleContext::new(),
        }
    }

    pub unsafe fn add_glyph(&mut self, font_data: &[u8], font_id: u64, glyph_id: u16) {
        // This transmute is dodgy because the definition in swash isn't repr(transparent).
        // I think the best solution is to have a from_u64 method, but we'll work that out
        // later.
        let font_id = FontId(std::mem::transmute(font_id));
        let encoder = self.make_glyph(font_data, font_id, glyph_id);
        self.render_ctx.encode_glyph(&encoder);
        // TODO: don't fill glyph if RGBA
        self.render_ctx.fill_glyph(0xff_ff_ff_ff);
    }

    fn make_glyph(&mut self, font_data: &[u8], font_id: FontId, glyph_id: u16) -> GlyphEncoder {
        let mut encoder = GlyphEncoder::default();
        let font_ref = FontRef {
            data: font_data,
            offset: 0,
            key: font_id.0,
        };
        let mut scaler = self.scale_context.builder(font_ref).size(2048.).build();
        if let Some(outline) = scaler.scale_outline(glyph_id) {
            crate::text::append_outline(&mut encoder, outline.verbs(), outline.points());
        }
        encoder
    }
}
