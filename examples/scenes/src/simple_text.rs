// Copyright 2022 The vello authors.
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

use std::sync::Arc;

use vello::{
    encoding::Glyph,
    glyph::{
        pinot,
        pinot::{FontRef, TableProvider},
        GlyphContext,
    },
    kurbo::Affine,
    peniko::{Blob, Brush, BrushRef, Font, StyleRef},
    SceneBuilder,
};

// This is very much a hack to get things working.
// On Windows, can set this to "c:\\Windows\\Fonts\\seguiemj.ttf" to get color emoji
const FONT_DATA: &[u8] = include_bytes!("../../assets/roboto/Roboto-Regular.ttf");

pub struct SimpleText {
    gcx: GlyphContext,
    font: Font,
}

impl SimpleText {
    pub fn new() -> Self {
        Self {
            gcx: GlyphContext::new(),
            font: Font::new(Blob::new(Arc::new(FONT_DATA)), 0),
        }
    }

    pub fn add_run<'a>(
        &mut self,
        builder: &mut SceneBuilder,
        size: f32,
        brush: impl Into<BrushRef<'a>>,
        transform: Affine,
        glyph_transform: Option<Affine>,
        style: impl Into<StyleRef<'a>>,
        text: &str,
    ) {
        let font = FontRef {
            data: FONT_DATA,
            offset: 0,
        };
        let brush = brush.into();
        let style = style.into();
        if let Some(cmap) = font.cmap() {
            if let Some(hmtx) = font.hmtx() {
                let upem = font.head().map(|head| head.units_per_em()).unwrap_or(1000) as f64;
                let scale = size as f64 / upem;
                let hmetrics = hmtx.hmetrics();
                let default_advance = hmetrics
                    .get(hmetrics.len().saturating_sub(1))
                    .map(|h| h.advance_width)
                    .unwrap_or(0);
                let mut pen_x = 0f64;
                builder
                    .draw_glyphs(&self.font)
                    .font_size(size)
                    .transform(transform)
                    .glyph_transform(glyph_transform)
                    .brush(brush)
                    .draw(
                        style,
                        text.chars().map(|ch| {
                            let gid = cmap.map(ch as u32).unwrap_or(0);
                            let advance = hmetrics
                                .get(gid as usize)
                                .map(|h| h.advance_width)
                                .unwrap_or(default_advance)
                                as f64
                                * scale;
                            let x = pen_x as f32;
                            pen_x += advance;
                            Glyph {
                                id: gid as u32,
                                x,
                                y: 0.0,
                            }
                        }),
                    )
            }
        }
    }

    pub fn add(
        &mut self,
        builder: &mut SceneBuilder,
        font: Option<&FontRef>,
        size: f32,
        brush: Option<&Brush>,
        transform: Affine,
        text: &str,
    ) {
        let font = font.unwrap_or(&FontRef {
            data: FONT_DATA,
            offset: 0,
        });
        if let Some(cmap) = font.cmap() {
            if let Some(hmtx) = font.hmtx() {
                let upem = font.head().map(|head| head.units_per_em()).unwrap_or(1000) as f64;
                let scale = size as f64 / upem;
                let vars: [(pinot::types::Tag, f32); 0] = [];
                let mut provider = self.gcx.new_provider(font, None, size, false, vars);
                let hmetrics = hmtx.hmetrics();
                let default_advance = hmetrics
                    .get(hmetrics.len().saturating_sub(1))
                    .map(|h| h.advance_width)
                    .unwrap_or(0);
                let mut pen_x = 0f64;
                for ch in text.chars() {
                    let gid = cmap.map(ch as u32).unwrap_or(0);
                    let advance = hmetrics
                        .get(gid as usize)
                        .map(|h| h.advance_width)
                        .unwrap_or(default_advance) as f64
                        * scale;
                    if let Some(glyph) = provider.get(gid, brush) {
                        let xform = transform
                            * Affine::translate((pen_x, 0.0))
                            * Affine::scale_non_uniform(1.0, -1.0);
                        builder.append(&glyph, Some(xform));
                    }
                    pen_x += advance;
                }
            }
        }
    }
}
