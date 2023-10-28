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
    fello::meta::MetadataProvider,
    fello::raw::FontRef,
    glyph::{Glyph, GlyphContext},
    kurbo::Affine,
    peniko::{Blob, Brush, BrushRef, Font, StyleRef},
    SceneBuilder,
};

// This is very much a hack to get things working.
// On Windows, can set this to "c:\\Windows\\Fonts\\seguiemj.ttf" to get color emoji
const ROBOTO_FONT: &[u8] = include_bytes!("../../assets/roboto/Roboto-Regular.ttf");
const INCONSOLATA_FONT: &[u8] = include_bytes!("../../assets/inconsolata/Inconsolata.ttf");

pub struct SimpleText {
    gcx: GlyphContext,
    roboto: Font,
    inconsolata: Font,
}

impl SimpleText {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            gcx: GlyphContext::new(),
            roboto: Font::new(Blob::new(Arc::new(ROBOTO_FONT)), 0),
            inconsolata: Font::new(Blob::new(Arc::new(INCONSOLATA_FONT)), 0),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_run<'a>(
        &mut self,
        builder: &mut SceneBuilder,
        font: Option<&Font>,
        size: f32,
        brush: impl Into<BrushRef<'a>>,
        transform: Affine,
        glyph_transform: Option<Affine>,
        style: impl Into<StyleRef<'a>>,
        text: &str,
    ) {
        self.add_var_run(
            builder,
            font,
            size,
            &[],
            brush,
            transform,
            glyph_transform,
            style,
            text,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_var_run<'a>(
        &mut self,
        builder: &mut SceneBuilder,
        font: Option<&Font>,
        size: f32,
        variations: &[(&str, f32)],
        brush: impl Into<BrushRef<'a>>,
        transform: Affine,
        glyph_transform: Option<Affine>,
        style: impl Into<StyleRef<'a>>,
        text: &str,
    ) {
        let default_font = if variations.is_empty() {
            &self.roboto
        } else {
            &self.inconsolata
        };
        let font = font.unwrap_or(default_font);
        let font_ref = to_font_ref(font).unwrap();
        let brush = brush.into();
        let style = style.into();
        let axes = font_ref.axes();
        let fello_size = vello::fello::Size::new(size);
        let coords = axes
            .normalize(variations.iter().copied())
            .collect::<Vec<_>>();
        let charmap = font_ref.charmap();
        let metrics = font_ref.metrics(fello_size, coords.as_slice().into());
        let line_height = metrics.ascent - metrics.descent + metrics.leading;
        let glyph_metrics = font_ref.glyph_metrics(fello_size, coords.as_slice().into());
        let mut pen_x = 0f32;
        let mut pen_y = 0f32;
        builder
            .draw_glyphs(font)
            .font_size(size)
            .transform(transform)
            .glyph_transform(glyph_transform)
            .normalized_coords(&coords)
            .brush(brush)
            .draw(
                style,
                text.chars().filter_map(|ch| {
                    if ch == '\n' {
                        pen_y += line_height;
                        pen_x = 0.0;
                        return None;
                    }
                    let gid = charmap.map(ch).unwrap_or_default();
                    let advance = glyph_metrics.advance_width(gid).unwrap_or_default();
                    let x = pen_x;
                    pen_x += advance;
                    Some(Glyph {
                        id: gid.to_u16() as u32,
                        x,
                        y: pen_y,
                    })
                }),
            );
    }

    pub fn add(
        &mut self,
        builder: &mut SceneBuilder,
        font: Option<&Font>,
        size: f32,
        brush: Option<&Brush>,
        transform: Affine,
        text: &str,
    ) {
        let default_font = FontRef::new(ROBOTO_FONT).unwrap();
        let font = font.and_then(to_font_ref).unwrap_or(default_font);
        let fello_size = vello::fello::Size::new(size);
        let charmap = font.charmap();
        let metrics = font.metrics(fello_size, Default::default());
        let line_height = metrics.ascent - metrics.descent + metrics.leading;
        let glyph_metrics = font.glyph_metrics(fello_size, Default::default());
        let mut pen_x = 0f64;
        let mut pen_y = 0f64;
        let vars: [(&str, f32); 0] = [];
        let mut provider = self.gcx.new_provider(&font, None, size, false, vars);
        for ch in text.chars() {
            if ch == '\n' {
                pen_y += line_height as f64;
                pen_x = 0.0;
                continue;
            }
            let gid = charmap.map(ch).unwrap_or_default();
            let advance = glyph_metrics.advance_width(gid).unwrap_or_default() as f64;
            if let Some(glyph) = provider.get(gid.to_u16(), brush) {
                let xform = transform
                    * Affine::translate((pen_x, pen_y))
                    * Affine::scale_non_uniform(1.0, -1.0);
                builder.append(&glyph, Some(xform));
            }
            pen_x += advance;
        }
    }
}

fn to_font_ref(font: &Font) -> Option<FontRef<'_>> {
    use vello::fello::raw::FileRef;
    let file_ref = FileRef::new(font.data.as_ref()).ok()?;
    match file_ref {
        FileRef::Font(font) => Some(font),
        FileRef::Collection(collection) => collection.get(font.index).ok(),
    }
}
