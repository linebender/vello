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

//! Support for glyph rendering.

use crate::scene::{SceneBuilder, SceneFragment};
use {
    fello::{
        raw::types::GlyphId,
        raw::FontRef,
        scale::{Context, Pen, Scaler},
        FontKey, Setting, Size,
    },
    peniko::kurbo::Affine,
    peniko::{Brush, Color, Fill, Style},
    vello_encoding::Encoding,
};

pub use vello_encoding::Glyph;

/// General context for creating scene fragments for glyph outlines.
pub struct GlyphContext {
    ctx: Context,
}

impl Default for GlyphContext {
    fn default() -> Self {
        Self::new()
    }
}

impl GlyphContext {
    /// Creates a new context.
    pub fn new() -> Self {
        Self {
            ctx: Context::new(),
        }
    }

    /// Creates a new provider for generating scene fragments for glyphs from
    /// the specified font and settings.
    pub fn new_provider<'a, V>(
        &'a mut self,
        font: &FontRef<'a>,
        font_id: Option<FontKey>,
        ppem: f32,
        hint: bool,
        variations: V,
    ) -> GlyphProvider<'a>
    where
        V: IntoIterator,
        V::Item: Into<Setting<f32>>,
    {
        let scaler = self
            .ctx
            .new_scaler()
            .size(Size::new(ppem))
            .hint(hint.then_some(fello::scale::Hinting::VerticalSubpixel))
            .key(font_id)
            .variations(variations)
            .build(font);
        GlyphProvider { scaler }
    }
}

/// Generator for scene fragments containing glyph outlines for a specific
/// font.
pub struct GlyphProvider<'a> {
    scaler: Scaler<'a>,
}

impl<'a> GlyphProvider<'a> {
    /// Returns a scene fragment containing the commands to render the
    /// specified glyph.
    pub fn get(&mut self, gid: u16, brush: Option<&Brush>) -> Option<SceneFragment> {
        let mut fragment = SceneFragment::default();
        let mut builder = SceneBuilder::for_fragment(&mut fragment);
        let mut path = BezPathPen::default();
        self.scaler.outline(GlyphId::new(gid), &mut path).ok()?;
        builder.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            brush.unwrap_or(&Brush::Solid(Color::rgb8(255, 255, 255))),
            None,
            &path.0,
        );
        Some(fragment)
    }

    pub fn encode_glyph(&mut self, gid: u16, style: &Style, encoding: &mut Encoding) -> Option<()> {
        match style {
            Style::Fill(Fill::NonZero) => encoding.encode_linewidth(-1.0),
            Style::Fill(Fill::EvenOdd) => encoding.encode_linewidth(-2.0),
            Style::Stroke(stroke) => encoding.encode_linewidth(stroke.width),
        }
        let mut path = encoding.encode_path(matches!(style, Style::Fill(_)));
        self.scaler.outline(GlyphId::new(gid), &mut path).ok()?;
        if path.finish(false) != 0 {
            Some(())
        } else {
            None
        }
    }
}

#[derive(Default)]
struct BezPathPen(peniko::kurbo::BezPath);

impl Pen for BezPathPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to((x as f64, y as f64))
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to((x as f64, y as f64))
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.0
            .quad_to((cx0 as f64, cy0 as f64), (x as f64, y as f64))
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.0.curve_to(
            (cx0 as f64, cy0 as f64),
            (cx1 as f64, cy1 as f64),
            (x as f64, y as f64),
        )
    }

    fn close(&mut self) {
        self.0.close_path()
    }
}
