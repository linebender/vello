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
    peniko::kurbo::Affine,
    peniko::{Brush, Color, Fill, Style},
    skrifa::{
        instance::{NormalizedCoord, Size},
        outline::OutlinePen,
        raw::FontRef,
        setting::Setting,
        GlyphId, OutlineGlyphCollection,
    },
    vello_encoding::Encoding,
};

use peniko::kurbo::Shape;
pub use skrifa;
use skrifa::{outline::DrawSettings, MetadataProvider};
pub use vello_encoding::Glyph;

/// General context for creating scene fragments for glyph outlines.
pub struct GlyphContext {
    coords: Vec<NormalizedCoord>,
}

impl Default for GlyphContext {
    fn default() -> Self {
        Self::new()
    }
}

impl GlyphContext {
    /// Creates a new context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new provider for generating scene fragments for glyphs from
    /// the specified font and settings.
    pub fn new_provider<'a, V>(
        &'a mut self,
        font: &FontRef<'a>,
        ppem: f32,
        _hint: bool,
        variations: V,
    ) -> GlyphProvider<'a>
    where
        V: IntoIterator,
        V::Item: Into<Setting<f32>>,
    {
        let outlines = font.outline_glyphs();
        let size = Size::new(ppem);
        let axes = font.axes();
        let axis_count = axes.len();
        self.coords.clear();
        self.coords.resize(axis_count, Default::default());
        axes.location_to_slice(variations, &mut self.coords);
        if self.coords.iter().all(|x| *x == NormalizedCoord::default()) {
            self.coords.clear();
        }
        GlyphProvider {
            outlines,
            size,
            coords: &self.coords,
        }
    }
}

/// Generator for scene fragments containing glyph outlines for a specific
/// font.
pub struct GlyphProvider<'a> {
    outlines: OutlineGlyphCollection<'a>,
    size: Size,
    coords: &'a [NormalizedCoord],
}

impl<'a> GlyphProvider<'a> {
    /// Returns a scene fragment containing the commands to render the
    /// specified glyph.
    pub fn get(&mut self, gid: u16, brush: Option<&Brush>) -> Option<SceneFragment> {
        let mut fragment = SceneFragment::default();
        let mut builder = SceneBuilder::for_fragment(&mut fragment);
        let mut path = BezPathPen::default();
        let outline = self.outlines.get(GlyphId::new(gid))?;
        let draw_settings = DrawSettings::unhinted(self.size, self.coords);
        outline.draw(draw_settings, &mut path).ok()?;
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
        let fill = match style {
            Style::Fill(fill) => *fill,
            Style::Stroke(_) => Fill::NonZero,
        };
        encoding.encode_fill_style(fill);
        let mut path = encoding.encode_path(true);
        let outline = self.outlines.get(GlyphId::new(gid))?;
        let draw_settings = DrawSettings::unhinted(self.size, self.coords);
        match style {
            Style::Fill(_) => {
                outline.draw(draw_settings, &mut path).ok()?;
            }
            Style::Stroke(stroke) => {
                const STROKE_TOLERANCE: f64 = 0.01;
                let mut pen = BezPathPen::default();
                outline.draw(draw_settings, &mut pen).ok()?;
                let stroked = peniko::kurbo::stroke(
                    pen.0.path_elements(STROKE_TOLERANCE),
                    stroke,
                    &Default::default(),
                    STROKE_TOLERANCE,
                );
                path.shape(&stroked);
            }
        }
        if path.finish(false) != 0 {
            Some(())
        } else {
            None
        }
    }
}

#[derive(Default)]
struct BezPathPen(peniko::kurbo::BezPath);

impl OutlinePen for BezPathPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to((x as f64, y as f64));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to((x as f64, y as f64));
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.0
            .quad_to((cx0 as f64, cy0 as f64), (x as f64, y as f64));
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.0.curve_to(
            (cx0 as f64, cy0 as f64),
            (cx1 as f64, cy1 as f64),
            (x as f64, y as f64),
        );
    }

    fn close(&mut self) {
        self.0.close_path();
    }
}
