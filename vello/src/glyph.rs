// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Support for glyph rendering.

use crate::scene::Scene;
use peniko::kurbo::Affine;
use peniko::{Brush, Color, Fill, Style};
use skrifa::instance::{NormalizedCoord, Size};
use skrifa::outline::OutlinePen;
use skrifa::raw::FontRef;
use skrifa::setting::Setting;
use skrifa::{GlyphId, OutlineGlyphCollection};
use vello_encoding::{Encoding, Index};

use peniko::kurbo::Shape;
pub use skrifa;
use skrifa::outline::DrawSettings;
use skrifa::MetadataProvider;
pub use vello_encoding::Glyph;

/// General context for creating scene fragments for glyph outlines.
#[derive(Default)]
pub struct GlyphContext {
    coords: Vec<NormalizedCoord>,
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
    pub fn get(&mut self, gid: u16, brush: Option<&Brush>) -> Option<Scene> {
        let mut scene = Scene::new();
        let mut path = BezPathPen::default();
        let outline = self.outlines.get(GlyphId::new(gid))?;
        let draw_settings = DrawSettings::unhinted(self.size, self.coords);
        outline.draw(draw_settings, &mut path).ok()?;
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            brush.unwrap_or(&Brush::Solid(Color::rgb8(255, 255, 255))),
            None,
            &path.0,
        );
        Some(scene)
    }

    pub fn encode_glyph(&mut self, gid: u16, style: &Style, encoding: &mut Encoding) -> Option<()> {
        let fill = match style {
            Style::Fill(fill) => *fill,
            Style::Stroke(_) => Fill::NonZero,
        };
        encoding.encode_fill_style(&mut Index::default(), fill);
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
