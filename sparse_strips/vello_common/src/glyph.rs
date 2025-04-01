// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Processing and drawing glyphs.

use crate::peniko::Font;
use skrifa::instance::Size;
use skrifa::outline::DrawSettings;
use skrifa::{
    GlyphId, MetadataProvider,
    outline::{HintingInstance, HintingOptions, OutlinePen},
};
use vello_api::kurbo::{Affine, BezPath, Vec2};

pub use vello_api::glyph::*;

/// A glyph prepared for rendering.
#[derive(Debug)]
pub enum PreparedGlyph {
    /// A glyph defined by its outline.
    Outline(OutlineGlyph),
    // TODO: Image and Colr variants.
}

/// A glyph defined by a path (its outline) and a local transform.
#[derive(Debug)]
pub struct OutlineGlyph {
    /// The path of the glyph.
    pub path: BezPath,
    /// The local transform of the glyph.
    pub local_transform: Affine,
}

/// Trait for types that can render glyphs.
pub trait GlyphRenderer {
    /// Fill glyphs with the current paint and fill rule.
    fn fill_glyphs(&mut self, glyphs: impl Iterator<Item = PreparedGlyph>);

    /// Stroke glyphs with the current paint and stroke settings.
    fn stroke_glyphs(&mut self, glyphs: impl Iterator<Item = PreparedGlyph>);
}

/// A builder for configuring and drawing glyphs.
#[derive(Debug)]
pub struct GlyphRunBuilder<'a, T: GlyphRenderer + 'a> {
    run: GlyphRun<'a>,
    renderer: &'a mut T,
}

impl<'a, T: GlyphRenderer + 'a> GlyphRunBuilder<'a, T> {
    /// Creates a new builder for drawing glyphs.
    pub fn new(font: Font, transform: Affine, renderer: &'a mut T) -> Self {
        Self {
            run: GlyphRun {
                font,
                font_size: 16.0,
                transform,
                glyph_transform: None,
                hint: true,
                normalized_coords: &[],
            },
            renderer,
        }
    }

    /// Set the font size in pixels per em.
    pub fn font_size(mut self, size: f32) -> Self {
        self.run.font_size = size;
        self
    }

    /// Set the per-glyph transform. Can be used to apply skew to simulate italic text.
    pub fn glyph_transform(mut self, transform: Affine) -> Self {
        self.run.glyph_transform = Some(transform);
        self
    }

    /// Set whether font hinting is enabled.
    pub fn hint(mut self, hint: bool) -> Self {
        self.run.hint = hint;
        self
    }

    /// Set normalized variation coordinates for variable fonts.
    pub fn normalized_coords(mut self, coords: &'a [NormalizedCoord]) -> Self {
        self.run.normalized_coords = bytemuck::cast_slice(coords);
        self
    }

    /// Consumes the builder and fills the glyphs with the current configuration.
    pub fn fill_glyphs(self, glyphs: impl Iterator<Item = &'a Glyph>) {
        self.renderer
            .fill_glyphs(Self::prepare_glyphs(&self.run, glyphs));
    }

    /// Consumes the builder and strokes the glyphs with the current configuration.
    pub fn stroke_glyphs(self, glyphs: impl Iterator<Item = &'a Glyph>) {
        self.renderer
            .stroke_glyphs(Self::prepare_glyphs(&self.run, glyphs));
    }

    fn prepare_glyphs(
        run: &GlyphRun<'a>,
        glyphs: impl Iterator<Item = &'a Glyph>,
    ) -> impl Iterator<Item = PreparedGlyph> {
        let font = skrifa::FontRef::from_index(run.font.data.as_ref(), run.font.index).unwrap();
        let outlines = font.outline_glyphs();
        let size = Size::new(run.font_size);
        let hinting_instance = if run.hint {
            // Rotated, skewed, or other transformations cannot be hinted.
            let [a, b, c, d, _, _] = run.transform.as_coeffs();
            if a == d && b == 0.0 && c == 0.0 {
                // TODO: Cache hinting instance.
                HintingInstance::new(&outlines, size, run.normalized_coords, HINTING_OPTIONS).ok()
            } else {
                None
            }
        } else {
            None
        };
        glyphs.filter_map(move |glyph| {
            let draw_settings = if let Some(hinting_instance) = &hinting_instance {
                DrawSettings::hinted(hinting_instance, false)
            } else {
                DrawSettings::unhinted(size, run.normalized_coords)
            };
            let outline = outlines.get(GlyphId::new(glyph.id))?;
            let mut path = OutlinePath(BezPath::new());
            outline.draw(draw_settings, &mut path).ok()?;
            let mut transform = Affine::translate(Vec2::new(glyph.x as f64, glyph.y as f64));
            if let Some(glyph_transform) = run.glyph_transform {
                transform *= glyph_transform;
            }
            Some(PreparedGlyph::Outline(OutlineGlyph {
                path: path.0,
                local_transform: transform,
            }))
        })
    }
}

/// A sequence of glyphs with shared rendering properties.
#[derive(Clone, Debug)]
struct GlyphRun<'a> {
    /// Font for all glyphs in the run.
    pub font: Font,
    /// Size of the font in pixels per em.
    pub font_size: f32,
    /// Global transform.
    pub transform: Affine,
    /// Per-glyph transform. Can be used to apply skew to simulate italic text.
    pub glyph_transform: Option<Affine>,
    /// Normalized variation coordinates for variable fonts.
    pub normalized_coords: &'a [skrifa::instance::NormalizedCoord],
    /// Controls whether font hinting is enabled.
    pub hint: bool,
}

// TODO: Although these are sane defaults, we might want to make them
// configurable.
const HINTING_OPTIONS: HintingOptions = HintingOptions {
    engine: skrifa::outline::Engine::AutoFallback,
    target: skrifa::outline::Target::Smooth {
        mode: skrifa::outline::SmoothMode::Lcd,
        symmetric_rendering: false,
        preserve_linear_metrics: true,
    },
};

struct OutlinePath(BezPath);

// Note that we flip the y-axis to match our coordinate system.
impl OutlinePen for OutlinePath {
    #[inline]
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to((x, -y));
    }

    #[inline]
    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to((x, -y));
    }

    #[inline]
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.0.curve_to((cx0, -cy0), (cx1, -cy1), (x, -y));
    }

    #[inline]
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.0.quad_to((cx, -cy), (x, -y));
    }

    #[inline]
    fn close(&mut self) {
        self.0.close_path();
    }
}

/// A normalized variation coordinate (for variable fonts) in 2.14 fixed point format.
///
/// In most cases, this can be [cast](bytemuck::cast_slice) from the
/// normalised coords provided by your text layout library.
///
/// Equivalent to [`skrifa::instance::NormalizedCoord`], but defined
/// in Vello so that Skrifa is not part of Vello's public API.
/// This allows Vello to update its Skrifa in a patch release, and limits
/// the need for updates only to align Skrifa versions.
pub type NormalizedCoord = i16;

#[cfg(test)]
mod tests {
    use super::*;

    const _NORMALISED_COORD_SIZE_MATCHES: () =
        assert!(size_of::<skrifa::instance::NormalizedCoord>() == size_of::<NormalizedCoord>());
}
