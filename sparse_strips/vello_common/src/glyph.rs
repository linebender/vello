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
pub enum PreparedGlyph<'a> {
    /// A glyph defined by its outline.
    Outline(OutlineGlyph<'a>),
    // TODO: Image and Colr variants.
}

/// A glyph defined by a path (its outline) and a local transform.
#[derive(Debug)]
pub struct OutlineGlyph<'a> {
    /// The path of the glyph.
    pub path: &'a BezPath,
    /// The global transform of the glyph.
    pub transform: Affine,
}

/// Trait for types that can render glyphs.
pub trait GlyphRenderer {
    /// Fill glyphs with the current paint and fill rule.
    fn fill_glyph(&mut self, glyph: PreparedGlyph<'_>);

    /// Stroke glyphs with the current paint and stroke settings.
    fn stroke_glyph(&mut self, glyph: PreparedGlyph<'_>);
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
    pub fn fill_glyphs(self, glyphs: impl Iterator<Item = Glyph>) {
        self.render(glyphs, Style::Fill);
    }

    /// Consumes the builder and strokes the glyphs with the current configuration.
    pub fn stroke_glyphs(self, glyphs: impl Iterator<Item = Glyph>) {
        self.render(glyphs, Style::Stroke);
    }

    fn render(self, glyphs: impl Iterator<Item = Glyph>, style: Style) {
        let run = self.run;
        let font = skrifa::FontRef::from_index(run.font.data.as_ref(), run.font.index).unwrap();
        let outlines = font.outline_glyphs();
        let (transform, size, scale, hinting_instance) = if run.hint {
            // Hinting doesn't make sense if we later scale the glyphs via `transform`. So, if this glyph can be
            // scaled uniformly, we scale its font size and use that for hinting. If the glyph is rotated or skewed,
            // hinting is not applicable.
            if let Some((scale, transform)) = take_uniform_scale(run.transform) {
                let size = Size::new(run.font_size * scale as f32);
                (
                    transform,
                    size,
                    scale,
                    HintingInstance::new(&outlines, size, run.normalized_coords, HINTING_OPTIONS)
                        .ok(),
                )
            } else {
                (run.transform, Size::new(run.font_size), 1.0, None)
            }
        } else {
            (run.transform, Size::new(run.font_size), 1.0, None)
        };

        let render_glyph = match style {
            Style::Fill => GlyphRenderer::fill_glyph,
            Style::Stroke => GlyphRenderer::stroke_glyph,
        };
        // Reuse the same `path` allocation for each glyph.
        let mut path = OutlinePath(BezPath::new());
        for glyph in glyphs {
            let draw_settings = if let Some(hinting_instance) = &hinting_instance {
                DrawSettings::hinted(hinting_instance, false)
            } else {
                DrawSettings::unhinted(size, run.normalized_coords)
            };
            let Some(outline) = outlines.get(GlyphId::new(glyph.id)) else {
                continue;
            };
            path.0.truncate(0);
            if outline.draw(draw_settings, &mut path).is_err() {
                continue;
            }
            let mut local_transform =
                Affine::translate(Vec2::new(glyph.x as f64 * scale, glyph.y as f64 * scale));
            if let Some(glyph_transform) = run.glyph_transform {
                local_transform *= glyph_transform;
            }

            render_glyph(
                self.renderer,
                PreparedGlyph::Outline(OutlineGlyph {
                    path: &path.0,
                    transform: transform * local_transform,
                }),
            );
        }
    }
}

enum Style {
    Fill,
    Stroke,
}

/// A sequence of glyphs with shared rendering properties.
#[derive(Clone, Debug)]
struct GlyphRun<'a> {
    /// Font for all glyphs in the run.
    font: Font,
    /// Size of the font in pixels per em.
    font_size: f32,
    /// Global transform.
    transform: Affine,
    /// Per-glyph transform. Can be used to apply skew to simulate italic text.
    glyph_transform: Option<Affine>,
    /// Normalized variation coordinates for variable fonts.
    normalized_coords: &'a [skrifa::instance::NormalizedCoord],
    /// Controls whether font hinting is enabled.
    hint: bool,
}

/// If `transform` has a uniform scale without rotation or skew, return the scale factor and the
/// transform with the scale factored out. Translation is unchanged.
fn take_uniform_scale(transform: Affine) -> Option<(f64, Affine)> {
    let [a, b, c, d, e, f] = transform.as_coeffs();
    if a == d && b == 0.0 && c == 0.0 {
        let scale = a;
        let transform_without_scale = Affine::new([1.0, 0.0, 0.0, 1.0, e, f]);

        Some((scale, transform_without_scale))
    } else {
        None
    }
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

    mod take_uniform_scale {
        use super::*;

        #[test]
        fn identity_transform() {
            let identity = Affine::IDENTITY;
            let result = take_uniform_scale(identity);
            assert!(result.is_some());
            let (scale, transform) = result.unwrap();
            assert!((scale - 1.0).abs() < 1e-10);
            assert!(transform == Affine::IDENTITY);
        }

        #[test]
        fn pure_uniform_scale() {
            let scale_transform = Affine::scale(2.5);
            let result = take_uniform_scale(scale_transform);
            assert!(result.is_some());
            let (scale, transform) = result.unwrap();
            assert!((scale - 2.5).abs() < 1e-10);
            assert!(transform == Affine::IDENTITY);
        }

        #[test]
        fn scale_with_translation() {
            let scale_translate = Affine::scale(3.0).then_translate(Vec2::new(10.0, 20.0));
            let result = take_uniform_scale(scale_translate);
            assert!(result.is_some());
            let (scale, transform) = result.unwrap();
            assert!((scale - 3.0).abs() < 1e-10);
            // The translation should be adjusted by the scale factor
            assert!(transform == Affine::translate(Vec2::new(10.0, 20.0)));
        }

        #[test]
        fn pure_translation() {
            let translation = Affine::translate(Vec2::new(5.0, 7.0));
            let result = take_uniform_scale(translation);
            assert!(result.is_some());
            let (scale, transform) = result.unwrap();
            assert!((scale - 1.0).abs() < 1e-10);
            assert!(transform == translation);
        }

        #[test]
        fn non_uniform_scale() {
            let non_uniform = Affine::scale_non_uniform(2.0, 3.0);
            assert!(take_uniform_scale(non_uniform).is_none());
        }

        #[test]
        fn rotation_transform() {
            let rotation = Affine::rotate(std::f64::consts::PI / 4.0);
            assert!(take_uniform_scale(rotation).is_none());
        }

        #[test]
        fn skew_transform() {
            let skew = Affine::skew(0.5, 0.0);
            assert!(take_uniform_scale(skew).is_none());
        }

        #[test]
        fn complex_transform() {
            let complex = Affine::translate(Vec2::new(10.0, 20.0))
                .then_rotate(std::f64::consts::PI / 6.0)
                .then_scale(2.0);
            assert!(take_uniform_scale(complex).is_none());
        }
    }
}
