// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Processing and drawing glyphs.

use crate::peniko::Font;
use skrifa::OutlineGlyphCollection;
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

    /// Set the per-glyph transform. Use `Affine::skew` with a horizontal-only skew to simulate
    /// italic text.
    pub fn glyph_transform(mut self, transform: Affine) -> Self {
        self.run.glyph_transform = Some(transform);
        self
    }

    /// Set whether font hinting is enabled.
    ///
    /// This performs vertical hinting only. Hinting is performed only if the combined `transform`
    /// and `glyph_transform` have a uniform scale and no vertical skew or rotation.
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
        let font =
            skrifa::FontRef::from_index(self.run.font.data.as_ref(), self.run.font.index).unwrap();
        let outlines = font.outline_glyphs();

        let PreparedGlyphRun {
            transform,
            size,
            normalized_coords,
            hinting_instance,
        } = prepare_glyph_run(&self.run, &outlines);

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
                DrawSettings::unhinted(size, normalized_coords)
            };
            let Some(outline) = outlines.get(GlyphId::new(glyph.id)) else {
                continue;
            };
            path.0.truncate(0);
            if outline.draw(draw_settings, &mut path).is_err() {
                continue;
            }

            // Calculate the global glyph translation based on the glyph's local position within
            // the run and the run's global transform.
            //
            // This is a partial affine matrix multiplication, calculating only the translation
            // component that we need. It is added below to calculate the total transform of this
            // glyph.
            let [a, b, c, d, _, _] = self.run.transform.as_coeffs();
            let translation = Vec2::new(
                a * glyph.x as f64 + c * glyph.y as f64,
                b * glyph.x as f64 + d * glyph.y as f64,
            );

            // When hinting, ensure the y-offset is integer. The x-offset doesn't matter, as we
            // perform vertical-only hinting.
            let mut total_transform = transform.then_translate(translation).as_coeffs();
            if hinting_instance.is_some() {
                total_transform[5] = total_transform[5].round();
            }

            render_glyph(
                self.renderer,
                PreparedGlyph::Outline(OutlineGlyph {
                    path: &path.0,
                    transform: Affine::new(total_transform),
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
    /// Per-glyph transform. Use [`Affine::skew`] with horizontal-skew only to simulate italic
    /// text.
    glyph_transform: Option<Affine>,
    /// Normalized variation coordinates for variable fonts.
    normalized_coords: &'a [skrifa::instance::NormalizedCoord],
    /// Controls whether font hinting is enabled.
    hint: bool,
}

struct PreparedGlyphRun<'a> {
    /// The total transform (`global_transform * glyph_transform`), not accounting for glyph
    /// translation.
    transform: Affine,
    /// The font size to generate glyph outlines for.
    size: Size,
    normalized_coords: &'a [skrifa::instance::NormalizedCoord],
    hinting_instance: Option<HintingInstance>,
}

/// Prepare a glyph run for rendering.
///
/// This function calculates the appropriate transform, size, and scaling parameters
/// for proper font hinting when enabled and possible.
fn prepare_glyph_run<'a>(
    run: &GlyphRun<'a>,
    outlines: &OutlineGlyphCollection<'_>,
) -> PreparedGlyphRun<'a> {
    if !run.hint {
        return PreparedGlyphRun {
            transform: run.transform * run.glyph_transform.unwrap_or(Affine::IDENTITY),
            size: Size::new(run.font_size),
            normalized_coords: run.normalized_coords,
            hinting_instance: None,
        };
    }

    // We perform vertical-only hinting.
    //
    // Hinting doesn't make sense if we later scale the glyphs via some transform. So we extract
    // the scale from the global transform and glyph transform and apply it to the font size for
    // hinting. We do require the scaling to be uniform: simply using the vertical scale as font
    // size and then transforming by the relative horizontal scale can cause, e.g., overlapping
    // glyphs. Note that this extracted scale should be later applied to the glyph's position.
    //
    // As the hinting is vertical-only, we can handle horizontal skew, but not vertical skew or
    // rotations.

    let total_transform = run.transform * run.glyph_transform.unwrap_or(Affine::IDENTITY);
    let [t_a, t_b, t_c, t_d, t_e, t_f] = total_transform.as_coeffs();

    let uniform_scale = t_a == t_d;
    let vertically_uniform = t_b == 0.;

    if uniform_scale && vertically_uniform {
        let vertical_font_size = run.font_size * t_d as f32;
        let size = Size::new(vertical_font_size);
        let hinting_instance =
            HintingInstance::new(outlines, size, run.normalized_coords, HINTING_OPTIONS).ok();
        PreparedGlyphRun {
            transform: Affine::new([1., 0., t_c, 1., t_e, t_f]),
            size,
            normalized_coords: run.normalized_coords,
            hinting_instance,
        }
    } else {
        PreparedGlyphRun {
            transform: run.transform * run.glyph_transform.unwrap_or(Affine::IDENTITY),
            size: Size::new(run.font_size),
            normalized_coords: run.normalized_coords,
            hinting_instance: None,
        }
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
}
