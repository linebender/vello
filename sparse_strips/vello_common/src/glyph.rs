// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Processing and drawing glyphs.

use crate::kurbo::{Affine, BezPath, Vec2};
use crate::peniko::FontData;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt::{Debug, Formatter};
use hashbrown::hash_map::{Entry, RawEntryMut};
use hashbrown::{Equivalent, HashMap};
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlineGlyphFormat};
use skrifa::raw::TableProvider;
use skrifa::{FontRef, OutlineGlyphCollection};
use skrifa::{
    GlyphId, MetadataProvider,
    outline::{HintingInstance, HintingOptions, OutlinePen},
};

use crate::colr::convert_bounding_box;
use crate::encode::x_y_advances;
use crate::kurbo::Rect;
use crate::pixmap::Pixmap;
use skrifa::bitmap::{BitmapData, BitmapFormat, BitmapStrikes, Origin};

#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;

/// Positioned glyph.
#[derive(Copy, Clone, Default, Debug)]
pub struct Glyph {
    /// The font-specific identifier for this glyph.
    ///
    /// This ID is specific to the font being used and corresponds to the
    /// glyph index within that font. It is *not* a Unicode code point.
    pub id: u32,
    /// X-offset in run, relative to transform.
    pub x: f32,
    /// Y-offset in run, relative to transform.
    pub y: f32,
}

/// A type of glyph.
#[derive(Debug)]
pub enum GlyphType<'a> {
    /// An outline glyph.
    Outline(OutlineGlyph<'a>),
    /// A bitmap glyph.
    Bitmap(BitmapGlyph),
    /// A COLR glyph.
    Colr(Box<ColorGlyph<'a>>),
}

/// A simplified representation of a glyph, prepared for easy rendering.
#[derive(Debug)]
pub struct PreparedGlyph<'a> {
    /// The type of glyph.
    pub glyph_type: GlyphType<'a>,
    /// The global transform of the glyph.
    pub transform: Affine,
}

/// A glyph defined by a path (its outline) and a local transform.
#[derive(Debug)]
pub struct OutlineGlyph<'a> {
    /// The path of the glyph.
    pub path: &'a BezPath,
}

/// A glyph defined by a bitmap.
#[derive(Debug)]
pub struct BitmapGlyph {
    /// The pixmap of the glyph.
    pub pixmap: Pixmap,
    /// The rectangular area that should be filled with the bitmap when painting.
    pub area: Rect,
}

/// A glyph defined by a COLR glyph description.
///
/// Clients are supposed to first draw the glyph into an intermediate image texture/pixmap
/// and then render that into the actual scene, in a similar fashion to
/// bitmap glyphs.
pub struct ColorGlyph<'a> {
    pub(crate) skrifa_glyph: skrifa::color::ColorGlyph<'a>,
    pub(crate) location: LocationRef<'a>,
    pub(crate) font_ref: &'a FontRef<'a>,
    pub(crate) draw_transform: Affine,
    /// The rectangular area that should be filled with the rendered representation of the
    /// COLR glyph when painting.
    pub area: Rect,
    /// The width of the pixmap/texture in pixels to which the glyph should be rendered to.
    pub pix_width: u16,
    /// The height of the pixmap/texture in pixels to which the glyph should be rendered to.
    pub pix_height: u16,
}

impl Debug for ColorGlyph<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "ColorGlyph")
    }
}

/// Trait for types that can render glyphs.
pub trait GlyphRenderer {
    /// Fill glyphs with the current paint and fill rule.
    fn fill_glyph(&mut self, glyph: PreparedGlyph<'_>);

    /// Stroke glyphs with the current paint and stroke settings.
    fn stroke_glyph(&mut self, glyph: PreparedGlyph<'_>);

    /// Takes the glyph caches from the renderer for use in a glyph run.
    ///
    /// NOTE: The caller must restore the caches after the glyph run is done.
    fn take_glyph_caches(&mut self) -> GlyphCaches;

    /// Restores the glyph caches after a glyph run.
    ///
    /// The caches must have been previously taken with `take_glyph_caches`.
    fn restore_glyph_caches(&mut self, caches: GlyphCaches);
}

/// A builder for configuring and drawing glyphs.
#[derive(Debug)]
#[must_use = "Methods on the builder don't do anything until `render` is called."]
pub struct GlyphRunBuilder<'a, T: GlyphRenderer + 'a> {
    run: GlyphRun<'a>,
    renderer: &'a mut T,
}

impl<'a, T: GlyphRenderer + 'a> GlyphRunBuilder<'a, T> {
    /// Creates a new builder for drawing glyphs.
    pub fn new(font: FontData, transform: Affine, renderer: &'a mut T) -> Self {
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
        let font_ref =
            FontRef::from_index(self.run.font.data.as_ref(), self.run.font.index).unwrap();

        let upem: f32 = font_ref.head().map(|h| h.units_per_em()).unwrap().into();

        let outlines = font_ref.outline_glyphs();
        let color_glyphs = font_ref.color_glyphs();
        let bitmaps = font_ref.bitmap_strikes();

        // TODO: Consider using a drop guard so that panics return the caches to the renderer.
        let GlyphCaches {
            mut hinting_cache,
            mut outline_cache,
        } = self.renderer.take_glyph_caches();
        let mut outline_cache_session =
            OutlineCacheSession::new(&mut outline_cache, VarLookupKey(self.run.normalized_coords));
        let PreparedGlyphRun {
            transform: initial_transform,
            size,
            normalized_coords,
            hinting_instance,
        } = prepare_glyph_run(&self.run, &outlines, &mut hinting_cache);

        let render_glyph = match style {
            Style::Fill => GlyphRenderer::fill_glyph,
            Style::Stroke => GlyphRenderer::stroke_glyph,
        };

        for glyph in glyphs {
            let bitmap_data = bitmaps
                .glyph_for_size(Size::new(self.run.font_size), GlyphId::new(glyph.id))
                .and_then(|g| match g.data {
                    #[cfg(feature = "png")]
                    BitmapData::Png(data) => Pixmap::from_png(data).ok().map(|d| (g, d)),
                    #[cfg(not(feature = "png"))]
                    BitmapData::Png(_) => None,
                    // The others are not worth implementing for now (unless we can find a test case),
                    // they should be very rare.
                    BitmapData::Bgra(_) => None,
                    BitmapData::Mask(_) => None,
                });

            let (glyph_type, transform) =
                if let Some(color_glyph) = color_glyphs.get(GlyphId::new(glyph.id)) {
                    prepare_colr_glyph(
                        &font_ref,
                        glyph,
                        self.run.font_size,
                        upem,
                        initial_transform,
                        color_glyph,
                        normalized_coords,
                    )
                } else if let Some((bitmap_glyph, pixmap)) = bitmap_data {
                    prepare_bitmap_glyph(
                        &bitmaps,
                        glyph,
                        pixmap,
                        self.run.font_size,
                        upem,
                        initial_transform,
                        bitmap_glyph,
                    )
                } else {
                    let Some(outline) = outlines.get(GlyphId::new(glyph.id)) else {
                        continue;
                    };

                    prepare_outline_glyph(
                        glyph,
                        self.run.font.data.id(),
                        self.run.font.index,
                        &mut outline_cache_session,
                        size,
                        initial_transform,
                        self.run.transform,
                        &outline,
                        hinting_instance,
                        normalized_coords,
                    )
                };

            let prepared_glyph = PreparedGlyph {
                glyph_type,
                transform,
            };

            render_glyph(self.renderer, prepared_glyph);
        }

        self.renderer.restore_glyph_caches(GlyphCaches {
            outline_cache,
            hinting_cache,
        });
    }
}

fn prepare_outline_glyph<'a>(
    glyph: Glyph,
    font_id: u64,
    font_index: u32,
    outline_cache: &'a mut OutlineCacheSession<'_>,
    size: Size,
    // The transform of the run + the per-glyph transform.
    initial_transform: Affine,
    // The transform of the run, without the per-glyph transform.
    run_transform: Affine,
    outline_glyph: &skrifa::outline::OutlineGlyph<'a>,
    hinting_instance: Option<&HintingInstance>,
    normalized_coords: &[skrifa::instance::NormalizedCoord],
) -> (GlyphType<'a>, Affine) {
    let path = outline_cache.get_or_insert(
        glyph.id,
        font_id,
        font_index,
        size,
        VarLookupKey(normalized_coords),
        outline_glyph,
        hinting_instance,
    );

    // Calculate the global glyph translation based on the glyph's local position within
    // the run and the run's global transform.
    //
    // This is a partial affine matrix multiplication, calculating only the translation
    // component that we need. It is added below to calculate the total transform of this
    // glyph.
    let [a, b, c, d, _, _] = run_transform.as_coeffs();
    let translation = Vec2::new(
        a * f64::from(glyph.x) + c * f64::from(glyph.y),
        b * f64::from(glyph.x) + d * f64::from(glyph.y),
    );

    // When hinting, ensure the y-offset is integer. The x-offset doesn't matter, as we
    // perform vertical-only hinting.
    let mut final_transform = initial_transform
        .then_translate(translation)
        // Account for the fact that the coordinate system of fonts
        // is upside down.
        .pre_scale_non_uniform(1.0, -1.0)
        .as_coeffs();

    if hinting_instance.is_some() {
        final_transform[5] = final_transform[5].round();
    }

    (
        GlyphType::Outline(OutlineGlyph { path: &path.0 }),
        Affine::new(final_transform),
    )
}

fn prepare_bitmap_glyph<'a>(
    bitmaps: &BitmapStrikes<'_>,
    glyph: Glyph,
    pixmap: Pixmap,
    font_size: f32,
    upem: f32,
    initial_transform: Affine,
    bitmap_glyph: skrifa::bitmap::BitmapGlyph<'a>,
) -> (GlyphType<'a>, Affine) {
    let x_scale_factor = font_size / bitmap_glyph.ppem_x;
    let y_scale_factor = font_size / bitmap_glyph.ppem_y;
    let font_units_to_size = font_size / upem;

    // CoreText appears to special case Apple Color Emoji, adding
    // a 100 font unit vertical offset. We do the same but only
    // when both vertical offsets are 0 to avoid incorrect
    // rendering if Apple ever does encode the offset directly in
    // the font.
    let bearing_y = if bitmap_glyph.bearing_y == 0.0 && bitmaps.format() == Some(BitmapFormat::Sbix)
    {
        100.0
    } else {
        bitmap_glyph.bearing_y
    };

    let origin_shift = match bitmap_glyph.placement_origin {
        Origin::TopLeft => Vec2::default(),
        Origin::BottomLeft => Vec2 {
            x: 0.,
            y: -f64::from(pixmap.height()),
        },
    };

    let transform = initial_transform
        .pre_translate(Vec2::new(glyph.x.into(), glyph.y.into()))
        // Apply outer bearings.
        .pre_translate(Vec2 {
            x: (-bitmap_glyph.bearing_x * font_units_to_size).into(),
            y: (bearing_y * font_units_to_size).into(),
        })
        // Scale to pixel-space.
        .pre_scale_non_uniform(f64::from(x_scale_factor), f64::from(y_scale_factor))
        // Apply inner bearings.
        .pre_translate(Vec2 {
            x: (-bitmap_glyph.inner_bearing_x).into(),
            y: (-bitmap_glyph.inner_bearing_y).into(),
        })
        .pre_translate(origin_shift);

    // Scale factor already accounts for ppem, so we can just draw in the size of the
    // actual image
    let area = Rect::new(
        0.0,
        0.0,
        f64::from(pixmap.width()),
        f64::from(pixmap.height()),
    );

    (GlyphType::Bitmap(BitmapGlyph { pixmap, area }), transform)
}

fn prepare_colr_glyph<'a>(
    font_ref: &'a FontRef<'a>,
    glyph: Glyph,
    font_size: f32,
    upem: f32,
    run_transform: Affine,
    color_glyph: skrifa::color::ColorGlyph<'a>,
    normalized_coords: &'a [skrifa::instance::NormalizedCoord],
) -> (GlyphType<'a>, Affine) {
    // A couple of notes on the implementation here:
    //
    // Firstly, COLR glyphs, similarly to normal outline
    // glyphs, are by default specified in an upside-down coordinate system. They operate
    // on a layer-based push/pop system, where you push new clip or blend layers and then
    // fill the whole available area (within the current clipping area) with a specific paint.
    // Rendering those glyphs in the main scene would be very expensive, as we have to push/pop
    // layers on the whole canvas just to draw a small glyph (at least with the current architecture).
    // Because of this, clients are instead supposed to create an intermediate texture to render the
    // glyph onto and then render it similarly to a bitmap glyph. This also makes it possible to cache
    // the glyphs.
    //
    // Next, there is a problem when rendering COLR glyphs to an intermediate pixmap: The bounding box
    // of a glyph can reach into the negative, meaning that parts of it might be cut off when
    // rendering it directly. Because of this, before drawing we first apply a shift transform so
    // that the bounding box of the glyph starts at (0, 0), then we draw the whole glyph, and
    // finally when positioning the actual pixmap in the scene, we reverse that transform so that
    // the position stays the same as the original one.

    // The scale factor we need to apply to scale from font units to our font size.
    let font_size_scale = (font_size / upem) as f64;

    let transform = run_transform.pre_translate(Vec2::new(glyph.x.into(), glyph.y.into()));

    // Estimate the size of the intermediate pixmap. Ideally, the intermediate bitmap should have
    // exactly one pixel (or more) per device pixel, to ensure that no quality is lost. Therefore,
    // we simply use the scaling/skewing factor to calculate how much to scale by, and use the
    // maximum of both dimensions.
    let (scale_factor_x, scale_factor_y) = {
        let (x_vec, y_vec) = x_y_advances(&transform.pre_scale(f64::from(font_size_scale)));
        (x_vec.length(), y_vec.length())
    };

    let bbox = color_glyph
        .bounding_box(LocationRef::default(), Size::unscaled())
        .map(convert_bounding_box)
        .unwrap_or(Rect::new(0.0, 0.0, f64::from(upem), f64::from(upem)));

    // Calculate the position of the rectangle that will contain the rendered pixmap in device
    // coordinates.
    let scaled_bbox = Rect {
        x0: bbox.x0 * scale_factor_x,
        y0: bbox.y0 * scale_factor_y,
        x1: bbox.x1 * scale_factor_x,
        y1: bbox.y1 * scale_factor_y,
    };

    let glyph_transform = transform
        // There are two things going on here:
        // - On the one hand, for images, the position (0, 0) will be at the top-left, while
        //   for images, the position will be at the bottom-left.
        // - COLR glyphs have a flipped y-axis, so in the intermediate image they will be
        //   upside down.
        // Because of both of these, all we simply need to do is to flip the image on the y-axis.
        // This will ensure that the glyph in the image isn't upside down anymore, and at the same
        // time also flips from having the origin in the top-left to having the origin in the
        // bottom-right.
        * Affine::scale_non_uniform(1.0, -1.0)
        // Overall, the whole pixmap is scaled by `scale_factor_x` and `scale_factor_y`. `scale_factor_x`
        // and `scale_factor_y` are composed by the scale necessary to adjust for the glyph size,
        // as well as the scale that has been applied to the whole glyph run. However, the scale
        // of the whole glyph run will be applied layer on in the render context. If
        // we didn't do anything, the scales would be applied twice (see https://github.com/linebender/vello/pull/1370).
        // Therefore, we apply another scale factor that unapplies the effect of the glyph run transform
        // and only retains the transform necessary to account for the size of the glyph.
        * Affine::scale_non_uniform(font_size_scale / scale_factor_x, font_size_scale / scale_factor_y)
        // Shift the pixmap back so that the bbox aligns with the original position
        // of where the glyph should be placed.
        * Affine::translate((scaled_bbox.x0, scaled_bbox.y0));

    let (pix_width, pix_height) = (
        scaled_bbox.width().ceil() as u16,
        scaled_bbox.height().ceil() as u16,
    );

    let draw_transform =
        // Shift everything so that the bbox starts at (0, 0) and the whole visible area of
        // the glyph will be contained in the intermediate pixmap.
        Affine::translate((-scaled_bbox.x0, -scaled_bbox.y0)) *
        // Scale down to the actual size that the COLR glyph will have in device units.
        Affine::scale_non_uniform(scale_factor_x, scale_factor_y);

    // The shift-back happens in `glyph_transform`, so here we can assume (0.0, 0.0) as the origin
    // of the area we want to draw to.
    let area = Rect::new(0.0, 0.0, scaled_bbox.width(), scaled_bbox.height());

    (
        GlyphType::Colr(Box::new(ColorGlyph {
            skrifa_glyph: color_glyph,
            font_ref,
            location: LocationRef::new(normalized_coords),
            area,
            pix_width,
            pix_height,
            draw_transform,
        })),
        glyph_transform,
    )
}

/// Rendering style for glyphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    /// Fill the glyph.
    Fill,
    /// Stroke the glyph.
    Stroke,
}

/// A sequence of glyphs with shared rendering properties.
#[derive(Clone, Debug)]
struct GlyphRun<'a> {
    /// Font for all glyphs in the run.
    font: FontData,
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
    hinting_instance: Option<&'a HintingInstance>,
}

/// Prepare a glyph run for rendering.
///
/// This function calculates the appropriate transform, size, and scaling parameters
/// for proper font hinting when enabled and possible.
fn prepare_glyph_run<'a>(
    run: &GlyphRun<'a>,
    outlines: &OutlineGlyphCollection<'_>,
    hint_cache: &'a mut HintCache,
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

        let hinting_instance = hint_cache.get(&HintKey {
            font_id: run.font.data.id(),
            font_index: run.font.index,
            outlines,
            size,
            coords: run.normalized_coords,
        });

        PreparedGlyphRun {
            transform: Affine::new([1., 0., t_c, 1., t_e, t_f]),
            size,
            normalized_coords: run.normalized_coords,
            hinting_instance,
        }
    } else {
        PreparedGlyphRun {
            transform: total_transform,
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

#[derive(Clone, Default)]
pub(crate) struct OutlinePath(pub(crate) BezPath);

impl OutlinePath {
    pub(crate) fn new() -> Self {
        Self(BezPath::new())
    }
}

// Note that we flip the y-axis to match our coordinate system.
impl OutlinePen for OutlinePath {
    #[inline]
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to((x, y));
    }

    #[inline]
    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to((x, y));
    }

    #[inline]
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.0.curve_to((cx0, cy0), (cx1, cy1), (x, y));
    }

    #[inline]
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.0.quad_to((cx, cy), (x, y));
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

/// Caches used for glyph rendering.
// TODO: Consider capturing cache performance metrics like hit rate, etc.
#[derive(Debug, Default)]
pub struct GlyphCaches {
    outline_cache: OutlineCache,
    hinting_cache: HintCache,
}

impl GlyphCaches {
    /// Creates a new `GlyphCaches` instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears the glyph caches.
    pub fn clear(&mut self) {
        self.outline_cache.clear();
        self.hinting_cache.clear();
    }

    /// Maintains the glyph caches by evicting unused cache entries.
    ///
    /// Should be called once per scene rendering.
    pub fn maintain(&mut self) {
        self.outline_cache.maintain();
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Default, Debug)]
struct OutlineKey {
    font_id: u64,
    font_index: u32,
    glyph_id: u32,
    size_bits: u32,
    hint: bool,
}

struct OutlineEntry {
    path: OutlinePath,
    serial: u32,
}

impl OutlineEntry {
    const fn new(path: OutlinePath, serial: u32) -> Self {
        Self { path, serial }
    }
}

/// Caches glyph outlines for reuse.
/// Heavily inspired by `vello_encoding::glyph_cache`.
#[derive(Default)]
struct OutlineCache {
    free_list: Vec<OutlinePath>,
    static_map: HashMap<OutlineKey, OutlineEntry>,
    variable_map: HashMap<VarKey, HashMap<OutlineKey, OutlineEntry>>,
    cached_count: usize,
    serial: u32,
    last_prune_serial: u32,
}

impl Debug for OutlineCache {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OutlineCache")
            .field("free_list", &self.free_list.len())
            .field("static_map", &self.static_map.len())
            .field("variable_map", &self.variable_map.len())
            .field("cached_count", &self.cached_count)
            .field("serial", &self.serial)
            .field("last_prune_serial", &self.last_prune_serial)
            .finish()
    }
}

impl OutlineCache {
    fn maintain(&mut self) {
        // Maximum number of full renders where we'll retain an unused glyph
        const MAX_ENTRY_AGE: u32 = 64;
        // Maximum number of full renders before we force a prune
        const PRUNE_FREQUENCY: u32 = 64;
        // Always prune if the cached count is greater than this value
        const CACHED_COUNT_THRESHOLD: usize = 256;
        // Number of encoding buffers we'll keep on the free list
        const MAX_FREE_LIST_SIZE: usize = 128;

        let free_list = &mut self.free_list;
        let serial = self.serial;
        self.serial += 1;
        // Don't iterate over the whole cache every frame
        if serial - self.last_prune_serial < PRUNE_FREQUENCY
            && self.cached_count < CACHED_COUNT_THRESHOLD
        {
            return;
        }
        self.last_prune_serial = serial;
        self.static_map.retain(|_, entry| {
            if serial - entry.serial > MAX_ENTRY_AGE {
                if free_list.len() < MAX_FREE_LIST_SIZE {
                    free_list.push(core::mem::take(&mut entry.path));
                }
                self.cached_count -= 1;
                false
            } else {
                true
            }
        });
        self.variable_map.retain(|_, map| {
            map.retain(|_, entry| {
                if serial - entry.serial > MAX_ENTRY_AGE {
                    if free_list.len() < MAX_FREE_LIST_SIZE {
                        free_list.push(core::mem::take(&mut entry.path));
                    }
                    self.cached_count -= 1;
                    false
                } else {
                    true
                }
            });
            !map.is_empty()
        });
    }

    fn clear(&mut self) {
        self.free_list.clear();
        self.static_map.clear();
        self.variable_map.clear();
        self.cached_count = 0;
        self.serial = 0;
        self.last_prune_serial = 0;
    }
}

struct OutlineCacheSession<'a> {
    map: &'a mut HashMap<OutlineKey, OutlineEntry>,
    free_list: &'a mut Vec<OutlinePath>,
    serial: u32,
    cached_count: &'a mut usize,
}

impl<'a> OutlineCacheSession<'a> {
    fn new(outline_cache: &'a mut OutlineCache, var_key: VarLookupKey<'_>) -> Self {
        let map = if var_key.0.is_empty() {
            &mut outline_cache.static_map
        } else {
            match outline_cache
                .variable_map
                .raw_entry_mut()
                .from_key(&var_key)
            {
                RawEntryMut::Occupied(entry) => entry.into_mut(),
                RawEntryMut::Vacant(entry) => entry.insert(var_key.into(), HashMap::new()).1,
            }
        };
        Self {
            map,
            free_list: &mut outline_cache.free_list,
            serial: outline_cache.serial,
            cached_count: &mut outline_cache.cached_count,
        }
    }

    fn get_or_insert(
        &mut self,
        glyph_id: u32,
        font_id: u64,
        font_index: u32,
        size: Size,
        var_key: VarLookupKey<'_>,
        outline_glyph: &skrifa::outline::OutlineGlyph<'_>,
        hinting_instance: Option<&HintingInstance>,
    ) -> &OutlinePath {
        let key = OutlineKey {
            glyph_id,
            font_id,
            font_index,
            size_bits: size.ppem().unwrap().to_bits(),
            hint: hinting_instance.is_some(),
        };

        match self.map.entry(key) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().serial = self.serial;
                &entry.into_mut().path
            }
            Entry::Vacant(entry) => {
                let mut path = self.free_list.pop().unwrap_or_default();

                let draw_settings = if let Some(hinting_instance) = hinting_instance {
                    DrawSettings::hinted(hinting_instance, false)
                } else {
                    DrawSettings::unhinted(size, var_key.0)
                };

                path.0.truncate(0);
                outline_glyph.draw(draw_settings, &mut path).unwrap();

                let entry = entry.insert(OutlineEntry::new(path, self.serial));
                *self.cached_count += 1;
                &entry.path
            }
        }
    }
}

/// Key for variable font caches.
type VarKey = Vec<skrifa::instance::NormalizedCoord>;

/// Lookup key for variable font caches.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct VarLookupKey<'a>(&'a [skrifa::instance::NormalizedCoord]);

impl Equivalent<VarKey> for VarLookupKey<'_> {
    fn equivalent(&self, other: &VarKey) -> bool {
        self.0 == *other
    }
}

impl From<VarLookupKey<'_>> for VarKey {
    fn from(key: VarLookupKey<'_>) -> Self {
        key.0.to_vec()
    }
}

/// We keep this small to enable a simple LRU cache with a linear
/// search. Regenerating hinting data is low to medium cost so it's fine
/// to redo it occasionally.
const MAX_CACHED_HINT_INSTANCES: usize = 16;

struct HintKey<'a> {
    font_id: u64,
    font_index: u32,
    outlines: &'a OutlineGlyphCollection<'a>,
    size: Size,
    coords: &'a [skrifa::instance::NormalizedCoord],
}

impl HintKey<'_> {
    fn instance(&self) -> Option<HintingInstance> {
        HintingInstance::new(self.outlines, self.size, self.coords, HINTING_OPTIONS).ok()
    }
}

/// LRU cache for hinting instances.
///
/// Heavily inspired by `vello_encoding::glyph_cache`.
#[derive(Default)]
struct HintCache {
    // Split caches for glyf/cff because the instance type can reuse
    // internal memory when reconfigured for the same format.
    glyf_entries: Vec<HintEntry>,
    cff_entries: Vec<HintEntry>,
    serial: u64,
}

impl Debug for HintCache {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HintCache")
            .field("glyf_entries", &self.glyf_entries.len())
            .field("cff_entries", &self.cff_entries.len())
            .field("serial", &self.serial)
            .finish()
    }
}

impl HintCache {
    fn get(&mut self, key: &HintKey<'_>) -> Option<&HintingInstance> {
        let entries = match key.outlines.format()? {
            OutlineGlyphFormat::Glyf => &mut self.glyf_entries,
            OutlineGlyphFormat::Cff | OutlineGlyphFormat::Cff2 => &mut self.cff_entries,
        };
        let (entry_ix, is_current) = find_hint_entry(entries, key)?;
        let entry = entries.get_mut(entry_ix)?;
        self.serial += 1;
        entry.serial = self.serial;
        if !is_current {
            entry.font_id = key.font_id;
            entry.font_index = key.font_index;
            entry
                .instance
                .reconfigure(key.outlines, key.size, key.coords, HINTING_OPTIONS)
                .ok()?;
        }
        Some(&entry.instance)
    }

    fn clear(&mut self) {
        self.glyf_entries.clear();
        self.cff_entries.clear();
        self.serial = 0;
    }
}

struct HintEntry {
    font_id: u64,
    font_index: u32,
    instance: HintingInstance,
    serial: u64,
}

fn find_hint_entry(entries: &mut Vec<HintEntry>, key: &HintKey<'_>) -> Option<(usize, bool)> {
    let mut found_serial = u64::MAX;
    let mut found_index = 0;
    for (ix, entry) in entries.iter().enumerate() {
        if entry.font_id == key.font_id
            && entry.font_index == key.font_index
            && entry.instance.size() == key.size
            && entry.instance.location().coords() == key.coords
        {
            return Some((ix, true));
        }
        if entry.serial < found_serial {
            found_serial = entry.serial;
            found_index = ix;
        }
    }
    if entries.len() < MAX_CACHED_HINT_INSTANCES {
        let instance = key.instance()?;
        let ix = entries.len();
        entries.push(HintEntry {
            font_id: key.font_id,
            font_index: key.font_index,
            instance,
            // This should be updated by the caller.
            serial: 0,
        });
        Some((ix, true))
    } else {
        Some((found_index, false))
    }
}
