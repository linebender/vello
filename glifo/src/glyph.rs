// Copyright 2025 the Vello Authors and the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Processing and drawing glyphs.

#![allow(
    clippy::cast_possible_truncation,
    reason = "We temporarily ignore these because the casts\
only break in edge cases, and some of them are also only related to conversions from f64 to f32."
)]

use crate::Pixmap;
use crate::atlas::AtlasSlot;
use crate::atlas::GlyphCacheKey;
use crate::atlas::key::{SUBPIXEL_BITMAP, SUBPIXEL_COLR, pack_color};
use crate::atlas::{GlyphAtlas, ImageCache};
use crate::color::PremulRgba8;
use crate::color::palette::css::BLACK;
use crate::colr::convert_bounding_box;
use crate::kurbo::Point;
use crate::kurbo::Rect;
use crate::kurbo::Vec2;
use crate::kurbo::{Affine, BezPath};
use crate::kurbo::{Line, ParamCurve as _, PathSeg};
use crate::peniko::FontData;
use crate::renderer::{fill_glyph, render_cached_glyph, stroke_glyph};
use crate::util::AffineExt;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt::{Debug, Formatter};
use core::ops::RangeInclusive;
#[cfg(not(feature = "std"))]
use core_maths::CoreFloat as _;
use hashbrown::hash_map::{Entry, RawEntryMut};
use hashbrown::{Equivalent, HashMap};
use skrifa::bitmap::{BitmapData, BitmapFormat, BitmapStrikes, Origin};
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlineGlyphFormat};
use skrifa::outline::{HintingInstance, HintingOptions, OutlinePen};
use skrifa::raw::TableProvider;
use skrifa::{FontRef, OutlineGlyphCollection};
use skrifa::{GlyphId, MetadataProvider};
use smallvec::SmallVec;

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

/// Pre-packed `BLACK` color as a `u32` for use in `GlyphCacheKey`.
const BLACK_PACKED: u32 = PremulRgba8 {
    r: 0,
    g: 0,
    b: 0,
    a: 255,
}
.to_u32();

/// A type of glyph.
#[derive(Debug)]
pub(crate) enum GlyphType<'a> {
    /// An outline glyph.
    Outline(GlyphOutline),
    /// A bitmap glyph.
    Bitmap(GlyphBitmap),
    /// A COLR glyph.
    Colr(Box<GlyphColr<'a>>),
}

/// Type hint for cached glyph rendering.
///
/// Used when rendering directly from the atlas cache to skip glyph preparation.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CachedGlyphType {
    /// An outline glyph cached in the atlas.
    Outline,
    /// A bitmap glyph cached in the atlas.
    Bitmap,
    /// A COLR glyph cached in the atlas.
    /// The `Rect` parameter contains the fractional area dimensions
    /// to preserve sub-pixel accuracy during rendering.
    Colr(Rect),
}

/// A simplified representation of a glyph, prepared for easy rendering.
#[derive(Debug)]
pub(crate) struct PreparedGlyph<'a> {
    /// The type of glyph.
    pub(crate) glyph_type: GlyphType<'a>,
    /// The global transform of the glyph.
    pub(crate) transform: Affine,
    /// Cache key for renderers that implement glyph caching.
    /// This is `Some` for glyphs that can be cached, `None` otherwise.
    ///
    /// For COLR glyphs, `context_color` is extracted from the renderer's
    /// current paint during cache key creation.
    pub(crate) cache_key: Option<GlyphCacheKey>,
}

/// A glyph defined by a path (its outline) and a local transform.
#[derive(Debug)]
pub(crate) struct GlyphOutline {
    /// The path of the glyph (shared with the outline cache via `Arc`).
    pub(crate) path: Arc<BezPath>,
}

/// A glyph defined by a bitmap.
#[derive(Debug)]
pub(crate) struct GlyphBitmap {
    /// The pixmap of the glyph.
    pub(crate) pixmap: Arc<Pixmap>,
    /// The rectangular area that should be filled with the bitmap when painting.
    pub(crate) area: Rect,
}

/// A glyph defined by a COLR glyph description.
///
/// Clients are supposed to first draw the glyph into an intermediate image texture/pixmap
/// and then render that into the actual scene, in a similar fashion to
/// bitmap glyphs.
pub struct GlyphColr<'a> {
    /// The original skrifa color glyph.
    pub skrifa_glyph: skrifa::color::ColorGlyph<'a>,
    /// The location of the glyph.
    pub location: LocationRef<'a>,
    /// The font reference.
    pub font_ref: &'a FontRef<'a>,
    /// The transform to apply to the glyph.
    pub draw_transform: Affine,
    /// The rectangular area that should be filled with the rendered representation of the
    /// COLR glyph when painting.
    pub area: Rect,
    /// The width of the pixmap/texture in pixels to which the glyph should be rendered to.
    pub pix_width: u16,
    /// The height of the pixmap/texture in pixels to which the glyph should be rendered to.
    pub pix_height: u16,
}

impl Debug for GlyphColr<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "GlyphColr")
    }
}

/// Caches used for preparing glyph drawing.
#[derive(Debug, Default)]
pub struct GlyphPrepCache {
    /// Caches glyph outlines.
    pub(crate) outline_cache: OutlineCache,
    /// Caches hinting instances.
    pub(crate) hinting_cache: HintCache,
    /// Horizontal spans excluded from "ink-skipping" underlines.
    pub(crate) underline_exclusions: Vec<(f64, f64)>,
}

impl GlyphPrepCache {
    /// Borrow this cache bundle mutable for glyph run construction.
    pub fn as_mut(&mut self) -> GlyphPrepCacheMut<'_> {
        GlyphPrepCacheMut {
            outline_cache: &mut self.outline_cache,
            hinting_cache: &mut self.hinting_cache,
            underline_exclusions: &mut self.underline_exclusions,
        }
    }

    /// Clear the glyph preparation caches.
    pub fn clear(&mut self) {
        self.outline_cache.clear();
        self.hinting_cache.clear();
        self.underline_exclusions.clear();
    }

    /// Maintain the glyph preparation caches.
    pub fn maintain(&mut self) {
        self.outline_cache.maintain();
    }
}

/// Mutably borrowed caches used for preparing glyph drawing.
#[derive(Debug)]
pub struct GlyphPrepCacheMut<'a> {
    /// Caches glyph outlines.
    pub(crate) outline_cache: &'a mut OutlineCache,
    /// Caches hinting instances .
    pub(crate) hinting_cache: &'a mut HintCache,
    /// Horizontal spans excluded from "ink-skipping" underlines.
    pub(crate) underline_exclusions: &'a mut Vec<(f64, f64)>,
}

/// Determines whether atlas-backed glyph caching is available for a draw.
#[derive(Debug)]
pub enum AtlasCacher<'a> {
    /// Draw directly without using the atlas cache.
    Disabled,
    /// Enable atlas-backed caching using the provided glyph atlas and image
    /// allocator.
    Enabled(&'a mut GlyphAtlas, &'a mut ImageCache),
}

impl AtlasCacher<'_> {
    fn config(&self) -> Option<&crate::atlas::GlyphCacheConfig> {
        match self {
            Self::Disabled => None,
            Self::Enabled(glyph_atlas, _) => Some(glyph_atlas.config()),
        }
    }

    fn get(&mut self, key: &GlyphCacheKey) -> Option<AtlasSlot> {
        match self {
            Self::Disabled => None,
            Self::Enabled(glyph_atlas, _) => glyph_atlas.get(key),
        }
    }
}

/// A backend for glyph run builders.
pub trait GlyphRunBackend<'a>: Sized {
    /// Enable or disable atlas-backed glyph caching for the glyph run.
    fn atlas_cache(self, enabled: bool) -> Self;

    /// Fill the given glyph sequence using the configured builder state.
    fn fill_glyphs<Glyphs>(self, run: GlyphRun<'a>, glyphs: Glyphs)
    where
        Glyphs: Iterator<Item = Glyph> + Clone;

    /// Stroke the given glyph sequence using the configured builder state.
    fn stroke_glyphs<Glyphs>(self, run: GlyphRun<'a>, glyphs: Glyphs)
    where
        Glyphs: Iterator<Item = Glyph> + Clone;
}

/// Helper struct for rendering a prepared glyph run.
#[derive(Debug)]
pub struct GlyphRunRenderer<'a, 'b, Glyphs: Iterator<Item = Glyph> + Clone> {
    prepared_run: PreparedGlyphRun<'a>,
    outline_cache: &'b mut OutlineCache,
    underline_span_cache: &'b mut Vec<(f64, f64)>,
    glyph_iterator: Glyphs,
    atlas_cacher: AtlasCacher<'b>,
}

impl<'a, 'b, Glyphs: Iterator<Item = Glyph> + Clone> GlyphRunRenderer<'a, 'b, Glyphs> {
    /// Fills the glyphs with the current configuration.
    pub fn fill_glyphs(&mut self, renderer: &mut impl crate::GlyphRenderer) {
        self.draw_glyphs(Style::Fill, renderer);
    }

    /// Strokes the glyphs with the current configuration.
    pub fn stroke_glyphs(&mut self, renderer: &mut impl crate::GlyphRenderer) {
        self.draw_glyphs(Style::Stroke, renderer);
    }

    /// Core rendering loop shared by [`fill_glyphs`](Self::fill_glyphs) and
    /// [`stroke_glyphs`](Self::stroke_glyphs).
    ///
    /// Each glyph is resolved through a priority cascade: COLR > bitmap > outline.
    /// The first matching representation wins. Within each branch the atlas cache
    /// is checked before falling through to the slow path (rasterization / path
    /// construction).
    fn draw_glyphs(&mut self, style: Style, renderer: &mut impl crate::GlyphRenderer) {
        let font_ref = self.prepared_run.font.as_skrifa();
        let upem: f32 = font_ref.head().map(|h| h.units_per_em()).unwrap().into();

        let outlines = font_ref.outline_glyphs();
        let color_glyphs = font_ref.color_glyphs();
        let bitmaps = font_ref.bitmap_strikes();

        let mut outline_cache_session = OutlineCacheSession::new(
            self.outline_cache,
            VarLookupKey(self.prepared_run.normalized_coords),
        );
        let PreparedGlyphRun {
            draw_props,
            run_size: _,
            normalized_coords,
            hinting_instance,
            ..
        } = self.prepared_run;

        let font_id = self.prepared_run.font.data.id();
        let font_index = self.prepared_run.font.index;
        let hinted = hinting_instance.is_some();

        let colr_bitmap_cache_enabled = self
            .atlas_cacher
            .config()
            .is_some_and(|config| draw_props.font_size <= config.max_cached_font_size);
        let outline_cache_enabled = colr_bitmap_cache_enabled
            // Due to the various parameters that would need to be considered in the cache key,
            // we never cache stroked outlines for now. For COLR and bitmap, this doesn't matter
            // because they are always filled anyway.
            && style == Style::Fill;

        let context_color = renderer.get_context_color();
        let context_color_packed = pack_color(context_color);
        for glyph in self.glyph_iterator.clone() {
            let glyph_id = GlyphId::new(glyph.id);

            // ── Speculative outline cache check ─────────────────────────
            // ~99% of glyphs are outlines. The transform and cache key are
            // pure arithmetic, so we probe the cache before the expensive
            // color_glyphs.get() / bitmaps.glyph_for_size() font-table lookups.
            // On a miss we keep both for reuse in the outline branch below.
            let outline_transform =
                calculate_outline_transform(glyph, draw_props, hinting_instance);
            let outline_cache_key = outline_cache_enabled.then(|| {
                let fractional_x = outline_transform.translation().x.fract() as f32;
                GlyphCacheKey::new(
                    font_id,
                    font_index,
                    glyph.id,
                    draw_props.font_size,
                    hinted,
                    fractional_x,
                    BLACK,
                    BLACK_PACKED,
                    normalized_coords,
                )
            });
            if let Some(ref key) = outline_cache_key
                && let Some(cached_slot) = self.atlas_cacher.get(key)
            {
                render_cached_glyph(
                    renderer,
                    cached_slot,
                    outline_transform,
                    CachedGlyphType::Outline,
                );
                continue;
            }

            // ── COLR Glyphs ───────────────────────────────────────────
            if let Some(color_glyph) = color_glyphs.get(glyph_id) {
                let metrics = calculate_colr_metrics(
                    draw_props.font_size,
                    upem,
                    draw_props,
                    glyph,
                    &color_glyph,
                );
                let transform = calculate_colr_transform(&metrics);

                // COLR glyphs are never hinted and have no sub-pixel offset;
                // context_color is part of the key because it affects painted layers.
                let cache_key = colr_bitmap_cache_enabled.then(|| GlyphCacheKey {
                    font_id,
                    font_index,
                    glyph_id: glyph.id,
                    size_bits: draw_props.font_size.to_bits(),
                    hinted: false,
                    subpixel_x: SUBPIXEL_COLR,
                    context_color,
                    context_color_packed,
                    var_coords: SmallVec::from_slice(normalized_coords),
                });

                if let Some(ref key) = cache_key
                    && let Some(cached_slot) = self.atlas_cacher.get(key)
                {
                    // Use fractional scaled_bbox dimensions to preserve sub-pixel accuracy.
                    let area = Rect::new(
                        0.0,
                        0.0,
                        metrics.scaled_bbox.width(),
                        metrics.scaled_bbox.height(),
                    );
                    render_cached_glyph(
                        renderer,
                        cached_slot,
                        transform,
                        CachedGlyphType::Colr(area),
                    );
                    continue;
                }

                // Cache miss — rasterize the COLR glyph from scratch.
                let glyph_type =
                    create_colr_glyph(&font_ref, &metrics, color_glyph, normalized_coords);

                let prepared_glyph = PreparedGlyph {
                    glyph_type,
                    transform,
                    cache_key,
                };
                match style {
                    Style::Fill => fill_glyph(renderer, prepared_glyph, &mut self.atlas_cacher),
                    Style::Stroke => stroke_glyph(renderer, prepared_glyph, &mut self.atlas_cacher),
                }
                continue;
            }

            // ── Bitmap Glyphs ────────────────────────────────────────────
            let bitmap_data: Option<(skrifa::bitmap::BitmapGlyph<'_>, Pixmap)> = bitmaps
                .glyph_for_size(Size::new(draw_props.font_size), glyph_id)
                .and_then(|g| match g.data {
                    #[cfg(feature = "png")]
                    BitmapData::Png(data) => Pixmap::from_png(std::io::Cursor::new(data))
                        .ok()
                        .map(|d| (g, d)),
                    #[cfg(not(feature = "png"))]
                    BitmapData::Png(_) => None,
                    // The others are not worth implementing for now (unless we can find a test case),
                    // they should be very rare.
                    BitmapData::Bgra(_) => None,
                    BitmapData::Mask(_) => None,
                });

            if let Some((bitmap_glyph, pixmap)) = bitmap_data {
                // Bitmaps use the strike's own ppem, not the run's, because the
                // image was pre-rendered at that specific size.
                let bitmap_ppem = bitmap_glyph.ppem_x;
                let transform = calculate_bitmap_transform(
                    glyph,
                    &pixmap,
                    draw_props,
                    draw_props.font_size,
                    upem,
                    &bitmap_glyph,
                    &bitmaps,
                );

                // Bitmaps are not hinted and have no sub-pixel offset or
                // context color; variation coords are irrelevant for fixed strikes.
                let cache_key = colr_bitmap_cache_enabled.then(|| GlyphCacheKey {
                    font_id,
                    font_index,
                    glyph_id: glyph.id,
                    size_bits: bitmap_ppem.to_bits(),
                    hinted: false,
                    subpixel_x: SUBPIXEL_BITMAP,
                    context_color: BLACK,
                    context_color_packed: BLACK_PACKED,
                    var_coords: SmallVec::new(),
                });

                if let Some(ref key) = cache_key
                    && let Some(cached_slot) = self.atlas_cacher.get(key)
                {
                    render_cached_glyph(renderer, cached_slot, transform, CachedGlyphType::Bitmap);
                    continue;
                }

                // Cache miss — wrap the decoded pixmap for rendering.
                let glyph_type = create_bitmap_glyph(pixmap);

                let prepared_glyph = PreparedGlyph {
                    glyph_type,
                    transform,
                    cache_key,
                };
                match style {
                    Style::Fill => fill_glyph(renderer, prepared_glyph, &mut self.atlas_cacher),
                    Style::Stroke => stroke_glyph(renderer, prepared_glyph, &mut self.atlas_cacher),
                }
                continue;
            }

            // ── Outline Glyphs ──────────────────────────────────────────
            // Transform and cache key were already computed at the top of the
            // loop for the speculative check. Reuse them here on a cache miss.

            // Cache miss — fetch the outline from skrifa (expensive: parses font
            // tables), then build the path. Deferred to here so cache hits skip it.
            let Some(outline) = outlines.get(glyph_id) else {
                continue;
            };

            let glyph_type = create_outline_glyph(
                glyph.id,
                font_id,
                font_index,
                &mut outline_cache_session,
                draw_props.font_size,
                &outline,
                hinting_instance,
                normalized_coords,
            );

            let prepared_glyph = PreparedGlyph {
                glyph_type,
                transform: outline_transform,
                cache_key: outline_cache_key,
            };
            match style {
                Style::Fill => fill_glyph(renderer, prepared_glyph, &mut self.atlas_cacher),
                Style::Stroke => stroke_glyph(renderer, prepared_glyph, &mut self.atlas_cacher),
            }
        }
    }

    /// Render a decoration (like an underline) that skips over glyph descenders.
    ///
    /// This implements `text-decoration-skip-ink`-like behavior, where the decoration line is interrupted where it
    /// would overlap with glyph outlines.
    ///
    /// The `x_range` specifies the horizontal position of the decoration, and the `offset` and `size` specify its
    /// vertical position and height (relative to the baseline). The `buffer` specifies how much horizontal space to
    /// leave around each descender.
    pub fn render_decoration(
        &mut self,
        x_range: RangeInclusive<f32>,
        baseline_y: f32,
        offset: f32,
        size: f32,
        buffer: f32,
        renderer: &mut impl crate::DrawSink,
    ) {
        self.decoration_spans(x_range, baseline_y, offset, size, buffer)
            .for_each(|rect| {
                renderer.fill_rect(&rect);
            });
    }

    fn decoration_spans<'c>(
        &'c mut self,
        x_range: RangeInclusive<f32>,
        baseline_y: f32,
        offset: f32,
        size: f32,
        buffer: f32,
    ) -> impl Iterator<Item = Rect> + 'c {
        let font_ref = self.prepared_run.font.as_skrifa();
        let outlines = font_ref.outline_glyphs();

        let PreparedGlyphRun {
            draw_props,
            hinting_instance,
            ..
        } = self.prepared_run;

        // The glyph_transform (e.g. skew for fake italics) affects where the outline points end up. We apply it along
        // with the Y flip to transform from font space (Y up) to layout space (Y down).
        //
        // During the preparation of the glyph run, the transform of the run may be absorbed into
        // `draw_props.font_size`, outlines are generated in that scaled coordinate space. We scale them back
        // to the nominal coordinate space. The glyph-drawing path handles this by
        // simply drawing in global space, but we need to invert it for drawing decorations.
        let outline_to_nominal_scale = f64::from(self.prepared_run.run_size / draw_props.font_size);
        let outline_transform = self
            .prepared_run
            .glyph_transform
            .unwrap_or(Affine::IDENTITY)
            * Affine::FLIP_Y
            * Affine::scale(outline_to_nominal_scale);

        // Buffer to add around each exclusion zone
        let buffer = f64::from(buffer);

        // X range for the decoration line
        let x0 = f64::from(*x_range.start());
        let x1 = f64::from(*x_range.end());

        // Convert offset/size to layout space (Y down).
        // offset is positive above baseline, so negate for layout coordinates.
        let layout_y0 = f64::from(-offset);
        let layout_y1 = f64::from(-offset + size);

        // Get a cache session for this font's variation coordinates
        let var_key = VarLookupKey(self.prepared_run.normalized_coords);
        let mut outline_cache_session = OutlineCacheSession::new(self.outline_cache, var_key);

        // Collect and merge exclusion zones from all glyphs.
        let exclusions = &mut self.underline_span_cache;
        // We `drain` this when creating the iterator, but just in case...
        exclusions.truncate(0);

        for glyph in self.glyph_iterator.clone() {
            // TODO: skip ink for color and bitmap glyphs
            let Some(outline) = outlines.get(GlyphId::new(glyph.id)) else {
                continue;
            };

            let cached = outline_cache_session.get_or_insert(
                glyph.id,
                self.prepared_run.font.data.id(),
                self.prepared_run.font.index,
                draw_props.font_size,
                var_key,
                &outline,
                hinting_instance,
            );

            // If the glyph's bounding box doesn't intersect the underline at all, we don't need to calculate
            // intersections. This saves a lot of time, since most glyphs don't have descenders.
            //
            // We only need the y-extent of the transformed bbox, so we compute it directly using the formula:
            // y' = b*x + d*y + f
            let [_, b, _, d, _, f] = outline_transform.as_coeffs();
            let (y_min, y_max) = {
                let bx0 = b * cached.bbox.x0;
                let bx1 = b * cached.bbox.x1;
                let dy0 = d * cached.bbox.y0;
                let dy1 = d * cached.bbox.y1;
                (
                    f + bx0.min(bx1) + dy0.min(dy1),
                    f + bx0.max(bx1) + dy0.max(dy1),
                )
            };
            if y_max < layout_y0 || y_min > layout_y1 {
                continue;
            }

            let mut rect = Rect {
                x0: f64::INFINITY,
                x1: f64::NEG_INFINITY,
                y0: layout_y0,
                y1: layout_y1,
            };

            for seg in cached.path.segments() {
                // Transform the segment to layout space
                let seg = outline_transform * seg;
                expand_rect_with_segment(&mut rect, seg, layout_y0..=layout_y1);
            }

            // Add glyph position and buffer, then clip to decoration x-range
            let excl_start = (rect.x0 + f64::from(glyph.x) - buffer).max(x0);
            let excl_end = (rect.x1 + f64::from(glyph.x) + buffer).min(x1);

            // Skip if no valid exclusion (empty intersection or outside x-range)
            if excl_start >= excl_end {
                continue;
            }

            // Insert in sorted order and merge with overlapping ranges
            insert_and_merge_range(exclusions, excl_start, excl_end);
        }

        // Draw decoration segments, skipping the exclusion zones
        let y0 = f64::from(baseline_y) + layout_y0;
        let y1 = f64::from(baseline_y) + layout_y1;

        let mut state = Some((exclusions.drain(..), x0));
        core::iter::from_fn(move || {
            let (iter, current_x) = state.as_mut()?;
            let Some((excl_start, excl_end)) = iter.next() else {
                // Draw the trailing rectangle
                let final_rect = Rect::new(*current_x, y0, x1, y1);
                state = None;
                return (final_rect.width() > 0.0).then_some(final_rect);
            };

            // Draw segment before this exclusion
            let rect = Rect::new(*current_x, y0, excl_start, y1);
            *current_x = excl_end;
            Some(rect)
        })
    }
}

/// A builder for configuring and drawing glyphs.
#[derive(Debug)]
#[must_use = "Methods on the builder don't do anything until `render` is called."]
pub struct GlyphRunBuilder<'a, B> {
    run: GlyphRun<'a>,
    backend: B,
}

impl<'a, B> GlyphRunBuilder<'a, B> {
    /// Creates a new builder for drawing glyphs with a pre-bound backend.
    pub fn new(font: FontData, transform: Affine, backend: B) -> Self {
        Self {
            // Note: This needs to be kept in sync with the default in vello_common!
            run: GlyphRun {
                font,
                font_size: 16.0,
                transform,
                glyph_transform: None,
                hint: true,
                normalized_coords: &[],
            },
            backend,
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
}

impl<'a> GlyphRun<'a> {
    // Note: Not sure if we should just remove that method and let each backend
    // call `prepare_glyph_run` manually, it might allow us to reduce the number of
    // generics we need to use. But for now, it seems nice to be able to abstract away
    // the `prepare_glyph_run` method call.
    /// Returns a renderer that can fill, stroke, and decorate this glyph run.
    #[doc(hidden)]
    pub fn build<'b: 'a, Glyphs: Iterator<Item = Glyph> + Clone>(
        self,
        glyphs: Glyphs,
        prep_cache: GlyphPrepCacheMut<'b>,
        atlas_cacher: AtlasCacher<'b>,
    ) -> GlyphRunRenderer<'a, 'b, Glyphs> {
        let prepared_run = prepare_glyph_run(self, prep_cache.hinting_cache);
        GlyphRunRenderer {
            prepared_run,
            glyph_iterator: glyphs,
            outline_cache: prep_cache.outline_cache,
            underline_span_cache: prep_cache.underline_exclusions,
            atlas_cacher,
        }
    }
}

impl<'a, B> GlyphRunBuilder<'a, B>
where
    B: GlyphRunBackend<'a>,
{
    /// Enable or disable the glyph atlas cache.
    pub fn atlas_cache(self, enabled: bool) -> Self {
        Self {
            run: self.run,
            backend: self.backend.atlas_cache(enabled),
        }
    }

    /// Fill the glyphs using the current settings.
    pub fn fill_glyphs<Glyphs>(self, glyphs: Glyphs)
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        let GlyphRunBuilder { run, backend } = self;
        backend.fill_glyphs(run, glyphs);
    }

    /// Stroke the glyphs using the current settings.
    pub fn stroke_glyphs<Glyphs>(self, glyphs: Glyphs)
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        let GlyphRunBuilder { run, backend } = self;
        backend.stroke_glyphs(run, glyphs);
    }
}

/// Insert a range into a sorted list, merging with any overlapping ranges.
fn insert_and_merge_range(ranges: &mut Vec<(f64, f64)>, start: f64, end: f64) {
    // Search backwards from the end to find insertion point. Since glyphs come in visual (left-to-right) order, new
    // ranges are usually at or near the end, making this O(1) in the common case.
    let insert_pos = ranges
        .iter()
        .rposition(|r| r.0 <= start)
        .map_or(0, |i| i + 1);

    // Check if we overlap with the previous range
    let merge_start = insert_pos
        .checked_sub(1)
        .filter(|&i| ranges[i].1 >= start)
        .unwrap_or(insert_pos);

    // Find all overlapping ranges and compute merged bounds
    let new_end = ranges[merge_start..]
        .iter()
        .take_while(|(s, _)| *s <= end)
        .fold(end, |acc, (_, e)| acc.max(*e));

    let merge_end = merge_start
        + ranges[merge_start..]
            .iter()
            .take_while(|(s, _)| *s <= new_end)
            .count();

    // Replace the overlapping ranges with the merged range
    if merge_start < merge_end {
        let new_start = start.min(ranges[merge_start].0);
        ranges.splice(merge_start..merge_end, [(new_start, new_end)]);
    } else {
        ranges.insert(insert_pos, (start, end));
    }
}

fn expand_rect_with_segment(rect: &mut Rect, seg: PathSeg, y_span: RangeInclusive<f64>) {
    // Calculate the rough bounds of the segment from its control points. This is *not* the same as
    // `kurbo::Shape::bounding_box`, which returns a precise bounding box but requires expensively calculating the curve
    // extrema.
    let (mut x_bounds, y_bounds) = match seg {
        PathSeg::Line(line) => (
            (line.p0.x.min(line.p1.x), line.p0.x.max(line.p1.x)),
            (line.p0.y.min(line.p1.y), line.p0.y.max(line.p1.y)),
        ),
        PathSeg::Quad(quad) => (
            (
                quad.p0.x.min(quad.p1.x).min(quad.p2.x),
                quad.p0.x.max(quad.p1.x).max(quad.p2.x),
            ),
            (
                quad.p0.y.min(quad.p1.y).min(quad.p2.y),
                quad.p0.y.max(quad.p1.y).max(quad.p2.y),
            ),
        ),
        PathSeg::Cubic(cubic) => (
            (
                cubic.p0.x.min(cubic.p1.x).min(cubic.p2.x).min(cubic.p3.x),
                cubic.p0.x.max(cubic.p1.x).max(cubic.p2.x).max(cubic.p3.x),
            ),
            (
                cubic.p0.y.min(cubic.p1.y).min(cubic.p2.y).min(cubic.p3.y),
                cubic.p0.y.max(cubic.p1.y).max(cubic.p2.y).max(cubic.p3.y),
            ),
        ),
    };
    // Skip segments entirely outside the y_span
    if y_bounds.1 < *y_span.start() || y_bounds.0 > *y_span.end() {
        return;
    }

    // All we care about are the x-intersections. The intersection methods don't work on infinitely-long lines, so we
    // construct a "long enough" line based on segment bounds. This expansion allows for a little bit of error.
    x_bounds.0 -= 1.0;
    x_bounds.1 += 1.0;
    let top_line = Line::new((x_bounds.0, *y_span.start()), (x_bounds.1, *y_span.start()));
    let bottom_line = Line::new((x_bounds.0, *y_span.end()), (x_bounds.1, *y_span.end()));

    for intersection in seg.intersect_line(top_line) {
        let point = top_line.eval(intersection.line_t);
        // There might be some slight inaccuracy calculating `point` from `line_t`, so we only adjust the x-values
        // instead of using `union_pt`, which may also expand the y-values.
        rect.x0 = rect.x0.min(point.x);
        rect.x1 = rect.x1.max(point.x);
    }

    for intersection in seg.intersect_line(bottom_line) {
        let point = bottom_line.eval(intersection.line_t);
        rect.x0 = rect.x0.min(point.x);
        rect.x1 = rect.x1.max(point.x);
    }

    // Also check segment endpoints that lie within the y-range
    let (seg_start, seg_end) = match seg {
        PathSeg::Line(line) => (line.p0, line.p1),
        PathSeg::Quad(quad) => (quad.p0, quad.p2),
        PathSeg::Cubic(cubic) => (cubic.p0, cubic.p3),
    };

    for point in [seg_start, seg_end] {
        if (*y_span.start()..=*y_span.end()).contains(&point.y) {
            rect.x0 = rect.x0.min(point.x);
            rect.x1 = rect.x1.max(point.x);
        }
    }
}

/// Create outline glyph data from cache.
///
/// This extracts the glyph path from the outline cache, creating a `GlyphType::Outline`
/// without any positioning information.
fn create_outline_glyph<'a>(
    glyph_id: u32,
    font_id: u64,
    font_index: u32,
    outline_cache: &'a mut OutlineCacheSession<'_>,
    size: f32,
    outline_glyph: &skrifa::outline::OutlineGlyph<'a>,
    hinting_instance: Option<&HintingInstance>,
    normalized_coords: &[skrifa::instance::NormalizedCoord],
) -> GlyphType<'a> {
    let cached = outline_cache.get_or_insert(
        glyph_id,
        font_id,
        font_index,
        size,
        VarLookupKey(normalized_coords),
        outline_glyph,
        hinting_instance,
    );

    GlyphType::Outline(GlyphOutline {
        path: Arc::clone(cached.path),
    })
}

/// Calculate transform for outline glyphs.
///
/// This computes the final positioning transform for an outline glyph, taking into account:
/// - Glyph position within the run
/// - Run-space glyph positioning
/// - Y-axis flip (fonts use upside-down coordinate system)
/// - Hinting adjustments (snap y-offset to integer)
fn calculate_outline_transform(
    glyph: Glyph,
    draw_props: DrawProps,
    hinting_instance: Option<&HintingInstance>,
) -> Affine {
    let mut final_transform = draw_props
        .positioned_transform(glyph)
        .pre_scale_non_uniform(1.0, -1.0)
        .as_coeffs();

    if hinting_instance.is_some() {
        final_transform[5] = final_transform[5].round();
    }

    Affine::new(final_transform)
}

/// Create bitmap glyph data.
///
/// This wraps the pixmap in a `GlyphType::Bitmap` with its display area,
/// without any positioning information.
fn create_bitmap_glyph(pixmap: Pixmap) -> GlyphType<'static> {
    // Scale factor already accounts for ppem, so we can just draw in the size of the
    // actual image
    let area = Rect::new(
        0.0,
        0.0,
        f64::from(pixmap.width()),
        f64::from(pixmap.height()),
    );

    GlyphType::Bitmap(GlyphBitmap {
        pixmap: Arc::new(pixmap),
        area,
    })
}

/// Calculate transform for bitmap glyphs.
///
/// This computes the final positioning transform for a bitmap glyph, taking into account:
/// - Glyph position within the run
/// - Bitmap scaling to match requested font size
/// - Bearing adjustments (outer and inner)
/// - Origin placement (top-left vs bottom-left)
/// - Special handling for Apple Color Emoji
fn calculate_bitmap_transform(
    glyph: Glyph,
    pixmap: &Pixmap,
    draw_props: DrawProps,
    font_size: f32,
    upem: f32,
    bitmap_glyph: &skrifa::bitmap::BitmapGlyph<'_>,
    bitmaps: &BitmapStrikes<'_>,
) -> Affine {
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

    draw_props
        .positioned_transform(glyph)
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
        .pre_translate(origin_shift)
}

/// Helper struct containing computed COLR glyph metrics.
struct ColrMetrics {
    /// Base transform with glyph position applied.
    transform: Affine,
    /// Scaled bounding box in device coordinates.
    scaled_bbox: Rect,
    /// Scale factor for x-axis.
    scale_factor_x: f64,
    /// Scale factor for y-axis.
    scale_factor_y: f64,
    /// Font size scale (`font_size` / `upem`).
    font_size_scale: f64,
}

/// Calculate COLR glyph metrics (scale factors, bounding box, etc.).
///
/// This computes the intermediate values needed for both creating the `GlyphColr`
/// and calculating its positioning transform.
fn calculate_colr_metrics(
    font_size: f32,
    upem: f32,
    draw_props: DrawProps,
    glyph: Glyph,
    color_glyph: &skrifa::color::ColorGlyph<'_>,
) -> ColrMetrics {
    // The scale factor we need to apply to scale from font units to our font size.
    let font_size_scale = (font_size / upem) as f64;
    let transform = draw_props.positioned_transform(glyph);

    // Estimate the size of the intermediate pixmap. Ideally, the intermediate bitmap should have
    // exactly one pixel (or more) per device pixel, to ensure that no quality is lost. Therefore,
    // we simply use the scaling/skewing factor to calculate how much to scale each axis by.
    let (scale_factor_x, scale_factor_y) = {
        let (x_vec, y_vec) = x_y_advances(&transform.pre_scale(font_size_scale));
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

    ColrMetrics {
        transform,
        scaled_bbox,
        scale_factor_x,
        scale_factor_y,
        font_size_scale,
    }
}

/// Calculate transform for COLR glyphs.
///
/// This uses pre-calculated metrics to compute the final positioning transform for a COLR glyph,
/// taking into account:
/// - Y-axis flip (fonts use upside-down coordinate system)
/// - Scale compensation (to avoid double-application of run transform scale)
/// - Bounding box alignment
fn calculate_colr_transform(metrics: &ColrMetrics) -> Affine {
    metrics.transform
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
        // of the whole glyph run will be applied later on in the render context. If
        // we didn't do anything, the scales would be applied twice (see https://github.com/linebender/vello/pull/1370).
        // Therefore, we apply another scale factor that unapplies the effect of the glyph run transform
        // and only retains the transform necessary to account for the size of the glyph.
        * Affine::scale_non_uniform(
            metrics.font_size_scale / metrics.scale_factor_x,
            metrics.font_size_scale / metrics.scale_factor_y,
        )
        // Shift the pixmap back so that the bbox aligns with the original position
        // of where the glyph should be placed.
        * Affine::translate((metrics.scaled_bbox.x0, metrics.scaled_bbox.y0))
}

/// Create COLR glyph data with intermediate texture parameters.
///
/// This uses pre-calculated metrics to create a `GlyphType::Colr` with all necessary
/// data for rendering to an intermediate texture.
fn create_colr_glyph<'a>(
    font_ref: &'a FontRef<'a>,
    metrics: &ColrMetrics,
    color_glyph: skrifa::color::ColorGlyph<'a>,
    normalized_coords: &'a [skrifa::instance::NormalizedCoord],
) -> GlyphType<'a> {
    let (pix_width, pix_height) = (
        metrics.scaled_bbox.width().ceil() as u16,
        metrics.scaled_bbox.height().ceil() as u16,
    );

    let draw_transform =
        // Shift everything so that the bbox starts at (0, 0) and the whole visible area of
        // the glyph will be contained in the intermediate pixmap.
        Affine::translate((-metrics.scaled_bbox.x0, -metrics.scaled_bbox.y0)) *
        // Scale down to the actual size that the COLR glyph will have in device units.
        Affine::scale_non_uniform(metrics.scale_factor_x, metrics.scale_factor_y);

    // The shift-back happens in `glyph_transform`, so here we can assume (0.0, 0.0) as the origin
    // of the area we want to draw to.
    let area = Rect::new(
        0.0,
        0.0,
        metrics.scaled_bbox.width(),
        metrics.scaled_bbox.height(),
    );

    GlyphType::Colr(Box::new(GlyphColr {
        skrifa_glyph: color_glyph,
        font_ref,
        location: LocationRef::new(normalized_coords),
        area,
        pix_width,
        pix_height,
        draw_transform,
    }))
}

trait FontDataExt {
    fn as_skrifa(&self) -> FontRef<'_>;
}

impl FontDataExt for FontData {
    fn as_skrifa(&self) -> FontRef<'_> {
        FontRef::from_index(self.data.data(), self.index).unwrap()
    }
}

/// Rendering style for glyphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Style {
    /// Fill the glyph.
    Fill,
    /// Stroke the glyph.
    Stroke,
}

/// A sequence of glyphs with shared rendering properties.
#[derive(Clone, Debug)]
pub struct GlyphRun<'a> {
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
    /// The underlying font data.
    font: FontData,
    // The fact that we store `run_size` and `glyph_transform` here, as well
    // as having more transforms and an effective font size inside of the `draw_props` field is pretty
    // confusing, so here is a brief explanation:
    // Basically, the reason why we need both `run_size` and `glyph_transform` here is that
    // we need to store some of the original metadata in scene space for certain functionality
    // (for example handling of underlines).
    /// The original run size supplied by the caller.
    run_size: f32,
    /// The original per-glyph transform supplied by the caller.
    glyph_transform: Option<Affine>,
    // Continuing the above comment, the problem is that we also need to precalculate data
    // that is needed specifically for glyph rendering. This includes:
    // 1) We need to concatenate run transform and glyph transform to compute the final transform
    // for the glyph outline.
    // 2) Whenever possible, we need to try to _absorb_ the font size into the draw transform,
    // such that we can just use the font size to uniquely identify a glyph cache hit (for example,
    // if we draw a glyph at font size 12 with scale 2, it's the same as drawing the glyph at font size 24).
    // While it would make things easier to just use the cache key in the transform and accept less
    // caching potential for easier code, we would still need scaling absorption to implement proper
    // hinting. Hence, it makes sense to just generalize the whole absorption procedure.
    // In any case, since we do scaling absorption, we cannot use `run_size`, `GlyphRun::transform` and
    // `glyph_transform` for glyph drawing purposes anymore. In particular, it can easily happen
    // that
    // 1) `run_size` != `draw_props.font_size`
    // 2) `run_transform` * `glyph_transform` != `draw_props.effective_transform`.
    // Therefore, we need to track a separate set of fields for glyph-drawing operations.
    /// Properties for turning glyph-local positions into final draw transforms.
    draw_props: DrawProps,
    normalized_coords: &'a [skrifa::instance::NormalizedCoord],
    hinting_instance: Option<&'a HintingInstance>,
}

/// Properties for easily calculating the transform of a positioned glyph.
#[derive(Clone, Copy, Debug)]
struct DrawProps {
    // Why do we need two separate transforms? Fundamentally, the problem is that the order
    // of application should be:
    // `run_transform` * `glyph_position` * `font_size` * `glyph_transform`.
    // As part of absorption, we are only left with a potentially new `font_size` and a merged
    // `effective_transform`. However, the translation that results form `glyph_position` logically
    // needs to be applied after `run_transform` but before `glyph_transform`.
    // Therefore, we need to store two separate transforms: One that is used only to transform
    // the original glyph position, and another one that is used to actually transform the glyph
    // outlines.
    /// A positioning transform for the glyph.
    positioning_transform: Affine,
    /// A transform to apply to the glyph after positioning.
    effective_transform: Affine,
    /// The actual font size that should be assumed for drawing and caching
    /// purposes.
    font_size: f32,
}

impl DrawProps {
    #[inline]
    fn positioned_transform(self, glyph: Glyph) -> Affine {
        // First, determine the "coarse" location of the glyph by applying the scaling/skewing
        // of the original run transform to the glyph position. Note that `positioning_transform`
        // has a translation factor of zero (since it has been absorbed into `effective_transform`), so
        // only the skewing and scaling factors are relevant.
        let translation = self.positioning_transform * Point::new(glyph.x as f64, glyph.y as f64);

        // Now, apply the final draw transform on top of that, which will also consider
        // the original glyph transform.
        Affine::translate(translation.to_vec2()) * self.effective_transform
    }
}

impl Debug for PreparedGlyphRun<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        // HintingInstance doesn't implement Debug so we have to do this manually :(
        f.debug_struct("PreparedGlyphRun")
            .field("font", &self.font)
            .field("run_size", &self.run_size)
            .field("glyph_transform", &self.glyph_transform)
            .field("transforms", &self.draw_props)
            .field("normalized_coords", &self.normalized_coords)
            .finish()
    }
}

/// Prepare a glyph run for rendering.
fn prepare_glyph_run<'a>(run: GlyphRun<'a>, hint_cache: &'a mut HintCache) -> PreparedGlyphRun<'a> {
    let full_transform = run.transform * run.glyph_transform.unwrap_or(Affine::IDENTITY);
    let [_, _, t_c, t_d, t_e, t_f] = full_transform.as_coeffs();

    /// The mode that should be used to handle transforms.
    #[derive(Clone, Copy, Debug)]
    enum PreparedGlyphRunMode {
        /// No absorption has happened, the font size stays the same and the effective transform
        /// is simply the concatenation of run transform and glyph transform.
        ///
        /// No hinting should be applied.
        Direct,
        /// The scaling factor has been absorbed, and hinting should be applied.
        AbsorbScaleUnhinted,
        /// The scaling factor has been absorbed, but not hinting should be applied.
        AbsorbScaleHinted,
    }

    let mode = if !run.hint {
        // TODO: We could explore generalizing this by decomposing the transform, such that
        // we always absorb it, even if there is a skewing factor in the transform. This won't
        // automatically make them eligible for caching because any skewing factor is currently
        // rejected for caching, but it might make the code a bit more consistent.
        if full_transform.is_positive_uniform_scale_without_skew() {
            PreparedGlyphRunMode::AbsorbScaleUnhinted
        } else {
            PreparedGlyphRunMode::Direct
        }
    } else {
        // We perform vertical-only hinting.
        //
        // Hinting doesn't make sense if we later scale the glyphs via some transform. So, similarly to
        // normal glyph runs, we try to extract the scale. As is currently done for unhinted glyph runs, we
        // also expect the scale to be uniform: Simply using the vertical scale as font
        // size and then transforming by the relative horizontal scale can cause, e.g., overlapping
        // glyphs. Note that this extracted scale should be later applied to the glyph's position.
        //
        // As the hinting is vertical-only, we can handle horizontal skew, but not vertical skew or
        // rotations.
        if full_transform.is_positive_uniform_scale_without_vertical_skew() {
            PreparedGlyphRunMode::AbsorbScaleHinted
        } else {
            PreparedGlyphRunMode::Direct
        }
    };

    let (effective_transform, draw_font_size, hinting_instance) = match mode {
        PreparedGlyphRunMode::Direct => (full_transform, run.font_size, None),
        PreparedGlyphRunMode::AbsorbScaleUnhinted => (
            Affine::new([1., 0., 0., 1., t_e, t_f]),
            run.font_size * t_d as f32,
            None,
        ),
        PreparedGlyphRunMode::AbsorbScaleHinted => {
            let vertical_font_size = run.font_size * t_d as f32;
            let font_ref = run.font.as_skrifa();
            let outlines = font_ref.outline_glyphs();
            let hinting_instance = hint_cache.get(&HintKey {
                font_id: run.font.data.id(),
                font_index: run.font.index,
                outlines: &outlines,
                size: vertical_font_size,
                coords: run.normalized_coords,
            });

            (
                // The scale has been absorbed into the font size, so we need to remove it from the skew
                // coefficient (t_c) as well. Otherwise the skew would be applied twice: once via the
                // larger outline, once via the transform. The translation (t_e, t_f) stays as-is since
                // it positions the run in scene coordinates.
                Affine::new([1., 0., t_c / t_d, 1., t_e, t_f]),
                vertical_font_size,
                hinting_instance,
            )
        }
    };

    PreparedGlyphRun {
        font: run.font,
        run_size: run.font_size,
        glyph_transform: run.glyph_transform,
        draw_props: DrawProps {
            positioning_transform: run
                .transform
                // Translation factor is already considered in `effective_transform`, so we need to remove
                // it here.
                .with_translation(Vec2::ZERO),
            effective_transform,
            font_size: draw_font_size,
        },
        normalized_coords: run.normalized_coords,
        hinting_instance,
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
pub(crate) struct OutlinePath {
    pub(crate) path: BezPath,
    pub(crate) bbox: Rect,
}

impl OutlinePath {
    pub(crate) fn new() -> Self {
        Self {
            path: BezPath::new(),
            bbox: Rect {
                x0: f64::INFINITY,
                y0: f64::INFINITY,
                x1: f64::NEG_INFINITY,
                y1: f64::NEG_INFINITY,
            },
        }
    }

    fn reuse(&mut self) {
        self.path.truncate(0);
        self.bbox = Rect {
            x0: f64::INFINITY,
            y0: f64::INFINITY,
            x1: f64::NEG_INFINITY,
            y1: f64::NEG_INFINITY,
        };
    }
}

// Note that we flip the y-axis to match our coordinate system.
impl OutlinePen for OutlinePath {
    #[inline]
    fn move_to(&mut self, x: f32, y: f32) {
        self.path.move_to((x, y));
        self.bbox = self.bbox.union_pt((x, y));
    }

    #[inline]
    fn line_to(&mut self, x: f32, y: f32) {
        self.path.line_to((x, y));
        self.bbox = self.bbox.union_pt((x, y));
    }

    #[inline]
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.path.curve_to((cx0, cy0), (cx1, cy1), (x, y));
        self.bbox = self.bbox.union_pt((cx0, cy0));
        self.bbox = self.bbox.union_pt((cx1, cy1));
        self.bbox = self.bbox.union_pt((x, y));
    }

    #[inline]
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.path.quad_to((cx, cy), (x, y));
        self.bbox = self.bbox.union_pt((cx, cy));
        self.bbox = self.bbox.union_pt((x, y));
    }

    #[inline]
    fn close(&mut self) {
        self.path.close_path();
    }
}

/// A normalized variation coordinate (for variable fonts) in 2.14 fixed point format.
///
/// In most cases, this can be [cast](bytemuck::cast_slice) from the
/// normalised coords provided by your text layout library.
///
/// Equivalent to [`skrifa::instance::NormalizedCoord`], but defined
/// in Glifo so that Skrifa is not part of Glifo's public API.
/// This allows Glifo to update its Skrifa in a patch release, and limits
/// the need for updates only to align Skrifa versions.
pub type NormalizedCoord = i16;

#[cfg(test)]
mod tests {
    use super::*;

    const _NORMALISED_COORD_SIZE_MATCHES: () =
        assert!(size_of::<skrifa::instance::NormalizedCoord>() == size_of::<NormalizedCoord>());
}

/// Caches used for glyph rendering.
///
/// Contains renderer-agnostic caches (outline paths, hinting instances)
/// alongside the glyph atlas bitmap cache.
// TODO: Consider capturing cache performance metrics like hit rate, etc.
#[derive(Debug, Default)]
pub struct GlyphCaches {
    /// Caches glyph outlines (paths) for reuse.
    pub(crate) outline_cache: OutlineCache,
    /// Caches hinting instances for reuse.
    pub(crate) hinting_cache: HintCache,
    /// Horizontal spans excluded from "ink-skipping" underlines. Cached to reuse one allocation.
    pub(crate) underline_exclusions: Vec<(f64, f64)>,
    /// Caches rasterized glyph bitmaps in atlas pages.
    pub(crate) glyph_atlas: GlyphAtlas,
}

impl GlyphCaches {
    /// Clears the glyph caches.
    pub fn clear(&mut self) {
        self.outline_cache.clear();
        self.hinting_cache.clear();
        self.underline_exclusions.clear();
        self.glyph_atlas.clear();
    }

    /// Maintains the glyph caches by evicting unused cache entries.
    ///
    /// The `image_cache` must be the same allocator passed to
    /// `GlyphRunBuilder::build` so that evicted entries are deallocated from
    /// the correct allocator.
    ///
    /// Should be called once per scene rendering.
    pub fn maintain(&mut self, image_cache: &mut ImageCache) {
        self.outline_cache.maintain();
        self.glyph_atlas.maintain(image_cache);
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
    path: Arc<BezPath>,
    bbox: Rect,
    serial: u32,
}

impl OutlineEntry {
    fn new(path: Arc<BezPath>, bbox: Rect, serial: u32) -> Self {
        Self { path, bbox, serial }
    }

    /// Takes the inner `BezPath` out of this entry if the `Arc` is uniquely owned.
    fn take_path(&mut self) -> Option<OutlinePath> {
        let arc = core::mem::replace(&mut self.path, Arc::new(BezPath::new()));
        Arc::try_unwrap(arc).ok().map(|path| OutlinePath {
            path,
            bbox: Rect::ZERO,
        })
    }
}

/// A cached outline glyph path with its approximate bounding box.
pub(crate) struct CachedOutline<'a> {
    pub(crate) path: &'a Arc<BezPath>,
    pub(crate) bbox: Rect,
}

/// Caches glyph outlines for reuse.
/// Heavily inspired by `vello_encoding::glyph_cache`.
#[derive(Default)]
pub struct OutlineCache {
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
    /// Maintains the outline cache by evicting unused cache entries.
    ///
    /// Should be called once per scene rendering.
    pub fn maintain(&mut self) {
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
                    // Try to recover the inner BezPath for reuse as a drawing buffer.
                    // This succeeds when the Arc has no other owners (refcount == 1).
                    if let Some(path) = entry.take_path() {
                        free_list.push(path);
                    }
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
                    if free_list.len() < MAX_FREE_LIST_SIZE
                        && let Some(path) = entry.take_path()
                    {
                        free_list.push(path);
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

    /// Clears the outline cache.
    pub fn clear(&mut self) {
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
        size: f32,
        var_key: VarLookupKey<'_>,
        outline_glyph: &skrifa::outline::OutlineGlyph<'_>,
        hinting_instance: Option<&HintingInstance>,
    ) -> CachedOutline<'_> {
        let key = OutlineKey {
            glyph_id,
            font_id,
            font_index,
            size_bits: size.to_bits(),
            hint: hinting_instance.is_some(),
        };

        match self.map.entry(key) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().serial = self.serial;
                let entry = entry.into_mut();
                CachedOutline {
                    path: &entry.path,
                    bbox: entry.bbox,
                }
            }
            Entry::Vacant(entry) => {
                // Pop a drawing buffer from the free list (or create a new one).
                let mut drawing_buf = self.free_list.pop().unwrap_or_default();

                let draw_settings = if let Some(hinting_instance) = hinting_instance {
                    DrawSettings::hinted(hinting_instance, false)
                } else {
                    DrawSettings::unhinted(Size::new(size), var_key.0)
                };

                drawing_buf.reuse();
                outline_glyph.draw(draw_settings, &mut drawing_buf).unwrap();

                let bbox = drawing_buf.bbox;
                let entry = entry.insert(OutlineEntry::new(
                    Arc::new(drawing_buf.path),
                    bbox,
                    self.serial,
                ));
                *self.cached_count += 1;
                CachedOutline {
                    path: &entry.path,
                    bbox: entry.bbox,
                }
            }
        }
    }
}

/// Key for variable font caches.
type VarKey = SmallVec<[skrifa::instance::NormalizedCoord; 4]>;

/// Lookup key for variable font caches.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct VarLookupKey<'a>(&'a [skrifa::instance::NormalizedCoord]);

impl Equivalent<VarKey> for VarLookupKey<'_> {
    fn equivalent(&self, other: &VarKey) -> bool {
        self.0 == other.as_slice()
    }
}

impl From<VarLookupKey<'_>> for VarKey {
    fn from(key: VarLookupKey<'_>) -> Self {
        Self::from_slice(key.0)
    }
}

/// We keep this small to enable a simple LRU cache with a linear
/// search. Regenerating hinting data is low to medium cost so it's fine
/// to redo it occasionally.
const MAX_CACHED_HINT_INSTANCES: usize = 16;

/// Hint key for hinting instances.
#[derive(Debug)]
pub struct HintKey<'a> {
    font_id: u64,
    font_index: u32,
    outlines: &'a OutlineGlyphCollection<'a>,
    size: f32,
    coords: &'a [skrifa::instance::NormalizedCoord],
}

impl HintKey<'_> {
    fn instance(&self) -> Option<HintingInstance> {
        HintingInstance::new(
            self.outlines,
            Size::new(self.size),
            self.coords,
            HINTING_OPTIONS,
        )
        .ok()
    }
}

/// LRU cache for hinting instances.
///
/// Heavily inspired by `vello_encoding::glyph_cache`.
#[derive(Default)]
pub struct HintCache {
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
    /// Gets a hinting instance for the given key.
    pub fn get(&mut self, key: &HintKey<'_>) -> Option<&HintingInstance> {
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
                .reconfigure(
                    key.outlines,
                    Size::new(key.size),
                    key.coords,
                    HINTING_OPTIONS,
                )
                .ok()?;
        }
        Some(&entry.instance)
    }

    /// Clears the hint cache.
    pub fn clear(&mut self) {
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
            && entry.instance.size() == Size::new(key.size)
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

fn x_y_advances(transform: &Affine) -> (Vec2, Vec2) {
    let scale_skew_transform = {
        let c = transform.as_coeffs();
        Affine::new([c[0], c[1], c[2], c[3], 0.0, 0.0])
    };

    let x_advance = scale_skew_transform * Point::new(1.0, 0.0);
    let y_advance = scale_skew_transform * Point::new(0.0, 1.0);

    (
        Vec2::new(x_advance.x, x_advance.y),
        Vec2::new(y_advance.x, y_advance.y),
    )
}
