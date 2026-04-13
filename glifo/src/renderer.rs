// Copyright 2026 the Vello Authors and the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared glyph rendering logic for rendering backends.

use crate::atlas::commands::{AtlasCommand, AtlasCommandRecorder};
use crate::atlas::key::subpixel_offset;
use crate::atlas::{AtlasSlot, GlyphAtlas, GlyphCacheKey, ImageCache, RasterMetrics};
use crate::colr::ColrPainter;
use crate::glyph::{
    AtlasCacher, CachedGlyphType, GlyphBitmap, GlyphColr, GlyphType, PreparedGlyph,
};
use crate::interface::{DrawSink, GlyphRenderer};
use crate::util::AffineExt;
use crate::{kurbo, peniko};
use alloc::sync::Arc;
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use core_maths::CoreFloat as _;
use kurbo::{Affine, BezPath, Rect, Shape};
use peniko::color::palette::css::BLACK;
use peniko::color::{AlphaColor, Srgb};
use peniko::{BlendMode, Extend, ImageQuality, ImageSampler};
use vello_common::paint::{Image, ImageSource, Tint, TintMode};

/// Outcome of a cache-first render attempt.
///
/// Callers use the variant to decide whether to fall back to direct rendering.
enum CacheResult {
    /// Glyph was rasterised, stored in the atlas, and drawn. No fallback needed.
    CachedAndRendered,
    /// Transform contains rotation or skew — cannot be cached at a single
    /// raster resolution, so the caller must render directly.
    UnsupportedTransform,
    /// Atlas allocator could not fit the glyph (page full, eviction didn't
    /// free enough space, or the glyph exceeds the page dimensions).
    AtlasFull,
}

/// Fill a prepared glyph, using the glyph atlas when possible and falling
/// back to direct rendering otherwise.
pub(crate) fn fill_glyph(
    renderer: &mut impl GlyphRenderer,
    prepared_glyph: PreparedGlyph<'_>,
    atlas_cacher: &mut AtlasCacher<'_>,
) {
    let AtlasCacher::Enabled(glyph_atlas, image_cache) = atlas_cacher else {
        let transform = prepared_glyph.transform;

        return match prepared_glyph.glyph_type {
            GlyphType::Outline(glyph) => {
                fill_uncached_outline_glyph(renderer, &glyph.path, transform);
            }
            GlyphType::Bitmap(glyph) => render_uncached_bitmap_glyph(renderer, glyph, transform),
            GlyphType::Colr(glyph) => {
                let context_color = renderer.get_context_color();
                render_uncached_colr_glyph(renderer, &glyph, transform, context_color);
            }
        };
    };

    let mut cache_key = prepared_glyph.cache_key;
    let transform = prepared_glyph.transform;

    match prepared_glyph.glyph_type {
        GlyphType::Outline(glyph) => {
            let tint_color = renderer.get_context_color();
            if let Some(key) = cache_key.take()
                && let CacheResult::CachedAndRendered = insert_and_render_outline(
                    renderer,
                    &glyph.path,
                    transform,
                    key,
                    glyph_atlas,
                    image_cache,
                    tint_color,
                )
            {
                return;
            }

            fill_uncached_outline_glyph(renderer, &glyph.path, transform);
        }
        GlyphType::Bitmap(glyph) => {
            if let Some(key) = cache_key.take()
                && let CacheResult::CachedAndRendered = insert_and_render_bitmap(
                    renderer,
                    &glyph,
                    transform,
                    key,
                    glyph_atlas,
                    image_cache,
                )
            {
                return;
            }

            render_uncached_bitmap_glyph(renderer, glyph, transform);
        }
        GlyphType::Colr(glyph) => {
            if let Some(key) = cache_key.take()
                && let CacheResult::CachedAndRendered = insert_and_render_colr(
                    renderer,
                    &glyph,
                    transform,
                    key,
                    glyph_atlas,
                    image_cache,
                )
            {
                return;
            }

            let context_color = renderer.get_context_color();
            render_uncached_colr_glyph(renderer, &glyph, transform, context_color);
        }
    }
}

/// Stroke a prepared glyph, using the glyph atlas when possible and falling
/// back to direct rendering otherwise.
pub(crate) fn stroke_glyph(
    renderer: &mut impl GlyphRenderer,
    prepared_glyph: PreparedGlyph<'_>,
    atlas_cacher: &mut AtlasCacher<'_>,
) {
    let AtlasCacher::Enabled(glyph_atlas, image_cache) = atlas_cacher else {
        let transform = prepared_glyph.transform;
        return match prepared_glyph.glyph_type {
            GlyphType::Outline(glyph) => {
                stroke_uncached_outline_glyph(renderer, &glyph.path, transform);
            }
            GlyphType::Bitmap(_) | GlyphType::Colr(_) => {
                fill_glyph(renderer, prepared_glyph, atlas_cacher);
            }
        };
    };

    match prepared_glyph.glyph_type {
        GlyphType::Outline(glyph) => {
            let mut cache_key = prepared_glyph.cache_key;
            let transform = prepared_glyph.transform;
            let tint_color = renderer.get_context_color();

            if let Some(key) = cache_key.take()
                && let CacheResult::CachedAndRendered = insert_and_render_outline(
                    renderer,
                    &glyph.path,
                    transform,
                    key,
                    glyph_atlas,
                    image_cache,
                    tint_color,
                )
            {
                return;
            }

            stroke_uncached_outline_glyph(renderer, &glyph.path, transform);
        }
        GlyphType::Bitmap(_) | GlyphType::Colr(_) => {
            fill_glyph(renderer, prepared_glyph, atlas_cacher);
        }
    }
}

fn fill_uncached_outline_glyph(
    renderer: &mut impl GlyphRenderer,
    path: &BezPath,
    transform: Affine,
) {
    let state = renderer.save_state();
    renderer.set_transform(transform);
    renderer.fill_path(path);
    renderer.restore_state(state);
}

fn stroke_uncached_outline_glyph(
    renderer: &mut impl GlyphRenderer,
    path: &BezPath,
    transform: Affine,
) {
    let state = renderer.save_state();
    renderer.set_transform(transform);
    renderer.stroke_path(path);
    renderer.restore_state(state);
}

fn render_uncached_bitmap_glyph(
    renderer: &mut impl GlyphRenderer,
    glyph: GlyphBitmap,
    transform: Affine,
) {
    let image = Image {
        image: ImageSource::Pixmap(glyph.pixmap),
        sampler: ImageSampler {
            x_extend: Extend::Pad,
            y_extend: Extend::Pad,
            quality: quality_for_scale(&transform),
            alpha: 1.0,
        },
    };

    let state = renderer.save_state();
    renderer.set_transform(transform);
    renderer.set_paint_image(image);
    renderer.fill_rect(&glyph.area);
    renderer.restore_state(state);
}

fn render_uncached_colr_glyph(
    renderer: &mut impl GlyphRenderer,
    glyph: &GlyphColr<'_>,
    transform: Affine,
    context_color: AlphaColor<Srgb>,
) {
    let state = renderer.save_state();
    renderer.set_transform(transform);
    // Wrap COLR glyphs in a layer, to make sure they are isolated and don't
    // blend into the main surface.
    renderer.push_blend_layer(BlendMode::default());

    let mut colr_painter = ColrPainter::new(glyph, context_color, renderer);
    colr_painter.paint();
    renderer.pop_layer();

    renderer.restore_state(state);
}

/// Render a cached glyph from the atlas.
pub(crate) fn render_cached_glyph(
    renderer: &mut impl GlyphRenderer,
    cached_slot: AtlasSlot,
    transform: Affine,
    glyph_type: CachedGlyphType,
) {
    match glyph_type {
        CachedGlyphType::Outline => {
            let tint = renderer.get_context_color();
            render_outline_glyph_from_atlas(renderer, cached_slot, transform, tint);
        }
        CachedGlyphType::Bitmap => {
            render_bitmap_glyph_from_atlas(renderer, cached_slot, transform);
        }
        CachedGlyphType::Colr(area) => {
            render_colr_glyph_from_atlas(renderer, cached_slot, transform, area);
        }
    }
}

/// Render from the atlas, constructing the appropriate image from the slot.
fn render_from_atlas(
    renderer: &mut impl GlyphRenderer,
    atlas_slot: AtlasSlot,
    rect_transform: Affine,
    area: Rect,
    quality: ImageQuality,
    tint: Option<Tint>,
) {
    let paint_transform = renderer.atlas_paint_transform(&atlas_slot);
    let image_source = renderer.atlas_image_source(&atlas_slot);
    let image = Image {
        image: image_source,
        sampler: ImageSampler {
            x_extend: Extend::Pad,
            y_extend: Extend::Pad,
            quality,
            alpha: 1.0,
        },
    };

    let state = renderer.save_state();
    renderer.set_tint(tint);
    renderer.set_transform(rect_transform);
    renderer.set_paint_image(image);
    renderer.set_paint_transform(paint_transform);
    renderer.fill_rect(&area);
    renderer.set_tint(None);
    renderer.restore_state(state);
}

/// Record outline glyph draw commands into the atlas command recorder.
fn render_outline_to_atlas(
    path: &Arc<BezPath>,
    subpixel_offset: f32,
    recorder: &mut AtlasCommandRecorder,
    atlas_slot: AtlasSlot,
    raster_metrics: RasterMetrics,
) {
    let outline_transform = Affine::scale_non_uniform(1.0, -1.0).then_translate(kurbo::Vec2::new(
        atlas_slot.x as f64 - raster_metrics.bearing_x as f64 + subpixel_offset as f64,
        atlas_slot.y as f64 - raster_metrics.bearing_y as f64,
    ));
    recorder.set_transform(outline_transform);
    recorder.set_paint(BLACK.into());
    recorder.fill_path(path);
}

/// Record COLR glyph draw commands into the atlas command recorder.
fn render_colr_to_atlas(
    glyph: &GlyphColr<'_>,
    context_color: AlphaColor<Srgb>,
    recorder: &mut AtlasCommandRecorder,
    atlas_slot: AtlasSlot,
) {
    recorder.set_transform(Affine::translate((
        atlas_slot.x as f64,
        atlas_slot.y as f64,
    )));
    // See the comment in `render_uncached_colr_glyph` for why we wrap COLR glyphs
    // in a layer.
    recorder.push_blend_layer(BlendMode::default());

    let mut colr_painter = ColrPainter::new(glyph, context_color, recorder);
    colr_painter.paint();
    recorder.pop_layer();
}

/// Insert an outline glyph into the atlas and render it from there.
///
/// Allocates atlas space (the insert returns the per-page command recorder)
/// and records rasterisation commands. The upstream caller is responsible for
/// checking the cache first and only calling this on a miss.
fn insert_and_render_outline(
    renderer: &mut impl GlyphRenderer,
    path: &Arc<BezPath>,
    transform: Affine,
    cache_key: GlyphCacheKey,
    glyph_atlas: &mut GlyphAtlas,
    image_cache: &mut ImageCache,
    tint_color: AlphaColor<Srgb>,
) -> CacheResult {
    if !supports_atlas_caching(&transform, CachedGlyphType::Outline) {
        return CacheResult::UnsupportedTransform;
    }

    let bounds = path.bounding_box();
    let raster_metrics = calculate_raster_metrics(&bounds);

    let subpixel_offset = subpixel_offset(cache_key.subpixel_x);

    let Some((atlas_slot, recorder)) = glyph_atlas.insert(image_cache, cache_key, raster_metrics)
    else {
        return CacheResult::AtlasFull;
    };

    render_outline_to_atlas(path, subpixel_offset, recorder, atlas_slot, raster_metrics);

    render_outline_glyph_from_atlas(renderer, atlas_slot, transform, tint_color);
    CacheResult::CachedAndRendered
}

fn insert_and_render_bitmap(
    renderer: &mut impl GlyphRenderer,
    glyph: &GlyphBitmap,
    transform: Affine,
    cache_key: GlyphCacheKey,
    glyph_atlas: &mut GlyphAtlas,
    image_cache: &mut ImageCache,
) -> CacheResult {
    if !supports_atlas_caching(&transform, CachedGlyphType::Bitmap) {
        return CacheResult::UnsupportedTransform;
    }

    let width = glyph.pixmap.width();
    let height = glyph.pixmap.height();

    let raster_metrics = RasterMetrics {
        width,
        height,
        bearing_x: 0,
        bearing_y: 0,
    };

    // Bitmap glyphs already have pixel data — no draw commands to record,
    // so we discard the returned recorder.
    let Some((atlas_slot, _)) = glyph_atlas.insert(image_cache, cache_key, raster_metrics) else {
        return CacheResult::AtlasFull;
    };

    // Both backends defer the actual pixel copy/upload; it completes before
    // the render pass that resolves image references.
    glyph_atlas.push_pending_upload(atlas_slot.image_id, Arc::clone(&glyph.pixmap), atlas_slot);

    render_from_atlas(
        renderer,
        atlas_slot,
        transform,
        glyph.area,
        quality_for_scale(&transform),
        None,
    );
    CacheResult::CachedAndRendered
}

fn insert_and_render_colr(
    renderer: &mut impl GlyphRenderer,
    glyph: &GlyphColr<'_>,
    transform: Affine,
    cache_key: GlyphCacheKey,
    glyph_atlas: &mut GlyphAtlas,
    image_cache: &mut ImageCache,
) -> CacheResult {
    if !supports_atlas_caching(&transform, CachedGlyphType::Colr(Rect::ZERO)) {
        return CacheResult::UnsupportedTransform;
    }

    let width = glyph.pix_width;
    let height = glyph.pix_height;

    let raster_metrics = RasterMetrics {
        width,
        height,
        bearing_x: 0,
        bearing_y: 0,
    };

    let area = glyph.area;

    let context_color = cache_key.context_color;
    let Some((atlas_slot, recorder)) = glyph_atlas.insert(image_cache, cache_key, raster_metrics)
    else {
        return CacheResult::AtlasFull;
    };

    render_colr_to_atlas(glyph, context_color, recorder, atlas_slot);

    render_from_atlas(
        renderer,
        atlas_slot,
        transform,
        area,
        quality_for_skew(&transform),
        None,
    );
    CacheResult::CachedAndRendered
}

/// Render an outline glyph from the atlas using bearing-based positioning.
#[inline]
fn render_outline_glyph_from_atlas(
    renderer: &mut impl GlyphRenderer,
    atlas_slot: AtlasSlot,
    transform: Affine,
    tint_color: AlphaColor<Srgb>,
) {
    let [_, _, _, _, tx, ty] = transform.as_coeffs();
    let rect_transform = Affine::translate((
        tx.floor() + atlas_slot.bearing_x as f64,
        ty.floor() + atlas_slot.bearing_y as f64,
    ));
    let area = Rect::new(0.0, 0.0, atlas_slot.width as f64, atlas_slot.height as f64);
    render_from_atlas(
        renderer,
        atlas_slot,
        rect_transform,
        area,
        ImageQuality::Low,
        Some(Tint {
            color: tint_color,
            mode: TintMode::AlphaMask,
        }),
    );
}

/// Render a bitmap glyph from the atlas cache.
#[inline]
fn render_bitmap_glyph_from_atlas(
    renderer: &mut impl GlyphRenderer,
    atlas_slot: AtlasSlot,
    transform: Affine,
) {
    let area = Rect::new(0.0, 0.0, atlas_slot.width as f64, atlas_slot.height as f64);
    render_from_atlas(
        renderer,
        atlas_slot,
        transform,
        area,
        quality_for_scale(&transform),
        None,
    );
}

/// Render a COLR glyph from the atlas cache.
///
/// This version accepts a pre-calculated fractional area to preserve
/// sub-pixel accuracy during rendering, avoiding scaling artifacts.
#[inline]
fn render_colr_glyph_from_atlas(
    renderer: &mut impl GlyphRenderer,
    atlas_slot: AtlasSlot,
    transform: Affine,
    area: Rect,
) {
    render_from_atlas(
        renderer,
        atlas_slot,
        transform,
        area,
        quality_for_skew(&transform),
        None,
    );
}

/// Calculate raster metrics (pixel bounds, bearings) from a glyph's bounding box.
#[expect(
    clippy::cast_possible_truncation,
    reason = "glyph bounds fit in i32/u16/i16 at reasonable ppem values"
)]
#[inline]
pub(crate) fn calculate_raster_metrics(bounds: &Rect) -> RasterMetrics {
    // Floor/ceil round outward from the fractional bounding box. Width gets an
    // extra pixel to accommodate the horizontal subpixel offset (up to 0.75 px)
    // applied when rasterising into the atlas; the Y axis has no subpixel shift
    // so floor/ceil alone is sufficient. GLYPH_PADDING in the atlas allocator
    // provides the guard band needed by the hybrid renderer's Extend::Pad sampling.
    let min_x = bounds.x0.floor() as i32;
    let max_x = bounds.x1.ceil() as i32 + 1;

    // For Y, we flip the coordinate system: font Y up -> screen Y down
    // After flipping Y, min_y becomes -max_y and max_y becomes -min_y
    let flipped_min_y = (-bounds.y1).floor() as i32;
    let flipped_max_y = (-bounds.y0).ceil() as i32;

    let width = (max_x - min_x) as u16;
    let height = (flipped_max_y - flipped_min_y) as u16;

    RasterMetrics {
        width,
        height,
        bearing_x: min_x as i16,
        bearing_y: flipped_min_y as i16,
    }
}

/// Choose image sampling quality based on downscale factor.
///
/// Returns `High` when the transform scales below 50% (where aliasing is
/// visible), `Medium` otherwise.
#[inline]
pub fn quality_for_scale(transform: &Affine) -> ImageQuality {
    let [a, _, _, d, _, _] = transform.as_coeffs();
    if a < 0.5 || d < 0.5 {
        ImageQuality::High
    } else {
        ImageQuality::Medium
    }
}

/// Choose image sampling quality based on skew presence.
///
/// Skewed transforms need `Medium` quality to avoid aliasing; axis-aligned
/// transforms use `Low` (nearest-neighbour) since the content was already
/// rasterized at pixel boundaries.
#[inline]
pub(crate) fn quality_for_skew(transform: &Affine) -> ImageQuality {
    if transform.has_skew() {
        ImageQuality::Medium
    } else {
        ImageQuality::Low
    }
}

/// Replay recorded atlas commands into a [`DrawSink`].
///
/// The commands `Vec` is drained, freeing memory as each command is consumed.
pub fn replay_atlas_commands(commands: &mut Vec<AtlasCommand>, target: &mut impl DrawSink) {
    for cmd in commands.drain(..) {
        match cmd {
            AtlasCommand::SetTransform(t) => target.set_transform(t),
            AtlasCommand::SetPaint(p) => target.set_paint(p),
            AtlasCommand::SetPaintTransform(t) => target.set_paint_transform(t),
            AtlasCommand::FillPath(p) => target.fill_path(&p),
            AtlasCommand::FillRect(r) => target.fill_rect(&r),
            AtlasCommand::PushClipLayer(c) => target.push_clip_layer(&c),
            AtlasCommand::PushBlendLayer(m) => target.push_blend_layer(m),
            AtlasCommand::PopLayer => target.pop_layer(),
        }
    }
}

/// Returns `true` if the transform is safe for atlas-cached glyph rendering.
#[inline]
pub(crate) fn supports_atlas_caching(transform: &Affine, glyph_type: CachedGlyphType) -> bool {
    // TODO: Investigate whether we can support arbitrary mirroring. From some
    // initial experiments, allowing x-mirroring leads to slightly shifted glyphs, so
    // we don't support this now. Y-mirroring also needs more consideration.

    let [a, _, _, d, _, _] = transform.as_coeffs();

    match glyph_type {
        // For those glyphs, we expect any scaling factor to have been completely absorbed. Due to the fact
        // that we had to apply a flip transform for outlines, the y-scaling factor is expected to be negative.
        CachedGlyphType::Outline | CachedGlyphType::Colr(_) => {
            !transform.has_non_unit_skew_or_scale() && a.is_sign_positive() && d.is_sign_negative()
        }
        // For bitmap glyphs, we need to relax the condition a bit, since bitmap glyphs already have a fixed
        // size and thus might not correspond 100% to the font size. Therefore, they likely don't have a unit
        // transform.
        CachedGlyphType::Bitmap => {
            !transform.has_skew() && a.is_sign_positive() && d.is_sign_positive()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::supports_atlas_caching;
    use crate::glyph::CachedGlyphType;
    use peniko::kurbo::Affine;
    use peniko::kurbo::Rect;

    #[test]
    fn supports_bitmap_caching_for_identity_and_translation() {
        assert!(supports_atlas_caching(
            &Affine::IDENTITY,
            CachedGlyphType::Bitmap
        ));
        assert!(supports_atlas_caching(
            &Affine::translate((12.0, -3.5)),
            CachedGlyphType::Bitmap
        ));
    }

    #[test]
    fn rejects_skewed_transforms() {
        assert!(!supports_atlas_caching(
            &Affine::new([1.0, 0.1, 0.0, 1.0, 0.0, 0.0]),
            CachedGlyphType::Bitmap
        ));
        assert!(!supports_atlas_caching(
            &Affine::skew(0.2, 0.0),
            CachedGlyphType::Outline
        ));
        assert!(!supports_atlas_caching(
            &Affine::skew(0.2, 0.0),
            CachedGlyphType::Colr(Rect::ZERO)
        ));
    }

    #[test]
    fn outline_and_colr_reject_non_unit_scales() {
        assert!(!supports_atlas_caching(
            &Affine::scale(2.0),
            CachedGlyphType::Outline
        ));
        assert!(!supports_atlas_caching(
            &Affine::scale_non_uniform(1.0, -0.5),
            CachedGlyphType::Outline
        ));
        assert!(!supports_atlas_caching(
            &Affine::scale(2.0),
            CachedGlyphType::Colr(Rect::ZERO)
        ));
    }

    #[test]
    fn outline_and_colr_requires_negative_y_and_positive_x() {
        assert!(supports_atlas_caching(
            &Affine::scale_non_uniform(1.0, -1.0),
            CachedGlyphType::Outline
        ));
        assert!(supports_atlas_caching(
            &Affine::scale_non_uniform(1.0, -1.0),
            CachedGlyphType::Colr(Rect::ZERO)
        ));
        assert!(!supports_atlas_caching(
            &Affine::scale_non_uniform(-1.0, -1.0),
            CachedGlyphType::Outline
        ));
        assert!(!supports_atlas_caching(
            &Affine::scale_non_uniform(1.0, 1.0),
            CachedGlyphType::Outline
        ));
    }

    #[test]
    fn bitmap_allows_positive_scales_only() {
        assert!(supports_atlas_caching(
            &Affine::scale_non_uniform(1.0, 1.0),
            CachedGlyphType::Bitmap
        ));
        assert!(supports_atlas_caching(
            &Affine::scale_non_uniform(2.0, 3.0),
            CachedGlyphType::Bitmap
        ));
        assert!(!supports_atlas_caching(
            &Affine::scale_non_uniform(1.0, -1.0),
            CachedGlyphType::Bitmap
        ));
        assert!(!supports_atlas_caching(
            &Affine::scale_non_uniform(-1.0, 1.0),
            CachedGlyphType::Bitmap
        ));
    }
}
