// Copyright 2026 the Vello Authors and the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared glyph rendering logic for all Vello backends.
//!
//! This module contains the backend-agnostic parts of glyph rendering:
//! utility functions, the `GlyphAtlasBackend` trait, the
//! `AtlasReplayTarget` trait, and the generic cache orchestration functions
//! (`fill_glyph`, `stroke_glyph`).
//!
//! Backend-specific implementations live in the sibling
//! [`vello_cpu`](super::vello_cpu) and [`vello_hybrid`](super::vello_hybrid)
//! modules.

use crate::atlas::commands::{AtlasCommand, AtlasCommandRecorder, AtlasPaint};
use crate::atlas::key::subpixel_offset;
use crate::atlas::{AtlasSlot, GlyphCache, GlyphCacheKey, ImageCache, RasterMetrics};
use crate::colr::ColrPainter;
use crate::glyph::{GlyphBitmap, GlyphColr, GlyphRenderer, GlyphType, PreparedGlyph};
use crate::{kurbo, peniko};
use alloc::sync::Arc;
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use core_maths::CoreFloat as _;
use kurbo::{Affine, BezPath, Rect, Shape};
use peniko::color::palette::css::BLACK;
use peniko::color::{AlphaColor, Srgb};
use peniko::{BlendMode, Gradient, ImageQuality};
use vello_common::paint::{Tint, TintMode};

/// Outcome of a cache-first render attempt.
///
/// Callers use the variant to decide whether to fall back to direct rendering.
enum CacheResult {
    /// Glyph was rasterised, stored in the atlas, and drawn. No fallback needed.
    CachedAndRendered,
    /// Transform contains rotation or skew — cannot be cached at a single
    /// raster resolution, so the caller must render directly.
    NotAxisAligned,
    /// Atlas allocator could not fit the glyph (page full, eviction didn't
    /// free enough space, or the glyph exceeds the page dimensions).
    AtlasFull,
}

/// Abstracts the differences between CPU and Hybrid rendering backends.
///
/// Backend-specific operations — atlas image construction, paint transforms,
/// and outline/COLR rendering into the [`AtlasCommandRecorder`] — are defined
/// as trait methods.  The shared cache orchestration logic lives in the
/// generic free functions [`fill_glyph`] and [`stroke_glyph`].
pub(crate) trait GlyphAtlasBackend {
    /// The renderer type for this backend (e.g. `RenderContext` or `Scene`).
    type Renderer;

    /// The glyph atlas cache type for this backend.
    type Cache: GlyphCache;

    // ---- Atlas rendering -------------------------------------------------

    /// Draw a cached glyph from the atlas into the scene.
    ///
    /// Constructs a backend-specific `Image` from the atlas slot and fills
    /// `area` with it.
    ///
    /// `tint` controls colouring:
    /// - `Some(Tint)` — treat the atlas image as an alpha mask and tint with
    ///   the given colour (used for monochrome outline glyphs).
    /// - `None` — draw the atlas RGBA content as-is (bitmap and COLR glyphs
    ///   already have colour baked in).
    fn render_from_atlas(
        renderer: &mut Self::Renderer,
        atlas_slot: AtlasSlot,
        rect_transform: Affine,
        area: Rect,
        quality: ImageQuality,
        paint_transform: Affine,
        tint: Option<Tint>,
    );

    /// Compute the paint transform that maps atlas-image UV coordinates to the
    /// glyph's fill rect. Backend-specific because the CPU and hybrid backends
    /// use different addressing schemes (page offset vs. allocation origin).
    fn paint_transform(atlas_slot: &AtlasSlot) -> Affine;

    // ---- Direct rendering (uncached) -------------------------------------

    /// Fill an outline glyph directly (not using cache).
    fn fill_outline_directly(renderer: &mut Self::Renderer, path: &BezPath, transform: Affine);

    /// Stroke an outline glyph directly (not using cache).
    fn stroke_outline_directly(renderer: &mut Self::Renderer, path: &BezPath, transform: Affine);

    /// Render a bitmap glyph directly (not using cache).
    fn render_bitmap_directly(renderer: &mut Self::Renderer, glyph: GlyphBitmap, transform: Affine);

    /// Render a COLR glyph directly (not using cache).
    fn render_colr_directly(
        renderer: &mut Self::Renderer,
        glyph: &GlyphColr<'_>,
        transform: Affine,
        context_color: AlphaColor<Srgb>,
    );
}

/// Fill a prepared glyph, using the glyph atlas when possible and falling
/// back to direct rendering otherwise.
pub(crate) fn fill_glyph<B: GlyphAtlasBackend>(
    renderer: &mut B::Renderer,
    prepared_glyph: PreparedGlyph<'_>,
    glyph_atlas: &mut B::Cache,
    image_cache: &mut ImageCache,
) where
    B::Renderer: GlyphRenderer<B::Cache>,
{
    let mut cache_key = prepared_glyph.cache_key;
    let transform = prepared_glyph.transform;

    match prepared_glyph.glyph_type {
        GlyphType::Outline(glyph) => {
            let tint_color = renderer.get_context_color();
            if let Some(key) = cache_key.take() {
                if let CacheResult::CachedAndRendered = insert_and_render_outline::<B>(
                    renderer,
                    &glyph.path,
                    transform,
                    key,
                    glyph_atlas,
                    image_cache,
                    tint_color,
                ) {
                    return;
                }
            }

            B::fill_outline_directly(renderer, &glyph.path, transform);
        }
        GlyphType::Bitmap(glyph) => {
            if let Some(key) = cache_key.take() {
                if let CacheResult::CachedAndRendered = insert_and_render_bitmap::<B>(
                    renderer,
                    &glyph,
                    transform,
                    key,
                    glyph_atlas,
                    image_cache,
                ) {
                    return;
                }
            }

            B::render_bitmap_directly(renderer, glyph, transform);
        }
        GlyphType::Colr(glyph) => {
            if let Some(key) = cache_key.take() {
                if let CacheResult::CachedAndRendered = insert_and_render_colr::<B>(
                    renderer,
                    &glyph,
                    transform,
                    key,
                    glyph_atlas,
                    image_cache,
                ) {
                    return;
                }
            }

            let context_color = renderer.get_context_color();
            B::render_colr_directly(renderer, &glyph, transform, context_color);
        }
    }
}

/// Stroke a prepared glyph, using the glyph atlas when possible and falling
/// back to direct rendering otherwise.
pub(crate) fn stroke_glyph<B: GlyphAtlasBackend>(
    renderer: &mut B::Renderer,
    prepared_glyph: PreparedGlyph<'_>,
    glyph_atlas: &mut B::Cache,
    image_cache: &mut ImageCache,
) where
    B::Renderer: GlyphRenderer<B::Cache>,
{
    match prepared_glyph.glyph_type {
        GlyphType::Outline(glyph) => {
            let mut cache_key = prepared_glyph.cache_key;
            let transform = prepared_glyph.transform;
            let tint_color = renderer.get_context_color();

            if let Some(key) = cache_key.take() {
                if let CacheResult::CachedAndRendered = insert_and_render_outline::<B>(
                    renderer,
                    &glyph.path,
                    transform,
                    key,
                    glyph_atlas,
                    image_cache,
                    tint_color,
                ) {
                    return;
                }
            }

            B::stroke_outline_directly(renderer, &glyph.path, transform);
        }
        GlyphType::Bitmap(_) | GlyphType::Colr(_) => {
            // The definitions of COLR and bitmap glyphs can't meaningfully support being stroked.
            // (COLR's imaging model only has fills)
            fill_glyph::<B>(renderer, prepared_glyph, glyph_atlas, image_cache);
        }
    }
}

/// Record outline glyph draw commands into the atlas command recorder.
fn render_outline_to_atlas(
    path: &Arc<BezPath>,
    subpixel_offset: f32,
    recorder: &mut AtlasCommandRecorder,
    dst_x: u16,
    dst_y: u16,
    raster_metrics: RasterMetrics,
) {
    let outline_transform = Affine::scale_non_uniform(1.0, -1.0).then_translate(kurbo::Vec2::new(
        dst_x as f64 - raster_metrics.bearing_x as f64 + subpixel_offset as f64,
        dst_y as f64 - raster_metrics.bearing_y as f64,
    ));
    recorder.set_transform(outline_transform);
    recorder.set_paint(BLACK);
    recorder.fill_path(path);
}

/// Record COLR glyph draw commands into the atlas command recorder.
fn render_colr_to_atlas(
    glyph: &GlyphColr<'_>,
    context_color: AlphaColor<Srgb>,
    recorder: &mut AtlasCommandRecorder,
    dst_x: u16,
    dst_y: u16,
) {
    recorder.set_transform(Affine::translate((dst_x as f64, dst_y as f64)));

    let mut colr_painter = ColrPainter::new(glyph, context_color, recorder);
    colr_painter.paint();
}

/// Insert an outline glyph into the atlas and render it from there.
///
/// Allocates atlas space (the insert returns the per-page command recorder)
/// and records rasterisation commands. The upstream caller is responsible for
/// checking the cache first and only calling this on a miss.
fn insert_and_render_outline<B: GlyphAtlasBackend>(
    renderer: &mut B::Renderer,
    path: &Arc<BezPath>,
    transform: Affine,
    cache_key: GlyphCacheKey,
    glyph_atlas: &mut B::Cache,
    image_cache: &mut ImageCache,
    tint_color: AlphaColor<Srgb>,
) -> CacheResult {
    if !is_axis_aligned(&transform) {
        return CacheResult::NotAxisAligned;
    }

    let bounds = path.bounding_box();
    let raster_metrics = calculate_raster_metrics(&bounds);

    let subpixel_offset = subpixel_offset(cache_key.subpixel_x);

    let Some((dst_x, dst_y, atlas_slot, recorder)) =
        glyph_atlas.insert(image_cache, cache_key, raster_metrics)
    else {
        return CacheResult::AtlasFull;
    };

    render_outline_to_atlas(
        path,
        subpixel_offset,
        recorder,
        dst_x,
        dst_y,
        raster_metrics,
    );

    render_outline_glyph_from_atlas::<B>(renderer, atlas_slot, transform, tint_color);
    CacheResult::CachedAndRendered
}

/// Insert a bitmap glyph into the atlas and render it from there.
///
/// Delegates atlas population via [`GlyphCache::push_pending_upload`].
/// The upstream caller is responsible for checking the cache first
/// and only calling this on a miss.
fn insert_and_render_bitmap<B: GlyphAtlasBackend>(
    renderer: &mut B::Renderer,
    glyph: &GlyphBitmap,
    transform: Affine,
    cache_key: GlyphCacheKey,
    glyph_atlas: &mut B::Cache,
    image_cache: &mut ImageCache,
) -> CacheResult {
    let width = glyph.pixmap.width();
    let height = glyph.pixmap.height();

    let raster_metrics = RasterMetrics {
        width,
        height,
        bearing_x: 0,
        bearing_y: 0,
    };

    // Bitmap glyphs already have pixel data — no draw commands to record,
    // so we discard the returned dst coordinates and recorder.
    let Some((_dst_x, _dst_y, atlas_slot, _)) =
        glyph_atlas.insert(image_cache, cache_key, raster_metrics)
    else {
        return CacheResult::AtlasFull;
    };

    // Both backends defer the actual pixel copy/upload; it completes before
    // the render pass that resolves image references.
    glyph_atlas.push_pending_upload(atlas_slot.image_id, Arc::clone(&glyph.pixmap), atlas_slot);

    let paint_transform = B::paint_transform(&atlas_slot);
    B::render_from_atlas(
        renderer,
        atlas_slot,
        transform,
        glyph.area,
        quality_for_scale(&transform),
        paint_transform,
        None,
    );
    CacheResult::CachedAndRendered
}

/// Insert a COLR glyph into the atlas and render it from there.
///
/// Allocates atlas space (the insert returns the per-page command recorder)
/// and records rasterisation commands via [`render_colr_to_atlas`].
/// The upstream caller is responsible for checking the cache first
/// and only calling this on a miss.
fn insert_and_render_colr<B: GlyphAtlasBackend>(
    renderer: &mut B::Renderer,
    glyph: &GlyphColr<'_>,
    transform: Affine,
    cache_key: GlyphCacheKey,
    glyph_atlas: &mut B::Cache,
    image_cache: &mut ImageCache,
) -> CacheResult {
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
    let Some((dst_x, dst_y, atlas_slot, recorder)) =
        glyph_atlas.insert(image_cache, cache_key, raster_metrics)
    else {
        return CacheResult::AtlasFull;
    };

    render_colr_to_atlas(glyph, context_color, recorder, dst_x, dst_y);

    let paint_transform = B::paint_transform(&atlas_slot);

    // Use the original fractional area to preserve sub-pixel accuracy
    B::render_from_atlas(
        renderer,
        atlas_slot,
        transform,
        area,
        quality_for_skew(&transform),
        paint_transform,
        None,
    );
    CacheResult::CachedAndRendered
}

/// Render an outline glyph from the atlas using bearing-based positioning.
///
/// The transform's translation is floored to an integer pixel to align with
/// the atlas raster, then the slot's bearing offsets are applied. Sampling
/// quality is always `Low` (nearest-neighbour) because the glyph was already
/// rasterised at the target resolution.
#[inline]
pub(crate) fn render_outline_glyph_from_atlas<B: GlyphAtlasBackend>(
    renderer: &mut B::Renderer,
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
    let paint_transform = B::paint_transform(&atlas_slot);
    B::render_from_atlas(
        renderer,
        atlas_slot,
        rect_transform,
        area,
        ImageQuality::Low,
        paint_transform,
        Some(Tint {
            color: tint_color,
            mode: TintMode::AlphaMask,
        }),
    );
}

/// Render a bitmap glyph from the atlas cache.
///
/// Called on the cache-hit fast path — no glyph preparation needed. Sampling
/// quality adapts to the scale factor to avoid aliasing on downscaled glyphs.
#[inline]
pub(crate) fn render_bitmap_glyph_from_atlas<B: GlyphAtlasBackend>(
    renderer: &mut B::Renderer,
    atlas_slot: AtlasSlot,
    transform: Affine,
) {
    let area = Rect::new(0.0, 0.0, atlas_slot.width as f64, atlas_slot.height as f64);
    let paint_transform = B::paint_transform(&atlas_slot);
    B::render_from_atlas(
        renderer,
        atlas_slot,
        transform,
        area,
        quality_for_scale(&transform),
        paint_transform,
        None,
    );
}

/// Render a COLR glyph from the atlas cache.
///
/// This version accepts a pre-calculated fractional area to preserve
/// sub-pixel accuracy during rendering, avoiding scaling artifacts.
#[inline]
pub(crate) fn render_colr_glyph_from_atlas<B: GlyphAtlasBackend>(
    renderer: &mut B::Renderer,
    atlas_slot: AtlasSlot,
    transform: Affine,
    area: Rect,
) {
    let paint_transform = B::paint_transform(&atlas_slot);
    B::render_from_atlas(
        renderer,
        atlas_slot,
        transform,
        area,
        quality_for_skew(&transform),
        paint_transform,
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

/// Returns `true` if the transform is axis-aligned (no rotation or skew).
///
/// Axis-aligned transforms can be cached in the atlas; rotated/skewed glyphs
/// fall back to direct rendering.
#[inline]
pub(crate) fn is_axis_aligned(transform: &Affine) -> bool {
    !has_skew(transform)
}

/// Returns `true` if the transform has any rotation or skew component.
#[inline]
pub(crate) fn has_skew(transform: &Affine) -> bool {
    let [_, b, c, _, _, _] = transform.as_coeffs();
    b.abs() > 1e-6 || c.abs() > 1e-6
}

/// Choose image sampling quality based on downscale factor.
///
/// Returns `High` when the transform scales below 50% (where aliasing is
/// visible), `Medium` otherwise.
#[inline]
pub(crate) fn quality_for_scale(transform: &Affine) -> ImageQuality {
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
    if has_skew(transform) {
        ImageQuality::Medium
    } else {
        ImageQuality::Low
    }
}

/// Trait for types that can execute atlas draw commands.
///
/// Both the actual renderers (`RenderContext`, `Scene`) implement this trait
/// so that recorded [`AtlasCommand`]s can be replayed into them at render time.
pub trait AtlasReplayTarget {
    /// Set the current transform.
    fn set_transform(&mut self, t: Affine);
    /// Set the current paint to a solid colour.
    fn set_paint_solid(&mut self, color: AlphaColor<Srgb>);
    /// Set the current paint to a gradient.
    fn set_paint_gradient(&mut self, gradient: Gradient);
    /// Set the paint transform.
    fn set_paint_transform(&mut self, t: Affine);
    /// Fill a path with the current paint and transform.
    fn fill_path(&mut self, path: &BezPath);
    /// Fill a rectangle with the current paint and transform.
    fn fill_rect(&mut self, rect: &Rect);
    /// Push a clip layer defined by a path.
    fn push_clip_layer(&mut self, clip: &BezPath);
    /// Push a blend/compositing layer.
    fn push_blend_layer(&mut self, blend_mode: BlendMode);
    /// Pop the most recent clip or blend layer.
    fn pop_layer(&mut self);
}

/// Replay recorded atlas commands into a target that implements [`AtlasReplayTarget`].
///
/// The commands `Vec` is drained, freeing memory as each command is consumed.
pub fn replay_atlas_commands(
    commands: &mut Vec<AtlasCommand>,
    target: &mut impl AtlasReplayTarget,
) {
    for cmd in commands.drain(..) {
        match cmd {
            AtlasCommand::SetTransform(t) => target.set_transform(t),
            AtlasCommand::SetPaint(AtlasPaint::Solid(c)) => target.set_paint_solid(c),
            AtlasCommand::SetPaint(AtlasPaint::Gradient(g)) => target.set_paint_gradient(g),
            AtlasCommand::SetPaintTransform(t) => target.set_paint_transform(t),
            AtlasCommand::FillPath(p) => target.fill_path(&p),
            AtlasCommand::FillRect(r) => target.fill_rect(&r),
            AtlasCommand::PushClipLayer(c) => target.push_clip_layer(&c),
            AtlasCommand::PushBlendLayer(m) => target.push_blend_layer(m),
            AtlasCommand::PopLayer => target.pop_layer(),
        }
    }
}
