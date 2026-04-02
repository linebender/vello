// Copyright 2026 the Vello Authors and the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello Hybrid (GPU) glyph rendering backend.
//!
//! Provides [`GpuGlyphAtlas`] and the `HybridBackend` implementation of
//! `GlyphAtlasBackend`.
//!
//! Unlike the CPU backend, no local `Pixmap` storage is allocated here — the
//! GPU renderer owns atlas textures and receives pixel data through the
//! pending-upload queue.

use crate::AtlasId;
use crate::Scene;
use alloc::sync::Arc;
use alloc::vec::Vec;
use glifo::atlas::{PendingBitmapUpload, PendingClearRect};
use glifo::renderers::vello_renderer;
use glifo::renderers::vello_renderer::replay_atlas_commands;
use glifo::renderers::vello_renderer::{AtlasReplayTarget, GlyphAtlasBackend, quality_for_scale};
use glifo::{
    AtlasCommandRecorder, AtlasSlot, CachedGlyphType, ColrPainter, ColrRenderer, GLYPH_PADDING,
    GlyphAtlas, GlyphBitmap, GlyphCache, GlyphCacheConfig, GlyphCacheKey, GlyphCaches, GlyphColr,
    GlyphRenderer, GlyphRunBackend, HintCache, ImageCache, OutlineCache, PreparedGlyph,
    RasterMetrics,
};
use peniko::color::palette::css::BLACK;
use peniko::color::{AlphaColor, Srgb};
use peniko::{BlendMode, Extend, Gradient, ImageQuality, ImageSampler};
use vello_common::glyph::Glyph;
use vello_common::kurbo::{Affine, BezPath, Rect};
use vello_common::multi_atlas::AtlasConfig;
use vello_common::paint::{Image, ImageId, ImageSource, PaintType, Tint};
use vello_common::peniko;
use vello_common::pixmap::Pixmap;

/// Glyph atlas cache for the hybrid (GPU) renderer.
///
/// Wraps the shared [`GlyphAtlas`] for cache bookkeeping and pending uploads
/// but does not allocate any local pixel storage — the GPU renderer manages
/// atlas textures itself via `Renderer::write_to_atlas`.
#[derive(Debug, Default)]
pub(crate) struct GpuGlyphAtlas {
    /// Shared allocator, LRU eviction state, and pending-command queues.
    inner: GlyphAtlas,
}

impl GpuGlyphAtlas {
    /// Creates a new hybrid glyph atlas cache with default eviction settings.
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: GlyphAtlas::new(),
        }
    }

    /// Creates a new hybrid glyph atlas cache with custom eviction settings.
    #[inline]
    pub fn with_config(eviction_config: GlyphCacheConfig) -> Self {
        Self {
            inner: GlyphAtlas::with_config(eviction_config),
        }
    }
}

/// Thin delegation to the inner [`GlyphAtlas`]. No page-level pixel storage
/// to manage here — the GPU owns atlas textures.
impl GlyphCache for GpuGlyphAtlas {
    #[inline(always)]
    fn get(&mut self, key: &GlyphCacheKey) -> Option<AtlasSlot> {
        self.inner.get(key)
    }

    #[inline]
    fn insert(
        &mut self,
        image_cache: &mut ImageCache,
        key: GlyphCacheKey,
        raster_metrics: RasterMetrics,
    ) -> Option<(u16, u16, AtlasSlot, &mut AtlasCommandRecorder)> {
        let (_page_index, x, y, atlas_slot) =
            self.inner.insert_entry(image_cache, key, raster_metrics)?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "atlas dimensions are configured to fit in u16"
        )]
        let (atlas_w, atlas_h) = {
            let (w, h) = image_cache.atlas_manager().config().atlas_size;
            (w as u16, h as u16)
        };
        let recorder = self
            .inner
            .recorder_for_page(atlas_slot.page_index, atlas_w, atlas_h);
        Some((x, y, atlas_slot, recorder))
    }

    #[inline]
    fn push_pending_upload(
        &mut self,
        image_id: ImageId,
        pixmap: Arc<Pixmap>,
        atlas_slot: AtlasSlot,
    ) {
        self.inner.push_pending_upload(image_id, pixmap, atlas_slot);
    }

    #[inline]
    fn drain_pending_uploads(&mut self) -> impl Iterator<Item = PendingBitmapUpload> + '_ {
        self.inner.drain_pending_uploads()
    }

    #[inline]
    fn replay_pending_atlas_commands(&mut self, f: impl FnMut(&mut AtlasCommandRecorder)) {
        self.inner.replay_pending_atlas_commands(f);
    }

    #[inline]
    fn drain_pending_clear_rects(&mut self) -> impl Iterator<Item = PendingClearRect> + '_ {
        self.inner.drain_pending_clear_rects()
    }

    #[inline]
    fn maintain(&mut self, image_cache: &mut ImageCache) {
        self.inner.maintain(image_cache);
    }

    #[inline]
    fn clear(&mut self) {
        self.inner.clear();
    }

    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    fn cache_hits(&self) -> u64 {
        self.inner.cache_hits()
    }

    #[inline]
    fn cache_misses(&self) -> u64 {
        self.inner.cache_misses()
    }

    #[inline]
    fn clear_stats(&mut self) {
        self.inner.clear_stats();
    }

    #[inline]
    fn config(&self) -> &GlyphCacheConfig {
        self.inner.config()
    }
}

/// All glyph caches needed by the hybrid (GPU) renderer.
#[derive(Debug)]
pub(crate) struct GpuGlyphCaches(pub(crate) GlyphCaches<GpuGlyphAtlas>);

impl GpuGlyphCaches {
    /// Creates a new `GpuGlyphCaches` instance with custom eviction settings.
    pub(crate) fn with_config(eviction_config: GlyphCacheConfig) -> Self {
        Self(GlyphCaches {
            outline_cache: OutlineCache::default(),
            hinting_cache: HintCache::default(),
            underline_exclusions: Vec::new(),
            glyph_atlas: GpuGlyphAtlas::with_config(eviction_config),
        })
    }
}

/// Auxiliary hybrid glyph resources.
#[derive(Debug)]
pub struct Resources {
    pub(crate) glyph_caches: GpuGlyphCaches,
    pub(crate) image_cache: ImageCache,
    pub(crate) glyph_renderer: Scene,
}

impl Resources {
    /// Create a new set of renderer resources.
    pub fn new() -> Self {
        let image_cache = ImageCache::new_with_config(Default::default());
        let (atlas_width, atlas_height) = image_cache.atlas_manager().config().atlas_size;
        Self {
            glyph_caches: GpuGlyphCaches::with_config(GlyphCacheConfig::default()),
            image_cache,
            glyph_renderer: Scene::new(atlas_width as u16, atlas_height as u16),
        }
    }
}

impl Default for Resources {
    fn default() -> Self {
        Self::new()
    }
}

impl Resources {
    #[doc(hidden)]
    pub fn atlas_config(&self) -> AtlasConfig {
        self.image_cache.atlas_manager().config().clone()
    }

    #[doc(hidden)]
    pub fn atlas_count(&self) -> u32 {
        self.image_cache.atlas_count() as u32
    }

    #[doc(hidden)]
    pub fn replay_pending_atlas_commands(&mut self, mut f: impl FnMut(&Scene, AtlasId)) {
        self.glyph_caches
            .0
            .glyph_atlas
            .replay_pending_atlas_commands(|recorder| {
                self.glyph_renderer.reset();
                let mut replay_target = ColrSceneWrapper::new(&mut self.glyph_renderer);
                replay_atlas_commands(&mut recorder.commands, &mut replay_target);
                f(&self.glyph_renderer, AtlasId::new(recorder.page_index));
            });
    }

    #[doc(hidden)]
    pub fn drain_pending_uploads(&mut self) -> impl Iterator<Item = PendingBitmapUpload> + '_ {
        self.glyph_caches.0.glyph_atlas.drain_pending_uploads()
    }

    #[doc(hidden)]
    pub fn take_pending_uploads(&mut self) -> Vec<PendingBitmapUpload> {
        self.glyph_caches
            .0
            .glyph_atlas
            .drain_pending_uploads()
            .collect()
    }

    #[doc(hidden)]
    pub fn maintain(&mut self, image_cache: &mut ImageCache) {
        self.glyph_caches.0.maintain(image_cache);
    }

    #[doc(hidden)]
    pub fn drain_pending_clear_rects(&mut self) -> impl Iterator<Item = PendingClearRect> + '_ {
        self.glyph_caches.0.glyph_atlas.drain_pending_clear_rects()
    }

    #[doc(hidden)]
    pub fn maintain_and_take_pending_clear_rects(&mut self) -> Vec<PendingClearRect> {
        self.glyph_caches.0.maintain(&mut self.image_cache);
        self.glyph_caches
            .0
            .glyph_atlas
            .drain_pending_clear_rects()
            .collect()
    }
}

// See this PR for a bit more context on why we have this.
// layer: https://github.com/linebender/vello/pull/1554
// In short, we want to ensure that COLR rendering works even when the
// `default_blending_only` scene constraint is enabled. In order to do so,
// we need to ensure that we don't use non-default blending to blend into
// the root layer.
struct ColrSceneWrapper<'a> {
    scene: &'a mut Scene,
    layer_depth: usize,
    inserted_root_wrapper: bool,
}

impl<'a> ColrSceneWrapper<'a> {
    fn new(scene: &'a mut Scene) -> Self {
        Self {
            scene,
            layer_depth: 0,
            inserted_root_wrapper: false,
        }
    }

    fn push_clip_layer_impl(&mut self, clip: &BezPath) {
        self.scene.push_layer(Some(clip), None, None, None, None);
        self.layer_depth += 1;
    }

    fn push_blend_layer_impl(&mut self, blend_mode: BlendMode) {
        if self.layer_depth == 0 && blend_mode != BlendMode::default() {
            self.scene.push_layer(None, None, None, None, None);
            self.inserted_root_wrapper = true;
        }

        self.scene
            .push_layer(None, Some(blend_mode), None, None, None);
        self.layer_depth += 1;
    }

    fn pop_layer_impl(&mut self) {
        self.scene.pop_layer();
        self.layer_depth -= 1;

        if self.layer_depth == 0 && self.inserted_root_wrapper {
            self.scene.pop_layer();
            self.inserted_root_wrapper = false;
        }
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct HybridGlyphRunBackend<'a> {
    pub scene: &'a mut Scene,
    pub resources: &'a mut Resources,
}

impl<'a> GlyphRunBackend<'a> for HybridGlyphRunBackend<'a> {
    fn fill_glyphs<Glyphs>(self, builder: glifo::GlyphRunBuilder<'a>, glyphs: Glyphs)
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        builder
            .build(
                glyphs,
                &mut self.resources.glyph_caches.0,
                &mut self.resources.image_cache,
            )
            .fill_glyphs(self.scene);
    }

    fn stroke_glyphs<Glyphs>(self, builder: glifo::GlyphRunBuilder<'a>, glyphs: Glyphs)
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        builder
            .build(
                glyphs,
                &mut self.resources.glyph_caches.0,
                &mut self.resources.image_cache,
            )
            .stroke_glyphs(self.scene);
    }
}

/// A glyph run builder.
pub type GlyphRunBuilder<'a> = glifo::GlyphRunBuilder<'a, HybridGlyphRunBackend<'a>>;

/// Bridges Parley's [`GlyphRenderer`] trait to the shared
/// [`vello_renderer`] cache orchestration for the hybrid backend.
impl GlyphRenderer<GpuGlyphAtlas> for Scene {
    #[inline]
    fn fill_glyph(
        &mut self,
        prepared_glyph: PreparedGlyph<'_>,
        glyph_atlas: &mut GpuGlyphAtlas,
        image_cache: &mut ImageCache,
    ) {
        vello_renderer::fill_glyph::<HybridBackend>(self, prepared_glyph, glyph_atlas, image_cache);
    }

    #[inline]
    fn stroke_glyph(
        &mut self,
        prepared_glyph: PreparedGlyph<'_>,
        glyph_atlas: &mut GpuGlyphAtlas,
        image_cache: &mut ImageCache,
    ) {
        vello_renderer::stroke_glyph::<HybridBackend>(
            self,
            prepared_glyph,
            glyph_atlas,
            image_cache,
        );
    }

    #[inline]
    fn render_cached_glyph(
        &mut self,
        cached_slot: AtlasSlot,
        transform: Affine,
        glyph_type: CachedGlyphType,
    ) {
        match glyph_type {
            CachedGlyphType::Outline => {
                let tint = self.get_context_color();
                vello_renderer::render_outline_glyph_from_atlas::<HybridBackend>(
                    self,
                    cached_slot,
                    transform,
                    tint,
                );
            }
            CachedGlyphType::Bitmap => {
                vello_renderer::render_bitmap_glyph_from_atlas::<HybridBackend>(
                    self,
                    cached_slot,
                    transform,
                );
            }
            CachedGlyphType::Colr(area) => {
                vello_renderer::render_colr_glyph_from_atlas::<HybridBackend>(
                    self,
                    cached_slot,
                    transform,
                    area,
                );
            }
        }
    }

    #[inline]
    fn fill_rect(&mut self, rect: Rect) {
        self.fill_rect(&rect);
    }

    #[inline]
    fn get_context_color(&self) -> AlphaColor<Srgb> {
        // Non-solid paints (gradients, images) have no single color to
        // extract, so fall back to black — the CSS default for `currentColor`.
        let paint = self.paint().clone();
        match paint {
            PaintType::Solid(s) => s,
            _ => BLACK,
        }
    }
}

/// Zero-sized marker that selects the Vello Hybrid rendering backend
/// in generic [`GlyphAtlasBackend`] code.
pub(crate) struct HybridBackend;

impl GlyphAtlasBackend for HybridBackend {
    type Renderer = Scene;
    type Cache = GpuGlyphAtlas;

    fn render_from_atlas(
        renderer: &mut Scene,
        atlas_slot: AtlasSlot,
        rect_transform: Affine,
        area: Rect,
        quality: ImageQuality,
        paint_transform: Affine,
        tint: Option<Tint>,
    ) {
        // Use the image_cache-assigned ImageId (not page_index). The GPU renderer
        // resolves this through image_cache.get() to obtain atlas layer + offset.
        let image = Image {
            image: ImageSource::opaque_id(atlas_slot.image_id),
            sampler: ImageSampler {
                x_extend: Extend::Pad,
                y_extend: Extend::Pad,
                quality,
                alpha: 1.0,
            },
        };

        let state = renderer.save_current_state();

        renderer.set_tint(tint);
        renderer.set_transform(rect_transform);
        renderer.set_paint(image);
        renderer.set_paint_transform(paint_transform);
        renderer.fill_rect(&area);

        renderer.reset_tint();
        renderer.restore_state(state);
    }

    fn paint_transform(_atlas_slot: &AtlasSlot) -> Affine {
        // The image cache resolves the slot to the allocation origin, but the
        // actual glyph bitmap is inset by GLYPH_PADDING on each side (a guard
        // band required by Extend::Pad sampling). Shift paint to compensate.
        let padding = GLYPH_PADDING as f64;
        Affine::translate((-padding, -padding))
    }

    fn fill_outline_directly(renderer: &mut Scene, path: &BezPath, transform: Affine) {
        let state = renderer.save_current_state();
        renderer.set_transform(transform);
        renderer.fill_path(path);
        renderer.restore_state(state);
    }

    fn stroke_outline_directly(renderer: &mut Scene, path: &BezPath, transform: Affine) {
        let state = renderer.save_current_state();
        renderer.set_transform(transform);
        renderer.stroke_path(path);
        renderer.restore_state(state);
    }

    fn render_bitmap_directly(renderer: &mut Scene, glyph: GlyphBitmap, transform: Affine) {
        let image = Image {
            image: ImageSource::Pixmap(glyph.pixmap),
            sampler: ImageSampler {
                x_extend: Extend::Pad,
                y_extend: Extend::Pad,
                quality: quality_for_scale(&transform),
                alpha: 1.0,
            },
        };

        let state = renderer.save_current_state();
        renderer.set_transform(transform);
        renderer.set_paint(image);
        renderer.fill_rect(&glyph.area);
        renderer.restore_state(state);
    }

    fn render_colr_directly(
        renderer: &mut Scene,
        glyph: &GlyphColr<'_>,
        transform: Affine,
        context_color: AlphaColor<Srgb>,
    ) {
        let state = renderer.save_current_state();
        renderer.set_transform(transform);

        let mut target = ColrSceneWrapper::new(renderer);
        let mut colr_painter = ColrPainter::new(glyph, context_color, &mut target);
        colr_painter.paint();

        renderer.restore_state(state);
    }
}

/// Maps COLR paint operations to Vello Hybrid draw calls.
///
/// `fill_solid` and `fill_gradient` fill the entire surface because COLR
/// compositing relies on clip layers to restrict the painted region.
impl ColrRenderer for ColrSceneWrapper<'_> {
    // TODO: Use `push_clip_path` instead of `push_layer` to take advantage of
    // Vello Hybrid fast paths. This requires tracking blend layers vs clip paths
    // separately in `colr.rs`.
    fn push_clip_layer(&mut self, clip: BezPath) {
        self.push_clip_layer_impl(&clip);
    }

    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.push_blend_layer_impl(blend_mode);
    }

    fn fill_solid(&mut self, color: AlphaColor<Srgb>) {
        self.scene.set_paint(color);
        self.scene.fill_rect(&Rect::new(
            0.0,
            0.0,
            f64::from(self.scene.width()),
            f64::from(self.scene.height()),
        ));
    }

    fn fill_gradient(&mut self, gradient: Gradient) {
        self.scene.set_paint(gradient);
        self.scene.fill_rect(&Rect::new(
            0.0,
            0.0,
            f64::from(self.scene.width()),
            f64::from(self.scene.height()),
        ));
    }

    fn set_paint_transform(&mut self, affine: Affine) {
        self.scene.set_paint_transform(affine);
    }

    fn pop_layer(&mut self) {
        self.pop_layer_impl();
    }
}

/// Allows recorded [`AtlasCommand`](crate::atlas::commands::AtlasCommand)s
/// to be replayed into a hybrid [`Scene`].
impl AtlasReplayTarget for ColrSceneWrapper<'_> {
    #[inline]
    fn set_transform(&mut self, t: Affine) {
        self.scene.set_transform(t);
    }

    #[inline]
    fn set_paint_solid(&mut self, color: AlphaColor<Srgb>) {
        self.scene.set_paint(color);
    }

    #[inline]
    fn set_paint_gradient(&mut self, gradient: Gradient) {
        self.scene.set_paint(gradient);
    }

    #[inline]
    fn set_paint_transform(&mut self, t: Affine) {
        self.scene.set_paint_transform(t);
    }

    #[inline]
    fn fill_path(&mut self, path: &BezPath) {
        self.scene.fill_path(path);
    }

    #[inline]
    fn fill_rect(&mut self, rect: &Rect) {
        self.scene.fill_rect(rect);
    }

    #[inline]
    fn push_clip_layer(&mut self, clip: &BezPath) {
        self.push_clip_layer_impl(clip);
    }

    #[inline]
    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.push_blend_layer_impl(blend_mode);
    }

    #[inline]
    fn pop_layer(&mut self) {
        self.pop_layer_impl();
    }
}
