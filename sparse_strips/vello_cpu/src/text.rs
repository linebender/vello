// Copyright 2025 the Vello Authors and the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello CPU glyph rendering backend.
//!
//! Provides [`GlyphAtlas`] (atlas backed by per-page [`Pixmap`]s) and the
//! `CpuBackend` implementation of `GlyphAtlasBackend` that rasterises
//! glyphs into CPU-accessible pixel buffers.
//!
//! The key difference from the hybrid backend is that atlas pages are owned as
//! [`Arc<Pixmap>`]s here, so the CPU renderer can read pixels directly without
//! any GPU upload step.

use crate::render::{ATLAS_IMAGE_ID_BASE, DEFAULT_GLYPH_ATLAS_SIZE};
use crate::{
    Image, ImageSource, PaintType, Pixmap, RenderContext, RenderMode, RenderSettings, Resources,
    color, kurbo, peniko,
};
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use color::palette::css::BLACK;
use core::fmt::{Debug, Formatter};
use glifo::atlas::{
    AtlasCommandRecorder, AtlasConfig, AtlasSlot, GlyphAtlas as GlifoGlyphAtlas, GlyphCache,
    GlyphCacheConfig, GlyphCacheKey, ImageCache, PendingBitmapUpload, PendingClearRect,
    RasterMetrics,
};
use glifo::renderers::vello_renderer::{
    self, AtlasReplayTarget, GlyphAtlasBackend, quality_for_scale,
};
use glifo::{
    AtlasCacher, CachedGlyphType, ColrPainter, ColrRenderer, GlyphBitmap, GlyphColr, GlyphRenderer,
    GlyphRunBackend, PreparedGlyph,
};
use kurbo::{Affine, BezPath, Rect};
use peniko::Extend;
use peniko::color::{AlphaColor, Srgb};
use peniko::{BlendMode, Gradient};
use peniko::{ImageQuality, ImageSampler};
use vello_common::fearless_simd::Level;
use vello_common::glyph::Glyph;
use vello_common::paint::{ImageId, Tint};

fn atlas_page_image_id(page_index: u32) -> ImageId {
    ImageId::new(ATLAS_IMAGE_ID_BASE + page_index)
}

/// CPU-side glyph atlas backed by per-page [`Pixmap`]s.
///
/// Wraps the shared [`GlyphAtlas`] allocator and adds owned pixel storage
/// so that glyphs can be rasterized directly into CPU-accessible memory.
pub(crate) struct GlyphAtlas {
    /// Shared cache data.
    pub(crate) inner: GlifoGlyphAtlas,
    /// One `Pixmap` per atlas page, grown on demand.
    // It's a bit annoying to have this in an `Arc`, but it needs to be this way. During fine
    // rasterization, we need to be able to easily clone the atlas page so that it can be shared
    // across multiple threads. However, we also need to be able to mutate the pixmap to
    // sync new glyphs. The way this is achieved is by calling `Arc::make_mut` when syncing
    // (at this point, the pixmap isn't shared anywhere else). Before fine rasterization, we
    // then share it with the image registry such that it can easily be fetched and cloned during
    // fine rasterization. After that, we remove it from the image registry, such that it's uniquely
    // owned again and can be mutated in the next frame.
    pub(crate) pixmaps: Vec<Arc<Pixmap>>,
    /// Width of each atlas page in pixels.
    page_width: u16,
    /// Height of each atlas page in pixels.
    page_height: u16,
}

impl GlyphAtlas {
    /// Creates a new atlas with the given page dimensions and default eviction settings.
    #[inline]
    pub(crate) fn new(page_width: u16, page_height: u16) -> Self {
        // TODO: Make glyph cache config configurable by user?
        Self::with_config(page_width, page_height, GlyphCacheConfig::default())
    }

    /// Creates a new atlas with custom page dimensions and eviction settings.
    #[inline]
    pub(crate) fn with_config(
        page_width: u16,
        page_height: u16,
        eviction_config: GlyphCacheConfig,
    ) -> Self {
        Self {
            inner: GlifoGlyphAtlas::with_config(eviction_config),
            pixmaps: Vec::new(),
            page_width,
            page_height,
        }
    }

    /// Returns a mutable reference to the pixmap for `page_index`.
    #[inline]
    pub(crate) fn page_pixmap_mut(&mut self, page_index: usize) -> Option<&mut Pixmap> {
        self.pixmaps.get_mut(page_index).and_then(Arc::get_mut)
    }

    /// Returns the number of atlas pages currently allocated.
    #[inline]
    pub(crate) fn page_count(&self) -> usize {
        self.pixmaps.len()
    }

    /// Replay pending atlas commands with access to the per-page pixmaps.
    ///
    /// The closure receives `(recorder, pixmaps)`, allowing the caller to
    /// composite into the target page pixmap without a borrow conflict.
    #[inline]
    pub(crate) fn replay_pending_atlas_commands_with_pixmaps(
        &mut self,
        mut f: impl FnMut(&mut AtlasCommandRecorder, &mut Vec<Arc<Pixmap>>),
    ) {
        self.inner
            .replay_pending_atlas_commands(|recorder| f(recorder, &mut self.pixmaps));
    }
}

impl Default for GlyphAtlas {
    fn default() -> Self {
        Self::new(DEFAULT_GLYPH_ATLAS_SIZE, DEFAULT_GLYPH_ATLAS_SIZE)
    }
}

impl Debug for GlyphAtlas {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GlyphAtlas")
            .field("inner", &self.inner)
            .field("page_count", &self.pixmaps.len())
            .field("page_width", &self.page_width)
            .field("page_height", &self.page_height)
            .finish()
    }
}

/// Delegates to the inner [`GlyphAtlas`], additionally growing the `pixmaps`
/// vector when `insert` opens a new atlas page.
impl GlyphCache for GlyphAtlas {
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
        let (page_index, x, y, atlas_slot) =
            self.inner.insert_entry(image_cache, key, raster_metrics)?;

        // Create a new pixmap if the allocator opened a new atlas page
        if self.pixmaps.len() <= page_index {
            debug_assert_eq!(
                self.pixmaps.len(),
                page_index,
                "atlas page indices must be contiguous"
            );
            self.pixmaps
                .push(Arc::new(Pixmap::new(self.page_width, self.page_height)));
        }

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
        self.pixmaps.clear();
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

#[derive(Debug)]
pub(crate) struct GlyphAtlasResources {
    pub(crate) glyph_atlas: GlyphAtlas,
    pub(crate) image_cache: ImageCache,
    pub(crate) glyph_renderer: Box<RenderContext>,
}

impl GlyphAtlasResources {
    pub(crate) fn with_config(
        page_width: u16,
        page_height: u16,
        level: Level,
        render_mode: RenderMode,
        eviction_config: GlyphCacheConfig,
    ) -> Self {
        Self {
            glyph_atlas: GlyphAtlas::with_config(page_width, page_height, eviction_config),
            image_cache: ImageCache::new_with_config(AtlasConfig::default()),
            glyph_renderer: Box::new(RenderContext::new_with(
                page_width,
                page_height,
                RenderSettings {
                    level,
                    num_threads: 0,
                    render_mode,
                },
            )),
        }
    }

    pub(crate) fn maintain(&mut self) {
        self.glyph_atlas.maintain(&mut self.image_cache);
    }
}

impl Resources {
    pub(crate) fn prepare_glyph_cache(&mut self) {
        if self.glyph_resources.is_some() {
            self.sync_glyph_cache();
        }
    }

    pub(crate) fn maintain_glyph_cache(&mut self) {
        self.glyph_prep_cache.maintain();

        if let Some(glyph_resources) = self.glyph_resources.as_mut() {
            glyph_resources.maintain();
            // See the comment in `GlyphAtlas`.
            let page_count = glyph_resources.glyph_atlas.page_count();
            for page_index in 0..page_count {
                self.image_registry.destroy_atlas_page(page_index as u32);
            }
            self.clear_evicted_glyph_atlas_regions();
        }
    }

    fn ensure_glyph_resources(&mut self, level: Level, render_mode: RenderMode) {
        if self.glyph_resources.is_none() {
            self.glyph_resources = Some(GlyphAtlasResources::with_config(
                DEFAULT_GLYPH_ATLAS_SIZE,
                DEFAULT_GLYPH_ATLAS_SIZE,
                level,
                render_mode,
                GlyphCacheConfig::default(),
            ));
        }
    }

    /// Upload all pending bitmaps, rasterize pending outline/COLR glyphs, etc.
    fn sync_glyph_cache(&mut self) {
        self.pending_glyph_uploads_scratch.clear();
        let glyph_resources = self
            .glyph_resources
            .as_mut()
            .expect("glyph atlas resources must exist before syncing");
        self.pending_glyph_uploads_scratch
            .extend(glyph_resources.glyph_atlas.drain_pending_uploads());

        // Upload all pending bitmap glyphs to the image atlas.
        for upload in &self.pending_glyph_uploads_scratch {
            let pixmap = glyph_resources
                .glyph_atlas
                .page_pixmap_mut(upload.atlas_slot.page_index as usize)
                .expect("atlas upload refers to a missing page");
            copy_pixmap_to_atlas(
                &upload.pixmap,
                pixmap,
                upload.atlas_slot.x,
                upload.atlas_slot.y,
                upload.atlas_slot.width,
                upload.atlas_slot.height,
            );
        }

        // Draw all new COLR/outline glyphs into the render context, and then composite them into the
        // existing atlas page.
        let glyph_renderer = glyph_resources.glyph_renderer.as_mut();
        let glyph_atlas = &mut glyph_resources.glyph_atlas;
        glyph_atlas.replay_pending_atlas_commands_with_pixmaps(|recorder, pixmaps| {
            let page = Arc::get_mut(
                pixmaps
                    .get_mut(recorder.page_index as usize)
                    .expect("atlas recorder refers to a missing page"),
            )
            .expect("atlas page pixmap must be uniquely owned during replay");

            glyph_renderer.reset();
            vello_renderer::replay_atlas_commands(&mut recorder.commands, glyph_renderer);
            glyph_renderer.flush();
            // Note: This method panics if multi-threading is enabled, but our glyph renderer is always
            // single-threaded anyway, so this shouldn't ever panic, even if the main render context
            // uses multi-threading.
            glyph_renderer.composite_to_pixmap_at_offset(&Self::default(), page, 0, 0);
        });

        // See the comment in `GlyphAtlas`.
        for (page_index, pixmap) in glyph_resources.glyph_atlas.pixmaps.iter().enumerate() {
            self.image_registry
                .register_atlas_page(page_index as u32, Arc::clone(pixmap));
        }
    }

    fn clear_evicted_glyph_atlas_regions(&mut self) {
        self.pending_glyph_clear_rects_scratch.clear();
        let glyph_resources = self
            .glyph_resources
            .as_mut()
            .expect("glyph atlas resources must exist before clearing");
        self.pending_glyph_clear_rects_scratch
            .extend(glyph_resources.glyph_atlas.drain_pending_clear_rects());

        for clear in &self.pending_glyph_clear_rects_scratch {
            let pixmap = glyph_resources
                .glyph_atlas
                .page_pixmap_mut(clear.page_index as usize)
                .expect("atlas clear rect refers to a missing page");
            clear_pixmap_region(pixmap, *clear);
        }
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct CpuGlyphRunBackend<'a> {
    pub ctx: &'a mut RenderContext,
    pub resources: &'a mut Resources,
    pub atlas_cache_enabled: bool,
}

impl<'a> CpuGlyphRunBackend<'a> {
    fn render_glyphs<Glyphs>(
        self,
        run: glifo::GlyphRun<'a>,
        glyphs: Glyphs,
        render: impl FnOnce(
            &mut glifo::GlyphRunRenderer<'a, 'a, Glyphs, GlyphAtlas>,
            &mut RenderContext,
        ),
    ) where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        let atlas_cacher = if self.atlas_cache_enabled {
            self.resources.ensure_glyph_resources(
                self.ctx.render_settings.level,
                self.ctx.render_settings.render_mode,
            );
            let glyph_resources = self
                .resources
                .glyph_resources
                .as_mut()
                .expect("glyph atlas resources must exist after initialization");
            AtlasCacher::Enabled(
                &mut glyph_resources.glyph_atlas,
                &mut glyph_resources.image_cache,
            )
        } else {
            AtlasCacher::Disabled
        };

        let mut glyph_run = run.build(
            glyphs,
            self.resources.glyph_prep_cache.as_mut(),
            atlas_cacher,
        );
        render(&mut glyph_run, self.ctx);
    }
}

impl<'a> GlyphRunBackend<'a> for CpuGlyphRunBackend<'a> {
    fn atlas_cache(mut self, enabled: bool) -> Self {
        self.atlas_cache_enabled = enabled;
        self
    }

    fn fill_glyphs<Glyphs>(self, run: glifo::GlyphRun<'a>, glyphs: Glyphs)
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        self.render_glyphs(run, glyphs, |glyph_run, ctx| glyph_run.fill_glyphs(ctx));
    }

    fn stroke_glyphs<Glyphs>(self, run: glifo::GlyphRun<'a>, glyphs: Glyphs)
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        self.render_glyphs(run, glyphs, |glyph_run, ctx| glyph_run.stroke_glyphs(ctx));
    }
}

/// A glyph run builder.
pub type GlyphRunBuilder<'a> = glifo::GlyphRunBuilder<'a, CpuGlyphRunBackend<'a>>;

/// Zero out a rectangular region in the atlas pixmap.
///
/// Necessary because `composite_to_pixmap_at_offset` uses `SrcOver` blending,
/// so stale pixels from evicted glyphs would bleed through if not cleared.
fn clear_pixmap_region(dst: &mut Pixmap, rect: PendingClearRect) {
    let dst_stride = dst.width() as usize;
    let dst_data = dst.data_as_u8_slice_mut();
    let clear_width = rect.width as usize;
    let clear_height = rect.height as usize;

    for y in 0..clear_height {
        let row_start = ((rect.y as usize + y) * dst_stride + rect.x as usize) * 4;
        let row_end = row_start + clear_width * 4;
        dst_data[row_start..row_end].fill(0);
    }
}

/// Copy bitmap glyph pixels into a rectangular region of an atlas page.
fn copy_pixmap_to_atlas(
    src: &Pixmap,
    dst: &mut Pixmap,
    dst_x: u16,
    dst_y: u16,
    width: u16,
    height: u16,
) {
    let copy_width = width as usize;
    let copy_height = height as usize;
    let src_stride = src.width() as usize;
    let dst_stride = dst.width() as usize;

    let src_data = src.data_as_u8_slice();
    let dst_data = dst.data_as_u8_slice_mut();

    for y in 0..copy_height {
        let src_row_start = y * src_stride * 4;
        let src_row_end = src_row_start + copy_width * 4;
        let dst_row_start = ((dst_y as usize + y) * dst_stride + dst_x as usize) * 4;
        let dst_row_end = dst_row_start + copy_width * 4;

        dst_data[dst_row_start..dst_row_end].copy_from_slice(&src_data[src_row_start..src_row_end]);
    }
}

/// Bridges Glifo's [`GlyphRenderer`] trait to the shared
/// [`vello_renderer`] cache orchestration for the CPU backend.
impl GlyphRenderer<GlyphAtlas> for RenderContext {
    #[inline]
    fn fill_glyph(
        &mut self,
        prepared_glyph: PreparedGlyph<'_>,
        atlas_cacher: &mut AtlasCacher<'_, GlyphAtlas>,
    ) {
        vello_renderer::fill_glyph::<CpuBackend>(self, prepared_glyph, atlas_cacher);
    }

    #[inline]
    fn stroke_glyph(
        &mut self,
        prepared_glyph: PreparedGlyph<'_>,
        atlas_cacher: &mut AtlasCacher<'_, GlyphAtlas>,
    ) {
        vello_renderer::stroke_glyph::<CpuBackend>(self, prepared_glyph, atlas_cacher);
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
                vello_renderer::render_outline_glyph_from_atlas::<CpuBackend>(
                    self,
                    cached_slot,
                    transform,
                    tint,
                );
            }
            CachedGlyphType::Bitmap => {
                vello_renderer::render_bitmap_glyph_from_atlas::<CpuBackend>(
                    self,
                    cached_slot,
                    transform,
                );
            }
            CachedGlyphType::Colr(area) => {
                vello_renderer::render_colr_glyph_from_atlas::<CpuBackend>(
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

/// Zero-sized marker that selects the Vello CPU rendering backend
/// in generic [`GlyphAtlasBackend`] code.
pub(crate) struct CpuBackend;

impl GlyphAtlasBackend for CpuBackend {
    type Renderer = RenderContext;
    type Cache = GlyphAtlas;

    fn render_from_atlas(
        renderer: &mut RenderContext,
        atlas_slot: AtlasSlot,
        rect_transform: Affine,
        area: Rect,
        quality: ImageQuality,
        paint_transform: Affine,
        tint: Option<Tint>,
    ) {
        let image = Image {
            image: ImageSource::opaque_id(atlas_page_image_id(atlas_slot.page_index)),
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

    fn paint_transform(atlas_slot: &AtlasSlot) -> Affine {
        // The CPU backend uses full-page pixmaps, so the paint origin must be
        // shifted to the glyph's slot within the page. Contrast with the hybrid
        // backend, which gets per-allocation origins from the image cache.
        Affine::translate((-(atlas_slot.x as f64), -(atlas_slot.y as f64)))
    }

    fn fill_outline_directly(renderer: &mut RenderContext, path: &BezPath, transform: Affine) {
        let state = renderer.save_current_state();
        renderer.set_transform(transform);
        renderer.fill_path(path);
        renderer.restore_state(state);
    }

    fn stroke_outline_directly(renderer: &mut RenderContext, path: &BezPath, transform: Affine) {
        let state = renderer.save_current_state();
        renderer.set_transform(transform);
        renderer.stroke_path(path);
        renderer.restore_state(state);
    }

    fn render_bitmap_directly(renderer: &mut RenderContext, glyph: GlyphBitmap, transform: Affine) {
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
        renderer.set_paint(image);
        renderer.set_transform(transform);
        renderer.fill_rect(&glyph.area);
        renderer.restore_state(state);
    }

    fn render_colr_directly(
        renderer: &mut RenderContext,
        glyph: &GlyphColr<'_>,
        transform: Affine,
        context_color: AlphaColor<Srgb>,
    ) {
        let state = renderer.save_current_state();
        renderer.set_transform(transform);

        let mut colr_painter = ColrPainter::new(glyph, context_color, renderer);
        colr_painter.paint();

        renderer.restore_state(state);
    }
}

/// Maps COLR paint operations to Vello CPU draw calls.
impl ColrRenderer for RenderContext {
    fn push_clip_layer(&mut self, clip: BezPath) {
        Self::push_clip_layer(self, &clip);
    }

    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        Self::push_blend_layer(self, blend_mode);
    }

    fn fill_solid(&mut self, color: AlphaColor<Srgb>) {
        self.set_paint(color);
        self.fill_rect(&Rect::new(
            0.0,
            0.0,
            f64::from(self.width()),
            f64::from(self.height()),
        ));
    }

    fn fill_gradient(&mut self, gradient: Gradient) {
        self.set_paint(gradient);
        self.fill_rect(&Rect::new(
            0.0,
            0.0,
            f64::from(self.width()),
            f64::from(self.height()),
        ));
    }

    fn set_paint_transform(&mut self, affine: Affine) {
        Self::set_paint_transform(self, affine);
    }

    fn pop_layer(&mut self) {
        Self::pop_layer(self);
    }
}

/// Allows recorded [`AtlasCommand`](glifo::atlas::commands::AtlasCommand)s
/// to be replayed into a CPU [`RenderContext`].
impl AtlasReplayTarget for RenderContext {
    #[inline]
    fn set_transform(&mut self, t: Affine) {
        Self::set_transform(self, t);
    }

    #[inline]
    fn set_paint_solid(&mut self, color: AlphaColor<Srgb>) {
        self.set_paint(color);
    }

    #[inline]
    fn set_paint_gradient(&mut self, gradient: Gradient) {
        self.set_paint(gradient);
    }

    #[inline]
    fn set_paint_transform(&mut self, t: Affine) {
        Self::set_paint_transform(self, t);
    }

    #[inline]
    fn fill_path(&mut self, path: &BezPath) {
        Self::fill_path(self, path);
    }

    #[inline]
    fn fill_rect(&mut self, rect: &Rect) {
        Self::fill_rect(self, rect);
    }

    #[inline]
    fn push_clip_layer(&mut self, clip: &BezPath) {
        Self::push_clip_layer(self, clip);
    }

    #[inline]
    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        Self::push_blend_layer(self, blend_mode);
    }

    #[inline]
    fn pop_layer(&mut self) {
        Self::pop_layer(self);
    }
}

/// Debug utilities for visualizing glyph bounds during rasterization.
#[cfg(feature = "debug_glyph_bounds")]
#[allow(
    dead_code,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    reason = "debug-only utilities called manually during development"
)]
mod debug {
    use core::sync::atomic::{AtomicUsize, Ordering};

    use crate::RenderContext;
    use crate::kurbo::{Affine, Rect};
    use crate::peniko;
    use glifo::atlas::RasterMetrics;

    static COLOR_INDEX: AtomicUsize = AtomicUsize::new(0);

    /// Twelve semi-transparent rotating colours for distinguishing adjacent
    /// glyph bounds. Each call to [`fill_glyph_bounds`] advances the index.
    const COLORS: [peniko::Color; 12] = [
        peniko::Color::new([1.0, 0.0, 0.0, 0.5]), // Red
        peniko::Color::new([0.0, 1.0, 0.0, 0.5]), // Green
        peniko::Color::new([0.0, 0.0, 1.0, 0.5]), // Blue
        peniko::Color::new([1.0, 1.0, 0.0, 0.5]), // Yellow
        peniko::Color::new([1.0, 0.0, 1.0, 0.5]), // Magenta
        peniko::Color::new([0.0, 1.0, 1.0, 0.5]), // Cyan
        peniko::Color::new([1.0, 0.5, 0.0, 0.5]), // Orange
        peniko::Color::new([0.5, 0.0, 1.0, 0.5]), // Purple
        peniko::Color::new([0.0, 1.0, 0.5, 0.5]), // Mint
        peniko::Color::new([1.0, 0.5, 0.5, 0.5]), // Pink
        peniko::Color::new([0.5, 1.0, 0.5, 0.5]), // Light green
        peniko::Color::new([0.5, 0.5, 1.0, 0.5]), // Light blue
    ];

    /// Fill the glyph's pixel bounds with the next rotating debug colour.
    /// Call before rendering the actual glyph content to visualise the extent.
    pub fn fill_glyph_bounds(renderer: &mut RenderContext, raster_metrics: &RasterMetrics) {
        let idx = COLOR_INDEX.fetch_add(1, Ordering::Relaxed) % COLORS.len();
        renderer.set_transform(Affine::IDENTITY);
        renderer.set_paint(COLORS[idx]);
        renderer.fill_rect(&Rect::new(
            0.0,
            0.0,
            raster_metrics.width as f64,
            raster_metrics.height as f64,
        ));
    }
}
