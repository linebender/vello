// Copyright 2025 the Vello Authors and the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello CPU glyph rendering backend.
//!
//! Provides [`GlyphAtlas`] (atlas backed by per-page [`Pixmap`]s) and the
//! [`GlyphRenderer`](glifo::GlyphRenderer) implementation for [`RenderContext`] that
//! rasterises glyphs into CPU-accessible pixel buffers.
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
use core::fmt::Debug;
use glifo::atlas::{
    AtlasConfig, AtlasSlot, GlyphAtlas, GlyphCacheConfig, ImageCache, PendingClearRect,
};
use glifo::{AtlasCacher, DrawSink, GlyphRunBackend};
use glifo::{Glyph, renderer};
use kurbo::{Affine, BezPath, Rect};
use peniko::BlendMode;
use peniko::color::{AlphaColor, Srgb};
use vello_common::fearless_simd::Level;
use vello_common::paint::ImageId;

fn atlas_page_image_id(page_index: u32) -> ImageId {
    ImageId::new(ATLAS_IMAGE_ID_BASE + page_index)
}

#[derive(Debug)]
pub(crate) struct GlyphAtlasResources {
    pub(crate) glyph_atlas: GlyphAtlas,
    pub(crate) image_cache: ImageCache,
    pub(crate) glyph_renderer: Box<RenderContext>,
    /// One `Pixmap` per atlas page, grown on demand.
    // It's a bit annoying to have this in an `Arc`, but it needs to be this way. During fine
    // rasterization, we need to be able to easily clone the atlas page so that it can be shared
    // across multiple threads. However, we also need to be able to mutate the pixmap to
    // sync new glyphs. The way this is achieved is by calling `Arc::get_mut` when syncing
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

impl GlyphAtlasResources {
    pub(crate) fn with_config(
        page_width: u16,
        page_height: u16,
        level: Level,
        render_mode: RenderMode,
        eviction_config: GlyphCacheConfig,
    ) -> Self {
        Self {
            glyph_atlas: GlyphAtlas::with_config(eviction_config),
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
            pixmaps: Vec::new(),
            page_width,
            page_height,
        }
    }

    pub(crate) fn maintain(&mut self) {
        self.glyph_atlas.maintain(&mut self.image_cache);
    }
}

/// Ensure a pixmap exists for the given page, creating it if needed.
fn ensure_page(
    pixmaps: &mut Vec<Arc<Pixmap>>,
    page_width: u16,
    page_height: u16,
    page_index: usize,
) {
    while pixmaps.len() <= page_index {
        pixmaps.push(Arc::new(Pixmap::new(page_width, page_height)));
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
            let page_count = glyph_resources.pixmaps.len();
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
        let glyph_resources = self
            .glyph_resources
            .as_mut()
            .expect("glyph atlas resources must exist before syncing");

        // Upload all pending bitmap glyphs to the image atlas.
        for upload in glyph_resources.glyph_atlas.drain_pending_uploads() {
            let page_index = upload.atlas_slot.page_index as usize;
            ensure_page(
                &mut glyph_resources.pixmaps,
                glyph_resources.page_width,
                glyph_resources.page_height,
                page_index,
            );
            let pixmap = Arc::get_mut(&mut glyph_resources.pixmaps[page_index])
                .expect("atlas pixmap should be uniquely owned during bitmap upload");
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
        glyph_resources
            .glyph_atlas
            .replay_pending_atlas_commands(|recorder| {
                let page_index = recorder.page_index as usize;
                ensure_page(
                    &mut glyph_resources.pixmaps,
                    glyph_resources.page_width,
                    glyph_resources.page_height,
                    page_index,
                );
                let page = Arc::get_mut(&mut glyph_resources.pixmaps[page_index])
                    .expect("atlas page pixmap must be uniquely owned during replay");

                glyph_renderer.reset();
                renderer::replay_atlas_commands(&mut recorder.commands, glyph_renderer);
                glyph_renderer.flush();
                glyph_renderer.composite_to_pixmap_at_offset(&Self::default(), page, 0, 0);
            });

        for (page_index, pixmap) in glyph_resources.pixmaps.iter().enumerate() {
            self.image_registry
                .register_atlas_page(page_index as u32, Arc::clone(pixmap));
        }
    }

    fn clear_evicted_glyph_atlas_regions(&mut self) {
        let glyph_resources = self
            .glyph_resources
            .as_mut()
            .expect("glyph atlas resources must exist before clearing");
        for clear in glyph_resources.glyph_atlas.drain_pending_clear_rects() {
            let pixmap = Arc::get_mut(&mut glyph_resources.pixmaps[clear.page_index as usize])
                .expect("atlas pixmap should be uniquely owned during region clearing");
            clear_pixmap_region(pixmap, clear);
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
        render: impl FnOnce(&mut glifo::GlyphRunRenderer<'a, 'a, Glyphs>, &mut RenderContext),
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

impl DrawSink for RenderContext {
    #[inline]
    fn set_transform(&mut self, t: Affine) {
        Self::set_transform(self, t);
    }

    #[inline]
    fn set_paint(&mut self, paint: glifo::AtlasPaint) {
        Self::set_paint(self, paint);
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
        // TODO: See the comment for the `colr_test_glyphs` test.
        if blend_mode != BlendMode::default() {
            panic!("COLR emojis with non-default blending are not supported yet.")
        }

        Self::push_blend_layer(self, blend_mode);
    }

    #[inline]
    fn pop_layer(&mut self) {
        Self::pop_layer(self);
    }

    #[inline]
    fn width(&self) -> u16 {
        Self::width(self)
    }

    #[inline]
    fn height(&self) -> u16 {
        Self::height(self)
    }
}

impl glifo::GlyphRenderer for RenderContext {
    type SavedState = vello_common::recording::RenderState;

    #[inline]
    fn save_state(&mut self) -> Self::SavedState {
        self.save_current_state()
    }

    #[inline]
    fn restore_state(&mut self, state: Self::SavedState) {
        Self::restore_state(self, state);
    }

    #[inline]
    fn stroke_path(&mut self, path: &BezPath) {
        Self::stroke_path(self, path);
    }

    #[inline]
    fn set_paint_image(&mut self, image: Image) {
        self.set_paint(image);
    }

    #[inline]
    fn set_tint(&mut self, tint: Option<vello_common::paint::Tint>) {
        Self::set_tint(self, tint);
    }

    #[inline]
    fn get_context_color(&self) -> AlphaColor<Srgb> {
        let paint = self.paint().clone();
        match paint {
            PaintType::Solid(s) => s,
            _ => BLACK,
        }
    }

    #[inline]
    fn atlas_image_source(&self, atlas_slot: &AtlasSlot) -> ImageSource {
        ImageSource::opaque_id(atlas_page_image_id(atlas_slot.page_index))
    }

    #[inline]
    fn atlas_paint_transform(&self, atlas_slot: &AtlasSlot) -> Affine {
        Affine::translate((-(atlas_slot.x as f64), -(atlas_slot.y as f64)))
    }
}

/// Debug utilities for visualizing glyph bounds during rasterization.
#[cfg(debug_assertions)]
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
