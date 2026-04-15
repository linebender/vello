// Copyright 2026 the Vello Authors and the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello Hybrid (GPU) glyph rendering backend.
//!
//! Provides [`GlyphAtlas`] and the [`GlyphRenderer`](glifo::GlyphRenderer)
//! implementation for [`Scene`].
//!
//! Unlike the CPU backend, no local `Pixmap` storage is allocated here — the
//! GPU renderer owns atlas textures and receives pixel data through the
//! pending-upload queue.

use crate::{AtlasId, Resources, Scene};
use glifo::atlas::{PendingBitmapUpload, PendingClearRect};
use glifo::renderer::replay_atlas_commands;
use glifo::{
    AtlasCacher, AtlasSlot, DrawSink, GLYPH_PADDING, Glyph, GlyphAtlas, GlyphCacheConfig,
    GlyphRunBackend, ImageCache,
};
use peniko::BlendMode;
use peniko::color::palette::css::BLACK;
use peniko::color::{AlphaColor, Srgb};
use vello_common::kurbo::{Affine, BezPath, Rect};
use vello_common::multi_atlas::AtlasConfig;
use vello_common::paint::{Image, ImageSource, PaintType};
use vello_common::peniko;

/// Glyph atlas cache for the hybrid (GPU) renderer.
#[derive(Debug)]
pub(crate) struct GlyphAtlasResources {
    pub(crate) glyph_atlas: GlyphAtlas,
    pub(crate) glyph_renderer: Scene,
}

impl GlyphAtlasResources {
    pub(crate) fn with_config(
        atlas_width: u16,
        atlas_height: u16,
        eviction_config: GlyphCacheConfig,
    ) -> Self {
        Self {
            glyph_atlas: GlyphAtlas::with_config(eviction_config),
            glyph_renderer: Scene::new(atlas_width, atlas_height),
        }
    }

    pub(crate) fn maintain(&mut self, image_cache: &mut ImageCache) {
        self.glyph_atlas.maintain(image_cache);
    }
}

impl Resources {
    fn ensure_glyph_resources(&mut self) {
        if self.glyph_resources.is_none() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "atlas dimensions are configured to fit in u16"
            )]
            let (atlas_width, atlas_height) = {
                let (width, height) = self.image_cache.atlas_manager().config().atlas_size;
                (width as u16, height as u16)
            };
            // TODO: Use a config optionally provided by the user!
            self.glyph_resources = Some(GlyphAtlasResources::with_config(
                atlas_width,
                atlas_height,
                GlyphCacheConfig::default(),
            ));
        }
    }

    fn atlas_config(&self) -> AtlasConfig {
        *self.image_cache.atlas_manager().config()
    }

    fn atlas_count(&self) -> u32 {
        u32::try_from(self.image_cache.atlas_count()).unwrap()
    }

    fn replay_pending_atlas_commands(&mut self, mut f: impl FnMut(&Scene, AtlasId)) {
        if let Some(glyph_resources) = self.glyph_resources.as_mut() {
            glyph_resources
                .glyph_atlas
                .replay_pending_atlas_commands(|recorder| {
                    glyph_resources.glyph_renderer.reset();
                    replay_atlas_commands(
                        &mut recorder.commands,
                        &mut glyph_resources.glyph_renderer,
                    );
                    f(
                        &glyph_resources.glyph_renderer,
                        AtlasId::new(recorder.page_index),
                    );
                });
        }
    }

    pub(crate) fn before_render<T>(
        &mut self,
        backend: &mut T,
        mut render_to_atlas: impl FnMut(&mut T, &Scene, u32, AtlasConfig, AtlasId),
        mut upload_to_atlas: impl FnMut(&mut T, &ImageCache, &PendingBitmapUpload, u32, u32),
    ) {
        let atlas_count = self.atlas_count();
        let atlas_config = self.atlas_config();
        self.replay_pending_atlas_commands(|glyph_renderer, atlas_id| {
            render_to_atlas(backend, glyph_renderer, atlas_count, atlas_config, atlas_id);
        });

        const PADDING: u32 = GLYPH_PADDING as u32;

        if let Some(glyph_resources) = self.glyph_resources.as_mut() {
            for upload in glyph_resources.glyph_atlas.drain_pending_uploads() {
                let resource = self.image_cache.get(upload.image_id).unwrap();
                let dst_x = resource.offset[0] as u32 + PADDING;
                let dst_y = resource.offset[1] as u32 + PADDING;
                upload_to_atlas(backend, &self.image_cache, &upload, dst_x, dst_y);
            }
        }
    }

    pub(crate) fn after_render<T>(
        &mut self,
        backend: &mut T,
        mut clear_rect: impl FnMut(&mut T, &PendingClearRect),
    ) {
        self.glyph_prep_cache.maintain();
        if let Some(glyph_resources) = self.glyph_resources.as_mut() {
            glyph_resources.maintain(&mut self.image_cache);
            for rect in glyph_resources.glyph_atlas.drain_pending_clear_rects() {
                clear_rect(backend, &rect);
            }
        }
    }
}

/// [`DrawSink`] for the hybrid [`Scene`].
impl DrawSink for Scene {
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

#[doc(hidden)]
#[derive(Debug)]
pub struct HybridGlyphRunBackend<'a> {
    pub(crate) scene: &'a mut Scene,
    pub(crate) resources: &'a mut Resources,
    pub(crate) atlas_cache_enabled: bool,
}

impl<'a> HybridGlyphRunBackend<'a> {
    fn render_glyphs<Glyphs>(
        self,
        run: glifo::GlyphRun<'a>,
        glyphs: Glyphs,
        render: impl FnOnce(&mut glifo::GlyphRunRenderer<'a, 'a, Glyphs>, &mut Scene),
    ) where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        let atlas_cacher = if self.atlas_cache_enabled {
            self.resources.ensure_glyph_resources();
            let glyph_resources = self
                .resources
                .glyph_resources
                .as_mut()
                .expect("glyph atlas resources must exist after initialization");
            AtlasCacher::Enabled(
                &mut glyph_resources.glyph_atlas,
                &mut self.resources.image_cache,
            )
        } else {
            AtlasCacher::Disabled
        };

        let mut glyph_run = run.build(
            glyphs,
            self.resources.glyph_prep_cache.as_mut(),
            atlas_cacher,
        );
        render(&mut glyph_run, self.scene);
    }
}

impl<'a> GlyphRunBackend<'a> for HybridGlyphRunBackend<'a> {
    fn atlas_cache(mut self, enabled: bool) -> Self {
        self.atlas_cache_enabled = enabled;
        self
    }

    fn fill_glyphs<Glyphs>(self, run: glifo::GlyphRun<'a>, glyphs: Glyphs)
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        self.render_glyphs(run, glyphs, |glyph_run, scene| glyph_run.fill_glyphs(scene));
    }

    fn stroke_glyphs<Glyphs>(self, run: glifo::GlyphRun<'a>, glyphs: Glyphs)
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        self.render_glyphs(run, glyphs, |glyph_run, scene| {
            let stroke_adjustment = glyph_run.stroke_adjustment();
            let original_width = scene.stroke().width;
            scene.stroke_mut().width *= stroke_adjustment;
            glyph_run.stroke_glyphs(scene);
            scene.stroke_mut().width = original_width;
        });
    }
}

/// A glyph run builder.
pub type GlyphRunBuilder<'a> = glifo::GlyphRunBuilder<'a, HybridGlyphRunBackend<'a>>;

impl glifo::GlyphRenderer for Scene {
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
        ImageSource::opaque_id(atlas_slot.image_id)
    }

    #[inline]
    fn atlas_paint_transform(&self, _atlas_slot: &AtlasSlot) -> Affine {
        let padding = GLYPH_PADDING as f64;
        Affine::translate((-padding, -padding))
    }
}
