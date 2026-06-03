// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::coarse::{CommandBucketer, LayerClip};
use crate::dispatch::{Dispatcher, replay_render_commands};
use crate::filter::context::FilterContext;
use crate::fine::FineKernel;
use crate::kurbo::{Affine, BezPath, Rect, Stroke};
use crate::peniko::{BlendMode, Fill};
use crate::record::{CommandRecorder, FilterLayerPlan, PoppedLayer};
use crate::{CompositeMode, RasterizerSettings};
use alloc::vec::Vec;
use core::cell::RefCell;
use vello_common::clip::{ClipContext, control_point_bbox_u16};
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd};
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint};
use vello_common::pixmap::{Pixmap, PixmapMut};
use vello_common::strip_generator::{GenerationMode, StripGenerator, StripStorage};

/// Single-threaded dispatcher for the row-bucket prototype.
#[derive(Debug)]
pub(crate) struct SingleThreadedDispatcher {
    bucketer: RefCell<CommandBucketer>,
    clip_context: ClipContext,
    recorder: CommandRecorder,
    strip_generator: StripGenerator,
    strip_storage: StripStorage,
    // TODO: Once `StripGenerator`s (in particular `Tiles`) can be resized,
    // we can reuse one strip generator across filter layers.
    strip_generator_stack: Vec<StripGenerator>,
    base_width: u16,
    base_height: u16,
    level: Level,
}

impl SingleThreadedDispatcher {
    pub(crate) fn new(width: u16, height: u16, level: Level) -> Self {
        Self {
            bucketer: RefCell::new(CommandBucketer::new(width, height)),
            clip_context: ClipContext::new(),
            recorder: CommandRecorder::new(),
            strip_generator: StripGenerator::new(width, height, level),
            strip_storage: StripStorage::new(GenerationMode::Append),
            strip_generator_stack: Vec::new(),
            base_width: width,
            base_height: height,
            level,
        }
    }

    #[cfg(feature = "f32_pipeline")]
    fn rasterize_f32(
        &self,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        use crate::fine::F32Kernel;
        use vello_common::fearless_simd::dispatch;
        dispatch!(self.level, simd => self.rasterize_with::<_, F32Kernel>(simd, target, scene_width, scene_height, settings, encoded_paints, image_resolver));
    }

    #[cfg(feature = "u8_pipeline")]
    fn rasterize_u8(
        &self,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        use crate::fine::U8Kernel;
        use vello_common::fearless_simd::dispatch;
        dispatch!(self.level, simd => self.rasterize_with::<_, U8Kernel>(simd, target, scene_width, scene_height, settings, encoded_paints, image_resolver));
    }

    // Note: We purposefully don't add `vectorize` to each of these helpers,
    // since vectorization is applied wherever necessary in child functions.
    fn rasterize_with<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        let layer_manager = self.render_filter_layers::<S, F>(simd, encoded_paints, image_resolver);
        let mut bucketer = self.bucketer.borrow_mut();
        bucketer.reset(scene_width, scene_height);
        replay_render_commands(
            &self.recorder.root_cmds,
            &self.strip_storage.strips,
            &mut bucketer,
            encoded_paints,
            (0, 0),
        );

        let unpack_dest = settings.composite_mode == CompositeMode::SrcOver;
        let alpha_buffers = &[self.strip_storage.alphas.as_slice()];

        crate::fine::rasterize_at_offset::<S, F>(
            simd,
            &bucketer,
            alpha_buffers,
            &layer_manager,
            target,
            scene_width,
            scene_height,
            settings.offset.0,
            settings.offset.1,
            unpack_dest,
            encoded_paints,
            image_resolver,
        );
    }

    fn record_fill(
        &mut self,
        strip_start: usize,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    ) {
        let strip_end = self.strip_storage.strips.len();
        let viewport_width = self.strip_generator.width();
        self.recorder.push_fill(
            strip_start..strip_end,
            &self.strip_storage.strips[strip_start..strip_end],
            viewport_width,
            paint,
            blend_mode,
            mask,
            0,
        );
    }

    fn render_filter_layers<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) -> FilterContext {
        let mut layer_manager = FilterContext::new(self.recorder.filter_layers.len());
        for id in (0..self.recorder.filter_layers.len()).rev() {
            let layer = &self.recorder.filter_layers[id];
            if layer.pixmap_bbox.is_empty() {
                continue;
            }

            let width = layer.pixmap_bbox.width();
            let height = layer.pixmap_bbox.height();
            let mut pixmap = Pixmap::new(width, height);
            let mut bucketer = self.bucketer.borrow_mut();
            bucketer.reset(width, height);
            replay_render_commands(
                &layer.cmds,
                &self.strip_storage.strips,
                &mut bucketer,
                encoded_paints,
                (layer.pixmap_bbox.x0, layer.pixmap_bbox.y0),
            );
            crate::fine::rasterize_at_offset::<S, F>(
                simd,
                &bucketer,
                &[self.strip_storage.alphas.as_slice()],
                &layer_manager,
                (&mut pixmap).into(),
                width,
                height,
                0,
                0,
                false,
                encoded_paints,
                image_resolver,
            );

            F::filter_layer(
                &mut pixmap,
                &layer.filter_plan.filter,
                layer_manager.scratch(),
                layer.filter_plan.transform,
            );
            layer_manager.set_layer(id, pixmap);
        }

        layer_manager
    }

    fn push_filter_viewport(&mut self, padding: RectU16) {
        let width = self
            .strip_generator
            .width()
            .saturating_add(padding.x0)
            .saturating_add(padding.x1);
        let height = self
            .strip_generator
            .height()
            .saturating_add(padding.y0)
            .saturating_add(padding.y1);
        let filter_generator = StripGenerator::new(width, height, self.level);
        let parent_generator = core::mem::replace(&mut self.strip_generator, filter_generator);
        self.strip_generator_stack.push(parent_generator);
    }

    fn pop_filter_viewport(&mut self) {
        self.strip_generator = self
            .strip_generator_stack
            .pop()
            .expect("filter viewport stack underflow");
    }
}

impl Dispatcher for SingleThreadedDispatcher {
    fn has_layers(&self) -> bool {
        self.recorder.has_layers()
    }

    fn fill_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    ) {
        let clip_path = self.clip_context.get();
        let strip_start = self.strip_storage.strips.len();
        let strip_generator = &mut self.strip_generator;
        let strip_storage = &mut self.strip_storage;
        strip_generator.generate_filled_path(
            path,
            fill_rule,
            transform,
            aliasing_threshold,
            strip_storage,
            clip_path,
        );
        self.record_fill(strip_start, paint, blend_mode, mask);
    }

    fn stroke_path(
        &mut self,
        path: &BezPath,
        stroke: &Stroke,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    ) {
        let clip_path = self.clip_context.get();
        let strip_start = self.strip_storage.strips.len();
        let strip_generator = &mut self.strip_generator;
        let strip_storage = &mut self.strip_storage;
        strip_generator.generate_stroked_path(
            path,
            stroke,
            transform,
            aliasing_threshold,
            strip_storage,
            clip_path,
        );
        self.record_fill(strip_start, paint, blend_mode, mask);
    }

    fn fill_rect_fast(
        &mut self,
        rect: &Rect,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    ) {
        let clip_path = self.clip_context.get();
        let strip_start = self.strip_storage.strips.len();
        let strip_generator = &mut self.strip_generator;
        let strip_storage = &mut self.strip_storage;
        strip_generator.generate_filled_rect_fast(rect, strip_storage, clip_path);
        self.record_fill(strip_start, paint, blend_mode, mask);
    }

    fn push_clip_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    ) {
        self.clip_context.push_clip(
            path,
            &mut self.strip_generator,
            fill_rule,
            transform,
            aliasing_threshold,
        );
    }

    fn pop_clip_path(&mut self) {
        self.clip_context.pop_clip();
    }

    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        fill_rule: Fill,
        clip_transform: Affine,
        blend_mode: BlendMode,
        opacity: f32,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
        filter_plan: Option<FilterLayerPlan>,
    ) {
        if let Some(plan) = &filter_plan {
            self.push_filter_viewport(plan.source_padding);
        }

        let clip_path = clip_path.map(|clip_path| {
            let existing_clip = self.clip_context.get();
            let mut bbox = control_point_bbox_u16(clip_path, clip_transform);
            if let Some(existing_clip) = existing_clip {
                bbox = bbox.intersect(existing_clip.bbox);
            } else {
                bbox.x1 = bbox.x1.min(self.strip_generator.width());
                bbox.y1 = bbox.y1.min(self.strip_generator.height());
            }

            let strip_start = self.strip_storage.strips.len();
            self.strip_generator.generate_filled_path(
                clip_path,
                fill_rule,
                clip_transform,
                aliasing_threshold,
                &mut self.strip_storage,
                existing_clip,
            );

            LayerClip {
                strip_range: strip_start..self.strip_storage.strips.len(),
                thread_idx: 0,
                bbox,
            }
        });

        self.recorder
            .push_layer(blend_mode, opacity, mask, clip_path, filter_plan);
    }

    fn pop_layer(&mut self) {
        match self.recorder.pop_layer() {
            PoppedLayer::Regular => {}
            PoppedLayer::Filter => {
                self.pop_filter_viewport();
            }
        }
    }

    fn reset(&mut self) {
        // Bucketer will be reset on demand.
        self.clip_context.reset();
        self.recorder.reset();
        self.strip_generator_stack.clear();
        self.strip_generator = StripGenerator::new(self.base_width, self.base_height, self.level);
        self.strip_generator.reset();
        self.strip_storage.clear();
    }

    fn flush(&mut self) {}

    fn rasterize(
        &self,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        #[cfg(all(feature = "u8_pipeline", not(feature = "f32_pipeline")))]
        {
            self.rasterize_u8(
                target,
                scene_width,
                scene_height,
                settings,
                encoded_paints,
                image_resolver,
            );
        }

        #[cfg(all(feature = "f32_pipeline", not(feature = "u8_pipeline")))]
        {
            self.rasterize_f32(
                target,
                scene_width,
                scene_height,
                settings,
                encoded_paints,
                image_resolver,
            );
        }

        #[cfg(all(feature = "u8_pipeline", feature = "f32_pipeline"))]
        match settings.quality {
            crate::RenderMode::OptimizeSpeed => {
                self.rasterize_u8(
                    target,
                    scene_width,
                    scene_height,
                    settings,
                    encoded_paints,
                    image_resolver,
                );
            }
            crate::RenderMode::OptimizeQuality => {
                self.rasterize_f32(
                    target,
                    scene_width,
                    scene_height,
                    settings,
                    encoded_paints,
                    image_resolver,
                );
            }
        }

        #[cfg(all(not(feature = "u8_pipeline"), not(feature = "f32_pipeline")))]
        {
            let _ = (
                target,
                scene_width,
                scene_height,
                settings,
                encoded_paints,
                image_resolver,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coarse::RenderCmd;
    use crate::kurbo::Shape;
    use vello_common::color::palette::css::BLUE;
    use vello_common::geometry::RectU16;
    use vello_common::paint::PremulColor;

    fn paint() -> Paint {
        Paint::Solid(PremulColor::from_alpha_color(BLUE))
    }

    fn layer_content_bbox(dispatcher: &SingleThreadedDispatcher, cmd_idx: usize) -> RectU16 {
        match &dispatcher.recorder.root_cmds[cmd_idx] {
            RenderCmd::PushLayer {
                bbox: content_bbox, ..
            } => *content_bbox,
            _ => panic!("expected push layer command"),
        }
    }

    #[test]
    fn tracks_layer_content_bbox_ignoring_layer_clip() {
        let mut dispatcher = SingleThreadedDispatcher::new(64, 64, Level::new());
        let clip = Rect::new(0.0, 0.0, 8.0, 8.0).to_path(0.1);

        dispatcher.push_layer(
            Some(&clip),
            Fill::NonZero,
            Affine::IDENTITY,
            BlendMode::default(),
            1.0,
            None,
            None,
            None,
        );
        dispatcher.fill_rect_fast(
            &Rect::new(20.0, 12.0, 36.0, 20.0),
            paint(),
            BlendMode::default(),
            None,
        );
        dispatcher.pop_layer();

        assert_eq!(
            layer_content_bbox(&dispatcher, 0),
            RectU16::new(20, 12, 40, 20)
        );
    }

    #[test]
    fn nested_layer_content_bbox_is_propagated_to_parent() {
        let mut dispatcher = SingleThreadedDispatcher::new(64, 64, Level::new());

        dispatcher.push_layer(
            None,
            Fill::NonZero,
            Affine::IDENTITY,
            BlendMode::default(),
            1.0,
            None,
            None,
            None,
        );
        dispatcher.fill_rect_fast(
            &Rect::new(4.0, 4.0, 12.0, 12.0),
            paint(),
            BlendMode::default(),
            None,
        );

        dispatcher.push_layer(
            None,
            Fill::NonZero,
            Affine::IDENTITY,
            BlendMode::default(),
            1.0,
            None,
            None,
            None,
        );
        dispatcher.fill_rect_fast(
            &Rect::new(24.0, 24.0, 32.0, 32.0),
            paint(),
            BlendMode::default(),
            None,
        );
        dispatcher.pop_layer();
        dispatcher.pop_layer();

        assert_eq!(
            layer_content_bbox(&dispatcher, 0),
            RectU16::new(4, 4, 36, 32)
        );
        assert_eq!(
            layer_content_bbox(&dispatcher, 2),
            RectU16::new(24, 24, 36, 32)
        );
    }
}
