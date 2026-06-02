// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::coarse::{CommandBucketer, LayerClip};
use crate::dispatch::{Dispatcher, replay_render_commands};
use crate::filter::context::FilterContext;
use crate::fine::FineKernel;
use crate::kurbo::{Affine, BezPath, Rect, Stroke};
use crate::peniko::{BlendMode, Fill};
use crate::record::{CommandRecorder, PoppedLayer, expansion_left_top, expansion_padding};
use crate::{CompositeMode, RasterizerSettings};
use alloc::vec::Vec;
use core::cell::RefCell;
use vello_common::clip::{ClipContext, control_point_bbox_u16};
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd};
use vello_common::filter_effects::Filter;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint};
use vello_common::pixmap::{Pixmap, PixmapMut};
use vello_common::strip::Strip;
use vello_common::strip_generator::{GenerationMode, StripGenerator, StripStorage};
use vello_common::tile::Tile;

/// Single-threaded dispatcher for the row-bucket prototype.
#[derive(Debug)]
pub(crate) struct SingleThreadedDispatcher {
    bucketer: RefCell<CommandBucketer>,
    clip_context: ClipContext,
    recorder: CommandRecorder,
    strip_generator: StripGenerator,
    strip_storage: StripStorage,
    viewport_stack: Vec<(u16, u16)>,
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
            viewport_stack: Vec::new(),
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
            self.recorder.root_cmds(),
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
        let content_bbox = strip_bbox(
            &self.strip_storage.strips[strip_start..],
            self.strip_generator.width(),
            self.strip_generator.height(),
        );
        self.recorder.record_fill(
            strip_start..strip_end,
            paint,
            blend_mode,
            mask,
            0,
            content_bbox,
        );
    }

    fn render_filter_layers<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) -> FilterContext {
        let mut layer_manager = FilterContext::new(self.recorder.filter_layers().len());
        for id in (0..self.recorder.filter_layers().len()).rev() {
            let layer = &self.recorder.filter_layers()[id];
            if layer.bbox.is_empty() {
                continue;
            }

            let width = layer.bbox.width();
            let height = layer.bbox.height();
            let mut pixmap = Pixmap::new(width, height);
            let mut bucketer = self.bucketer.borrow_mut();
            bucketer.reset(width, height);
            replay_render_commands(
                &layer.cmds,
                &self.strip_storage.strips,
                &mut bucketer,
                encoded_paints,
                (layer.bbox.x0, layer.bbox.y0),
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
                &layer.filter,
                layer_manager.scratch(),
                layer.transform,
            );
            layer_manager.set_layer(id, pixmap);
        }

        layer_manager
    }

    fn push_filter_viewport(&mut self, expansion: Rect) {
        let (left, top, right, bottom) = expansion_padding(expansion);
        let width = self
            .strip_generator
            .width()
            .saturating_add(left)
            .saturating_add(right);
        let height = self
            .strip_generator
            .height()
            .saturating_add(top)
            .saturating_add(bottom);
        self.viewport_stack
            .push((self.strip_generator.width(), self.strip_generator.height()));
        self.strip_generator = StripGenerator::new(width, height, self.level);
    }

    fn pop_filter_viewport(&mut self) {
        let (width, height) = self
            .viewport_stack
            .pop()
            .expect("filter viewport stack underflow");
        self.strip_generator = StripGenerator::new(width, height, self.level);
    }
}

fn strip_bbox(strips: &[Strip], width: u16, height: u16) -> RectU16 {
    if strips.len() < 2 {
        return RectU16::INVERTED;
    }

    let mut bbox = RectU16::INVERTED;
    for pair in strips.windows(2) {
        let strip = pair[0];
        let next_strip = pair[1];
        if strip.is_sentinel() {
            continue;
        }

        let strip_y = strip.strip_y();
        let row_y = strip_y.saturating_mul(Tile::HEIGHT);
        let row_y1 = row_y.saturating_add(Tile::HEIGHT).min(height);
        let col = strip.alpha_idx() / u32::from(Tile::HEIGHT);
        let next_col = next_strip.alpha_idx() / u32::from(Tile::HEIGHT);
        let strip_width = next_col.saturating_sub(col) as u16;
        let strip_x1 = strip.x.saturating_add(strip_width).min(width);

        if strip_width > 0 && strip.x < width && row_y < height {
            bbox.union(RectU16::new(strip.x, row_y, strip_x1, row_y1));
        }

        if next_strip.fill_gap() && strip_y == next_strip.strip_y() && strip_x1 < next_strip.x {
            let fill_x1 = next_strip.x.min(width);
            if strip_x1 < fill_x1 && row_y < height {
                bbox.union(RectU16::new(strip_x1, row_y, fill_x1, row_y1));
            }
        }
    }
    bbox
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
        filter: Option<Filter>,
    ) {
        let filter_expansion = filter
            .as_ref()
            .map(|filter| filter.filter_expansion(&clip_transform));
        let filter_source_expansion = filter
            .as_ref()
            .map(|filter| filter.source_expansion(&clip_transform));
        if let Some(expansion) = filter_source_expansion {
            self.push_filter_viewport(expansion);
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

        if let Some(filter) = filter {
            let expansion = filter_expansion.expect("filter expansion missing");
            let source_expansion =
                filter_source_expansion.expect("filter source expansion missing");
            self.recorder.push_filter_layer(
                filter,
                clip_transform,
                expansion,
                expansion_left_top(source_expansion),
                blend_mode,
                opacity,
                mask,
                clip_path,
            );
        } else {
            self.recorder
                .push_layer(blend_mode, opacity, mask, clip_path);
        }
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
        self.viewport_stack.clear();
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
    use vello_common::paint::PremulColor;

    fn paint() -> Paint {
        Paint::Solid(PremulColor::from_alpha_color(BLUE))
    }

    fn layer_content_bbox(dispatcher: &SingleThreadedDispatcher, cmd_idx: usize) -> RectU16 {
        match &dispatcher.recorder.root_cmds()[cmd_idx] {
            RenderCmd::PushLayer { content_bbox, .. } => *content_bbox,
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
