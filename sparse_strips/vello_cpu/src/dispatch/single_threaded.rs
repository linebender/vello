// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::RenderMode;
use crate::dispatch::{Dispatcher, RecordedCmd};
use crate::fine::FineKernel;
use crate::kurbo::{Affine, BezPath, PathEl, Rect, Stroke};
use crate::peniko::{BlendMode, Fill};
use crate::row::{CommandBucketer, LayerClip};
use crate::util::scalar::div_255;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use vello_common::clip::ClipContext;
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd};
use vello_common::filter_effects::Filter;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint};
use vello_common::strip::Strip;
use vello_common::strip_generator::{GenerationMode, StripGenerator, StripStorage};
use vello_common::tile::Tile;

#[derive(Debug)]
struct RecordedLayer {
    push_cmd_idx: usize,
    content_bbox: RectU16,
}

/// Single-threaded dispatcher for the row-bucket prototype.
#[derive(Debug)]
pub(crate) struct SingleThreadedDispatcher {
    bucketer: RefCell<CommandBucketer>,
    clip_context: ClipContext,
    cmds: Vec<RecordedCmd>,
    layer_stack: Vec<RecordedLayer>,
    strip_generator: StripGenerator,
    strip_storage: StripStorage,
    layer_depth: usize,
    level: Level,
}

impl SingleThreadedDispatcher {
    pub(crate) fn new(width: u16, height: u16, level: Level) -> Self {
        Self {
            bucketer: RefCell::new(CommandBucketer::new(width, height)),
            clip_context: ClipContext::new(),
            cmds: Vec::new(),
            layer_stack: Vec::new(),
            strip_generator: StripGenerator::new(width, height, level),
            strip_storage: StripStorage::new(GenerationMode::Append),
            layer_depth: 0,
            level,
        }
    }

    #[cfg(feature = "f32_pipeline")]
    fn rasterize_f32(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        use crate::fine::F32Kernel;
        use vello_common::fearless_simd::dispatch;
        dispatch!(self.level, simd => self.rasterize_with::<_, F32Kernel>(simd, buffer, width, height, encoded_paints, image_resolver));
    }

    #[cfg(feature = "u8_pipeline")]
    fn rasterize_u8(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        use crate::fine::U8Kernel;
        use vello_common::fearless_simd::dispatch;
        dispatch!(self.level, simd => self.rasterize_with::<_, U8Kernel>(simd, buffer, width, height, encoded_paints, image_resolver));
    }

    fn replay_commands(&self, bucketer: &mut CommandBucketer, encoded_paints: &[EncodedPaint]) {
        bucketer.reset();
        for cmd in &self.cmds {
            match cmd {
                RecordedCmd::Fill {
                    thread_idx,
                    strip_range,
                    paint,
                    blend_mode,
                    mask,
                } => bucketer.generate_fill(
                    &self.strip_storage.strips[strip_range.clone()],
                    paint.clone(),
                    *blend_mode,
                    mask.clone(),
                    *thread_idx,
                    encoded_paints,
                ),
                RecordedCmd::PushLayer {
                    blend_mode,
                    opacity,
                    mask,
                    clip,
                    ..
                } => bucketer.push_layer(*blend_mode, *opacity, mask.clone(), clip.clone()),
                RecordedCmd::PopLayer => bucketer.pop_layer(&self.strip_storage.strips),
            }
        }
    }

    fn rasterize_with<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        let mut bucketer = self.bucketer.borrow_mut();
        self.replay_commands(&mut bucketer, encoded_paints);

        crate::fine::rasterize::<S, F>(
            simd,
            &bucketer,
            &[self.strip_storage.alphas.as_slice()],
            buffer,
            width,
            height,
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
        self.cmds.push(RecordedCmd::Fill {
            thread_idx: 0,
            strip_range: strip_start..self.strip_storage.strips.len(),
            paint,
            blend_mode,
            mask,
        });
        let content_bbox = strip_bbox(&self.strip_storage.strips[strip_start..]);
        self.include_content_bbox(content_bbox);
    }

    fn include_content_bbox(&mut self, bbox: RectU16) {
        if bbox.is_empty() {
            return;
        }
        if let Some(layer) = self.layer_stack.last_mut() {
            layer.content_bbox.union(bbox);
        }
    }
}

fn composite_src_over_at_offset(
    src: &[u8],
    src_width: u16,
    src_height: u16,
    dst: &mut [u8],
    dst_x: u16,
    dst_y: u16,
    dst_width: u16,
    dst_height: u16,
) {
    if dst_x >= dst_width || dst_y >= dst_height {
        return;
    }

    let width = usize::from(src_width.min(dst_width - dst_x));
    let height = usize::from(src_height.min(dst_height - dst_y));
    let src_stride = usize::from(src_width) * 4;
    let dst_stride = usize::from(dst_width) * 4;
    let dst_x = usize::from(dst_x);
    let dst_y = usize::from(dst_y);

    for y in 0..height {
        let src_row = &src[y * src_stride..][..width * 4];
        let dst_start = (dst_y + y) * dst_stride + dst_x * 4;
        let dst_row = &mut dst[dst_start..][..width * 4];

        for (src, dst) in src_row.chunks_exact(4).zip(dst_row.chunks_exact_mut(4)) {
            let inv_alpha = u16::from(255 - src[3]);
            dst[0] = (u16::from(src[0]) + div_255(u16::from(dst[0]) * inv_alpha)).min(255) as u8;
            dst[1] = (u16::from(src[1]) + div_255(u16::from(dst[1]) * inv_alpha)).min(255) as u8;
            dst[2] = (u16::from(src[2]) + div_255(u16::from(dst[2]) * inv_alpha)).min(255) as u8;
            dst[3] = (u16::from(src[3]) + div_255(u16::from(dst[3]) * inv_alpha)).min(255) as u8;
        }
    }
}

fn control_point_bbox(path: &BezPath, transform: Affine) -> RectU16 {
    let mut bbox = Rect::new(
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for el in path.iter() {
        match el {
            PathEl::MoveTo(p) | PathEl::LineTo(p) => {
                bbox = bbox.union_pt(transform * p);
            }
            PathEl::QuadTo(p1, p2) => {
                bbox = bbox.union_pt(transform * p1);
                bbox = bbox.union_pt(transform * p2);
            }
            PathEl::CurveTo(p1, p2, p3) => {
                bbox = bbox.union_pt(transform * p1);
                bbox = bbox.union_pt(transform * p2);
                bbox = bbox.union_pt(transform * p3);
            }
            PathEl::ClosePath => {}
        }
    }

    RectU16::new(
        bbox.x0 as u16,
        bbox.y0 as u16,
        bbox.x1.ceil() as u16,
        bbox.y1.ceil() as u16,
    )
}

fn strip_bbox(strips: &[Strip]) -> RectU16 {
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
        let row_y1 = row_y.saturating_add(Tile::HEIGHT);
        let col = strip.alpha_idx() / u32::from(Tile::HEIGHT);
        let next_col = next_strip.alpha_idx() / u32::from(Tile::HEIGHT);
        let strip_width = next_col.saturating_sub(col) as u16;
        let strip_x1 = strip.x.saturating_add(strip_width);

        if strip_width > 0 {
            bbox.union(RectU16::new(strip.x, row_y, strip_x1, row_y1));
        }

        if next_strip.fill_gap() && strip_y == next_strip.strip_y() && strip_x1 < next_strip.x {
            bbox.union(RectU16::new(strip_x1, row_y, next_strip.x, row_y1));
        }
    }
    bbox
}

impl Dispatcher for SingleThreadedDispatcher {
    fn has_unpopped_layers(&self) -> bool {
        self.layer_depth != 0
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
        _encoded_paints: &[EncodedPaint],
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
        _encoded_paints: &[EncodedPaint],
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
        _encoded_paints: &[EncodedPaint],
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
        if filter.is_some() {
            unimplemented!("row-bucket prototype does not support filter layers");
        }
        let clip_path = clip_path.map(|clip_path| {
            let existing_clip = self.clip_context.get();
            let mut bbox = control_point_bbox(clip_path, clip_transform);
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

        let push_cmd_idx = self.cmds.len();
        self.cmds.push(RecordedCmd::PushLayer {
            blend_mode,
            opacity,
            mask,
            clip: clip_path,
            content_bbox: RectU16::INVERTED,
        });
        self.layer_stack.push(RecordedLayer {
            push_cmd_idx,
            content_bbox: RectU16::INVERTED,
        });
        self.layer_depth += 1;
    }

    fn pop_layer(&mut self) {
        let layer = self.layer_stack.pop().expect("layer stack underflow");
        let content_bbox = layer.content_bbox;
        match &mut self.cmds[layer.push_cmd_idx] {
            RecordedCmd::PushLayer {
                content_bbox: bbox, ..
            } => *bbox = content_bbox,
            _ => unreachable!("layer stack referenced a non-layer command"),
        }
        self.include_content_bbox(content_bbox);
        self.cmds.push(RecordedCmd::PopLayer);
        self.layer_depth = self
            .layer_depth
            .checked_sub(1)
            .expect("layer stack underflow");
    }

    fn reset(&mut self) {
        self.bucketer.borrow_mut().reset();
        self.clip_context.reset();
        self.cmds.clear();
        self.layer_stack.clear();
        self.strip_generator.reset();
        self.strip_storage.clear();
        self.layer_depth = 0;
    }

    fn flush(&mut self, _encoded_paints: &[EncodedPaint]) {}

    fn rasterize(
        &self,
        buffer: &mut [u8],
        render_mode: RenderMode,
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        #[cfg(all(feature = "u8_pipeline", not(feature = "f32_pipeline")))]
        {
            let _ = render_mode;
            self.rasterize_u8(buffer, width, height, encoded_paints, image_resolver);
        }

        #[cfg(all(feature = "f32_pipeline", not(feature = "u8_pipeline")))]
        {
            let _ = render_mode;
            self.rasterize_f32(buffer, width, height, encoded_paints, image_resolver);
        }

        #[cfg(all(feature = "u8_pipeline", feature = "f32_pipeline"))]
        match render_mode {
            RenderMode::OptimizeSpeed => {
                self.rasterize_u8(buffer, width, height, encoded_paints, image_resolver);
            }
            RenderMode::OptimizeQuality => {
                self.rasterize_f32(buffer, width, height, encoded_paints, image_resolver);
            }
        }

        #[cfg(all(not(feature = "u8_pipeline"), not(feature = "f32_pipeline")))]
        {
            let _ = (
                buffer,
                render_mode,
                width,
                height,
                encoded_paints,
                image_resolver,
            );
        }
    }

    fn composite_at_offset(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        dst_x: u16,
        dst_y: u16,
        dst_buffer_width: u16,
        dst_buffer_height: u16,
        render_mode: RenderMode,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        let mut src = vec![0; usize::from(width) * usize::from(height) * 4];
        self.rasterize(
            &mut src,
            render_mode,
            width,
            height,
            encoded_paints,
            image_resolver,
        );
        composite_src_over_at_offset(
            &src,
            width,
            height,
            buffer,
            dst_x,
            dst_y,
            dst_buffer_width,
            dst_buffer_height,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kurbo::Shape;
    use vello_common::color::palette::css::BLUE;
    use vello_common::paint::PremulColor;

    fn paint() -> Paint {
        Paint::Solid(PremulColor::from_alpha_color(BLUE))
    }

    fn layer_content_bbox(dispatcher: &SingleThreadedDispatcher, cmd_idx: usize) -> RectU16 {
        match &dispatcher.cmds[cmd_idx] {
            RecordedCmd::PushLayer { content_bbox, .. } => *content_bbox,
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
            &[],
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
            &[],
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
            &[],
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
