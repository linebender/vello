// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::RenderMode;
use crate::dispatch::Dispatcher;
use crate::fine::FineKernel;
use crate::kurbo::{Affine, BezPath, PathEl, Rect, Stroke};
use crate::peniko::{BlendMode, Fill};
use crate::row::{CommandBucketer, LayerClip, RowRenderKernel};
use vello_common::clip::ClipContext;
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd};
use vello_common::filter_effects::Filter;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint};
use vello_common::strip_generator::{StripGenerator, StripStorage};

/// Single-threaded dispatcher for the row-bucket prototype.
#[derive(Debug)]
pub(crate) struct SingleThreadedDispatcher {
    bucketer: CommandBucketer,
    clip_context: ClipContext,
    strip_generator: StripGenerator,
    strip_storage: StripStorage,
    level: Level,
}

impl SingleThreadedDispatcher {
    pub(crate) fn new(width: u16, height: u16, level: Level) -> Self {
        Self {
            bucketer: CommandBucketer::new(width, height),
            clip_context: ClipContext::new(),
            strip_generator: StripGenerator::new(width, height, level),
            strip_storage: StripStorage::default(),
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

    fn rasterize_with<S: Simd, F: FineKernel<S> + RowRenderKernel<S>>(
        &self,
        simd: S,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        crate::row::rasterize::<S, F>(
            simd,
            &self.bucketer,
            &self.strip_storage.alphas,
            buffer,
            width,
            height,
            encoded_paints,
            image_resolver,
        );
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

impl Dispatcher for SingleThreadedDispatcher {
    fn has_unpopped_layers(&self) -> bool {
        self.bucketer.has_unpopped_layers()
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
        encoded_paints: &[EncodedPaint],
    ) {
        let clip_path = self.clip_context.get();
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
        self.bucketer.generate_fill(
            &self.strip_storage.strips,
            paint,
            blend_mode,
            mask,
            encoded_paints,
        );
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
        encoded_paints: &[EncodedPaint],
    ) {
        let clip_path = self.clip_context.get();
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
        self.bucketer.generate_fill(
            &self.strip_storage.strips,
            paint,
            blend_mode,
            mask,
            encoded_paints,
        );
    }

    fn fill_rect_fast(
        &mut self,
        rect: &Rect,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
        encoded_paints: &[EncodedPaint],
    ) {
        let clip_path = self.clip_context.get();
        let strip_generator = &mut self.strip_generator;
        let strip_storage = &mut self.strip_storage;
        strip_generator.generate_filled_rect_fast(rect, strip_storage, clip_path);
        self.bucketer.generate_fill(
            &self.strip_storage.strips,
            paint,
            blend_mode,
            mask,
            encoded_paints,
        );
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

            self.strip_generator.generate_filled_path(
                clip_path,
                fill_rule,
                clip_transform,
                aliasing_threshold,
                &mut self.strip_storage,
                existing_clip,
            );

            LayerClip {
                strips: self.strip_storage.strips.clone(),
                bbox,
            }
        });

        self.bucketer
            .push_layer(blend_mode, opacity, mask, clip_path);
    }

    fn pop_layer(&mut self) {
        self.bucketer.pop_layer();
    }

    fn reset(&mut self) {
        self.bucketer.reset();
        self.clip_context.reset();
        self.strip_generator.reset();
        self.strip_storage.clear();
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
        _buffer: &mut [u8],
        _width: u16,
        _height: u16,
        _dst_x: u16,
        _dst_y: u16,
        _dst_buffer_width: u16,
        _dst_buffer_height: u16,
        _render_mode: RenderMode,
        _encoded_paints: &[EncodedPaint],
        _image_resolver: &dyn ImageResolver,
    ) {
        unimplemented!("row-bucket prototype does not support compositing at an offset");
    }
}
