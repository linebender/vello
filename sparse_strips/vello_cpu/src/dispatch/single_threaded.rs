// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::RenderMode;
use crate::dispatch::Dispatcher;
use crate::fine::FineKernel;
use crate::kurbo::{Affine, BezPath, Rect, Stroke};
use crate::peniko::{BlendMode, Fill};
use crate::row::{CommandBucketer, RowRenderKernel};
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd};
use vello_common::filter_effects::Filter;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint};
use vello_common::strip_generator::{StripGenerator, StripStorage};

/// Single-threaded dispatcher for the row-bucket prototype.
#[derive(Debug)]
pub(crate) struct SingleThreadedDispatcher {
    bucketer: CommandBucketer,
    strip_generator: StripGenerator,
    strip_storage: StripStorage,
    level: Level,
}

impl SingleThreadedDispatcher {
    pub(crate) fn new(width: u16, height: u16, level: Level) -> Self {
        Self {
            bucketer: CommandBucketer::new(width, height),
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
        _encoded_paints: &[EncodedPaint],
        _image_resolver: &dyn ImageResolver,
    ) {
        use crate::fine::F32Kernel;
        use vello_common::fearless_simd::dispatch;
        dispatch!(self.level, simd => self.rasterize_with::<_, F32Kernel>(simd, buffer, width, height));
    }

    #[cfg(feature = "u8_pipeline")]
    fn rasterize_u8(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        _encoded_paints: &[EncodedPaint],
        _image_resolver: &dyn ImageResolver,
    ) {
        use crate::fine::U8Kernel;
        use vello_common::fearless_simd::dispatch;
        dispatch!(self.level, simd => self.rasterize_with::<_, U8Kernel>(simd, buffer, width, height));
    }

    fn rasterize_with<S: Simd, F: FineKernel<S> + RowRenderKernel<S>>(
        &self,
        simd: S,
        buffer: &mut [u8],
        width: u16,
        height: u16,
    ) {
        crate::row::rasterize::<S, F>(
            simd,
            &self.bucketer,
            &self.strip_storage.alphas,
            buffer,
            width,
            height,
        );
    }

    fn assert_supported(&self, paint: &Paint, blend_mode: BlendMode, mask: &Option<Mask>) {
        if !matches!(paint, Paint::Solid(_)) {
            unimplemented!("row-bucket prototype only supports solid paints");
        }
        assert_eq!(
            blend_mode,
            BlendMode::default(),
            "row-bucket prototype only supports default source-over blending"
        );
        assert!(
            mask.is_none(),
            "row-bucket prototype does not support masks"
        );
    }
}

impl Dispatcher for SingleThreadedDispatcher {
    fn has_unpopped_layers(&self) -> bool {
        false
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
        self.assert_supported(&paint, blend_mode, &mask);
        self.strip_generator.generate_filled_path(
            path,
            fill_rule,
            transform,
            aliasing_threshold,
            &mut self.strip_storage,
            None,
        );
        self.bucketer.generate(&self.strip_storage.strips, paint);
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
        self.assert_supported(&paint, blend_mode, &mask);
        self.strip_generator.generate_stroked_path(
            path,
            stroke,
            transform,
            aliasing_threshold,
            &mut self.strip_storage,
            None,
        );
        self.bucketer.generate(&self.strip_storage.strips, paint);
    }

    fn fill_rect_fast(
        &mut self,
        rect: &Rect,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
        _encoded_paints: &[EncodedPaint],
    ) {
        self.assert_supported(&paint, blend_mode, &mask);
        self.strip_generator
            .generate_filled_rect_fast(rect, &mut self.strip_storage, None);
        self.bucketer.generate(&self.strip_storage.strips, paint);
    }

    fn push_clip_path(
        &mut self,
        _path: &BezPath,
        _fill_rule: Fill,
        _transform: Affine,
        _aliasing_threshold: Option<u8>,
    ) {
        unimplemented!("row-bucket prototype does not support clip paths");
    }

    fn pop_clip_path(&mut self) {
        unimplemented!("row-bucket prototype does not support clip paths");
    }

    fn push_layer(
        &mut self,
        _clip_path: Option<&BezPath>,
        _fill_rule: Fill,
        _clip_transform: Affine,
        _blend_mode: BlendMode,
        _opacity: f32,
        _aliasing_threshold: Option<u8>,
        _mask: Option<Mask>,
        _filter: Option<Filter>,
    ) {
        unimplemented!("row-bucket prototype does not support layers");
    }

    fn pop_layer(&mut self) {
        unimplemented!("row-bucket prototype does not support layers");
    }

    fn reset(&mut self) {
        self.bucketer.reset();
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
