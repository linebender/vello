// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::RenderMode;
use crate::dispatch::Dispatcher;
use crate::fine::{F32Kernel, Fine, FineKernel, U8Kernel};
use crate::kurbo::{Affine, BezPath, Stroke};
use crate::peniko::{BlendMode, Fill};
use crate::region::Regions;
use alloc::vec::Vec;
use vello_common::coarse::{MODE_CPU, Wide};
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd, simd_dispatch};
use vello_common::mask::Mask;
use vello_common::paint::Paint;
use vello_common::strip_generator::StripGenerator;

#[derive(Debug)]
pub(crate) struct SingleThreadedDispatcher {
    wide: Wide,
    strip_generator: StripGenerator,
    level: Level,
}

impl SingleThreadedDispatcher {
    pub(crate) fn new(width: u16, height: u16, level: Level) -> Self {
        let wide = Wide::<MODE_CPU>::new(width, height);
        let strip_generator = StripGenerator::new(width, height, level);

        Self {
            wide,
            strip_generator,
            level,
        }
    }

    fn rasterize_f32(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        rasterize_with_f32_dispatch(self.level, self, buffer, width, height, encoded_paints);
    }

    fn rasterize_u8(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        rasterize_with_u8_dispatch(self.level, self, buffer, width, height, encoded_paints);
    }

    fn rasterize_with<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        let mut buffer = Regions::new(width, height, buffer);
        let mut fine = Fine::<S, F>::new(simd);

        buffer.update_regions(|region| {
            let x = region.x;
            let y = region.y;

            let wtile = self.wide.get(x, y);
            fine.set_coords(x, y);

            fine.clear(wtile.bg);
            for cmd in &wtile.cmds {
                fine.run_cmd(cmd, self.strip_generator.alpha_buf(), encoded_paints);
            }

            fine.pack(region);
        });
    }
}

impl Dispatcher for SingleThreadedDispatcher {
    fn wide(&self) -> &Wide {
        &self.wide
    }

    fn wide_mut(&mut self) -> &mut Wide {
        &mut self.wide
    }

    fn fill_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        let wide = &mut self.wide;

        let func = |strips| wide.generate(strips, fill_rule, paint, 0);
        self.strip_generator.generate_filled_path(
            path,
            fill_rule,
            transform,
            aliasing_threshold,
            func,
        );
    }

    fn stroke_path(
        &mut self,
        path: &BezPath,
        stroke: &Stroke,
        transform: Affine,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        let wide = &mut self.wide;

        let func = |strips| wide.generate(strips, Fill::NonZero, paint, 0);
        self.strip_generator.generate_stroked_path(
            path,
            stroke,
            transform,
            aliasing_threshold,
            func,
        );
    }

    fn alpha_buf(&self) -> &[u8] {
        self.strip_generator.alpha_buf()
    }

    fn extend_alpha_buf(&mut self, alphas: &[u8]) {
        self.strip_generator.extend_alpha_buf(alphas);
    }

    fn replace_alpha_buf(&mut self, alphas: Vec<u8>) -> Vec<u8> {
        self.strip_generator.replace_alpha_buf(alphas)
    }

    fn set_alpha_buf(&mut self, alphas: Vec<u8>) {
        self.strip_generator.set_alpha_buf(alphas);
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
    ) {
        let clip = if let Some(c) = clip_path {
            // This variable will always be assigned to in the closure, but the compiler can't recognize that.
            // So just assign a dummy value here.
            let mut strip_buf = &[][..];

            self.strip_generator.generate_filled_path(
                c,
                fill_rule,
                clip_transform,
                aliasing_threshold,
                |strips| strip_buf = strips,
            );

            Some((strip_buf, fill_rule))
        } else {
            None
        };

        self.wide.push_layer(clip, blend_mode, mask, opacity, 0);
    }

    fn pop_layer(&mut self) {
        self.wide.pop_layer();
    }

    fn reset(&mut self) {
        self.wide.reset();
        self.strip_generator.reset();
    }

    fn flush(&mut self) {}

    fn rasterize(
        &self,
        buffer: &mut [u8],
        render_mode: RenderMode,
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        match render_mode {
            RenderMode::OptimizeSpeed => self.rasterize_u8(buffer, width, height, encoded_paints),
            RenderMode::OptimizeQuality => {
                self.rasterize_f32(buffer, width, height, encoded_paints);
            }
        }
    }
}

simd_dispatch!(
    pub fn rasterize_with_f32_dispatch(
        level,
        self_: &SingleThreadedDispatcher,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint]
    ) = rasterize_with_f32
);

simd_dispatch!(
    pub fn rasterize_with_u8_dispatch(
        level,
        self_: &SingleThreadedDispatcher,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint]
    ) = rasterize_with_u8
);

fn rasterize_with_f32<S: Simd>(
    simd: S,
    self_: &SingleThreadedDispatcher,
    buffer: &mut [u8],
    width: u16,
    height: u16,
    encoded_paints: &[EncodedPaint],
) {
    self_.rasterize_with::<S, F32Kernel>(simd, buffer, width, height, encoded_paints);
}

fn rasterize_with_u8<S: Simd>(
    simd: S,
    self_: &SingleThreadedDispatcher,
    buffer: &mut [u8],
    width: u16,
    height: u16,
    encoded_paints: &[EncodedPaint],
) {
    self_.rasterize_with::<S, U8Kernel>(simd, buffer, width, height, encoded_paints);
}
