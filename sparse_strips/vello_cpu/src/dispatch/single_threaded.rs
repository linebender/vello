// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::RenderMode;
use crate::dispatch::Dispatcher;
use crate::fine::{F32Kernel, Fine, FineKernel, U8Kernel};
use crate::kurbo::{Affine, BezPath, Stroke};
use crate::peniko::{BlendMode, Fill};
use crate::region::Regions;
use crate::strip_generator::StripGenerator;
use vello_common::coarse::Wide;
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Fallback, Level, Simd};
use vello_common::mask::Mask;
use vello_common::paint::Paint;

#[derive(Debug)]
pub(crate) struct SingleThreadedDispatcher {
    wide: Wide,
    strip_generator: StripGenerator,
    level: Level,
}

impl SingleThreadedDispatcher {
    pub(crate) fn new(width: u16, height: u16, level: Level) -> Self {
        let wide = Wide::new(width, height);
        let strip_generator = StripGenerator::new(width, height, level);

        Self {
            wide,
            strip_generator,
            level,
        }
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

    fn fill_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        paint: Paint,
        anti_alias: bool,
    ) {
        let wide = &mut self.wide;

        let func = |strips| wide.generate(strips, fill_rule, paint, 0);
        self.strip_generator
            .generate_filled_path(path, fill_rule, transform, anti_alias, func);
    }

    fn stroke_path(
        &mut self,
        path: &BezPath,
        stroke: &Stroke,
        transform: Affine,
        paint: Paint,
        anti_alias: bool,
    ) {
        let wide = &mut self.wide;

        let func = |strips| wide.generate(strips, Fill::NonZero, paint, 0);
        self.strip_generator
            .generate_stroked_path(path, stroke, transform, anti_alias, func);
    }

    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        fill_rule: Fill,
        clip_transform: Affine,
        blend_mode: BlendMode,
        opacity: f32,
        anti_alias: bool,
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
                anti_alias,
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
            RenderMode::OptimizeSpeed => match self.level {
                #[cfg(all(feature = "std", target_arch = "aarch64"))]
                Level::Neon(n) => {
                    self.rasterize_with::<vello_common::fearless_simd::Neon, U8Kernel>(
                        n,
                        buffer,
                        width,
                        height,
                        encoded_paints,
                    );
                }
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                Level::WasmSimd128(w) => {
                    self.rasterize_with::<vello_common::fearless_simd::WasmSimd128, U8Kernel>(
                        w,
                        buffer,
                        width,
                        height,
                        encoded_paints,
                    );
                }
                _ => self.rasterize_with::<Fallback, U8Kernel>(
                    Fallback::new(),
                    buffer,
                    width,
                    height,
                    encoded_paints,
                ),
            },
            RenderMode::OptimizeQuality => match self.level {
                #[cfg(all(feature = "std", target_arch = "aarch64"))]
                Level::Neon(n) => {
                    self.rasterize_with::<vello_common::fearless_simd::Neon, F32Kernel>(
                        n,
                        buffer,
                        width,
                        height,
                        encoded_paints,
                    );
                }
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                Level::WasmSimd128(w) => {
                    self.rasterize_with::<vello_common::fearless_simd::WasmSimd128, F32Kernel>(
                        w,
                        buffer,
                        width,
                        height,
                        encoded_paints,
                    );
                }
                _ => self.rasterize_with::<Fallback, F32Kernel>(
                    Fallback::new(),
                    buffer,
                    width,
                    height,
                    encoded_paints,
                ),
            },
        }
    }
}
