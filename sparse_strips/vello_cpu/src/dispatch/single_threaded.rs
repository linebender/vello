// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::RenderMode;
use crate::dispatch::Dispatcher;
use crate::fine::{F32Kernel, Fine, FineKernel, U8Kernel};
use crate::kurbo::{Affine, BezPath, Stroke};
use crate::peniko::{BlendMode, Fill};
use crate::region::Regions;
use vello_common::coarse::{MODE_CPU, Wide};
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd, simd_dispatch};
use vello_common::mask::Mask;
use vello_common::paint::Paint;
use vello_common::strip::Strip;
use vello_common::strip_generator::{StripGenerator, StripStorage};

#[derive(Debug)]
pub(crate) struct SingleThreadedDispatcher {
    wide: Wide,
    strip_generator: StripGenerator,
    strip_storage: StripStorage,
    level: Level,
}

impl SingleThreadedDispatcher {
    pub(crate) fn new(width: u16, height: u16, level: Level) -> Self {
        let wide = Wide::<MODE_CPU>::new(width, height);
        let strip_generator = StripGenerator::new(width, height, level);
        let strip_storage = StripStorage::default();

        Self {
            wide,
            strip_generator,
            strip_storage,
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
                fine.run_cmd(cmd, &self.strip_storage.alphas, encoded_paints);
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
        aliasing_threshold: Option<u8>,
    ) {
        let wide = &mut self.wide;

        self.strip_generator.generate_filled_path(
            path,
            fill_rule,
            transform,
            aliasing_threshold,
            &mut self.strip_storage,
        );

        wide.generate(&self.strip_storage.strips, paint, 0);
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

        self.strip_generator.generate_stroked_path(
            path,
            stroke,
            transform,
            aliasing_threshold,
            &mut self.strip_storage,
        );

        wide.generate(&self.strip_storage.strips, paint, 0);
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
            self.strip_generator.generate_filled_path(
                c,
                fill_rule,
                clip_transform,
                aliasing_threshold,
                &mut self.strip_storage,
            );

            Some(self.strip_storage.strips.as_slice())
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
        self.strip_storage.clear();
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

    fn generate_wide_cmd(&mut self, strip_buf: &[Strip], paint: Paint) {
        self.wide.generate(strip_buf, paint, 0);
    }

    fn strip_storage_mut(&mut self) -> &mut StripStorage {
        &mut self.strip_storage
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kurbo::Rect;
    use vello_common::color::palette::css::BLUE;
    use vello_common::kurbo::Shape;
    use vello_common::paint::PremulColor;

    #[test]
    fn buffers_cleared_on_reset() {
        let mut dispatcher = SingleThreadedDispatcher::new(100, 100, Level::new());

        dispatcher.fill_path(
            &Rect::new(0.0, 0.0, 50.0, 50.0).to_path(0.1),
            Fill::NonZero,
            Affine::IDENTITY,
            Paint::Solid(PremulColor::from_alpha_color(BLUE)),
            None,
        );

        // Ensure there is data to clear.
        assert!(!dispatcher.strip_storage.alphas.is_empty());
        assert!(!dispatcher.wide.get(0, 0).cmds.is_empty());

        dispatcher.reset();

        // Verify buffers are cleared.
        assert!(dispatcher.strip_storage.alphas.is_empty());
        assert!(dispatcher.wide.get(0, 0).cmds.is_empty());
    }
}
