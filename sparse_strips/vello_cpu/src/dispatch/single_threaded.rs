// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::RenderMode;
use crate::dispatch::Dispatcher;
use crate::fine::{F32Kernel, Fine, FineKernel, U8Kernel};
use crate::kurbo::{Affine, BezPath, Stroke};
use crate::peniko::{BlendMode, Fill};
use crate::region::Regions;
use vello_common::clip::ClipContext;
use vello_common::coarse::{MODE_CPU, Wide};
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd, dispatch};
use vello_common::mask::Mask;
use vello_common::paint::Paint;
use vello_common::strip::Strip;
use vello_common::strip_generator::{StripGenerator, StripStorage};

#[derive(Debug)]
pub(crate) struct SingleThreadedDispatcher {
    wide: Wide,
    clip_context: ClipContext,
    strip_generator: StripGenerator,
    strip_storage: StripStorage,
    level: Level,
}

impl SingleThreadedDispatcher {
    pub(crate) fn new(width: u16, height: u16, level: Level) -> Self {
        let wide = Wide::<MODE_CPU>::new(width, height);
        let strip_generator = StripGenerator::new(width, height, level);
        let clip_context = ClipContext::new();
        let strip_storage = StripStorage::default();

        Self {
            wide,
            clip_context,
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
        dispatch!(self.level, simd => self.rasterize_with::<_, F32Kernel>(simd, buffer, width, height, encoded_paints));
    }

    fn rasterize_u8(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        dispatch!(self.level, simd => self.rasterize_with::<_, U8Kernel>(simd, buffer, width, height, encoded_paints));
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
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    ) {
        let wide = &mut self.wide;

        self.strip_generator.generate_filled_path(
            path,
            fill_rule,
            transform,
            aliasing_threshold,
            &mut self.strip_storage,
            self.clip_context.get(),
        );

        wide.generate(&self.strip_storage.strips, paint, blend_mode, 0, mask);
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
        let wide = &mut self.wide;

        self.strip_generator.generate_stroked_path(
            path,
            stroke,
            transform,
            aliasing_threshold,
            &mut self.strip_storage,
            self.clip_context.get(),
        );

        wide.generate(&self.strip_storage.strips, paint, blend_mode, 0, mask);
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
                self.clip_context.get(),
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
        self.clip_context.reset();
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

    fn generate_wide_cmd(&mut self, strip_buf: &[Strip], paint: Paint, blend_mode: BlendMode) {
        // Masks are not supported in recordings, so just pass `None` for now.
        self.wide.generate(strip_buf, paint, blend_mode, 0, None);
    }

    fn strip_storage_mut(&mut self) -> &mut StripStorage {
        &mut self.strip_storage
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
            BlendMode::default(),
            None,
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
