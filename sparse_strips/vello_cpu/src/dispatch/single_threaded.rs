// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::RenderMode;
use crate::dispatch::Dispatcher;
use crate::fine::{Fine, FineType};
use crate::kurbo::{Affine, BezPath, Stroke};
use crate::peniko::{BlendMode, Fill};
use crate::region::Regions;
use crate::strip_generator::StripGenerator;
use vello_common::coarse::Wide;
use vello_common::encode::EncodedPaint;
use vello_common::mask::Mask;
use vello_common::paint::Paint;

#[derive(Debug)]
pub(crate) struct SingleThreadedDispatcher {
    wide: Wide,
    strip_generator: StripGenerator,
}

impl SingleThreadedDispatcher {
    pub(crate) fn new(width: u16, height: u16) -> Self {
        let wide = Wide::new(width, height);
        let strip_generator = StripGenerator::new(width, height);

        Self {
            wide,
            strip_generator,
        }
    }

    fn rasterize<F: FineType>(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        let mut buffer = Regions::new(width, height, buffer);
        let mut fine = Fine::new();

        buffer.update_regions(|region| {
            let x = region.x;
            let y = region.y;

            let wtile = self.wide.get(x, y);
            fine.set_coords(x, y);

            fine.clear(F::extract_color(&wtile.bg));
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

    fn fill_path(&mut self, path: &BezPath, fill_rule: Fill, transform: Affine, paint: Paint) {
        let wide = &mut self.wide;

        let func = |strips| wide.generate(strips, fill_rule, paint, 0);
        self.strip_generator
            .generate_filled_path(path, fill_rule, transform, func);
    }

    fn stroke_path(&mut self, path: &BezPath, stroke: &Stroke, transform: Affine, paint: Paint) {
        let wide = &mut self.wide;

        let func = |strips| wide.generate(strips, Fill::NonZero, paint, 0);
        self.strip_generator
            .generate_stroked_path(path, stroke, transform, func);
    }

    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        fill_rule: Fill,
        clip_transform: Affine,
        blend_mode: BlendMode,
        opacity: f32,
        mask: Option<Mask>,
    ) {
        let clip = if let Some(c) = clip_path {
            // This variable will always be assigned to in the closure, but the compiler can't recognize that.
            // So just assign a dummy value here.
            let mut strip_buf = &[][..];

            self.strip_generator
                .generate_filled_path(c, fill_rule, clip_transform, |strips| strip_buf = strips);

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
            RenderMode::OptimizeSpeed => {
                Self::rasterize::<u8>(self, buffer, width, height, encoded_paints);
            }
            RenderMode::OptimizeQuality => {
                Self::rasterize::<f32>(self, buffer, width, height, encoded_paints);
            }
        }
    }
}
