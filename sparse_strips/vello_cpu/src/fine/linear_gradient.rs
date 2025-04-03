//! Rendering linear gradients.

use crate::fine::{COLOR_COMPONENTS, Positive, Sign, TILE_HEIGHT_COMPONENTS, extend};
use crate::paint::{EncodedLinearGradient, GradientRange};
use vello_common::kurbo::Point;

#[derive(Debug)]
pub(crate) struct LinearGradientFiller<'a> {
    /// The position of the next x that should be processed.
    cur_pos: (f32, f32),
    /// The index of the current right stop we are processing.
    range_idx: usize,
    /// The underlying gradient.
    gradient: &'a EncodedLinearGradient,
    cur_range: &'a GradientRange,
}

impl<'a> LinearGradientFiller<'a> {
    pub(crate) fn new(gradient: &'a EncodedLinearGradient, start_x: u16, start_y: u16) -> Self {
        let filler = Self {
            cur_pos: (start_x as f32, start_y as f32),
            range_idx: 0,
            cur_range: &gradient.ranges[0],
            gradient,
        };

        filler
    }

    fn advance(&mut self, target_pos: f32) {
        while target_pos > self.cur_range.x1 || target_pos < self.cur_range.x0 {
            Positive::idx_advance(&mut self.range_idx, self.gradient.ranges.len());
            self.cur_range = &self.gradient.ranges[self.range_idx];
        }
    }

    fn cur_pos(&self, pos: (f32, f32)) -> f32 {
        (pos.0 * self.gradient.fact1 - pos.1 * self.gradient.fact2) / self.gradient.denom
    }

    pub(super) fn run(mut self, target: &mut [u8]) {
        let pad = self.gradient.pad;
        let end = self.gradient.end;
        let transform = self.gradient.transform;

        let extend = |val| extend(val, pad, 0.0, end);

        target
            .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
            .for_each(|column| {
                let mut pos = self.cur_pos;

                for pixel in column.chunks_exact_mut(COLOR_COMPONENTS) {
                    let actual_pos = transform * Point::new(pos.0 as f64, pos.1 as f64);
                    let points = (actual_pos.x as f32, actual_pos.y as f32);

                    let dist = extend(self.cur_pos(points));
                    self.advance(dist);
                    let range = self.cur_range;

                    for col_idx in 0..COLOR_COMPONENTS {
                        let im3 = dist - range.x0;
                        let combined = (range.im3[col_idx] * im3 + 0.5) as i16;

                        pixel[col_idx] = (range.c0[col_idx] as i16 + combined) as u8;
                    }

                    pos.1 += 1.0;
                }

                self.cur_pos.0 += 1.0;
            })
    }
}
