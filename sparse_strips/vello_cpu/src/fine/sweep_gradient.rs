use crate::fine::{COLOR_COMPONENTS, Positive, Sign, TILE_HEIGHT_COMPONENTS, extend};
use crate::paint::{EncodedSweepGradient, GradientRange};
use std::f32::consts::PI;
use vello_common::kurbo::{Affine, Point};

#[derive(Debug)]
pub(crate) struct SweepGradientFiller<'a> {
    /// The position of the next x that should be processed.
    cur_pos: (f32, f32),
    /// The underlying gradient.
    gradient: &'a EncodedSweepGradient,
    range_idx: usize,
    cur_range: &'a GradientRange,
}

impl<'a> SweepGradientFiller<'a> {
    pub(crate) fn new(gradient: &'a EncodedSweepGradient, start_x: u16, start_y: u16) -> Self {
        let mut start_point = Point::new(start_x as f64 + 0.5, start_y as f64 + 0.5);

        let filler = Self {
            cur_pos: (start_point.x as f32, start_point.y as f32),
            gradient,
            range_idx: 0,
            cur_range: &gradient.ranges[0],
        };

        filler
    }

    fn cur_angle(&self, pos: (f32, f32)) -> f32 {
        (-pos.1).atan2(pos.0).rem_euclid(2.0 * PI)
    }

    fn advance(&mut self, target_angle: f32) {
        while target_angle > self.cur_range.x1 || target_angle < self.cur_range.x0 {
            Positive::idx_advance(&mut self.range_idx, self.gradient.ranges.len());
            self.cur_range = &self.gradient.ranges[self.range_idx];
        }
    }

    pub(super) fn run(mut self, target: &mut [u8]) {
        let extend = |val| {
            extend(
                val,
                self.gradient.pad,
                self.gradient.start_angle,
                self.gradient.end_angle,
            )
        };

        target
            .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
            .for_each(|column| {
                let mut pos = self.cur_pos;

                for pixel in column.chunks_exact_mut(COLOR_COMPONENTS) {
                    let actual_pos = self.gradient.trans * Point::new(pos.0 as f64, pos.1 as f64);
                    let points = (actual_pos.x as f32, actual_pos.y as f32);

                    let angle = extend(self.cur_angle(points));
                    self.advance(angle);
                    let range = self.cur_range;

                    for col_idx in 0..COLOR_COMPONENTS {
                        let im3 = angle - range.x0;
                        let combined = (range.im3[col_idx] * im3 + 0.5) as i16;

                        pixel[col_idx] = (range.c0[col_idx] as i16 + combined) as u8;
                    }

                    pos.1 += 1.0;
                }

                self.cur_pos.0 += 1.0;
            })
    }
}
