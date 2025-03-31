//! Rendering linear gradients.

use crate::fine;
use crate::fine::{
    COLOR_COMPONENTS, Extend, Negative, Pad, Positive, Repeat, Sign, TILE_HEIGHT_COMPONENTS,
};
use crate::paint::{EncodedLinearGradient, GradientRange};
use vello_common::tile::Tile;

#[derive(Debug)]
pub(crate) struct LinearGradientFiller<'a> {
    /// The position of the next x that should be processed.
    cur_pos: f32,
    x_advance: f32,
    y_advance: f32,
    /// The index of the current right stop we are processing.
    range_idx: usize,
    temp_range_idx: usize,
    /// The underlying gradient.
    gradient: &'a EncodedLinearGradient,
    cur_range: &'a GradientRange,
    temp_range: &'a GradientRange,
}

impl<'a> LinearGradientFiller<'a> {
    pub(crate) fn new(gradient: &'a EncodedLinearGradient, start_x: u16, start_y: u16) -> Self {
        // The actual starting point of the strip.
        let x0 = start_x as f32 + gradient.offsets.0 + 0.5;
        let y0 = start_y as f32 + gradient.offsets.1 + 0.5;

        let cur_pos = (x0 * gradient.fact1 - y0 * gradient.fact2) / gradient.denom;

        let filler = Self {
            cur_pos,
            x_advance: gradient.advances.0,
            y_advance: gradient.advances.1,
            range_idx: 0,
            temp_range_idx: 0,
            cur_range: &gradient.ranges[0],
            temp_range: &gradient.ranges[0],
            gradient,
        };

        filler
    }

    fn advance<SI: Sign>(&mut self) {
        while self.cur_pos > self.cur_range.x1 || self.cur_pos < self.cur_range.x0 {
            SI::idx_advance(&mut self.range_idx, self.gradient.ranges.len());
            self.cur_range = &self.gradient.ranges[self.range_idx];
        }
    }

    fn advance_temp<SI: Sign>(&mut self, target_pos: f32) {
        while target_pos > self.temp_range.x1 || target_pos < self.temp_range.x0 {
            SI::idx_advance(&mut self.temp_range_idx, self.gradient.ranges.len());
            self.temp_range = &self.gradient.ranges[self.temp_range_idx];
        }
    }

    pub(super) fn run(self, target: &mut [u8]) {
        if self.gradient.pad {
            self.run_1::<Pad>(target);
        } else {
            self.run_1::<Repeat>(target);
        }
    }

    fn run_1<EX: Extend>(self, target: &mut [u8]) {
        if self.gradient.positive {
            self.run_2::<EX, Positive>(target);
        } else {
            self.run_2::<EX, Negative>(target);
        }
    }

    fn run_2<EX: Extend, XS: Sign>(self, target: &mut [u8]) {
        if self.gradient.positive {
            self.run_inner::<EX, XS, Positive>(target);
        } else {
            self.run_inner::<EX, XS, Negative>(target);
        }
    }

    fn run_inner<EX: Extend, XS: Sign, YS: Sign>(mut self, target: &mut [u8]) {
        let mut col_positions = [0.0; Tile::HEIGHT as usize];
        self.cur_pos = EX::extend(self.cur_pos, self.gradient.end);

        // Get to the initial position.
        self.advance::<XS>();

        target
            .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
            .for_each(|col| {
                let mut needs_advance = false;
                let range = self.cur_range;
                for i in 0..COLOR_COMPONENTS {
                    let base_pos = self.cur_pos + i as f32 * self.y_advance;
                    needs_advance |= YS::needs_advance(base_pos, range.x0, range.x1);
                    col_positions[i] = base_pos;
                }

                if needs_advance {
                    for i in 0..COLOR_COMPONENTS {
                        col_positions[i] = EX::extend(col_positions[i], self.gradient.end);
                    }

                    self.run_col::<Advancer, YS>(col, &col_positions);
                } else {
                    self.run_col::<NoAdvancer, YS>(col, &col_positions);
                }

                self.cur_pos = EX::extend(self.cur_pos + self.x_advance, self.gradient.end);
                self.advance::<XS>()
            })
    }

    #[inline(always)]
    fn run_col<AD: Advance, SI: Sign>(
        &mut self,
        column: &mut [u8],
        positions: &[f32; Tile::HEIGHT as usize],
    ) {
        for (pixel, target_pos) in column.chunks_exact_mut(COLOR_COMPONENTS).zip(positions) {
            let range = AD::get_range::<SI>(self, *target_pos);

            for col_idx in 0..COLOR_COMPONENTS {
                let im3 = target_pos - range.x0;
                let combined = (range.im3[col_idx] * im3 + 0.5) as i16;

                pixel[col_idx] = (range.c0[col_idx] as i16 + combined) as u8;
            }
        }
    }
}

trait Advance {
    fn get_range<'a, S: Sign>(
        gf: &mut LinearGradientFiller<'a>,
        target_pos: f32,
    ) -> &'a GradientRange;
}

struct Advancer;

impl Advance for Advancer {
    #[inline]
    fn get_range<'a, S: Sign>(
        gf: &mut LinearGradientFiller<'a>,
        target_pos: f32,
    ) -> &'a GradientRange {
        // It's possible that we have to skip multiple stops.
        gf.advance_temp::<S>(target_pos);

        gf.temp_range
    }
}

struct NoAdvancer;

impl Advance for NoAdvancer {
    fn get_range<'a, S: Sign>(gf: &mut LinearGradientFiller<'a>, _: f32) -> &'a GradientRange {
        gf.cur_range
    }
}
