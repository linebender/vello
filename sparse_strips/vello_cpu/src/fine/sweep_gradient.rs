use crate::fine::{COLOR_COMPONENTS, Sign, TILE_HEIGHT_COMPONENTS};
use crate::paint::{EncodedSweepGradient, GradientRange};
use std::f32::consts::PI;
use vello_common::kurbo::{Affine, Point};

#[derive(Debug)]
pub(crate) struct SweepGradientFiller<'a> {
    /// The position of the next x that should be processed.
    cur_pos: (f32, f32),
    x0: f32,
    /// The color of the left stop.
    c0: [u8; 4],
    im3: [f32; 4],
    /// The underlying gradient.
    gradient: &'a EncodedSweepGradient,
    range_idx: usize,
    temp_range_idx: usize,
    cur_range: &'a GradientRange,
    temp_range: &'a GradientRange,
}

impl<'a> SweepGradientFiller<'a> {
    pub(crate) fn new(gradient: &'a EncodedSweepGradient, start_x: u16, start_y: u16) -> Self {
        unimplemented!()
        // let mut start_point = Point::new(
        //     start_x as f64 + gradient.offsets.0 as f64,
        //     start_y as f64 + gradient.offsets.1 as f64,
        // );
        // start_point =
        //     Affine::rotate_about(gradient.rotation as f64, Point::new(0.0, 0.0)) * start_point;
        //
        // let left_stop = &gradient.ranges[0];
        // let right_stop = &gradient.ranges[1];
        //
        // let c0 = left_stop.color;
        // let c1 = right_stop.color;
        // let x0 = 0.0;
        // let x1 = gradient.end_angle;
        //
        // let mut im1 = [0.0; 4];
        // let im2 = x1 - x0;
        // let mut im3 = [0.0; 4];
        //
        // for i in 0..COLOR_COMPONENTS {
        //     im1[i] = c1[i] as f32 - c0[i] as f32;
        //     im3[i] = im1[i] / im2;
        // }
        //
        // let filler = Self {
        //     cur_pos: (start_point.x as f32, start_point.y as f32 + 0.5),
        //     c0,
        //     x0,
        //     im3,
        //     gradient,
        //     range_idx: 0,
        //     temp_range_idx: 0,
        //     cur_range: &gradient.ranges[0],
        //     temp_range: &gradient.ranges[0],
        // };
        //
        // filler
    }
    //
    // fn advance<SI: Sign>(&mut self) {
    //     while self.cur_pos > self.cur_range.x1 || self.cur_pos < self.cur_range.x0 {
    //         SI::idx_advance(&mut self.range_idx, self.gradient.ranges.len());
    //         self.cur_range = &self.gradient.ranges[self.range_idx];
    //     }
    // }
    //
    // fn advance_temp<SI: Sign>(&mut self, target_pos: f32) {
    //     while target_pos > self.temp_range.x1 || target_pos < self.temp_range.x0 {
    //         SI::idx_advance(&mut self.temp_range_idx, self.gradient.ranges.len());
    //         self.temp_range = &self.gradient.ranges[self.temp_range_idx];
    //     }
    // }

    pub(super) fn run(mut self, target: &mut [u8]) {
        // target
        //     .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
        //     .for_each(|column| {
        //         let mut pos = self.cur_pos;
        //
        //         for pixel in column.chunks_exact_mut(COLOR_COMPONENTS) {
        //             let angle = (-pos.1).atan2(pos.0).rem_euclid(2.0 * PI);
        //
        //             for col_idx in 0..COLOR_COMPONENTS {
        //                 let im3 = angle - self.x0;
        //                 let combined = (self.im3[col_idx] * im3 + 0.5) as i16;
        //
        //                 pixel[col_idx] = (self.c0[col_idx] as i16 + combined) as u8;
        //             }
        //
        //             pos.1 += 1.0;
        //         }
        //
        //         self.cur_pos.0 += 1.0;
        //     })
    }
}
