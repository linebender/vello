use crate::fine::{COLOR_COMPONENTS, Positive, Sign, TILE_HEIGHT_COMPONENTS, extend};
use crate::paint::{EncodedLinearGradient, EncodedRadialGradient, GradientRange};
use vello_common::kurbo::Point;

#[derive(Debug)]
pub(crate) struct RadialGradientFiller<'a> {
    /// The position of the next x that should be processed.
    cur_pos: (f32, f32),
    /// The index of the current right stop we are processing.
    range_idx: usize,
    /// The underlying gradient.
    gradient: &'a EncodedRadialGradient,
    cur_range: &'a GradientRange,
}

impl<'a> RadialGradientFiller<'a> {
    pub(crate) fn new(gradient: &'a EncodedRadialGradient, start_x: u16, start_y: u16) -> Self {
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

    // Find smallest t such that distance((x, y), center(t)) = radius(t)
    // (x - cx(t))^2 + (y - cy(t))^2 - r(t)^2 = 0
    // in the form Atˆ2 + Bt + C = 0
    // where
    // cx(t) = t * x1
    // cy(t) = t * y1
    // r(t) = (1 - t) * r0 + t * r1
    fn cur_pos(&self, pos: (f32, f32)) -> f32 {
        let r0 = self.gradient.r0;
        let dx = self.gradient.c1.0;
        let dy = self.gradient.c1.1;
        let dr = self.gradient.r1 - self.gradient.r0;

        let px = pos.0;
        let py = pos.1;

        let a = dx * dx + dy * dy - dr * dr;
        let b = -2.0 * (px * dx + py * dy + r0 * dr);
        let c = px * px + py * py - r0 * r0;

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            eprintln!("{:?}", pos);
            panic!("was zero");
        }

        let sqrt_d = discriminant.sqrt();
        let t1 = (-b - sqrt_d) / (2.0 * a);
        let t2 = (-b + sqrt_d) / (2.0 * a);

        t1
    }

    pub(super) fn run(mut self, target: &mut [u8]) {
        let pad = self.gradient.pad;
        let transform = self.gradient.transform;

        let extend = |val| extend(val, pad, 0.0, 1.0);

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
