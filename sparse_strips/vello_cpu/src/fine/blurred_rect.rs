//! Drawing blurred rectangles.
//! 
//! Implementation is taken from: https://git.sr.ht/~raph/blurrr/tree/master/src/distfield.rs

use crate::fine::{COLOR_COMPONENTS, Painter, TILE_HEIGHT_COMPONENTS};
use crate::util::scalar::div_255;
use vello_common::encode::EncodedBlurredRectangle;
use vello_common::kurbo::Point;
use vello_common::math::compute_erf7;

#[derive(Debug)]
pub(crate) struct BlurredRectFiller<'a> {
    /// The current position that should be processed.
    cur_pos: Point,
    /// The underlying gradient.
    rect: &'a EncodedBlurredRectangle,
}

impl<'a> BlurredRectFiller<'a> {
    pub(crate) fn new(rect: &'a EncodedBlurredRectangle, start_x: u16, start_y: u16) -> Self {
        Self {
            // We want to sample values of the pixels at the center, so add an offset of 0.5.
            cur_pos: rect.transform * Point::new(start_x as f64, start_y as f64),
            rect,
        }
    }

    pub(super) fn run(mut self, target: &mut [u8]) {
        let h = self.rect.h;
        let w = self.rect.w;
        let width = self.rect.width;
        let height = self.rect.height;
        let r1 = self.rect.r1;
        let exponent = self.rect.exponent;
        let recip_exponent = self.rect.recip_exponent;
        let scale = self.rect.scale;
        let min_edge = self.rect.min_edge;
        let std_dev_inv = self.rect.std_dev_inv;

        let mut col = self.rect.color.as_premul_rgba8().to_u8_array();

        for column in target.chunks_exact_mut(TILE_HEIGHT_COMPONENTS) {
            let mut col_pos = self.cur_pos;

            for pixel in column.chunks_exact_mut(COLOR_COMPONENTS) {
                let mut pixel_color = col;

                let j = col_pos.y as f32;
                let i = col_pos.x as f32;

                let alpha_val = {
                    let y = j + 0.5 - 0.5 * height;
                    let y0 = y.abs() - (h * 0.5 - r1);
                    let y1 = y0.max(0.0);

                    let x = i + 0.5 - 0.5 * width;
                    let x0 = x.abs() - (w * 0.5 - r1);
                    let x1 = x0.max(0.0);
                    let d_pos = (x1.powf(exponent) + y1.powf(exponent)).powf(recip_exponent);
                    let d_neg = x0.max(y0).min(0.0);
                    let d = d_pos + d_neg - r1;
                    let z = scale
                        * (compute_erf7(std_dev_inv * (min_edge + d))
                            - compute_erf7(std_dev_inv * d));

                    ((z * 255.0) + 0.5) as u8
                };

                for col in &mut pixel_color {
                    *col = div_255(*col as u16 * alpha_val as u16) as u8;
                }
                
                pixel.copy_from_slice(&pixel_color);
                
                col_pos += self.rect.y_advance;
            }

            self.cur_pos += self.rect.x_advance;
        }
    }
}

impl Painter for BlurredRectFiller<'_> {
    fn paint(self, target: &mut [u8]) {
        self.run(target);
    }
}
