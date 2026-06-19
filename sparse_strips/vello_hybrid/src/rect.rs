// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_common::geometry::RectU16;
use vello_common::kurbo::Rect;

/// The threshold of the rectangle size after which a rectangle should be split up
/// into multiple smaller ones.
const LARGE_RECT_SPLIT_THRESHOLD: u16 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RectPart {
    pub(crate) rect: RectU16,
    pub(crate) frac: u32,
}

impl RectPart {
    pub(crate) fn shift(self, shift: (i32, i32)) -> Self {
        Self {
            rect: self.rect.shift(shift),
            ..self
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SplitRect {
    pub(crate) main: RectPart,
    pub(crate) top: Option<RectPart>,
    pub(crate) bottom: Option<RectPart>,
    pub(crate) left: Option<RectPart>,
    pub(crate) right: Option<RectPart>,
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "recorded rect coordinates are clipped to the u16 viewport domain before packing"
)]
pub(crate) fn split_rect(rect: &Rect) -> SplitRect {
    let sx0 = rect.x0.floor();
    let sy0 = rect.y0.floor();
    let sx1 = rect.x1.ceil();
    let sy1 = rect.y1.ceil();

    let x = sx0 as u16;
    let y = sy0 as u16;
    let width = (sx1 - sx0) as u16;
    let height = (sy1 - sy0) as u16;

    let left_frac = (rect.x0 - sx0) as f32;
    let top_frac = (rect.y0 - sy0) as f32;
    let right_frac = (sx1 - rect.x1) as f32;
    let bottom_frac = (sy1 - rect.y1) as f32;

    if rect.x1 - rect.x0 < f64::from(LARGE_RECT_SPLIT_THRESHOLD)
        || rect.y1 - rect.y0 < f64::from(LARGE_RECT_SPLIT_THRESHOLD)
    {
        return SplitRect {
            main: RectPart {
                rect: RectU16::new(x, y, x + width, y + height),
                frac: pack_unorm4x8([left_frac, top_frac, right_frac, bottom_frac]),
            },
            top: None,
            bottom: None,
            left: None,
            right: None,
        };
    }

    let has_left_aa = left_frac > 0.0;
    let has_top_aa = top_frac > 0.0;
    let has_right_aa = right_frac > 0.0;
    let has_bottom_aa = bottom_frac > 0.0;
    let has_top_strip = has_top_aa || has_left_aa || has_right_aa;
    let has_bottom_strip = has_bottom_aa || has_left_aa || has_right_aa;
    let left_inset = u16::from(has_left_aa);
    let right_inset = u16::from(has_right_aa);
    let top_inset = u16::from(has_top_strip);
    let bottom_inset = u16::from(has_bottom_strip);
    let inner_x = x + left_inset;
    let inner_y = y + top_inset;
    let inner_width = width - left_inset - right_inset;
    let inner_height = height - top_inset - bottom_inset;

    SplitRect {
        main: RectPart {
            rect: RectU16::new(
                inner_x,
                inner_y,
                inner_x + inner_width,
                inner_y + inner_height,
            ),
            frac: 0,
        },
        top: has_top_strip.then_some(RectPart {
            rect: RectU16::new(x, y, x + width, y + 1),
            frac: pack_unorm4x8([left_frac, top_frac, right_frac, 0.0]),
        }),
        bottom: has_bottom_strip.then_some(RectPart {
            rect: RectU16::new(x, y + height - 1, x + width, y + height),
            frac: pack_unorm4x8([left_frac, 0.0, right_frac, bottom_frac]),
        }),
        left: has_left_aa.then_some(RectPart {
            rect: RectU16::new(x, inner_y, x + 1, inner_y + inner_height),
            frac: pack_unorm4x8([left_frac, 0.0, 0.0, 0.0]),
        }),
        right: has_right_aa.then_some(RectPart {
            rect: RectU16::new(x + width - 1, inner_y, x + width, inner_y + inner_height),
            frac: pack_unorm4x8([0.0, 0.0, right_frac, 0.0]),
        }),
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "normalized color channels are packed into u8 lanes"
)]
fn pack_unorm4x8(v: [f32; 4]) -> u32 {
    let q = |f: f32| -> u8 { (f * 255.0 + 0.5) as u8 };
    u32::from(q(v[0]))
        | (u32::from(q(v[1])) << 8)
        | (u32::from(q(v[2])) << 16)
        | (u32::from(q(v[3])) << 24)
}

#[cfg(test)]
mod tests {
    use super::{RectPart, SplitRect, pack_unorm4x8, split_rect};

    use vello_common::geometry::RectU16;
    use vello_common::kurbo::Rect;

    fn part(x: u16, y: u16, width: u16, height: u16, frac: [f32; 4]) -> RectPart {
        RectPart {
            rect: RectU16::new(x, y, x + width, y + height),
            frac: pack_unorm4x8(frac),
        }
    }

    #[test]
    fn splitter_keeps_small_rect_whole() {
        let rect = Rect::new(10.25, 20.5, 25.75, 35.25);
        let split = split_rect(&rect);

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 16, 16, [0.25, 0.5, 0.25, 0.75]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_splits_large_rect_into_five_parts() {
        let rect = Rect::new(10.25, 20.5, 42.75, 52.75);
        let split = split_rect(&rect);

        assert_eq!(
            split,
            SplitRect {
                main: part(11, 21, 31, 31, [0.0, 0.0, 0.0, 0.0]),
                top: Some(part(10, 20, 33, 1, [0.25, 0.5, 0.25, 0.0])),
                bottom: Some(part(10, 52, 33, 1, [0.25, 0.0, 0.25, 0.25])),
                left: Some(part(10, 21, 1, 31, [0.25, 0.0, 0.0, 0.0])),
                right: Some(part(42, 21, 1, 31, [0.0, 0.0, 0.25, 0.0])),
            }
        );
    }

    #[test]
    fn splitter_omits_unneeded_edge_parts() {
        let rect = Rect::new(10.0, 20.5, 42.0, 53.0);
        let split = split_rect(&rect);

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 21, 32, 32, [0.0, 0.0, 0.0, 0.0]),
                top: Some(part(10, 20, 32, 1, [0.0, 0.5, 0.0, 0.0])),
                bottom: None,
                left: None,
                right: None,
            }
        );
    }
}
