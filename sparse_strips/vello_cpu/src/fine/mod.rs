// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fine rasterization runs the commands in each wide tile to determine the final RGBA value
//! of each pixel and pack it into the pixmap.

mod blend;
mod gradient;
mod image;
mod rounded_blurred_rect;

use crate::fine::gradient::GradientFiller;
use crate::fine::image::ImageFiller;
use crate::fine::rounded_blurred_rect::BlurredRoundedRectFiller;
use crate::region::Region;
use crate::util::scalar::div_255;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::iter;
use core::ops::{Add, Div, Mul, Sub};
use vello_common::encode::{EncodedKind, EncodedPaint};
use vello_common::paint::{Paint, PremulColor};
use vello_common::peniko::{BlendMode, Compose, Mix};
use vello_common::{
    coarse::{Cmd, WideTile},
    tile::Tile,
};

pub(crate) const COLOR_COMPONENTS: usize = 4;
pub(crate) const TILE_HEIGHT_COMPONENTS: usize = Tile::HEIGHT as usize * COLOR_COMPONENTS;
#[doc(hidden)]
pub const SCRATCH_BUF_SIZE: usize =
    WideTile::WIDTH as usize * Tile::HEIGHT as usize * COLOR_COMPONENTS;

pub type ScratchBuf<F> = [F; SCRATCH_BUF_SIZE];

pub type FineU8 = ScratchBuf<u8>;
pub type FineF32 = ScratchBuf<f32>;

#[derive(Debug)]
#[doc(hidden)]
/// This is an internal struct, do not access directly.
pub struct Fine<F: FineType> {
    pub(crate) wide_coords: (u16, u16),
    pub(crate) blend_buf: Vec<ScratchBuf<F>>,
    pub(crate) color_buf: ScratchBuf<F>,
}

impl<F: FineType> Default for Fine<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FineType> Fine<F> {
    /// Create a new fine rasterizer.
    pub fn new() -> Self {
        let blend_buf = [F::ZERO; SCRATCH_BUF_SIZE];
        let color_buf = [F::ZERO; SCRATCH_BUF_SIZE];

        Self {
            wide_coords: (0, 0),
            blend_buf: vec![blend_buf],
            color_buf,
        }
    }

    /// Set the coordinates of the current wide tile that is being processed (in tile units).
    pub fn set_coords(&mut self, x: u16, y: u16) {
        self.wide_coords = (x, y);
    }

    pub fn clear(&mut self, premul_color: [F; 4]) {
        let blend_buf = self.blend_buf.last_mut().unwrap();

        if premul_color[0] == premul_color[1]
            && premul_color[1] == premul_color[2]
            && premul_color[2] == premul_color[3]
        {
            // All components are the same, so we can use memset instead.
            blend_buf.fill(premul_color[0]);
        } else {
            for z in blend_buf.chunks_exact_mut(COLOR_COMPONENTS) {
                z.copy_from_slice(&premul_color);
            }
        }
    }

    #[doc(hidden)]
    pub fn pack(&mut self, region: &mut Region<'_>) {
        let blend_buf = self.blend_buf.last().unwrap();

        for y in 0..Tile::HEIGHT {
            for (x, pixel) in region
                .row_mut(y)
                .chunks_exact_mut(COLOR_COMPONENTS)
                .enumerate()
            {
                let idx = COLOR_COMPONENTS * (usize::from(Tile::HEIGHT) * x + usize::from(y));
                pixel.copy_from_slice(&F::to_rgba8(&blend_buf[idx..][..COLOR_COMPONENTS]));
            }
        }
    }

    pub(crate) fn run_cmd(&mut self, cmd: &Cmd, alphas: &[u8], paints: &[EncodedPaint]) {
        match cmd {
            Cmd::Fill(f) => {
                self.fill(
                    usize::from(f.x),
                    usize::from(f.width),
                    &f.paint,
                    f.blend_mode
                        .unwrap_or(BlendMode::new(Mix::Normal, Compose::SrcOver)),
                    paints,
                );
            }
            Cmd::AlphaFill(s) => {
                let a_slice = &alphas[s.alpha_idx..];
                self.strip(
                    usize::from(s.x),
                    usize::from(s.width),
                    a_slice,
                    &s.paint,
                    s.blend_mode
                        .unwrap_or(BlendMode::new(Mix::Normal, Compose::SrcOver)),
                    paints,
                );
            }
            Cmd::PushBuf => {
                self.blend_buf.push([F::ZERO; SCRATCH_BUF_SIZE]);
            }
            Cmd::PopBuf => {
                self.blend_buf.pop();
            }
            Cmd::ClipFill(cf) => {
                self.clip_fill(cf.x as usize, cf.width as usize);
            }
            Cmd::ClipStrip(cs) => {
                let aslice = &alphas[cs.alpha_idx..];
                self.clip_strip(cs.x as usize, cs.width as usize, aslice);
            }
            Cmd::Blend(cb) => {
                self.apply_blend(*cb);
            }
            Cmd::Opacity(o) => {
                if *o != 1.0 {
                    self.blend_buf
                        .last_mut()
                        .unwrap()
                        .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
                        .for_each(|s| {
                            for c in s {
                                *c = F::from_normalized_f32(*o).normalized_mul(*c);
                            }
                        });
                }
            }
            Cmd::Mask(m) => {
                let start_x = self.wide_coords.0 * WideTile::WIDTH;
                let start_y = self.wide_coords.1 * Tile::HEIGHT;

                for (x, col) in self
                    .blend_buf
                    .last_mut()
                    .unwrap()
                    .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
                    .enumerate()
                {
                    for (y, pix) in col.chunks_exact_mut(COLOR_COMPONENTS).enumerate() {
                        let x = start_x + x as u16;
                        let y = start_y + y as u16;

                        if x < m.width() && y < m.height() {
                            let val = F::from_normalized_u8(m.sample(x, y));

                            for comp in pix.iter_mut() {
                                *comp = comp.normalized_mul(val);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Fill at a given x and with a width using the given paint.
    pub fn fill(
        &mut self,
        x: usize,
        width: usize,
        fill: &Paint,
        blend_mode: BlendMode,
        encoded_paints: &[EncodedPaint],
    ) {
        let blend_buf = &mut self.blend_buf.last_mut().unwrap()[x * TILE_HEIGHT_COMPONENTS..]
            [..TILE_HEIGHT_COMPONENTS * width];
        let color_buf =
            &mut self.color_buf[x * TILE_HEIGHT_COMPONENTS..][..TILE_HEIGHT_COMPONENTS * width];

        let start_x = self.wide_coords.0 * WideTile::WIDTH + x as u16;
        let start_y = self.wide_coords.1 * Tile::HEIGHT;

        let default_blend = blend_mode == BlendMode::new(Mix::Normal, Compose::SrcOver);

        fn fill_complex_paint<T: FineType>(
            color_buf: &mut [T],
            blend_buf: &mut [T],
            has_opacities: bool,
            blend_mode: BlendMode,
            filler: impl Painter,
        ) {
            if has_opacities {
                filler.paint(color_buf);
                fill::blend(
                    blend_buf,
                    color_buf.chunks_exact(4).map(|e| [e[0], e[1], e[2], e[3]]),
                    blend_mode,
                );
            } else {
                // Similarly to solid colors we can just override the previous values
                // if all colors in the gradient are fully opaque.
                filler.paint(blend_buf);
            }
        }

        match fill {
            Paint::Solid(color) => {
                let color = F::extract_color(color);

                // If color is completely opaque we can just memcopy the colors.
                if color[3] == F::ONE && default_blend {
                    for t in blend_buf.chunks_exact_mut(COLOR_COMPONENTS) {
                        t.copy_from_slice(&color);
                    }

                    return;
                }

                fill::blend(blend_buf, iter::repeat(color), blend_mode);
            }
            Paint::Indexed(paint) => {
                let encoded_paint = &encoded_paints[paint.index()];

                match encoded_paint {
                    EncodedPaint::Gradient(g) => match &g.kind {
                        EncodedKind::Linear(l) => {
                            let filler = GradientFiller::new(g, l, start_x, start_y);
                            fill_complex_paint(
                                color_buf,
                                blend_buf,
                                g.has_opacities,
                                blend_mode,
                                filler,
                            );
                        }
                        EncodedKind::Radial(r) => {
                            let filler = GradientFiller::new(g, r, start_x, start_y);
                            fill_complex_paint(
                                color_buf,
                                blend_buf,
                                g.has_opacities,
                                blend_mode,
                                filler,
                            );
                        }
                        EncodedKind::Sweep(s) => {
                            let filler = GradientFiller::new(g, s, start_x, start_y);
                            fill_complex_paint(
                                color_buf,
                                blend_buf,
                                g.has_opacities,
                                blend_mode,
                                filler,
                            );
                        }
                    },
                    EncodedPaint::Image(i) => {
                        let filler = ImageFiller::new(i, start_x, start_y);
                        fill_complex_paint(
                            color_buf,
                            blend_buf,
                            i.has_opacities,
                            blend_mode,
                            filler,
                        );
                    }
                    EncodedPaint::BlurredRoundedRect(b) => {
                        let filler = BlurredRoundedRectFiller::new(b, start_x, start_y);
                        fill_complex_paint(color_buf, blend_buf, true, blend_mode, filler);
                    }
                }
            }
        }
    }

    /// Strip at a given x and with a width using the given paint and alpha values.
    pub fn strip(
        &mut self,
        x: usize,
        width: usize,
        alphas: &[u8],
        fill: &Paint,
        blend_mode: BlendMode,
        paints: &[EncodedPaint],
    ) {
        debug_assert!(
            alphas.len() >= width,
            "alpha buffer doesn't contain sufficient elements"
        );

        let blend_buf = &mut self.blend_buf.last_mut().unwrap()[x * TILE_HEIGHT_COMPONENTS..]
            [..TILE_HEIGHT_COMPONENTS * width];
        let color_buf =
            &mut self.color_buf[x * TILE_HEIGHT_COMPONENTS..][..TILE_HEIGHT_COMPONENTS * width];

        let start_x = self.wide_coords.0 * WideTile::WIDTH + x as u16;
        let start_y = self.wide_coords.1 * Tile::HEIGHT;

        fn strip_complex_paint<F: FineType>(
            color_buf: &mut [F],
            blend_buf: &mut [F],
            blend_mode: BlendMode,
            filler: impl Painter,
            alphas: &[u8],
        ) {
            filler.paint(color_buf);
            strip::blend(
                blend_buf,
                color_buf.chunks_exact(4).map(|e| [e[0], e[1], e[2], e[3]]),
                blend_mode,
                alphas.chunks_exact(4).map(|e| [e[0], e[1], e[2], e[3]]),
            );
        }

        match fill {
            Paint::Solid(color) => {
                strip::blend(
                    blend_buf,
                    iter::repeat(F::extract_color(color)),
                    blend_mode,
                    alphas.chunks_exact(4).map(|e| [e[0], e[1], e[2], e[3]]),
                );
            }
            Paint::Indexed(paint) => {
                let encoded_paint = &paints[paint.index()];

                match encoded_paint {
                    EncodedPaint::Gradient(g) => match &g.kind {
                        EncodedKind::Linear(l) => {
                            let filler = GradientFiller::new(g, l, start_x, start_y);
                            strip_complex_paint(color_buf, blend_buf, blend_mode, filler, alphas);
                        }
                        EncodedKind::Radial(r) => {
                            let filler = GradientFiller::new(g, r, start_x, start_y);
                            strip_complex_paint(color_buf, blend_buf, blend_mode, filler, alphas);
                        }
                        EncodedKind::Sweep(s) => {
                            let filler = GradientFiller::new(g, s, start_x, start_y);
                            strip_complex_paint(color_buf, blend_buf, blend_mode, filler, alphas);
                        }
                    },
                    EncodedPaint::Image(i) => {
                        let filler = ImageFiller::new(i, start_x, start_y);
                        strip_complex_paint(color_buf, blend_buf, blend_mode, filler, alphas);
                    }
                    EncodedPaint::BlurredRoundedRect(b) => {
                        let filler = BlurredRoundedRectFiller::new(b, start_x, start_y);
                        strip_complex_paint(color_buf, blend_buf, blend_mode, filler, alphas);
                    }
                }
            }
        }
    }

    fn apply_blend(&mut self, blend_mode: BlendMode) {
        let (source_buffer, rest) = self.blend_buf.split_last_mut().unwrap();
        let target_buffer = rest.last_mut().unwrap();

        fill::blend(
            target_buffer,
            source_buffer
                .chunks_exact(4)
                .map(|e| [e[0], e[1], e[2], e[3]]),
            blend_mode,
        );
    }

    fn clip_fill(&mut self, x: usize, width: usize) {
        let (source_buffer, rest) = self.blend_buf.split_last_mut().unwrap();
        let target_buffer = rest.last_mut().unwrap();

        let source_buffer =
            &mut source_buffer[x * TILE_HEIGHT_COMPONENTS..][..TILE_HEIGHT_COMPONENTS * width];
        let target_buffer =
            &mut target_buffer[x * TILE_HEIGHT_COMPONENTS..][..TILE_HEIGHT_COMPONENTS * width];

        fill::alpha_composite(
            target_buffer,
            source_buffer
                .chunks_exact(4)
                .map(|e| [e[0], e[1], e[2], e[3]]),
        );
    }

    fn clip_strip(&mut self, x: usize, width: usize, alphas: &[u8]) {
        let (source_buffer, rest) = self.blend_buf.split_last_mut().unwrap();
        let target_buffer = rest.last_mut().unwrap();

        let source_buffer =
            &mut source_buffer[x * TILE_HEIGHT_COMPONENTS..][..TILE_HEIGHT_COMPONENTS * width];
        let target_buffer =
            &mut target_buffer[x * TILE_HEIGHT_COMPONENTS..][..TILE_HEIGHT_COMPONENTS * width];

        strip::alpha_composite(
            target_buffer,
            source_buffer
                .chunks_exact(4)
                .map(|e| [e[0], e[1], e[2], e[3]]),
            alphas.chunks_exact(4).map(|e| [e[0], e[1], e[2], e[3]]),
        );
    }
}

pub(crate) mod fill {
    // See https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators for the
    // formulas.

    use crate::fine::{COLOR_COMPONENTS, FineType, TILE_HEIGHT_COMPONENTS, blend};
    use vello_common::peniko::{BlendMode, Compose, Mix};

    pub(crate) fn blend<F: FineType, T: Iterator<Item = [F; COLOR_COMPONENTS]>>(
        target: &mut [F],
        source: T,
        blend_mode: BlendMode,
    ) {
        match (blend_mode.mix, blend_mode.compose) {
            (Mix::Normal, Compose::SrcOver) => alpha_composite(target, source),
            _ => blend::fill::blend(target, source, blend_mode),
        }
    }

    pub(crate) fn alpha_composite<F: FineType, T: Iterator<Item = [F; COLOR_COMPONENTS]>>(
        target: &mut [F],
        mut source: T,
    ) {
        for strip in target.chunks_exact_mut(TILE_HEIGHT_COMPONENTS) {
            for bg_c in strip.chunks_exact_mut(COLOR_COMPONENTS) {
                let src_c = source.next().unwrap();
                for i in 0..COLOR_COMPONENTS {
                    bg_c[i] = src_c[i].add(bg_c[i].normalized_mul(src_c[3].one_minus()));
                }
            }
        }
    }
}

pub(crate) mod strip {
    use crate::fine::{COLOR_COMPONENTS, FineType, TILE_HEIGHT_COMPONENTS, Widened, blend};
    use vello_common::peniko::{BlendMode, Compose, Mix};
    use vello_common::tile::Tile;

    pub(crate) fn blend<
        F: FineType,
        T: Iterator<Item = [F; COLOR_COMPONENTS]>,
        A: Iterator<Item = [u8; Tile::HEIGHT as usize]>,
    >(
        target: &mut [F],
        source: T,
        blend_mode: BlendMode,
        alphas: A,
    ) {
        match (blend_mode.mix, blend_mode.compose) {
            (Mix::Normal, Compose::SrcOver) => alpha_composite(target, source, alphas),
            _ => blend::strip::blend(target, source, blend_mode, alphas),
        }
    }

    pub(crate) fn alpha_composite<
        F: FineType,
        T: Iterator<Item = [F; COLOR_COMPONENTS]>,
        A: Iterator<Item = [u8; Tile::HEIGHT as usize]>,
    >(
        target: &mut [F],
        mut source: T,
        mut alphas: A,
    ) {
        for bg_c in target.chunks_exact_mut(TILE_HEIGHT_COMPONENTS) {
            let masks = alphas.next().unwrap();

            for j in 0..usize::from(Tile::HEIGHT) {
                let src_c = source.next().unwrap();
                let mask_a = F::from_normalized_u8(masks[j]);
                let inv_src_a_mask_a = mask_a.normalized_mul(src_c[3]).one_minus();

                for i in 0..COLOR_COMPONENTS {
                    let p1 = bg_c[j * COLOR_COMPONENTS + i].widen() * inv_src_a_mask_a.widen();
                    let p2 = src_c[i].widen() * mask_a.widen();

                    bg_c[j * COLOR_COMPONENTS + i] = (p1 + p2).normalize().narrow();
                }
            }
        }
    }
}

trait Painter {
    fn paint<F: FineType>(self, target: &mut [F]);
}

/// A numeric type that can act as a substitute for another underlying type in case
/// the results are too big. Currently, this is only used for u8, where certain operations
/// are first cast to u16 and then cast back to u8.
pub trait Widened<T: FineType>:
    Sized
    + Copy
    + PartialEq<Self>
    + PartialOrd<Self>
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Div<Self, Output = Self>
    + Debug
{
    /// Clamp the current value to the boundaries **of the underlying narrowed type**.
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn clamp(self) -> Self;
    /// Normalize the current value to the range of the underlying narrowed type.
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn normalize(self) -> Self;
    /// Get the minimum between this number and another number.
    #[must_use = "this returns the result of the comparison, without modifying either input"]
    fn min(self, other: Self) -> Self;
    /// Get the maximum between this number and another number.
    #[must_use = "this returns the result of the comparison, without modifying either input"]
    fn max(self, other: Self) -> Self;
    /// Perform a normalizing multiplication between this number and another number.
    #[must_use = "this returns the result of the multiplication, without modifying either input"]
    fn normalized_mul(self, other: Self) -> Self;
    /// Cast the current type to its narrowed representation.
    fn narrow(self) -> T;
}

impl Widened<Self> for f32 {
    #[inline(always)]
    fn clamp(self) -> Self {
        Self::clamp(self, Self::ZERO, Self::ONE)
    }

    #[inline(always)]
    fn normalize(self) -> Self {
        // f32 values are always normalized between 0.0 and 1.0.
        self
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::max(self, other)
    }

    #[inline(always)]
    fn normalized_mul(self, other: Self) -> Self {
        self * other
    }

    #[inline(always)]
    fn narrow(self) -> Self {
        self
    }
}

impl Widened<u8> for u16 {
    #[inline(always)]
    fn clamp(self) -> Self {
        Ord::clamp(self, Self::from(u8::ZERO), Self::from(u8::ONE))
    }

    #[inline(always)]
    fn normalize(self) -> Self {
        div_255(self)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Ord::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Ord::max(self, other)
    }

    #[inline(always)]
    fn normalized_mul(self, other: Self) -> Self {
        (self * other).normalize()
    }

    #[inline(always)]
    fn narrow(self) -> u8 {
        debug_assert!(
            self <= Self::from(u8::MAX),
            "cannot narrow integers larger than u8::MAX"
        );

        self as u8
    }
}

/// A type that can be used as the underlying storage for fine rasterization.
pub trait FineType:
    Sized
    + Copy
    + PartialEq<Self>
    + PartialOrd<Self>
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Debug
    + Send
    + Sync
{
    type Widened: Widened<Self>;

    /// The number that is considered to be the minimum of the normalized range of this type.
    const ZERO: Self;
    /// The number that is considered to be in the "center" of the normalized range of this type.
    const MID: Self;
    /// The number considered to be the maximum of the normalized range of the type.
    const ONE: Self;

    /// Return the minimum number.
    #[must_use = "this returns the result of the comparison, without modifying either input"]
    fn min(self, other: Self) -> Self;
    /// Return the maximum number.
    #[must_use = "this returns the result of the comparison, without modifying either input"]
    fn max(self, other: Self) -> Self;
    /// Extract the underlying color from a premultiplied color.
    fn extract_color(color: &PremulColor) -> [Self; COLOR_COMPONENTS];
    /// Convert a normalized u8 integer to this type.
    fn from_normalized_u8(num: u8) -> Self;
    /// Convert a plain u8 integer to this type.
    fn from_u8(num: u8) -> Self;
    /// Convert this number to a normalized f32.
    fn to_normalized_f32(self) -> f32;
    /// Convert this number to a normalized u8.
    fn to_normalized_u8(self) -> u8;
    /// Convert to this number from a normalized f32.
    fn from_normalized_f32(num: f32) -> Self;
    /// Get the widened representation of the current number.
    fn widen(self) -> Self::Widened;

    /// Perform a normalized multiplication between this number and another
    #[inline(always)]
    #[must_use = "this returns the result of the multiplication, without modifying either input"]
    fn normalized_mul(self, other: Self) -> Self {
        (self.widen() * other.widen()).normalize().narrow()
    }
    /// Perform a widening multiplication and then divide by a third number.
    #[inline(always)]
    fn widened_mul_div(self, other: Self, other2: Self) -> Self::Widened {
        (self.widen() * other.widen()) / other2.widen()
    }
    // TODO: These RGBA conversions should be sized to COLOR_COMPONENTS, but will leave that for
    // the future.
    /// Convert a slice to a RGBA8 slice.
    #[inline(always)]
    fn to_rgba8(src: &[Self]) -> [u8; COLOR_COMPONENTS] {
        [
            src[0].to_normalized_u8(),
            src[1].to_normalized_u8(),
            src[2].to_normalized_u8(),
            src[3].to_normalized_u8(),
        ]
    }
    /// Convert a RGBA8 slice to a slice of this type.
    #[inline(always)]
    fn from_rgba8(src: &[u8]) -> [Self; COLOR_COMPONENTS] {
        [
            Self::from_normalized_u8(src[0]),
            Self::from_normalized_u8(src[1]),
            Self::from_normalized_u8(src[2]),
            Self::from_normalized_u8(src[3]),
        ]
    }
    /// Convert a RGBAF32 slice to a slice of this type.
    #[inline(always)]
    fn from_rgbaf32(src: &[f32]) -> [Self; COLOR_COMPONENTS] {
        [
            Self::from_normalized_f32(src[0]),
            Self::from_normalized_f32(src[1]),
            Self::from_normalized_f32(src[2]),
            Self::from_normalized_f32(src[3]),
        ]
    }
    /// Calculate "one minus" this number, i.e., `Self::ONE - self`.
    #[inline(always)]
    #[must_use = "method returns a new number and does not mutate the original value"]
    fn one_minus(self) -> Self {
        Self::ONE - self
    }
}

impl FineType for u8 {
    type Widened = u16;
    const ZERO: Self = 0;
    const MID: Self = 127;
    const ONE: Self = 255;

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Ord::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Ord::max(self, other)
    }

    #[inline(always)]
    fn extract_color(color: &PremulColor) -> [Self; COLOR_COMPONENTS] {
        color.as_premul_rgba8().to_u8_array()
    }

    #[inline(always)]
    fn from_normalized_u8(num: u8) -> Self {
        num
    }

    #[inline(always)]
    fn from_u8(num: u8) -> Self {
        num
    }

    #[inline(always)]
    fn to_normalized_f32(self) -> f32 {
        f32::from(self) / 255.0
    }

    #[inline(always)]
    fn to_normalized_u8(self) -> u8 {
        self
    }

    #[inline(always)]
    fn from_normalized_f32(num: f32) -> Self {
        (num * 255.0 + 0.5) as Self
    }

    #[inline(always)]
    fn widen(self) -> Self::Widened {
        u16::from(self)
    }
}

impl FineType for f32 {
    type Widened = Self;
    const ZERO: Self = 0.0;
    const MID: Self = 0.5;
    const ONE: Self = 1.0;

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::max(self, other)
    }

    #[inline(always)]
    fn extract_color(color: &PremulColor) -> [Self; COLOR_COMPONENTS] {
        color.as_premul_f32().components
    }

    #[inline(always)]
    fn from_normalized_u8(num: u8) -> Self {
        Self::from(num) / 255.0
    }

    #[inline(always)]
    fn from_u8(num: u8) -> Self {
        Self::from(num)
    }

    #[inline(always)]
    fn to_normalized_f32(self) -> f32 {
        self
    }

    #[inline(always)]
    fn to_normalized_u8(self) -> u8 {
        (self * 255.0 + 0.5) as u8
    }

    #[inline(always)]
    fn from_normalized_f32(num: f32) -> Self {
        num
    }

    #[inline(always)]
    fn widen(self) -> Self::Widened {
        self
    }
}
