// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::fill_single;
use criterion::{Bencher, Criterion};
use vello_common::coarse::WideTile;
use vello_common::color::palette::css::ROYAL_BLUE;
use vello_common::fearless_simd::Simd;
use vello_common::paint::{Paint, PremulColor};
use vello_common::peniko::{BlendMode, Compose, Mix};
use vello_cpu::fine::{Fine, FineKernel};
use vello_dev_macros::vello_bench;

pub fn blend(c: &mut Criterion) {
    normal(c);
    multiply(c);
    screen(c);
    overlay(c);
    darken(c);
    lighten(c);
    color_dodge(c);
    color_burn(c);
    hard_light(c);
    soft_light(c);
    difference(c);
    exclusion(c);
    hue(c);
    saturation(c);
    color(c);
    luminosity(c);

    clear(c);
    copy(c);
    dest(c);
    dest_over(c);
    src_out(c);
    src_in(c);
    dest_in(c);
    dest_out(c);
    src_atop(c);
    dest_atop(c);
    xor(c);
    plus(c);
    plus_lighter(c);
}

#[vello_bench]
fn normal<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::SrcOver));
}

#[vello_bench]
fn multiply<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Multiply, Compose::SrcOver));
}

#[vello_bench]
fn screen<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Screen, Compose::SrcOver));
}

#[vello_bench]
fn overlay<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Overlay, Compose::SrcOver));
}

#[vello_bench]
fn darken<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Darken, Compose::SrcOver));
}

#[vello_bench]
fn lighten<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Lighten, Compose::SrcOver));
}

#[vello_bench]
fn color_dodge<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::ColorDodge, Compose::SrcOver));
}

#[vello_bench]
fn color_burn<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::ColorBurn, Compose::SrcOver));
}

#[vello_bench]
fn hard_light<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::HardLight, Compose::SrcOver));
}

#[vello_bench]
fn soft_light<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::SoftLight, Compose::SrcOver));
}

#[vello_bench]
fn difference<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Difference, Compose::SrcOver));
}

#[vello_bench]
fn exclusion<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Exclusion, Compose::SrcOver));
}

#[vello_bench]
fn hue<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Hue, Compose::SrcOver));
}

#[vello_bench]
fn saturation<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Saturation, Compose::SrcOver));
}

#[vello_bench]
fn color<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Color, Compose::SrcOver));
}

#[vello_bench]
fn luminosity<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Luminosity, Compose::SrcOver));
}

#[vello_bench]
fn clear<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::Clear));
}

#[vello_bench]
fn copy<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::Copy));
}

#[vello_bench]
fn dest<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::Dest));
}

#[vello_bench]
fn dest_over<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::DestOver));
}

#[vello_bench]
fn src_in<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::SrcIn));
}

#[vello_bench]
fn dest_in<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::DestIn));
}

#[vello_bench]
fn src_out<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::SrcOut));
}

#[vello_bench]
fn dest_out<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::DestOut));
}

#[vello_bench]
fn src_atop<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::SrcAtop));
}

#[vello_bench]
fn dest_atop<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::DestAtop));
}

#[vello_bench]
fn xor<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::Xor));
}

#[vello_bench]
fn plus<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::Plus));
}

#[vello_bench]
fn plus_lighter<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::PlusLighter));
}

fn base<S: Simd, T: FineKernel<S>>(
    b: &mut Bencher<'_>,
    fine: &mut Fine<S, T>,
    blend_mode: BlendMode,
) {
    let paint = Paint::Solid(PremulColor::from_alpha_color(ROYAL_BLUE));
    let width = WideTile::WIDTH as usize;

    fill_single(&paint, &[], width, b, blend_mode, fine);
}
