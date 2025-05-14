// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::fill_single;
use criterion::{Bencher, Criterion};
use vello_common::coarse::WideTile;
use vello_common::color::palette::css::ROYAL_BLUE;
use vello_common::paint::{Paint, PremulColor};
use vello_common::peniko::{BlendMode, Compose, Mix};
use vello_cpu::fine::{Fine, FineType};
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
fn normal<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::SrcOver));
}

#[vello_bench]
fn multiply<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Multiply, Compose::SrcOver));
}

#[vello_bench]
fn screen<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Screen, Compose::SrcOver));
}

#[vello_bench]
fn overlay<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Overlay, Compose::SrcOver));
}

#[vello_bench]
fn darken<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Darken, Compose::SrcOver));
}

#[vello_bench]
fn lighten<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Lighten, Compose::SrcOver));
}

#[vello_bench]
fn color_dodge<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::ColorDodge, Compose::SrcOver));
}

#[vello_bench]
fn color_burn<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::ColorBurn, Compose::SrcOver));
}

#[vello_bench]
fn hard_light<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::HardLight, Compose::SrcOver));
}

#[vello_bench]
fn soft_light<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::SoftLight, Compose::SrcOver));
}

#[vello_bench]
fn difference<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Difference, Compose::SrcOver));
}

#[vello_bench]
fn exclusion<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Exclusion, Compose::SrcOver));
}

#[vello_bench]
fn hue<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Hue, Compose::SrcOver));
}

#[vello_bench]
fn saturation<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Saturation, Compose::SrcOver));
}

#[vello_bench]
fn color<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Color, Compose::SrcOver));
}

#[vello_bench]
fn luminosity<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Luminosity, Compose::SrcOver));
}

#[vello_bench]
fn clear<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::Clear));
}

#[vello_bench]
fn copy<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::Copy));
}

#[vello_bench]
fn dest<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::Dest));
}

#[vello_bench]
fn dest_over<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::DestOver));
}

#[vello_bench]
fn src_in<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::SrcIn));
}

#[vello_bench]
fn dest_in<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::DestIn));
}

#[vello_bench]
fn src_out<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::SrcOut));
}

#[vello_bench]
fn dest_out<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::DestOut));
}

#[vello_bench]
fn src_atop<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::SrcAtop));
}

#[vello_bench]
fn dest_atop<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::DestAtop));
}

#[vello_bench]
fn xor<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::Xor));
}

#[vello_bench]
fn plus<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::Plus));
}

#[vello_bench]
fn plus_lighter<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    base(b, fine, BlendMode::new(Mix::Normal, Compose::PlusLighter));
}

fn base<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>, blend_mode: BlendMode) {
    let paint = Paint::Solid(PremulColor::from_alpha_color(ROYAL_BLUE));
    let width = WideTile::WIDTH as usize;

    fill_single(&paint, &[], width, b, blend_mode, fine);
}
