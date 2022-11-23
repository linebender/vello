// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

use piet_scene::kurbo::{Affine, Ellipse, PathEl, Point, Rect};
use piet_scene::{
    BlendMode, Brush, Color, Fill, LinearGradient, Mix, RadialGradient, Scene, SceneBuilder,
    SceneFragment, Stroke,
};

use crate::pico_svg::PicoSvg;

pub fn gen_test_scene() -> Scene {
    let mut scene = Scene::default();
    let mut builder = SceneBuilder::for_scene(&mut scene);
    let scene_ix = 1;
    match scene_ix {
        0 => {
            let path = [
                PathEl::MoveTo(Point::new(100.0, 100.0)),
                PathEl::LineTo(Point::new(500.0, 120.0)),
                PathEl::LineTo(Point::new(300.0, 150.0)),
                PathEl::LineTo(Point::new(200.0, 260.0)),
                PathEl::LineTo(Point::new(150.0, 210.0)),
            ];
            let brush = Brush::Solid(Color::rgb8(0x40, 0x40, 0xff));
            builder.fill(Fill::NonZero, Affine::IDENTITY, &brush, None, &path);
            let transform = Affine::translate((50.0, 50.0));
            let brush = Brush::Solid(Color::rgba8(0xff, 0xff, 0x00, 0x80));
            builder.fill(Fill::NonZero, transform, &brush, None, &path);
            let transform = Affine::translate((100.0, 100.0));
            let style = Stroke::new(1.0);
            let brush = Brush::Solid(Color::rgb8(0xa0, 0x00, 0x00));
            builder.stroke(&style, transform, &brush, None, &path);
        }
        1 => {
            render_blend_grid(&mut builder);
        }
        _ => {
            let xml_str =
                std::str::from_utf8(include_bytes!("../../piet-gpu/Ghostscript_Tiger.svg"))
                    .unwrap();
            let svg = PicoSvg::load(xml_str, 6.0).unwrap();
            render_svg(&mut builder, &svg, false);
        }
    }
    builder.finish();
    scene
}

#[allow(unused)]
pub fn dump_scene_info(scene: &Scene) {
    let data = scene.data();
    println!("tags {:?}", data.tag_stream);
    println!(
        "pathsegs {:?}",
        bytemuck::cast_slice::<u8, f32>(&data.pathseg_stream)
    );
}

pub fn render_svg(sb: &mut SceneBuilder, svg: &PicoSvg, print_stats: bool) {
    use crate::pico_svg::*;
    let start = std::time::Instant::now();
    for item in svg.items.iter() {
        match item {
            Item::Fill(fill) => {
                sb.fill(
                    Fill::NonZero,
                    Affine::IDENTITY,
                    fill.color,
                    None,
                    &fill.path,
                );
            }
            Item::Stroke(stroke) => {
                sb.stroke(
                    &Stroke::new(stroke.width as f32),
                    Affine::IDENTITY,
                    stroke.color,
                    None,
                    &stroke.path,
                );
            }
        }
    }
    if print_stats {
        println!("flattening and encoding time: {:?}", start.elapsed());
    }
}

#[allow(unused)]
pub fn render_blend_grid(sb: &mut SceneBuilder) {
    const BLEND_MODES: &[Mix] = &[
        Mix::Normal,
        Mix::Multiply,
        Mix::Darken,
        Mix::Screen,
        Mix::Lighten,
        Mix::Overlay,
        Mix::ColorDodge,
        Mix::ColorBurn,
        Mix::HardLight,
        Mix::SoftLight,
        Mix::Difference,
        Mix::Exclusion,
        Mix::Hue,
        Mix::Saturation,
        Mix::Color,
        Mix::Luminosity,
    ];
    for (ix, &blend) in BLEND_MODES.iter().enumerate() {
        let i = ix % 4;
        let j = ix / 4;
        let transform = Affine::translate((i as f64 * 225., j as f64 * 225.));
        let square = blend_square(blend.into());
        sb.append(&square, Some(transform));
    }
}

#[allow(unused)]
fn render_blend_square(sb: &mut SceneBuilder, blend: BlendMode, transform: Affine) {
    // Inspired by https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    let rect = Rect::from_origin_size(Point::new(0., 0.), (200., 200.));
    let linear = LinearGradient::new((0.0, 0.0), (200.0, 0.0)).stops([Color::BLACK, Color::WHITE]);
    sb.fill(Fill::NonZero, transform, &linear, None, &rect);
    const GRADIENTS: &[(f64, f64, Color)] = &[
        (150., 0., Color::rgb8(255, 240, 64)),
        (175., 100., Color::rgb8(255, 96, 240)),
        (125., 200., Color::rgb8(64, 192, 255)),
    ];
    for (x, y, c) in GRADIENTS {
        let mut color2 = c.clone();
        color2.a = 0;
        let radial = RadialGradient::new((*x, *y), 100.0).stops([*c, color2]);
        sb.fill(Fill::NonZero, transform, &radial, None, &rect);
    }
    const COLORS: &[Color] = &[
        Color::rgb8(255, 0, 0),
        Color::rgb8(0, 255, 0),
        Color::rgb8(0, 0, 255),
    ];
    sb.push_layer(Mix::Normal, transform, &rect);
    for (i, c) in COLORS.iter().enumerate() {
        let linear = LinearGradient::new((0.0, 0.0), (0.0, 200.0)).stops([Color::WHITE, *c]);
        sb.push_layer(blend, transform, &rect);
        // squash the ellipse
        let a = transform
            * Affine::translate((100., 100.))
            * Affine::rotate(std::f64::consts::FRAC_PI_3 * (i * 2 + 1) as f64)
            * Affine::scale_non_uniform(1.0, 0.357)
            * Affine::translate((-100., -100.));
        sb.fill(
            Fill::NonZero,
            a,
            &linear,
            None,
            &Ellipse::new((100., 100.), (90., 90.), 0.),
        );
        sb.pop_layer();
    }
    sb.pop_layer();
}

#[allow(unused)]
fn blend_square(blend: BlendMode) -> SceneFragment {
    let mut fragment = SceneFragment::default();
    let mut sb = SceneBuilder::for_fragment(&mut fragment);
    render_blend_square(&mut sb, blend, Affine::IDENTITY);
    sb.finish();
    fragment
}
