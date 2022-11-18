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

use kurbo::BezPath;
use piet_scene::{
    Affine, BlendMode, Brush, Color, Compose, ExtendMode, Fill, GradientStop, LinearGradient, Mix,
    PathElement, Point, RadialGradient, Rect, Scene, SceneBuilder, SceneFragment, Stroke,
};

use crate::pico_svg::PicoSvg;

pub fn gen_test_scene() -> Scene {
    let mut scene = Scene::default();
    let mut builder = SceneBuilder::for_scene(&mut scene);
    let scene_ix = 1;
    match scene_ix {
        0 => {
            let path = [
                PathElement::MoveTo(Point::new(100.0, 100.0)),
                PathElement::LineTo(Point::new(500.0, 120.0)),
                PathElement::LineTo(Point::new(300.0, 150.0)),
                PathElement::LineTo(Point::new(200.0, 260.0)),
                PathElement::LineTo(Point::new(150.0, 210.0)),
                PathElement::Close,
            ];
            let brush = Brush::Solid(Color::rgb8(0x40, 0x40, 0xff));
            builder.fill(Fill::NonZero, Affine::IDENTITY, &brush, None, &path);
            let transform = Affine::translate(50.0, 50.0);
            let brush = Brush::Solid(Color::rgba8(0xff, 0xff, 0x00, 0x80));
            builder.fill(Fill::NonZero, transform, &brush, None, &path);
            let transform = Affine::translate(100.0, 100.0);
            let style = simple_stroke(1.0);
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
                    &fill.color.into(),
                    None,
                    convert_bez_path(&fill.path),
                );
            }
            Item::Stroke(stroke) => {
                sb.stroke(
                    &simple_stroke(stroke.width as f32),
                    Affine::IDENTITY,
                    &stroke.color.into(),
                    None,
                    convert_bez_path(&stroke.path),
                );
            }
        }
    }
    if print_stats {
        println!("flattening and encoding time: {:?}", start.elapsed());
    }
}

fn convert_bez_path<'a>(path: &'a BezPath) -> impl Iterator<Item = PathElement> + 'a + Clone {
    path.elements()
        .iter()
        .map(|el| PathElement::from_kurbo(*el))
}

fn simple_stroke(width: f32) -> Stroke<[f32; 0]> {
    Stroke {
        width,
        join: piet_scene::Join::Round,
        miter_limit: 1.4,
        start_cap: piet_scene::Cap::Round,
        end_cap: piet_scene::Cap::Round,
        dash_pattern: [],
        dash_offset: 0.0,
        scale: true,
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
        let transform = Affine::translate(i as f32 * 225., j as f32 * 225.);
        let square = blend_square(blend.into());
        sb.append(&square, Some(transform));
    }
}

#[allow(unused)]
fn render_blend_square(sb: &mut SceneBuilder, blend: BlendMode, transform: Affine) {
    // Inspired by https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    let rect = Rect::from_origin_size(Point::new(0., 0.), 200., 200.);
    let stops = &[
        GradientStop {
            color: Color::rgb8(0, 0, 0),
            offset: 0.0,
        },
        GradientStop {
            color: Color::rgb8(255, 255, 255),
            offset: 1.0,
        },
    ][..];
    let linear = Brush::LinearGradient(LinearGradient {
        start: Point::new(0.0, 0.0),
        end: Point::new(200.0, 0.0),
        stops: stops.into(),
        extend: ExtendMode::Pad,
    });
    sb.fill(Fill::NonZero, transform, &linear, None, rect.elements());
    const GRADIENTS: &[(f32, f32, Color)] = &[
        (150., 0., Color::rgb8(255, 240, 64)),
        (175., 100., Color::rgb8(255, 96, 240)),
        (125., 200., Color::rgb8(64, 192, 255)),
    ];
    for (x, y, c) in GRADIENTS {
        let mut color2 = c.clone();
        color2.a = 0;
        let stops = &[
            GradientStop {
                color: c.clone(),
                offset: 0.0,
            },
            GradientStop {
                color: color2,
                offset: 1.0,
            },
        ][..];
        let rad = Brush::RadialGradient(RadialGradient {
            center0: Point::new(*x, *y),
            center1: Point::new(*x, *y),
            radius0: 0.0,
            radius1: 100.0,
            stops: stops.into(),
            extend: ExtendMode::Pad,
        });
        sb.fill(Fill::NonZero, transform, &rad, None, rect.elements());
    }
    const COLORS: &[Color] = &[
        Color::rgb8(255, 0, 0),
        Color::rgb8(0, 255, 0),
        Color::rgb8(0, 0, 255),
    ];
    sb.push_layer(Mix::Normal.into(), transform, rect.elements());
    for (i, c) in COLORS.iter().enumerate() {
        let stops = &[
            GradientStop {
                color: Color::rgb8(255, 255, 255),
                offset: 0.0,
            },
            GradientStop {
                color: c.clone(),
                offset: 1.0,
            },
        ][..];
        let linear = Brush::LinearGradient(LinearGradient {
            start: Point::new(0.0, 0.0),
            end: Point::new(0.0, 200.0),
            stops: stops.into(),
            extend: ExtendMode::Pad,
        });
        sb.push_layer(blend, transform, rect.elements());
        // squash the ellipse
        let a = transform
            * Affine::translate(100., 100.)
            * Affine::rotate(std::f32::consts::FRAC_PI_3 * (i * 2 + 1) as f32)
            * Affine::scale(1.0, 0.357)
            * Affine::translate(-100., -100.);
        sb.fill(
            Fill::NonZero,
            a,
            &linear,
            None,
            make_ellipse(100., 100., 90., 90.),
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

fn make_ellipse(cx: f32, cy: f32, rx: f32, ry: f32) -> impl Iterator<Item = PathElement> + Clone {
    let a = 0.551915024494;
    let arx = a * rx;
    let ary = a * ry;
    let elements = [
        PathElement::MoveTo(Point::new(cx + rx, cy)),
        PathElement::CurveTo(
            Point::new(cx + rx, cy + ary),
            Point::new(cx + arx, cy + ry),
            Point::new(cx, cy + ry),
        ),
        PathElement::CurveTo(
            Point::new(cx - arx, cy + ry),
            Point::new(cx - rx, cy + ary),
            Point::new(cx - rx, cy),
        ),
        PathElement::CurveTo(
            Point::new(cx - rx, cy - ary),
            Point::new(cx - arx, cy - ry),
            Point::new(cx, cy - ry),
        ),
        PathElement::CurveTo(
            Point::new(cx + arx, cy - ry),
            Point::new(cx + rx, cy - ary),
            Point::new(cx + rx, cy),
        ),
        PathElement::Close,
    ];
    (0..elements.len()).map(move |i| elements[i])
}
