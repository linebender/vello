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
use piet_scene::{Affine, Brush, Color, Fill, LinearGradient, PathElement, Point, Scene, SceneBuilder, Stroke, GradientStop};

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
            let path = [
                PathElement::MoveTo(Point::new(100.0, 100.0)),
                PathElement::LineTo(Point::new(300.0, 100.0)),
                PathElement::LineTo(Point::new(300.0, 300.0)),
                PathElement::LineTo(Point::new(100.0, 300.0)),
                PathElement::Close,
            ];
            let gradient = Brush::LinearGradient(LinearGradient {
                start: Point::new(100.0, 100.0),
                end: Point::new(300.0, 300.0),
                extend: piet_scene::ExtendMode::Pad,
                stops: vec![
                    GradientStop {
                        offset: 0.0,
                        color: Color::rgb8(255, 0, 0),
                    },
                    GradientStop {
                        offset: 0.5,
                        color: Color::rgb8(0, 255, 0),
                    },
                    GradientStop {
                        offset: 1.0,
                        color: Color::rgb8(0, 0, 255),
                    },
                ].into()
            });
            builder.fill(Fill::NonZero, Affine::scale(3.0, 3.0), &gradient, None, &path);
        }
        _ => {
            let xml_str =
                std::str::from_utf8(include_bytes!("../../piet-gpu/Ghostscript_Tiger.svg"))
                    .unwrap();
            let svg = PicoSvg::load(xml_str, 6.0).unwrap();
            render_svg(&mut builder, &svg, false);
        }
    }
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
