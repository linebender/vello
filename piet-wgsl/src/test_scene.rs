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

use piet_scene::{Affine, Brush, Color, Fill, PathElement, Point, Scene, SceneBuilder, Stroke};

pub fn gen_test_scene() -> Scene {
    let mut scene = Scene::default();
    let mut builder = SceneBuilder::for_scene(&mut scene);
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
    let style = Stroke {
        width: 1.0,
        join: piet_scene::Join::Round,
        miter_limit: 1.4,
        start_cap: piet_scene::Cap::Round,
        end_cap: piet_scene::Cap::Round,
        dash_pattern: [],
        dash_offset: 0.0,
        scale: true,
    };
    let brush = Brush::Solid(Color::rgb8(0xa0, 0x00, 0x00));
    builder.stroke(&style, transform, &brush, None, &path);
    scene
}

pub fn dump_scene_info(scene: &Scene) {
    let data = scene.data();
    println!("tags {:?}", data.tag_stream);
    println!(
        "pathsegs {:?}",
        bytemuck::cast_slice::<u8, f32>(&data.pathseg_stream)
    );
}
