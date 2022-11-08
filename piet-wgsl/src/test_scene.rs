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

use piet_scene::{Affine, Brush, Color, Fill, PathElement, Point, Scene, SceneBuilder};

pub fn gen_test_scene() -> Scene {
    let mut scene = Scene::default();
    let mut builder = SceneBuilder::for_scene(&mut scene);
    let path = [
        PathElement::MoveTo(Point::new(100.0, 100.0)),
        PathElement::LineTo(Point::new(1000.0, 240.0)),
        PathElement::LineTo(Point::new(600.0, 300.0)),
        PathElement::LineTo(Point::new(400.0, 520.0)),
        PathElement::LineTo(Point::new(300.0, 420.0)),
        PathElement::Close,
    ];
    let brush = Brush::Solid(Color::rgb8(0x80, 0x80, 0x80));
    builder.fill(Fill::NonZero, Affine::IDENTITY, &brush, None, &path);
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
