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

//! Load rendering shaders.

use std::{fs, path::Path};

use wgpu::Device;

use crate::engine::{BindType, Engine, ShaderId};

pub fn reduced_shader(device: &Device, engine: &mut Engine) -> ShaderId {
    let shader_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/shader"));
    let read_shader =
        |path: &str| fs::read_to_string(shader_dir.join(path.to_string() + ".wgsl")).unwrap();
    engine.add_shader(
        device,
        read_shader("tile_alloc").into(),
        &[
            BindType::Buffer,
            BindType::Buffer,
        ],
    ).unwrap()
}
