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

mod preprocess;

use serde_json::json;
use wgpu::Device;

use crate::{
    engine::{BindType, Engine, Error, ShaderId},
    template::ShaderTemplate,
};

pub const PATHTAG_REDUCE_WG: u32 = 256;
pub const PATH_COARSE_WG: u32 = 256;

pub struct Shaders {
    pub pathtag_reduce: ShaderId,
    pub pathtag_scan: ShaderId,
    pub path_coarse: ShaderId,
    pub backdrop: ShaderId,
    pub fine: ShaderId,
}

pub fn init_shaders(device: &Device, engine: &mut Engine) -> Result<Shaders, Error> {
    let shaders = ShaderTemplate::new();
    let pathtag_reduce = engine.add_shader(
        device,
        shaders.get_shader("pathtag_reduce", &()).into(),
        &[BindType::BufReadOnly, BindType::Buffer],
    )?;
    let pathtag_scan = engine.add_shader(
        device,
        shaders.get_shader("pathtag_scan", &()).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
        ],
    )?;
    let path_coarse_config = json!({"cubics_out": false});
    let path_coarse = engine.add_shader(
        device,
        shaders
            .get_shader("path_coarse", &path_coarse_config)
            .into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;
    let backdrop = engine.add_shader(
        device,
        shaders.get_shader("backdrop", &()).into(),
        &[BindType::BufReadOnly, BindType::Buffer],
    )?;
    let fine = engine.add_shader(
        device,
        shaders.get_shader("fine", &()).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
        ],
    )?;
    Ok(Shaders {
        pathtag_reduce,
        pathtag_scan,
        path_coarse,
        backdrop,
        fine,
    })
}
