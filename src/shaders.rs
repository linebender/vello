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

use std::collections::HashSet;

#[cfg(feature = "wgpu")]
use wgpu::Device;

use crate::{
    cpu_shader,
    engine::{BindType, Error, ImageFormat, ShaderId},
};

#[cfg(feature = "wgpu")]
use crate::wgpu_engine::WgpuEngine;

macro_rules! shader {
    ($name:expr) => {&{
        let shader = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shader/",
            $name,
            ".wgsl"
        ));
        #[cfg(feature = "hot_reload")]
        let shader = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shader/",
            $name,
            ".wgsl"
        ))
        .unwrap_or_else(|e| {
            eprintln!(
                "Failed to read shader {name}, error falling back to version at compilation time. Error: {e:?}",
                name = $name
            );
            shader.to_string()
        });
        shader
    }};
}

// Shaders for the full pipeline
pub struct FullShaders {
    pub pathtag_reduce: ShaderId,
    pub pathtag_reduce2: ShaderId,
    pub pathtag_scan1: ShaderId,
    pub pathtag_scan: ShaderId,
    pub pathtag_scan_large: ShaderId,
    pub bbox_clear: ShaderId,
    pub flatten: ShaderId,
    pub draw_reduce: ShaderId,
    pub draw_leaf: ShaderId,
    pub clip_reduce: ShaderId,
    pub clip_leaf: ShaderId,
    pub binning: ShaderId,
    pub tile_alloc: ShaderId,
    pub backdrop: ShaderId,
    pub path_count_setup: ShaderId,
    pub path_count: ShaderId,
    pub coarse: ShaderId,
    pub path_tiling_setup: ShaderId,
    pub path_tiling: ShaderId,
    pub fine: ShaderId,
    // 2-level dispatch works for CPU pathtag scan even for large
    // inputs, 3-level is not yet implemented.
    pub pathtag_is_cpu: bool,
}

#[cfg(feature = "wgpu")]
pub fn full_shaders(device: &Device, engine: &mut WgpuEngine) -> Result<FullShaders, Error> {
    use crate::ANTIALIASING;

    let imports = SHARED_SHADERS
        .iter()
        .copied()
        .collect::<std::collections::HashMap<_, _>>();
    let empty = HashSet::new();
    let mut full_config = HashSet::new();
    full_config.insert("full".into());
    match crate::ANTIALIASING {
        crate::AaConfig::Msaa16 => {
            full_config.insert("msaa".into());
            full_config.insert("msaa16".into());
        }
        crate::AaConfig::Msaa8 => {
            full_config.insert("msaa".into());
            full_config.insert("msaa8".into());
        }
        crate::AaConfig::Area => (),
    }
    let mut small_config = HashSet::new();
    small_config.insert("full".into());
    small_config.insert("small".into());
    let pathtag_reduce = engine.add_shader(
        device,
        "pathtag_reduce",
        preprocess::preprocess(shader!("pathtag_reduce"), &full_config, &imports).into(),
        &[BindType::Uniform, BindType::BufReadOnly, BindType::Buffer],
    )?;
    let pathtag_reduce2 = engine.add_shader(
        device,
        "pathtag_reduce2",
        preprocess::preprocess(shader!("pathtag_reduce2"), &full_config, &imports).into(),
        &[BindType::BufReadOnly, BindType::Buffer],
    )?;
    let pathtag_scan1 = engine.add_shader(
        device,
        "pathtag_scan1",
        preprocess::preprocess(shader!("pathtag_scan1"), &full_config, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
        ],
    )?;
    let pathtag_scan = engine.add_shader(
        device,
        "pathtag_scan",
        preprocess::preprocess(shader!("pathtag_scan"), &small_config, &imports).into(),
        &[
            BindType::Uniform,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
        ],
    )?;
    let pathtag_scan_large = engine.add_shader(
        device,
        "pathtag_scan",
        preprocess::preprocess(shader!("pathtag_scan"), &full_config, &imports).into(),
        &[
            BindType::Uniform,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
        ],
    )?;
    let bbox_clear = engine.add_shader(
        device,
        "bbox_clear",
        preprocess::preprocess(shader!("bbox_clear"), &empty, &imports).into(),
        &[BindType::Uniform, BindType::Buffer],
    )?;
    let flatten = engine.add_shader(
        device,
        "flatten",
        preprocess::preprocess(shader!("flatten"), &full_config, &imports).into(),
        &[
            BindType::Uniform,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;
    let draw_reduce = engine.add_shader(
        device,
        "draw_reduce",
        preprocess::preprocess(shader!("draw_reduce"), &empty, &imports).into(),
        &[BindType::Uniform, BindType::BufReadOnly, BindType::Buffer],
    )?;
    let draw_leaf = engine.add_shader(
        device,
        "draw_leaf",
        preprocess::preprocess(shader!("draw_leaf"), &empty, &imports).into(),
        &[
            BindType::Uniform,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;
    let clip_reduce = engine.add_shader(
        device,
        "clip_reduce",
        preprocess::preprocess(shader!("clip_reduce"), &empty, &imports).into(),
        &[
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;
    let clip_leaf = engine.add_shader(
        device,
        "clip_leaf",
        preprocess::preprocess(shader!("clip_leaf"), &empty, &imports).into(),
        &[
            BindType::Uniform,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;
    let binning = engine.add_shader(
        device,
        "binning",
        preprocess::preprocess(shader!("binning"), &empty, &imports).into(),
        &[
            BindType::Uniform,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;
    let tile_alloc = engine.add_shader(
        device,
        "tile_alloc",
        preprocess::preprocess(shader!("tile_alloc"), &empty, &imports).into(),
        &[
            BindType::Uniform,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;
    let path_count_setup = engine.add_shader(
        device,
        "path_count_setup",
        preprocess::preprocess(shader!("path_count_setup"), &empty, &imports).into(),
        &[BindType::Buffer, BindType::Buffer],
    )?;
    let path_count = engine.add_shader(
        device,
        "path_count",
        preprocess::preprocess(shader!("path_count"), &full_config, &imports).into(),
        &[
            BindType::Uniform,
            BindType::Buffer,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;
    let backdrop = engine.add_shader(
        device,
        "backdrop_dyn",
        preprocess::preprocess(shader!("backdrop_dyn"), &empty, &imports).into(),
        &[BindType::Uniform, BindType::BufReadOnly, BindType::Buffer],
    )?;
    let coarse = engine.add_shader(
        device,
        "coarse",
        preprocess::preprocess(shader!("coarse"), &empty, &imports).into(),
        &[
            BindType::Uniform,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;
    let path_tiling_setup = engine.add_shader(
        device,
        "path_tiling_setup",
        preprocess::preprocess(shader!("path_tiling_setup"), &empty, &imports).into(),
        &[BindType::Buffer, BindType::Buffer],
    )?;
    let path_tiling = engine.add_shader(
        device,
        "path_tiling",
        preprocess::preprocess(shader!("path_tiling"), &empty, &imports).into(),
        &[
            BindType::Buffer,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
        ],
    )?;
    let fine = match ANTIALIASING {
        crate::AaConfig::Area => engine.add_shader(
            device,
            "fine",
            preprocess::preprocess(shader!("fine"), &full_config, &imports).into(),
            &[
                BindType::Uniform,
                BindType::BufReadOnly,
                BindType::BufReadOnly,
                BindType::BufReadOnly,
                BindType::Image(ImageFormat::Rgba8),
                BindType::ImageRead(ImageFormat::Rgba8),
                BindType::ImageRead(ImageFormat::Rgba8),
            ],
        )?,
        _ => {
            engine.add_shader(
                device,
                "fine",
                preprocess::preprocess(shader!("fine"), &full_config, &imports).into(),
                &[
                    BindType::Uniform,
                    BindType::BufReadOnly,
                    BindType::BufReadOnly,
                    BindType::BufReadOnly,
                    BindType::Image(ImageFormat::Rgba8),
                    BindType::ImageRead(ImageFormat::Rgba8),
                    BindType::ImageRead(ImageFormat::Rgba8),
                    BindType::BufReadOnly, // mask buffer
                ],
            )?
        }
    };
    Ok(FullShaders {
        pathtag_reduce,
        pathtag_reduce2,
        pathtag_scan,
        pathtag_scan1,
        pathtag_scan_large,
        bbox_clear,
        flatten,
        draw_reduce,
        draw_leaf,
        clip_reduce,
        clip_leaf,
        binning,
        tile_alloc,
        path_count_setup,
        path_count,
        backdrop,
        coarse,
        path_tiling_setup,
        path_tiling,
        fine,
        pathtag_is_cpu: false,
    })
}

#[cfg(feature = "wgpu")]
impl FullShaders {
    /// Install the CPU shaders.
    ///
    /// There are a couple things to note here. The granularity provided by
    /// this method is coarse; it installs all the shaders. There are many
    /// use cases (including debugging), where a mix is desired, or the
    /// choice between GPU and CPU dispatch might be dynamic.
    ///
    /// Second, the actual mapping to CPU shaders is not really specific to
    /// the engine, and should be split out into a back-end agnostic struct.
    pub fn install_cpu_shaders(&mut self, engine: &mut WgpuEngine) {
        engine.set_cpu_shader(self.pathtag_reduce, cpu_shader::pathtag_reduce);
        engine.set_cpu_shader(self.pathtag_scan, cpu_shader::pathtag_scan);
        engine.set_cpu_shader(self.bbox_clear, cpu_shader::bbox_clear);
        engine.set_cpu_shader(self.flatten, cpu_shader::flatten);
        engine.set_cpu_shader(self.draw_reduce, cpu_shader::draw_reduce);
        engine.set_cpu_shader(self.draw_leaf, cpu_shader::draw_leaf);
        engine.set_cpu_shader(self.clip_reduce, cpu_shader::clip_reduce);
        engine.set_cpu_shader(self.clip_leaf, cpu_shader::clip_leaf);
        engine.set_cpu_shader(self.binning, cpu_shader::binning);
        engine.set_cpu_shader(self.tile_alloc, cpu_shader::tile_alloc);
        engine.set_cpu_shader(self.path_count_setup, cpu_shader::path_count_setup);
        engine.set_cpu_shader(self.path_count, cpu_shader::path_count);
        engine.set_cpu_shader(self.backdrop, cpu_shader::backdrop);
        engine.set_cpu_shader(self.coarse, cpu_shader::coarse);
        engine.set_cpu_shader(self.path_tiling_setup, cpu_shader::path_tiling_setup);
        engine.set_cpu_shader(self.path_tiling, cpu_shader::path_tiling);
        self.pathtag_is_cpu = true;
    }
}

macro_rules! shared_shader {
    ($name:expr) => {
        (
            $name,
            include_str!(concat!("../shader/shared/", $name, ".wgsl")),
        )
    };
}

const SHARED_SHADERS: &[(&str, &str)] = &[
    shared_shader!("bbox"),
    shared_shader!("blend"),
    shared_shader!("bump"),
    shared_shader!("clip"),
    shared_shader!("config"),
    shared_shader!("cubic"),
    shared_shader!("drawtag"),
    shared_shader!("pathtag"),
    shared_shader!("ptcl"),
    shared_shader!("segment"),
    shared_shader!("tile"),
    shared_shader!("transform"),
    shared_shader!("util"),
];
