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
use crate::{wgpu_engine::WgpuEngine, RendererOptions};

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
    pub fine_area: Option<ShaderId>,
    pub fine_msaa8: Option<ShaderId>,
    pub fine_msaa16: Option<ShaderId>,
    // 2-level dispatch works for CPU pathtag scan even for large
    // inputs, 3-level is not yet implemented.
    pub pathtag_is_cpu: bool,
}

#[cfg(feature = "wgpu")]
pub fn full_shaders(
    device: &Device,
    engine: &mut WgpuEngine,
    options: &RendererOptions,
) -> Result<FullShaders, Error> {
    use crate::wgpu_engine::CpuShaderType;
    use BindType::*;
    let imports = SHARED_SHADERS
        .iter()
        .copied()
        .collect::<std::collections::HashMap<_, _>>();
    let empty = HashSet::new();
    let mut full_config = HashSet::new();
    full_config.insert("full".into());
    let mut small_config = HashSet::new();
    small_config.insert("full".into());
    small_config.insert("small".into());

    let mut force_gpu = false;

    let force_gpu_from: Option<&str> = None;

    // Uncomment this to force use of GPU shaders from the specified shader and later even
    // if `engine.use_cpu` is specified.
    //let force_gpu_from = Some("binning");

    macro_rules! add_shader {
        ($name:ident, $label:expr, $bindings:expr, $defines:expr, $cpu:expr) => {{
            if force_gpu_from == Some(stringify!($name)) {
                force_gpu = true;
            }
            engine.add_compute_shader(
                device,
                $label,
                preprocess::preprocess(shader!(stringify!($name)), &$defines, &imports).into(),
                &$bindings,
                if force_gpu {
                    CpuShaderType::Missing
                } else {
                    $cpu
                },
            )?
        }};
        ($name:ident, $bindings:expr, $defines:expr, $cpu:expr) => {{
            add_shader!($name, stringify!($name), $bindings, &$defines, $cpu)
        }};
        ($name:ident, $bindings:expr, $defines:expr) => {
            add_shader!(
                $name,
                $bindings,
                &$defines,
                CpuShaderType::Present(cpu_shader::$name)
            )
        };
        ($name:ident, $bindings:expr) => {
            add_shader!($name, $bindings, &full_config)
        };
    }

    let pathtag_reduce = add_shader!(pathtag_reduce, [Uniform, BufReadOnly, Buffer]);
    let pathtag_reduce2 = add_shader!(
        pathtag_reduce2,
        [BufReadOnly, Buffer],
        &full_config,
        CpuShaderType::Skipped
    );
    let pathtag_scan1 = add_shader!(
        pathtag_scan1,
        [BufReadOnly, BufReadOnly, Buffer],
        &full_config,
        CpuShaderType::Skipped
    );
    let pathtag_scan = add_shader!(
        pathtag_scan,
        [Uniform, BufReadOnly, BufReadOnly, Buffer],
        &small_config
    );
    let pathtag_scan_large = add_shader!(
        pathtag_scan,
        [Uniform, BufReadOnly, BufReadOnly, Buffer],
        &full_config,
        CpuShaderType::Skipped
    );
    let bbox_clear = add_shader!(bbox_clear, [Uniform, Buffer], &empty);
    let flatten = add_shader!(
        flatten,
        [Uniform, BufReadOnly, BufReadOnly, Buffer, Buffer, Buffer]
    );
    let draw_reduce = add_shader!(draw_reduce, [Uniform, BufReadOnly, Buffer], &empty);
    let draw_leaf = add_shader!(
        draw_leaf,
        [
            Uniform,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
            Buffer,
        ],
        &empty
    );
    let clip_reduce = add_shader!(
        clip_reduce,
        [BufReadOnly, BufReadOnly, Buffer, Buffer],
        &empty
    );
    let clip_leaf = add_shader!(
        clip_leaf,
        [
            Uniform,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
        ],
        &empty
    );
    let binning = add_shader!(
        binning,
        [
            Uniform,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
            Buffer,
            Buffer,
        ],
        &empty
    );
    let tile_alloc = add_shader!(
        tile_alloc,
        [Uniform, BufReadOnly, BufReadOnly, Buffer, Buffer, Buffer],
        &empty
    );
    let path_count_setup = add_shader!(path_count_setup, [Buffer, Buffer], &empty);
    let path_count = add_shader!(
        path_count,
        [Buffer, BufReadOnly, BufReadOnly, Buffer, Buffer]
    );
    let backdrop = add_shader!(
        backdrop_dyn,
        [Uniform, BufReadOnly, Buffer],
        &empty,
        CpuShaderType::Present(cpu_shader::backdrop)
    );
    let coarse = add_shader!(
        coarse,
        [
            Uniform,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
            Buffer,
        ],
        &empty
    );
    let path_tiling_setup = add_shader!(path_tiling_setup, [Buffer, Buffer], &empty);
    let path_tiling = add_shader!(
        path_tiling,
        [
            Buffer,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
        ],
        &empty
    );
    let fine_resources = [
        BindType::Uniform,
        BindType::BufReadOnly,
        BindType::BufReadOnly,
        BindType::BufReadOnly,
        BindType::Image(ImageFormat::Rgba8),
        BindType::ImageRead(ImageFormat::Rgba8),
        BindType::ImageRead(ImageFormat::Rgba8),
        // Mask LUT buffer, used only when MSAA is enabled.
        BindType::BufReadOnly,
    ];
    let [fine_area, fine_msaa8, fine_msaa16] = {
        let aa_support = &options.antialiasing_support;
        let aa_modes = [
            (aa_support.area, 1, "fine_area", None),
            (aa_support.msaa8, 0, "fine_msaa8", Some("msaa8")),
            (aa_support.msaa16, 0, "fine_msaa16", Some("msaa16")),
        ];
        let mut pipelines = [None, None, None];
        for (i, (enabled, offset, label, aa_config)) in aa_modes.iter().enumerate() {
            if !enabled {
                continue;
            }
            let mut config = full_config.clone();
            if let Some(aa_config) = *aa_config {
                config.insert("msaa".into());
                config.insert(aa_config.into());
            }
            pipelines[i] = Some(add_shader!(
                fine,
                label,
                fine_resources[..fine_resources.len() - offset],
                config,
                CpuShaderType::Missing
            ));
        }
        pipelines
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
        fine_area,
        fine_msaa8,
        fine_msaa16,
        pathtag_is_cpu: options.use_cpu,
    })
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
