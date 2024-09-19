// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Load rendering shaders.

#[cfg(feature = "wgpu")]
use wgpu::Device;

use crate::ShaderId;

#[cfg(feature = "wgpu")]
use crate::{
    recording::{BindType, ImageFormat},
    wgpu_engine::WgpuEngine,
    Error, RendererOptions,
};

// Shaders for the full pipeline
pub struct FullShaders {
    pub pathtag_scan_csdldf: ShaderId,
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
}

#[cfg(feature = "wgpu")]
pub(crate) fn full_shaders(
    device: &Device,
    engine: &mut WgpuEngine,
    options: &RendererOptions,
) -> Result<FullShaders, Error> {
    use crate::wgpu_engine::CpuShaderType;
    use BindType::*;

    let mut force_gpu = false;
    let force_gpu_from: Option<&str> = None;
    // Uncomment this to force use of GPU shaders from the specified shader and later even
    // if `engine.use_cpu` is specified.
    //let force_gpu_from = Some("binning");

    #[cfg(feature = "hot_reload")]
    let mut shaders = vello_shaders::compile::ShaderInfo::from_default()?;
    #[cfg(not(feature = "hot_reload"))]
    let shaders = vello_shaders::SHADERS;

    macro_rules! add_shader {
        ($name:ident, $label:expr, $bindings:expr, $cpu:expr) => {{
            if force_gpu_from == Some(stringify!($name)) {
                force_gpu = true;
            }
            #[cfg(feature = "hot_reload")]
            let source = shaders
                .remove(stringify!($name))
                .expect(stringify!($name))
                .source
                .into();
            #[cfg(not(feature = "hot_reload"))]
            let source = shaders.$name.wgsl.code;
            engine.add_compute_shader(
                device,
                concat!("vello.", $label),
                source,
                &$bindings,
                if force_gpu {
                    CpuShaderType::Missing
                } else {
                    $cpu
                },
            )
        }};
        ($name:ident, $bindings:expr, $cpu:expr) => {{
            add_shader!($name, stringify!($name), $bindings, $cpu)
        }};
        ($name:ident, $bindings:expr) => {
            add_shader!(
                $name,
                $bindings,
                CpuShaderType::Present(vello_shaders::cpu::$name)
            )
        };
    }

    let pathtag_scan_csdldf = add_shader!(
        pathtag_scan_csdldf,
        [Uniform, BufReadOnly, Buffer, Buffer, Buffer],
        CpuShaderType::Present(vello_shaders::cpu::pathtag_scan_single)
    );

    let bbox_clear = add_shader!(bbox_clear, [Uniform, Buffer]);
    let flatten = add_shader!(
        flatten,
        [Uniform, BufReadOnly, BufReadOnly, Buffer, Buffer, Buffer]
    );
    let draw_reduce = add_shader!(draw_reduce, [Uniform, BufReadOnly, Buffer]);
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
        ]
    );
    let clip_reduce = add_shader!(clip_reduce, [BufReadOnly, BufReadOnly, Buffer, Buffer]);
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
        ]
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
        ]
    );
    let tile_alloc = add_shader!(
        tile_alloc,
        [Uniform, BufReadOnly, BufReadOnly, Buffer, Buffer, Buffer]
    );
    let path_count_setup = add_shader!(path_count_setup, [Buffer, Buffer]);
    let path_count = add_shader!(
        path_count,
        [Uniform, Buffer, BufReadOnly, BufReadOnly, Buffer, Buffer]
    );
    let backdrop = add_shader!(
        backdrop_dyn,
        [Uniform, Buffer, BufReadOnly, Buffer],
        CpuShaderType::Present(vello_shaders::cpu::backdrop)
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
        ]
    );
    let path_tiling_setup = add_shader!(path_tiling_setup, [Buffer, Buffer, Buffer]);
    let path_tiling = add_shader!(
        path_tiling,
        [
            Buffer,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
        ]
    );
    let fine_resources = [
        BindType::Uniform,
        BindType::BufReadOnly,
        BindType::BufReadOnly,
        BindType::BufReadOnly,
        BindType::Buffer,
        BindType::Image(ImageFormat::Rgba8),
        BindType::ImageRead(ImageFormat::Rgba8),
        BindType::ImageRead(ImageFormat::Rgba8),
        // Mask LUT buffer, used only when MSAA is enabled.
        BindType::BufReadOnly,
    ];

    let aa_support = &options.antialiasing_support;
    let fine_area = if aa_support.area {
        Some(add_shader!(
            fine_area,
            fine_resources[..fine_resources.len() - 1],
            CpuShaderType::Missing
        ))
    } else {
        None
    };
    let fine_msaa8 = if aa_support.msaa8 {
        Some(add_shader!(
            fine_msaa8,
            fine_resources,
            CpuShaderType::Missing
        ))
    } else {
        None
    };
    let fine_msaa16 = if aa_support.msaa16 {
        Some(add_shader!(
            fine_msaa16,
            fine_resources,
            CpuShaderType::Missing
        ))
    } else {
        None
    };

    Ok(FullShaders {
        pathtag_scan_csdldf,
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
    })
}
