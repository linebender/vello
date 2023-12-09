// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{
    engine::{BindType, ImageProxy, Recording, ResourceProxy, ShaderId},
    render::CapturedBuffers,
    wgpu_engine::WgpuEngine,
    RenderParams,
};
use {
    bytemuck::{offset_of, Pod, Zeroable},
    vello_encoding::{BumpAllocators, LineSoup, PathBbox},
};

pub(crate) struct DebugLayers {
    bboxes: ShaderId,
    linesoup: ShaderId,
}

impl DebugLayers {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        engine: &mut WgpuEngine,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("debug layers"),
            source: wgpu::ShaderSource::Wgsl(SHADERS.into()),
        });
        let bboxes = engine.add_render_shader(
            device,
            "bbox-debug",
            &module,
            "bbox_vert",
            "fs_main",
            wgpu::PrimitiveTopology::LineStrip,
            wgpu::ColorTargetState {
                format: target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            },
            // This mirrors the layout of the PathBbox structure.
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<PathBbox>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Sint32x2,
                        offset: offset_of!(PathBbox, x0) as u64,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Sint32x2,
                        offset: offset_of!(PathBbox, x1) as u64,
                        shader_location: 1,
                    },
                ],
            }),
            &[(BindType::Uniform, wgpu::ShaderStages::VERTEX)],
        );
        let linesoup = engine.add_render_shader(
            device,
            "linesoup-debug",
            &module,
            "linesoup_vert",
            "fs_main",
            wgpu::PrimitiveTopology::LineList,
            wgpu::ColorTargetState {
                format: target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            },
            // This mirrors the layout of the LineSoup structure.
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<LineSoup>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: offset_of!(LineSoup, p0) as u64,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: offset_of!(LineSoup, p1) as u64,
                        shader_location: 1,
                    },
                ],
            }),
            &[(BindType::Uniform, wgpu::ShaderStages::VERTEX)],
        );

        Self { bboxes, linesoup }
    }

    pub fn render(
        &self,
        recording: &mut Recording,
        target: ImageProxy,
        captured: &CapturedBuffers,
        bump: Option<&BumpAllocators>,
        params: &RenderParams,
    ) {
        let uniforms = Uniforms {
            width: params.width,
            height: params.height,
        };
        let uniforms_buf =
            ResourceProxy::Buf(recording.upload_uniform("uniforms", bytemuck::bytes_of(&uniforms)));
        recording.draw(
            self.bboxes,
            captured.sizes.path_bboxes.len(),
            5,
            Some(captured.path_bboxes),
            [uniforms_buf],
            target,
            None,
        );
        recording.draw(
            self.linesoup,
            bump.unwrap().lines,
            2,
            Some(captured.lines),
            [uniforms_buf],
            target,
            None,
        );

        recording.free_resource(uniforms_buf);
    }
}

#[derive(Copy, Clone, Zeroable, Pod)]
#[repr(C)]
struct Uniforms {
    width: u32,
    height: u32,
}

const SHADERS: &str = r#"

alias QuadVertices = array<vec2f, 4>;
var<private> quad_vertices: QuadVertices = QuadVertices(
    vec2<f32>(0., 1.),
    vec2<f32>(0., 0.),
    vec2<f32>(1., 0.),
    vec2<f32>(1., 1.),
);

struct Uniforms {
    width: u32,
    height: u32,
}
@binding(0) @group(0) var<uniform> uniforms: Uniforms;

struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec3f,
}

struct BboxIn {
	@location(0) p0: vec2i,
	@location(1) p1: vec2i,
}

@vertex
fn bbox_vert(@builtin(vertex_index) vid: u32, bbox: BboxIn) -> VSOut {
    let ul = vec2f(f32(bbox.p0.x), f32(bbox.p0.y));
    let br = vec2f(f32(bbox.p1.x), f32(bbox.p1.y));
    let dim = br - ul;
    let p = (ul + dim * quad_vertices[vid % 4u]) / vec2f(f32(uniforms.width), f32(uniforms.height));

    // Map from y-down viewport coordinates to NDC:
    return VSOut(vec4(vec2(1., -1.) * (2. * p - vec2f(1.f)), 0., 1.), vec3f(0., 1., 0.));;
}

struct LinesoupIn {
    @location(0) p0: vec2f,
    @location(1) p1: vec2f,
}

@vertex
fn linesoup_vert(@builtin(vertex_index) vid: u32, line: LinesoupIn) -> VSOut {
    let p = select(line.p0, line.p1, vid == 1u) / vec2f(f32(uniforms.width), f32(uniforms.height));

    // Map from y-down viewport coordinates to NDC:
    return VSOut(vec4(vec2(1., -1.) * (2. * p - vec2f(1.f)), 0., 1.), vec3f(1., 1., 0.));
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
    return vec4(in.color, 1.);
}

"#;
