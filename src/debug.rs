// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

mod validate;

use crate::{
    debug::validate::{validate_line_soup, LineEndpoint},
    engine::{BindType, ImageProxy, Recording, ResourceProxy, ShaderId},
    render::CapturedBuffers,
    wgpu_engine::WgpuEngine,
    DebugDownloads, RenderParams,
};
use {
    bytemuck::{offset_of, Pod, Zeroable},
    peniko::Color,
    vello_encoding::{BumpAllocators, LineSoup, PathBbox},
};

#[derive(Copy, Clone)]
pub struct DebugLayers(u8);

// TODO: Currently all layers require read-back of the BumpAllocators buffer. This isn't strictly
// necessary for layers other than `VALIDATION`. The debug visualizations use the bump buffer only
// to obtain various instance counts for draws and these could instead get written out to an
// indirect draw buffer. OTOH `VALIDATION` should always require readback since we want to be able
// to run the same CPU-side tests for both CPU and GPU shaders.
impl DebugLayers {
    /// Visualize the bounding box of every path.
    pub const BOUNDING_BOXES: DebugLayers = DebugLayers(1 << 0);

    /// Visualize the post-flattening line segments using line primitives.
    pub const LINESOUP_SEGMENTS: DebugLayers = DebugLayers(1 << 1);

    /// Visualize the post-flattening line endpoints.
    pub const LINESOUP_POINTS: DebugLayers = DebugLayers(1 << 2);

    /// Enable validation of internal buffer contents and visualize errors. Validation tests are
    /// run on the CPU and require buffer contents to be read-back.
    ///
    /// Supported validation tests:
    ///
    ///    - Watertightness: validate that every line segment within a path is connected without
    ///      any gaps. Line endpoints that don't precisely overlap another endpoint get visualized
    ///      as red circles and logged to stderr.
    ///
    pub const VALIDATION: DebugLayers = DebugLayers(1 << 3);

    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    pub const fn none() -> Self {
        Self(0)
    }

    pub const fn all() -> Self {
        Self(
            Self::BOUNDING_BOXES.0
                | Self::LINESOUP_SEGMENTS.0
                | Self::LINESOUP_POINTS.0
                | Self::VALIDATION.0,
        )
    }

    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub fn check_bits(&self, mask: DebugLayers) -> bool {
        self.0 & mask.0 == mask.0
    }

    pub fn toggle(&mut self, mask: DebugLayers) {
        self.0 ^= mask.0
    }
}

impl std::ops::BitOr for DebugLayers {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

pub(crate) struct DebugRenderer {
    // `clear_tint` slightly darkens the output from the vello renderer to make the debug overlays
    // more distinguishable.
    clear_tint: ShaderId,
    bboxes: ShaderId,
    linesoup: ShaderId,
    linesoup_points: ShaderId,
    unpaired_points: ShaderId,
}

impl DebugRenderer {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        engine: &mut WgpuEngine,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("debug layers"),
            source: wgpu::ShaderSource::Wgsl(SHADERS.into()),
        });

        let clear_tint = engine.add_render_shader(
            device,
            "clear-tint",
            &module,
            "full_screen_quad_vert",
            "solid_color_frag",
            wgpu::PrimitiveTopology::TriangleStrip,
            wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent::OVER,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            },
            None,
            &[],
        );
        let bboxes = engine.add_render_shader(
            device,
            "bbox-debug",
            &module,
            "bbox_vert",
            "solid_color_frag",
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
            "solid_color_frag",
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
        let linesoup_points = engine.add_render_shader(
            device,
            "linepoints-debug",
            &module,
            "linepoints_vert",
            "sdf_circle_frag",
            wgpu::PrimitiveTopology::TriangleStrip,
            wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent::OVER,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            },
            // This mirrors the layout of the LineSoup structure. The pipeline only processes the
            // first point of each line. Since all points should be paired, this is enough to
            // render all points. All unpaired points alone get drawn by the `unpaired_points`
            // pipeline, so no point should get missed.
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<LineSoup>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: offset_of!(LineSoup, p0) as u64,
                    shader_location: 0,
                }],
            }),
            &[
                (BindType::Uniform, wgpu::ShaderStages::VERTEX),
                (
                    BindType::Uniform,
                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ),
            ],
        );
        let unpaired_points = engine.add_render_shader(
            device,
            "linepoints-debug",
            &module,
            "linepoints_vert",
            "sdf_circle_frag",
            wgpu::PrimitiveTopology::TriangleStrip,
            wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent::OVER,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            },
            // This mirrors the layout of the LineSoup structure.
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<LineEndpoint>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: offset_of!(LineEndpoint, x) as u64,
                    shader_location: 0,
                }],
            }),
            &[
                (BindType::Uniform, wgpu::ShaderStages::VERTEX),
                (
                    BindType::Uniform,
                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ),
            ],
        );

        Self {
            clear_tint,
            bboxes,
            linesoup,
            linesoup_points,
            unpaired_points,
        }
    }

    pub fn render(
        &self,
        recording: &mut Recording,
        target: ImageProxy,
        captured: &CapturedBuffers,
        bump: &BumpAllocators,
        params: &RenderParams,
        downloads: &DebugDownloads,
    ) {
        if params.debug.is_empty() {
            return;
        }

        let (unpaired_pts_len, unpaired_pts_buf) =
            if params.debug.check_bits(DebugLayers::VALIDATION) {
                // TODO: have this write directly to a GPU buffer?
                let unpaired_pts: Vec<LineEndpoint> =
                    validate_line_soup(bytemuck::cast_slice(&downloads.lines.get_mapped_range()));
                if unpaired_pts.is_empty() {
                    (0, None)
                } else {
                    (
                        unpaired_pts.len(),
                        Some(
                            recording
                                .upload("unpaired points", bytemuck::cast_slice(&unpaired_pts[..])),
                        ),
                    )
                }
            } else {
                (0, None)
            };

        let uniforms = Uniforms {
            width: params.width,
            height: params.height,
        };
        let uniforms_buf =
            ResourceProxy::Buf(recording.upload_uniform("uniforms", bytemuck::bytes_of(&uniforms)));

        let linepoints_uniforms = [
            LinepointsUniforms::new(Color::CYAN, 10.),
            LinepointsUniforms::new(Color::RED, 80.),
        ];
        let linepoints_uniforms_buf = recording.upload_uniform(
            "linepoints uniforms",
            bytemuck::bytes_of(&linepoints_uniforms),
        );

        recording.draw::<[ResourceProxy; 0]>(self.clear_tint, 1, 4, None, [], target, None);
        if params.debug.check_bits(DebugLayers::BOUNDING_BOXES) {
            recording.draw(
                self.bboxes,
                captured.sizes.path_bboxes.len(),
                5,
                Some(captured.path_bboxes),
                [uniforms_buf],
                target,
                None,
            );
        }
        if params.debug.check_bits(DebugLayers::LINESOUP_SEGMENTS) {
            recording.draw(
                self.linesoup,
                bump.lines,
                2,
                Some(captured.lines),
                [uniforms_buf],
                target,
                None,
            );
        }
        if params.debug.check_bits(DebugLayers::LINESOUP_POINTS) {
            recording.draw(
                self.linesoup_points,
                bump.lines,
                4,
                Some(captured.lines),
                [
                    uniforms_buf,
                    ResourceProxy::BufRange {
                        proxy: linepoints_uniforms_buf,
                        offset: 0,
                        size: std::mem::size_of::<LinepointsUniforms>() as u64,
                    },
                ],
                target,
                None,
            );
        }
        if let Some(unpaired_pts_buf) = unpaired_pts_buf {
            recording.draw(
                self.unpaired_points,
                unpaired_pts_len.try_into().unwrap(),
                4,
                Some(unpaired_pts_buf),
                [
                    uniforms_buf,
                    ResourceProxy::BufRange {
                        proxy: linepoints_uniforms_buf,
                        offset: std::mem::size_of::<LinepointsUniforms>() as u64,
                        size: std::mem::size_of::<LinepointsUniforms>() as u64,
                    },
                ],
                target,
                None,
            );
            recording.free_buf(unpaired_pts_buf);
        }

        recording.free_resource(uniforms_buf);
        recording.free_buf(linepoints_uniforms_buf);
    }
}

#[derive(Copy, Clone, Zeroable, Pod)]
#[repr(C)]
struct Uniforms {
    width: u32,
    height: u32,
}

#[derive(Copy, Clone, Zeroable, Pod)]
#[repr(C)]
struct LinepointsUniforms {
    point_color: [f32; 3],
    point_size: f32,
    // Uniform parameters for individual SDF point draws are stored in a single buffer.
    // This 240 byte padding is here to bring the element ffset alignment of 256 bytes.
    // (see https://www.w3.org/TR/webgpu/#dom-supported-limits-minuniformbufferoffsetalignment)
    _pad0: [u32; 30],
    _pad1: [u32; 30],
}

impl LinepointsUniforms {
    fn new(color: Color, point_size: f32) -> Self {
        Self {
            point_color: [
                color.r as f32 / 255.,
                color.g as f32 / 255.,
                color.b as f32 / 255.,
            ],
            point_size,
            _pad0: [0; 30],
            _pad1: [0; 30],
        }
    }
}

const SHADERS: &str = r#"

// Map from y-down normalized coordinates to NDC:
fn map_to_ndc(p: vec2f) -> vec4f {
    return vec4(vec2(1., -1.) * (2. * p - vec2(1.)), 0., 1.);
}

alias QuadVertices = array<vec2f, 4>;
var<private> quad_vertices: QuadVertices = QuadVertices(
    vec2<f32>(0., 1.),
    vec2<f32>(0., 0.),
    vec2<f32>(1., 0.),
    vec2<f32>(1., 1.),
);

var<private> quad_fill_indices: array<u32, 4> = array<u32, 4>(0u, 3u, 1u, 2u);

struct Uniforms {
    width: u32,
    height: u32,
}
@binding(0) @group(0) var<uniform> uniforms: Uniforms;

struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
}

////////////

@vertex
fn full_screen_quad_vert(@builtin(vertex_index) vid: u32) -> VSOut {
    let p = quad_vertices[quad_fill_indices[vid]];
    return VSOut(map_to_ndc(p), vec4(0., 0., 0., 0.5));
}

////////////

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
    return VSOut(map_to_ndc(p), vec4(0., 1., 0., 1.));
}

////////////

struct LinesoupIn {
    @location(0) p0: vec2f,
    @location(1) p1: vec2f,
}

@vertex
fn linesoup_vert(@builtin(vertex_index) vid: u32, line: LinesoupIn) -> VSOut {
    let p = select(line.p0, line.p1, vid == 1u) / vec2f(f32(uniforms.width), f32(uniforms.height));
    return VSOut(map_to_ndc(p), vec4(0.7, 0.5, 0., 1.));
}

////////////

struct LinepointsUniforms {
    point_color: vec3f,
    point_size: f32,
}
@binding(1) @group(0) var<uniform> linepoints_uniforms: LinepointsUniforms;

struct SDFCircleOut {
    @builtin(position) pos: vec4f,

    // Unpremultiplied color of the circle.
    @location(0) color: vec3f,

    // The 2D position of the pixel fragment relative to the center of the quad. The quad edges
    // are at coordinates (±1, 0) and (0, ±1).
    @location(1) quad_relative: vec2f,
}

@vertex
fn linepoints_vert(@builtin(vertex_index) vid: u32, @location(0) point: vec2f) -> SDFCircleOut {
    let quad_corner = quad_vertices[quad_fill_indices[vid]] - vec2(0.5);
    let rect_dim = vec2(linepoints_uniforms.point_size);
    let p = (point + rect_dim * quad_corner) / vec2(f32(uniforms.width), f32(uniforms.height));

    return SDFCircleOut(
        map_to_ndc(p),
        linepoints_uniforms.point_color,
        // Normalize the corners of the quad such that they form a vector of length √2. This should
        // align the edge fragments to ±1. The post-interpolation values of `quad_relative` will
        // then form a distance field that can represent a circle of radius 1 within the quad
        // (where the distance is relative to the center of the circle).
        normalize(quad_corner) * sqrt(2.),
    );
}

@fragment
fn solid_color_frag(in: VSOut) -> @location(0) vec4f {
    return in.color;
}

@fragment
fn sdf_circle_frag(in: SDFCircleOut) -> @location(0) vec4f {
    // Draw an antialiased circle with a fading margin as a visual effect. `THRESHOLD` is the
    // distance from the center of the circle to the edge where the fade begins.
    let THRESHOLD = 0.6;
    let d = saturate(length(in.quad_relative));
    let alpha = select(1., 1. - smoothstep(THRESHOLD, 1., d), d > THRESHOLD);
    return vec4(in.color.rgb, alpha);
}
"#;
