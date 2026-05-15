// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Headless symbol rendering example.

#![allow(
    clippy::cast_possible_truncation,
    clippy::allow_attributes_without_reason
)]

use std::fs::{File, create_dir_all};
#[cfg(target_os = "macos")]
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use vello::kurbo::{
    Affine, BezPath, Cap, Circle, Join, Line, Point, Rect, RoundedRect, Shape, Stroke, Vec2,
};
use vello::peniko::{Color, ColorStop, Extend, Fill, Gradient};
use vello::util::{RenderContext, block_on_wgpu};
use vello::wgpu::{
    self, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, TexelCopyBufferInfo,
    TextureDescriptor, TextureFormat, TextureUsages,
};
use vello::{AaConfig, AaSupport, RenderParams, Renderer, RendererOptions, Scene};

const MAP_WIDTH: f64 = 1200.0;
const MAP_HEIGHT: f64 = 820.0;

#[derive(Parser, Debug)]
#[command(about, long_about = None, bin_name = "cargo run -p symbols --")]
struct Args {
    /// Output image width in pixels.
    #[arg(long, default_value_t = 1200)]
    width: u32,
    /// Output image height in pixels.
    #[arg(long, default_value_t = 820)]
    height: u32,
    /// Antialiasing method used by Vello.
    #[arg(long, value_enum, default_value = "msaa16")]
    aa: AaMode,
    /// Where to write the PNG.
    #[arg(long, short, default_value_os_t = default_output())]
    output: PathBuf,
    /// Whether to use CPU to render.
    #[arg(long)]
    use_cpu: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum AaMode {
    Area,
    Msaa8,
    Msaa16,
}

impl From<AaMode> for AaConfig {
    fn from(value: AaMode) -> Self {
        match value {
            AaMode::Area => Self::Area,
            AaMode::Msaa8 => Self::Msaa8,
            AaMode::Msaa16 => Self::Msaa16,
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();
    let mut scene = Scene::new();
    build_scene(&mut scene, args.width, args.height);

    pollster::block_on(render_to_png(
        &scene,
        args.width,
        args.height,
        args.aa.into(),
        args.use_cpu,
        &args.output,
    ))
}

fn build_scene(scene: &mut Scene, width: u32, height: u32) {
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        rgb(242, 244, 239),
        None,
        &Rect::new(0.0, 0.0, f64::from(width), f64::from(height)),
    );

    let scale = (f64::from(width) / MAP_WIDTH).min(f64::from(height) / MAP_HEIGHT);
    let offset = Vec2::new(
        (f64::from(width) - MAP_WIDTH * scale) * 0.5,
        (f64::from(height) - MAP_HEIGHT * scale) * 0.5,
    );
    let mut map = Scene::new();
    draw_map(&mut map);
    scene.append(&map, Some(Affine::translate(offset) * Affine::scale(scale)));
}

fn draw_map(scene: &mut Scene) {
    let extent = RoundedRect::new(24.0, 24.0, MAP_WIDTH - 24.0, MAP_HEIGHT - 24.0, 18.0);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        rgb(250, 249, 242),
        None,
        &extent,
    );
    draw_grid(scene);
    draw_polygon_layer(scene);
    draw_water_layer(scene);
    draw_transport_layer(scene);
    draw_point_layer(scene);
    scene.stroke(
        &Stroke::new(2.0),
        Affine::IDENTITY,
        rgb(0, 0, 0),
        None,
        &extent,
    );
}

fn draw_grid(scene: &mut Scene) {
    let grid_stroke = Stroke::new(1.0);
    for x in (80..=1120).step_by(80) {
        scene.stroke(
            &grid_stroke,
            Affine::IDENTITY,
            rgba(188, 198, 190, 70),
            None,
            &Line::new((f64::from(x), 50.0), (f64::from(x), MAP_HEIGHT - 50.0)),
        );
    }
    for y in (80..=760).step_by(80) {
        scene.stroke(
            &grid_stroke,
            Affine::IDENTITY,
            rgba(188, 198, 190, 70),
            None,
            &Line::new((50.0, f64::from(y)), (MAP_WIDTH - 50.0, f64::from(y))),
        );
    }
}

fn draw_polygon_layer(scene: &mut Scene) {
    let park = polygon_path(&[
        (120.0, 150.0),
        (280.0, 92.0),
        (505.0, 126.0),
        (610.0, 255.0),
        (545.0, 410.0),
        (340.0, 455.0),
        (170.0, 360.0),
        (92.0, 245.0),
    ]);
    draw_hatched_polygon(
        scene,
        &park,
        rgb(199, 226, 175),
        rgb(113, 154, 96),
        rgba(96, 138, 82, 90),
    );

    let neighborhood = polygon_path(&[
        (692.0, 105.0),
        (1015.0, 80.0),
        (1102.0, 250.0),
        (1068.0, 502.0),
        (862.0, 555.0),
        (710.0, 435.0),
        (635.0, 245.0),
    ]);
    let gradient = Gradient::new_linear((700.0, 80.0), (1050.0, 550.0))
        .with_extend(Extend::Pad)
        .with_stops([
            ColorStop {
                offset: 0.0,
                color: Color::from_rgba8(234, 212, 172, 155).into(),
            },
            ColorStop {
                offset: 0.3,
                color: Color::from_rgba8(210, 160, 100, 200).into(),
            },
            ColorStop {
                offset: 0.7,
                color: Color::from_rgba8(150, 200, 120, 200).into(),
            },
            ColorStop {
                offset: 1.0,
                color: Color::from_rgba8(80, 140, 180, 255).into(),
            },
        ]);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &gradient,
        None,
        &neighborhood,
    );
    scene.stroke(
        &Stroke::new(5.0).with_join(Join::Round),
        Affine::IDENTITY,
        rgb(187, 151, 98),
        None,
        &neighborhood,
    );
    scene.stroke(
        &Stroke::new(1.5)
            .with_join(Join::Round)
            .with_dashes(0.0, [10.0, 8.0]),
        Affine::IDENTITY,
        rgba(120, 91, 54, 160),
        None,
        &neighborhood,
    );
}

fn draw_water_layer(scene: &mut Scene) {
    let lake = polygon_path(&[
        (178.0, 575.0),
        (288.0, 512.0),
        (438.0, 548.0),
        (502.0, 655.0),
        (426.0, 750.0),
        (258.0, 740.0),
        (142.0, 670.0),
    ]);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        rgb(151, 203, 224),
        None,
        &lake,
    );
    scene.stroke(
        &Stroke::new(11.0).with_join(Join::Round),
        Affine::IDENTITY,
        rgba(82, 152, 185, 130),
        None,
        &lake,
    );
    scene.stroke(
        &Stroke::new(3.0).with_join(Join::Round),
        Affine::IDENTITY,
        rgb(49, 125, 165),
        None,
        &lake,
    );

    let river = polyline_path(&[
        Point::new(650.0, 765.0),
        Point::new(720.0, 648.0),
        Point::new(855.0, 620.0),
        Point::new(940.0, 526.0),
        Point::new(1040.0, 462.0),
        Point::new(1165.0, 404.0),
    ]);
    scene.stroke(
        &Stroke::new(22.0)
            .with_caps(Cap::Round)
            .with_join(Join::Round),
        Affine::IDENTITY,
        rgba(64, 137, 178, 115),
        None,
        &river,
    );
    scene.stroke(
        &Stroke::new(12.0)
            .with_caps(Cap::Round)
            .with_join(Join::Round),
        Affine::IDENTITY,
        rgb(126, 197, 223),
        None,
        &river,
    );
}

fn draw_transport_layer(scene: &mut Scene) {
    let arterial = [
        Point::new(72.0, 498.0),
        Point::new(185.0, 443.0),
        Point::new(333.0, 470.0),
        Point::new(486.0, 428.0),
        Point::new(635.0, 470.0),
        Point::new(820.0, 396.0),
        Point::new(1120.0, 372.0),
    ];
    draw_cased_road(scene, &arterial);

    let connector = [
        Point::new(732.0, 83.0),
        Point::new(705.0, 208.0),
        Point::new(756.0, 338.0),
        Point::new(735.0, 490.0),
        Point::new(788.0, 706.0),
    ];
    draw_cased_road(scene, &connector);
}

fn draw_point_layer(scene: &mut Scene) {
    draw_point_symbol(
        scene,
        Point::new(333.0, 470.0),
        1.1,
        rgb(53, 124, 104),
        rgb(255, 207, 92),
    );
    draw_point_symbol(
        scene,
        Point::new(756.0, 338.0),
        1.0,
        rgb(170, 76, 93),
        rgb(255, 238, 166),
    );
    draw_point_symbol(
        scene,
        Point::new(940.0, 526.0),
        0.9,
        rgb(60, 104, 172),
        rgb(178, 219, 246),
    );
}

fn draw_hatched_polygon(
    scene: &mut Scene,
    path: &BezPath,
    fill: Color,
    outline: Color,
    hatch: Color,
) {
    scene.fill(Fill::NonZero, Affine::IDENTITY, fill, None, path);
    scene.push_clip_layer(Fill::NonZero, Affine::IDENTITY, path);
    let rect = path.bounding_box().inflate(150.0, 150.0);
    for x in (rect.x0.ceil() as i32..=rect.x1.floor() as i32).step_by(30) {
        scene.stroke(
            &Stroke::new(2.0),
            Affine::IDENTITY,
            hatch,
            None,
            &Line::new((f64::from(x + 150), rect.y0), (f64::from(x - 150), rect.y1)),
        );
    }
    scene.pop_layer();
    scene.stroke(
        &Stroke::new(5.0).with_join(Join::Round),
        Affine::IDENTITY,
        outline,
        None,
        path,
    );
    scene.stroke(
        &Stroke::new(1.5).with_join(Join::Round),
        Affine::IDENTITY,
        rgba(255, 255, 255, 150),
        None,
        path,
    );
}

fn draw_cased_road(scene: &mut Scene, points: &[Point]) {
    let road = polyline_path(points);
    scene.stroke(
        &Stroke::new(34.0)
            .with_caps(Cap::Round)
            .with_join(Join::Round),
        Affine::IDENTITY,
        rgb(76, 82, 76),
        None,
        &road,
    );
    scene.stroke(
        &Stroke::new(28.0)
            .with_caps(Cap::Round)
            .with_join(Join::Round),
        Affine::IDENTITY,
        rgb(210, 156, 84),
        None,
        &road,
    );
    scene.stroke(
        &Stroke::new(20.0)
            .with_caps(Cap::Round)
            .with_join(Join::Round),
        Affine::IDENTITY,
        rgb(248, 234, 181),
        None,
        &road,
    );

    let left_side = polyline_path(&offset_polyline(points, 10.5));
    let right_side = polyline_path(&offset_polyline(points, -10.5));
    let side_stroke = Stroke::new(3.0)
        .with_caps(Cap::Round)
        .with_join(Join::Round);
    scene.stroke(
        &side_stroke,
        Affine::IDENTITY,
        rgb(142, 83, 55),
        None,
        &left_side,
    );
    scene.stroke(
        &side_stroke,
        Affine::IDENTITY,
        rgb(142, 83, 55),
        None,
        &right_side,
    );

    scene.stroke(
        &Stroke::new(3.0)
            .with_caps(Cap::Round)
            .with_join(Join::Round)
            .with_dashes(0.0, [24.0, 18.0]),
        Affine::IDENTITY,
        rgb(255, 255, 255),
        None,
        &road,
    );
}

fn draw_point_symbol(scene: &mut Scene, center: Point, scale: f64, primary: Color, accent: Color) {
    let shadow = Circle::new(center + Vec2::new(3.0, 5.0) * scale, 24.0 * scale);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        rgba(48, 48, 48, 70),
        None,
        &shadow,
    );

    let outer = Circle::new(center, 23.0 * scale);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        rgb(255, 255, 255),
        None,
        &outer,
    );
    scene.stroke(
        &Stroke::new(4.0 * scale),
        Affine::IDENTITY,
        rgb(52, 58, 54),
        None,
        &outer,
    );

    let diamond = diamond_path(center, 17.0 * scale);
    scene.fill(Fill::NonZero, Affine::IDENTITY, primary, None, &diamond);
    scene.stroke(
        &Stroke::new(2.0 * scale).with_join(Join::Round),
        Affine::IDENTITY,
        rgb(255, 255, 255),
        None,
        &diamond,
    );

    let star = star_path(center, 9.0 * scale, 4.2 * scale, 5);
    scene.fill(Fill::NonZero, Affine::IDENTITY, accent, None, &star);

    let cross = Stroke::new(2.4 * scale).with_caps(Cap::Round);
    scene.stroke(
        &cross,
        Affine::IDENTITY,
        rgba(35, 40, 38, 190),
        None,
        &Line::new(
            center + Vec2::new(-8.0, 0.0) * scale,
            center + Vec2::new(8.0, 0.0) * scale,
        ),
    );
    scene.stroke(
        &cross,
        Affine::IDENTITY,
        rgba(35, 40, 38, 190),
        None,
        &Line::new(
            center + Vec2::new(0.0, -8.0) * scale,
            center + Vec2::new(0.0, 8.0) * scale,
        ),
    );
}

fn polygon_path(points: &[(f64, f64)]) -> BezPath {
    let mut path = BezPath::new();
    if let Some((first, rest)) = points.split_first() {
        path.move_to(*first);
        for point in rest {
            path.line_to(*point);
        }
        path.close_path();
    }
    path
}

fn polyline_path(points: &[Point]) -> BezPath {
    let mut path = BezPath::new();
    if let Some((first, rest)) = points.split_first() {
        path.move_to(*first);
        for point in rest {
            path.line_to(*point);
        }
    }
    path
}

fn diamond_path(center: Point, radius: f64) -> BezPath {
    let mut path = BezPath::new();
    path.move_to((center.x, center.y - radius));
    path.line_to((center.x + radius, center.y));
    path.line_to((center.x, center.y + radius));
    path.line_to((center.x - radius, center.y));
    path.close_path();
    path
}

fn star_path(center: Point, outer: f64, inner: f64, points: usize) -> BezPath {
    let mut path = BezPath::new();
    for i in 0..(points * 2) {
        let radius = if i % 2 == 0 { outer } else { inner };
        let angle =
            -std::f64::consts::FRAC_PI_2 + (i as f64) * std::f64::consts::PI / (points as f64);
        let point = Point::new(
            center.x + radius * angle.cos(),
            center.y + radius * angle.sin(),
        );
        if i == 0 {
            path.move_to(point);
        } else {
            path.line_to(point);
        }
    }
    path.close_path();
    path
}

fn offset_polyline(points: &[Point], offset: f64) -> Vec<Point> {
    if points.len() < 2 {
        return points.to_vec();
    }

    let dirs = points
        .windows(2)
        .map(|pair| normalize(pair[1] - pair[0]))
        .collect::<Vec<_>>();
    let normals = dirs
        .iter()
        .map(|dir| Vec2::new(-dir.y, dir.x))
        .collect::<Vec<_>>();

    let mut out = Vec::with_capacity(points.len());
    for (idx, point) in points.iter().enumerate() {
        let offset_vec = if idx == 0 {
            normals[0] * offset
        } else if idx == points.len() - 1 {
            normals[normals.len() - 1] * offset
        } else {
            let tangent = normalize(dirs[idx - 1] + dirs[idx]);
            let miter = Vec2::new(-tangent.y, tangent.x);
            let denom = dot(miter, normals[idx]);
            if denom.abs() < 0.15 {
                normalize(normals[idx - 1] + normals[idx]) * offset
            } else {
                let length = (offset / denom).clamp(-28.0, 28.0);
                miter * length
            }
        };
        out.push(*point + offset_vec);
    }
    out
}

fn normalize(vec: Vec2) -> Vec2 {
    let length = (vec.x * vec.x + vec.y * vec.y).sqrt();
    if length <= f64::EPSILON {
        Vec2::ZERO
    } else {
        Vec2::new(vec.x / length, vec.y / length)
    }
}

fn dot(a: Vec2, b: Vec2) -> f64 {
    a.x * b.x + a.y * b.y
}

async fn render_to_png(
    scene: &Scene,
    output_width: u32,
    output_height: u32,
    aa: AaConfig,
    use_cpu: bool,
    out_path: &Path,
) -> Result<()> {
    let mut context = RenderContext::new();
    let device_id = context
        .device(None)
        .await
        .ok_or_else(|| anyhow!("No compatible device found"))?;
    let device_handle = &mut context.devices[device_id];
    let device = &device_handle.device;
    let queue = &device_handle.queue;

    #[cfg(target_os = "macos")]
    let num_init_threads = NonZeroUsize::new(1);
    #[cfg(not(target_os = "macos"))]
    let num_init_threads = None;
    let mut renderer = Renderer::new(
        device,
        RendererOptions {
            use_cpu,
            num_init_threads,
            antialiasing_support: AaSupport::all(),
            ..Default::default()
        },
    )
    .or_else(|_| bail!("Got error from creating renderer"))?;

    let size = Extent3d {
        width: output_width,
        height: output_height,
        depth_or_array_layers: 1,
    };
    let target = device.create_texture(&TextureDescriptor {
        label: Some("Symbols target texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    renderer
        .render_to_texture(
            device,
            queue,
            scene,
            &view,
            &RenderParams {
                base_color: rgb(242, 244, 239),
                width: output_width,
                height: output_height,
                antialiasing_method: aa,
            },
        )
        .or_else(|_| bail!("Got error from rendering"))?;

    let padded_byte_width = (output_width * 4).next_multiple_of(256);
    let buffer_size = u64::from(padded_byte_width) * u64::from(output_height);
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Symbols readback buffer"),
        size: buffer_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Symbols copy encoder"),
    });
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_byte_width),
                rows_per_image: None,
            },
        },
        size,
    );

    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    queue.submit([encoder.finish()]);
    if let Some(recv_result) = block_on_wgpu(device, receiver.receive()) {
        recv_result?;
    } else {
        bail!("readback channel was closed");
    }

    let data = buffer.slice(..).get_mapped_range();
    let mut png_data = Vec::<u8>::with_capacity((output_width * output_height * 4) as usize);
    for row in 0..output_height {
        let start = (row * padded_byte_width) as usize;
        png_data.extend(&data[start..start + (output_width * 4) as usize]);
    }
    drop(data);
    buffer.unmap();

    if let Some(parent) = out_path.parent() {
        create_dir_all(parent).with_context(|| format!("Creating {}", parent.display()))?;
    }
    let mut file =
        File::create(out_path).with_context(|| format!("Creating {}", out_path.display()))?;
    let mut png_encoder = png::Encoder::new(&mut file, output_width, output_height);
    png_encoder.set_color(png::ColorType::Rgba);
    png_encoder.set_depth(png::BitDepth::Eight);
    let mut writer = png_encoder.write_header()?;
    writer.write_image_data(&png_data)?;
    writer.finish()?;

    println!(
        "Wrote symbols ({output_width}x{output_height}, {aa:?}) to {}",
        out_path.display()
    );
    Ok(())
}

fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color::from_rgb8(r, g, b)
}

fn rgba(r: u8, g: u8, b: u8, a: u8) -> Color {
    Color::from_rgba8(r, g, b, a)
}

fn default_output() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("outputs/symbols.png")
}
