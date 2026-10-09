// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Compare GPU sparse strips against the CPU `strip::render` for an SVG.
//!
//! Usage: `sparse_strips_compare <svg> [scale] [dx] [dy] [width] [height]`
//!
//! Flattens, tiles and sorts every path on the CPU, then renders the strips both with
//! `strip::render` and with the GPU passes, and reports differences (strips must match exactly,
//! alphas within one unit) and timings. The GPU timing is wall clock around submit and wait and
//! includes buffer uploads; the CPU timing is `strip::render` alone.

#![allow(
    clippy::cast_possible_truncation,
    reason = "viewport sizes are given on the command line"
)]

use std::time::Instant;
use vello_common::fearless_simd::Level;
use vello_common::flatten::{self, FlattenCtx, Line};
use vello_common::geometry::RectU16;
use vello_common::kurbo::{Affine, BezPath, Stroke, StrokeCtx};
use vello_common::peniko::Fill;
use vello_common::pico_svg::{Item, PicoSvg};
use vello_common::strip::{self, Strip};
use vello_common::tile::Tiles;
use vello_gpu::sparse_strips::gpu::{SparseStripPipelines, read_back};
use vello_gpu::sparse_strips::{AddPath, StripInput, model};

struct PathItem {
    path: BezPath,
    transform: Affine,
    stroke: Option<Stroke>,
}

fn collect(items: &[Item], transform: Affine, out: &mut Vec<PathItem>) {
    for item in items {
        match item {
            Item::Fill(f) => out.push(PathItem {
                path: f.path.clone(),
                transform,
                stroke: None,
            }),
            Item::Stroke(s) => out.push(PathItem {
                path: s.path.clone(),
                transform,
                stroke: Some(Stroke::new(s.width)),
            }),
            Item::Group(g) => collect(&g.children, transform * g.affine, out),
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let svg_filename = args.next().expect("svg filename is the first argument");
    let mut num = |default: f64| -> f64 { args.next().map_or(default, |a| a.parse().unwrap()) };
    let scale = num(1.0);
    let dx = num(0.0);
    let dy = num(0.0);
    let width = num(0.0);
    let height = num(0.0);

    let svg = std::fs::read_to_string(&svg_filename).expect("error reading file");
    let parsed = PicoSvg::load(&svg, 1.0).expect("error parsing SVG");
    let width = if width > 0.0 {
        width as u16
    } else {
        (parsed.size.width * scale).ceil() as u16
    };
    let height = if height > 0.0 {
        height as u16
    } else {
        (parsed.size.height * scale).ceil() as u16
    };
    let transform = Affine::translate((dx, dy)) * Affine::scale(scale);
    let mut items = Vec::new();
    collect(&parsed.items, transform, &mut items);

    // CPU front end, shared by both paths.
    let level = Level::new();
    let mut input = StripInput::new(width, height);
    let mut cpu_strips: Vec<Strip> = Vec::new();
    let mut cpu_alphas: Vec<u8> = Vec::new();
    let mut lines: Vec<Line> = Vec::new();
    let mut ctx = FlattenCtx::default();
    let mut stroke_ctx = StrokeCtx::default();
    let mut tiles = Tiles::new(level, width, height);
    let bbox = RectU16::new(0, 0, width, height);
    let mut cpu_render_time = std::time::Duration::ZERO;
    let mut front_time = std::time::Duration::ZERO;
    let mut fallbacks = 0;
    for item in &items {
        let t0 = Instant::now();
        let els = item.path.elements().iter().copied();
        if let Some(stroke) = &item.stroke {
            flatten::stroke(
                level,
                els,
                stroke,
                item.transform,
                &mut lines,
                &mut ctx,
                &mut stroke_ctx,
                bbox,
            );
        } else {
            flatten::fill(level, els, item.transform, &mut lines, &mut ctx, bbox);
        }
        tiles.reset(width, height);
        tiles.make_tiles_analytic_aa(level, &lines, width, height);
        tiles.sort_tiles();
        front_time += t0.elapsed();
        match input.add_path(&tiles, &lines, Fill::NonZero, None) {
            AddPath::CpuFallback => {
                fallbacks += 1;
                continue;
            }
            AddPath::Empty => continue,
            AddPath::Added => {}
        }
        let t1 = Instant::now();
        strip::render(
            level,
            &tiles,
            &mut cpu_strips,
            &mut cpu_alphas,
            Fill::NonZero,
            None,
            &lines,
        );
        cpu_render_time += t1.elapsed();
    }
    println!(
        "{}x{}: {} paths ({} CPU fallbacks), {} lines, {} entries, {} CPU strips, {} alpha bytes",
        width,
        height,
        input.paths.len(),
        fallbacks,
        input.lines.len(),
        input.entries.len(),
        cpu_strips.len(),
        cpu_alphas.len()
    );
    println!(
        "CPU: flatten+tile+sort {:.3} ms, strip::render {:.3} ms",
        front_time.as_secs_f64() * 1e3,
        cpu_render_time.as_secs_f64() * 1e3
    );

    // GPU.
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .expect("Failed to find an appropriate adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Device"),
        required_features: wgpu::Features::empty(),
        ..Default::default()
    }))
    .expect("Failed to create device");
    let pipelines = SparseStripPipelines::new(&device);

    // Warm up, then time a few iterations without readback.
    let iterations = 10;
    let mut best = f64::MAX;
    for _ in 0..iterations {
        let t0 = Instant::now();
        let buffers = pipelines.create_buffers(&device, &input);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        pipelines.encode(&mut encoder, &buffers);
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    println!("GPU: upload+passes+wait best of {iterations}: {best:.3} ms");

    let buffers = pipelines.create_buffers(&device, &input);
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    pipelines.encode(&mut encoder, &buffers);
    let out = read_back(&device, &queue, encoder, &buffers);

    let t0 = Instant::now();
    let (model_strips, model_alphas) = model::render(&input);
    println!(
        "model (CPU copy of the GPU algorithm): {:.3} ms",
        t0.elapsed().as_secs_f64() * 1e3
    );

    report(
        "GPU vs CPU",
        (&out.strips, &out.alphas),
        (&cpu_strips, &cpu_alphas),
    );
    report(
        "GPU vs model",
        (&out.strips, &out.alphas),
        (&model_strips, &model_alphas),
    );
    report(
        "model vs CPU",
        (&model_strips, &model_alphas),
        (&cpu_strips, &cpu_alphas),
    );
}

fn report(what: &str, got: (&[Strip], &[u8]), want: (&[Strip], &[u8])) {
    let (strips, alphas) = got;
    let (want_strips, want_alphas) = want;
    let mut strip_mismatches = 0;
    let mut first_strip = None;
    for (i, (g, w)) in strips.iter().zip(want_strips).enumerate() {
        if g != w {
            strip_mismatches += 1;
            first_strip.get_or_insert_with(|| format!("strip {i}: {g:?} vs {w:?}"));
        }
    }
    let mut hist = [0_usize; 4];
    let mut first_alpha = None;
    for (i, (g, w)) in alphas.iter().zip(want_alphas).enumerate() {
        let diff = (i32::from(*g) - i32::from(*w)).unsigned_abs() as usize;
        hist[diff.min(3)] += 1;
        if diff > 1 {
            first_alpha.get_or_insert_with(|| format!("alpha {i}: {g} vs {w}"));
        }
    }
    let ok = strips.len() == want_strips.len()
        && strip_mismatches == 0
        && alphas.len() == want_alphas.len()
        && hist[2] == 0
        && hist[3] == 0;
    println!(
        "{what}: {}; strips {} vs {} ({strip_mismatches} differ), alphas {} vs {}, diff histogram [0: {}, 1: {}, 2: {}, >2: {}]",
        if ok { "OK" } else { "MISMATCH" },
        strips.len(),
        want_strips.len(),
        alphas.len(),
        want_alphas.len(),
        hist[0],
        hist[1],
        hist[2],
        hist[3]
    );
    if let Some(s) = first_strip {
        println!("  first strip mismatch: {s}");
    }
    if let Some(a) = first_alpha {
        println!("  first alpha difference above 1: {a}");
    }
}
