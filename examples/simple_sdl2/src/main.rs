// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello can also be used with non-Winit crates which provide a `RawWindowHandle`.
//! This example uses it with [`sdl2`].
//!
//! Vello however is primarily designed for Xilem, which uses Winit, and so support for non-Winit crates
//! is on a tier-2 basis, i.e. it is checked in CI, but is not rarely manually validated.
extern crate sdl2;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;

use std::num::NonZeroUsize;

use vello::kurbo::{Affine, Circle, Ellipse, Line, RoundedRect, Stroke};
use vello::peniko::Color;
use vello::peniko::color::palette;
use vello::util::{RenderContext, RenderSurface};
use vello::{AaConfig, Renderer, RendererOptions, Scene};

use vello::wgpu;

fn main() {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let width: u32 = 800;
    let height: u32 = 600;

    let window = video_subsystem
        .window("Vello SDL2 Demo", width, height)
        .position_centered()
        .metal_view()
        .build()
        .unwrap();

    let mut context = RenderContext::new();

    let surface_future = unsafe {
        context.create_render_surface(
            context
                .instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::from_window(&window).unwrap())
                .unwrap(),
            width,
            height,
            wgpu::PresentMode::AutoVsync,
        )
    };

    let surface = pollster::block_on(surface_future).expect("Error creating surface.");

    let mut renderers: Vec<Option<Renderer>> = vec![];

    renderers.resize_with(context.devices.len(), || None);
    let _ = renderers[surface.dev_id].insert(create_vello_renderer(&context, &surface));

    let mut scene = Scene::new();

    let mut event_pump = sdl_context.event_pump().unwrap();

    'running: loop {
        scene.reset();

        add_shapes_to_scene(&mut scene);

        let device_handle = &context.devices[surface.dev_id];

        renderers[surface.dev_id]
            .as_mut()
            .unwrap()
            .render_to_texture(
                &device_handle.device,
                &device_handle.queue,
                &scene,
                &surface.target_view,
                &vello::RenderParams {
                    base_color: palette::css::BLACK, // Background color
                    width,
                    height,
                    antialiasing_method: AaConfig::Msaa16,
                },
            )
            .expect("failed to render to surface");

        let surface_texture = surface
            .surface
            .get_current_texture()
            .expect("failed to get surface texture");

        let mut encoder =
            device_handle
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Surface Blit"),
                });
        surface.blitter.copy(
            &device_handle.device,
            &mut encoder,
            &surface.target_view,
            &surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
        );
        device_handle.queue.submit([encoder.finish()]);
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }

        surface_texture.present();
    }
}

fn create_vello_renderer(render_cx: &RenderContext, surface: &RenderSurface<'_>) -> Renderer {
    Renderer::new(
        &render_cx.devices[surface.dev_id].device,
        RendererOptions {
            use_cpu: false,
            antialiasing_support: vello::AaSupport::all(),
            num_init_threads: NonZeroUsize::new(1),
            pipeline_cache: None,
        },
    )
    .expect("Couldn't create renderer")
}

fn add_shapes_to_scene(scene: &mut Scene) {
    // Draw an outlined rectangle
    let stroke = Stroke::new(6.0);
    let rect = RoundedRect::new(10.0, 10.0, 240.0, 240.0, 20.0);
    let rect_stroke_color = Color::new([0.9804, 0.702, 0.5294, 1.]);
    scene.stroke(&stroke, Affine::IDENTITY, rect_stroke_color, None, &rect);

    // Draw a filled circle
    let circle = Circle::new((420.0, 200.0), 120.0);
    let circle_fill_color = Color::new([0.9529, 0.5451, 0.6588, 1.]);
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        circle_fill_color,
        None,
        &circle,
    );

    // Draw a filled ellipse
    let ellipse = Ellipse::new((250.0, 420.0), (100.0, 160.0), -90.0);
    let ellipse_fill_color = Color::new([0.7961, 0.651, 0.9686, 1.]);
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        ellipse_fill_color,
        None,
        &ellipse,
    );

    // Draw a straight line
    let line = Line::new((260.0, 20.0), (620.0, 100.0));
    let line_stroke_color = Color::new([0.5373, 0.7059, 0.9804, 1.]);
    scene.stroke(&stroke, Affine::IDENTITY, line_stroke_color, None, &line);
}
