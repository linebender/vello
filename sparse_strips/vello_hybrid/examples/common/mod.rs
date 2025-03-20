// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(
    dead_code,
    reason = "This is a shared module between examples; not all examples use all functionality from it"
)]

use std::sync::Arc;

use vello_common::kurbo::{Affine, Stroke};
use vello_common::pico_svg::Item;
use vello_hybrid::{Renderer, RendererOptions, Scene};
use wgpu::{
    Adapter, Device, Features, Instance, Limits, MemoryHints, Queue, Surface, SurfaceConfiguration,
    SurfaceTarget, TextureFormat,
};
use winit::{event_loop::ActiveEventLoop, window::Window};

/// Define a render function that works with our `pico_svg::Item` type
pub(crate) fn render_svg(ctx: &mut Scene, scale: f64, items: &[Item]) {
    fn render_svg_inner(ctx: &mut Scene, items: &[Item], transform: Affine) {
        ctx.set_transform(transform);
        for item in items {
            match item {
                Item::Fill(fill_item) => {
                    ctx.set_paint(fill_item.color.into());
                    ctx.fill_path(&fill_item.path);
                }
                Item::Stroke(stroke_item) => {
                    let style = Stroke::new(stroke_item.width);
                    ctx.set_stroke(style);
                    ctx.set_paint(stroke_item.color.into());
                    ctx.stroke_path(&stroke_item.path);
                }
                Item::Group(group_item) => {
                    render_svg_inner(ctx, &group_item.children, transform * group_item.affine);
                    ctx.set_transform(transform);
                }
            }
        }
    }

    render_svg_inner(ctx, items, Affine::scale(scale));
}

/// Helper function that creates a Winit window and returns it (wrapped in an Arc for sharing)
pub(crate) fn create_winit_window(
    event_loop: &ActiveEventLoop,
    width: u32,
    height: u32,
    initially_visible: bool,
) -> Arc<Window> {
    let attr = Window::default_attributes()
        .with_inner_size(winit::dpi::PhysicalSize::new(width, height))
        .with_resizable(false)
        .with_title("Vello SVG Renderer")
        .with_visible(initially_visible)
        .with_active(true);
    Arc::new(event_loop.create_window(attr).unwrap())
}

/// Helper function that creates a Vello Hybrid renderer
pub(crate) fn create_vello_renderer(
    render_cx: &RenderContext,
    surface: &RenderSurface<'_>,
) -> Renderer {
    Renderer::new(
        &render_cx.devices[surface.dev_id].device,
        &RendererOptions {
            format: surface.config.format,
        },
    )
}

/// Simple render context that maintains wgpu state for rendering the pipeline.
#[derive(Debug)]
pub(crate) struct RenderContext {
    /// The instance of the wgpu instance
    pub(crate) instance: Instance,
    /// The devices of the wgpu instance
    pub(crate) devices: Vec<DeviceHandle>,
}

/// A handle to a device
#[derive(Debug)]
pub(crate) struct DeviceHandle {
    /// The adapter of the device
    pub(crate) adapter: Adapter,
    /// The device
    pub(crate) device: Device,
    /// The queue of the device
    pub(crate) queue: Queue,
}

impl RenderContext {
    /// Creates a new render context
    pub(crate) fn new() -> Self {
        let backends = wgpu::Backends::from_env().unwrap_or_default();
        let flags = wgpu::InstanceFlags::from_build_config().with_env();
        let backend_options = wgpu::BackendOptions::from_env_or_default();
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends,
            flags,
            backend_options,
        });
        Self {
            instance,
            devices: Vec::new(),
        }
    }

    /// Creates a new surface for the specified window and dimensions.
    pub(crate) async fn create_surface<'w>(
        &mut self,
        window: impl Into<SurfaceTarget<'w>>,
        width: u32,
        height: u32,
        present_mode: wgpu::PresentMode,
        format: TextureFormat,
    ) -> RenderSurface<'w> {
        self.create_render_surface(
            self.instance
                .create_surface(window.into())
                .expect("Error creating surface"),
            width,
            height,
            present_mode,
            format,
        )
        .await
    }

    /// Creates a new render surface for the specified window and dimensions.
    pub(crate) async fn create_render_surface<'w>(
        &mut self,
        surface: Surface<'w>,
        width: u32,
        height: u32,
        present_mode: wgpu::PresentMode,
        format: TextureFormat,
    ) -> RenderSurface<'w> {
        let dev_id = self
            .device(Some(&surface))
            .await
            .expect("No compatible device");

        let config = SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        let surface = RenderSurface {
            surface,
            config,
            dev_id,
        };
        self.configure_surface(&surface);
        surface
    }

    /// Resizes the surface to the new dimensions.
    pub(crate) fn resize_surface(&self, surface: &mut RenderSurface<'_>, width: u32, height: u32) {
        surface.config.width = width;
        surface.config.height = height;
        self.configure_surface(surface);
    }

    /// Sets the present mode for the surface
    pub(crate) fn set_present_mode(
        &self,
        surface: &mut RenderSurface<'_>,
        present_mode: wgpu::PresentMode,
    ) {
        surface.config.present_mode = present_mode;
        self.configure_surface(surface);
    }

    fn configure_surface(&self, surface: &RenderSurface<'_>) {
        let device = &self.devices[surface.dev_id].device;
        surface.surface.configure(device, &surface.config);
    }

    /// Finds or creates a compatible device handle id.
    pub(crate) async fn device(
        &mut self,
        compatible_surface: Option<&Surface<'_>>,
    ) -> Option<usize> {
        let compatible = match compatible_surface {
            Some(s) => self
                .devices
                .iter()
                .enumerate()
                .find(|(_, d)| d.adapter.is_surface_supported(s))
                .map(|(i, _)| i),
            None => (!self.devices.is_empty()).then_some(0),
        };
        if compatible.is_none() {
            return self.new_device(compatible_surface).await;
        }
        compatible
    }

    /// Creates a compatible device handle id.
    async fn new_device(&mut self, compatible_surface: Option<&Surface<'_>>) -> Option<usize> {
        let adapter =
            wgpu::util::initialize_adapter_from_env_or_default(&self.instance, compatible_surface)
                .await?;
        let limits = Limits::default();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: Features::empty(),
                    required_limits: limits,
                    memory_hints: MemoryHints::default(),
                },
                None,
            )
            .await
            .ok()?;
        let device_handle = DeviceHandle {
            adapter,
            device,
            queue,
        };
        self.devices.push(device_handle);
        Some(self.devices.len() - 1)
    }
}

/// Combination of surface and its configuration.
#[derive(Debug)]
pub(crate) struct RenderSurface<'s> {
    /// The surface
    pub(crate) surface: Surface<'s>,
    /// The configuration of the surface
    pub(crate) config: SurfaceConfiguration,
    /// The device id
    pub(crate) dev_id: usize,
}
