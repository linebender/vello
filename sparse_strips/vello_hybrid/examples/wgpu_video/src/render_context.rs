// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Minimal wgpu setup for the example: instance, adapter, device, queue and a configured surface
//! tied to a winit window.

use std::sync::Arc;

use vello_hybrid::{RenderTargetConfig, Renderer};
use wgpu::{
    Adapter, Device, Features, Instance, Limits, Queue, Surface, SurfaceConfiguration,
    TextureFormat,
};
use winit::window::Window;

/// Surface format used everywhere in this example.
pub(crate) const SURFACE_FORMAT: TextureFormat = TextureFormat::Bgra8Unorm;

/// All the wgpu state the example needs, plus a configured surface for the winit window.
#[derive(Debug)]
pub(crate) struct RenderContext<'window> {
    pub(crate) device: Device,
    pub(crate) queue: Queue,
    pub(crate) surface: Surface<'window>,
    pub(crate) surface_config: SurfaceConfiguration,
    pub(crate) renderer: Renderer,
    /// Held so the macOS video pipeline can confirm the device backend is Metal
    /// before reaching into the wgpu hal Metal API for `IOSurface` interop.
    pub(crate) adapter: Adapter,
}

impl<'window> RenderContext<'window> {
    /// Synchronously construct an instance/adapter/device for the given window and create a
    /// Vello Hybrid `Renderer` configured for it.
    pub(crate) fn new(window: Arc<Window>) -> Self {
        pollster::block_on(Self::new_async(window))
    }

    async fn new_async(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY);
        let instance = Instance::new(wgpu::InstanceDescriptor {
            display: None,
            backends,
            flags: wgpu::InstanceFlags::from_build_config().with_env(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::from_env_or_default(),
        });
        let surface = instance
            .create_surface(window)
            .expect("Failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("No compatible adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("vello_hybrid_wgpu_video device"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                ..Default::default()
            })
            .await
            .expect("Failed to request device");

        let surface_config = SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: SURFACE_FORMAT,
            width,
            height,
            // `Immediate` = uncapped FPS so the FPS counter actually measures render
            // work and not just the display refresh rate (VSync). May tear on resize.
            present_mode: wgpu::PresentMode::Immediate,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let renderer = Renderer::new(
            &device,
            &RenderTargetConfig {
                format: SURFACE_FORMAT,
                width,
                height,
            },
        );

        Self {
            device,
            queue,
            surface,
            surface_config,
            renderer,
            adapter,
        }
    }

    /// Reconfigure the surface for a new window size.
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface.configure(&self.device, &self.surface_config);
    }
}
