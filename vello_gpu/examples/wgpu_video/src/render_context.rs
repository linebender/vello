// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Minimal wgpu setup: a device, a queue and a surface for the winit window.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use vello_gpu::{RenderSize, RenderTargetConfig, Renderer, Resources};
use wgpu::{
    Adapter, CommandBuffer, Device, Features, Instance, Limits, Queue, Surface,
    SurfaceConfiguration, TextureFormat, TextureView,
};
use winit::window::Window;

pub(crate) const SURFACE_FORMAT: TextureFormat = TextureFormat::Bgra8Unorm;

/// Numbers queue submissions and records how many the GPU has finished, so resources
/// the CPU no longer needs can be released once the GPU is done with them too.
#[derive(Debug, Default)]
pub(crate) struct SubmissionTracker {
    submitted: u64,
    /// Written from wgpu's completion callbacks, which run inside `submit` or `poll`.
    completed: Arc<AtomicU64>,
}

impl SubmissionTracker {
    /// Serial of the most recent submission; 0 before the first.
    pub(crate) fn submitted(&self) -> u64 {
        self.submitted
    }

    /// Every submission with a serial up to this one has finished on the GPU.
    pub(crate) fn completed(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    fn submit(&mut self, queue: &Queue, command_buffer: CommandBuffer) {
        queue.submit([command_buffer]);
        self.submitted += 1;
        let serial = self.submitted;
        let completed = Arc::clone(&self.completed);
        queue.on_submitted_work_done(move || {
            completed.fetch_max(serial, Ordering::Release);
        });
    }
}

/// The wgpu state and Vello renderer for one window.
#[derive(Debug)]
pub(crate) struct RenderContext<'window> {
    pub(crate) device: Device,
    pub(crate) queue: Queue,
    pub(crate) surface: Surface<'window>,
    pub(crate) surface_config: SurfaceConfiguration,
    pub(crate) renderer: Renderer,
    pub(crate) depth_texture_view: TextureView,
    /// Lets the video pipeline check for the Metal backend before using wgpu's Metal
    /// hal API.
    pub(crate) adapter: Adapter,
    pub(crate) submissions: SubmissionTracker,
}

impl<'window> RenderContext<'window> {
    /// Creates the device and surface for `window`, plus a Vello renderer targeting it.
    /// Blocks until the device is ready.
    pub(crate) fn new(window: Arc<Window>) -> (Self, Resources) {
        pollster::block_on(Self::new_async(window))
    }

    async fn new_async(window: Arc<Window>) -> (Self, Resources) {
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
                label: Some("vello_gpu_wgpu_video device"),
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
            // Uncapped, so the FPS counter measures rendering rather than the display's
            // refresh rate. May tear.
            present_mode: wgpu::PresentMode::Immediate,
            color_space: wgpu::SurfaceColorSpace::Auto,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let render_size = RenderSize {
            width: u16::try_from(width).unwrap_or(u16::MAX),
            height: u16::try_from(height).unwrap_or(u16::MAX),
        };
        let (renderer, resources) = Renderer::new(
            &device,
            &RenderTargetConfig {
                format: SURFACE_FORMAT,
                width: render_size.width,
                height: render_size.height,
            },
        );
        let depth_texture_view = Renderer::create_depth_texture_view(&device, &render_size);

        (
            Self {
                device,
                queue,
                surface,
                surface_config,
                renderer,
                depth_texture_view,
                adapter,
                submissions: SubmissionTracker::default(),
            },
            resources,
        )
    }

    /// Submits `command_buffer`, tracking it in [`Self::submissions`].
    pub(crate) fn submit(&mut self, command_buffer: CommandBuffer) {
        self.submissions.submit(&self.queue, command_buffer);
    }

    /// Resizes the surface and depth buffer to the new window size.
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface.configure(&self.device, &self.surface_config);
        let render_size = RenderSize {
            width: u16::try_from(self.surface_config.width).unwrap_or(u16::MAX),
            height: u16::try_from(self.surface_config.height).unwrap_or(u16::MAX),
        };
        self.depth_texture_view = Renderer::create_depth_texture_view(&self.device, &render_size);
    }
}
