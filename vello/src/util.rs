// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simple helpers for managing wgpu state and surfaces.

use std::future::Future;
use wgpu::{
    Adapter, Device, Features, Instance, Limits, MemoryHints, Queue, Surface, SurfaceConfiguration,
    SurfaceTarget, Texture, TextureFormat, TextureView, util::TextureBlitter,
};

use crate::Error as VelloError;

/// Simple render context that maintains wgpu state for rendering the pipeline.
pub struct RenderContext {
    /// A WGPU `Instance`. This only needs to be created once per application.
    pub instance: Instance,
    /// A pool of already-created devices so that we can avoid recreating devices
    /// when we already have a suitable one available
    pub device_pool: Vec<DeviceHandle>,

    // Config
    extra_features: Option<Features>,
    override_limits: Option<Limits>,
}

/// A wgpu `Device`, it's associated `Queue`, and the adapter used to create it.
/// Q: could we drop the adapter here? wgpu docs say adapters do not need to be kept around...
#[derive(Clone, Debug)]
pub struct DeviceHandle {
    adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl DeviceHandle {
    /// Returns the adapter associated with the device.
    pub fn adapter(&self) -> &Adapter {
        &self.adapter
    }
}

impl Default for RenderContext {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderContext {
    pub fn new() -> Self {
        Self::with_features_and_limits(None, None)
    }

    pub fn with_features_and_limits(
        extra_features: Option<Features>,
        override_limits: Option<Limits>,
    ) -> Self {
        Self {
            instance: Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::from_env().unwrap_or_default(),
                flags: wgpu::InstanceFlags::from_build_config().with_env(),
                backend_options: wgpu::BackendOptions::from_env_or_default(),
            }),
            device_pool: Vec::new(),
            extra_features,
            override_limits,
        }
    }

    /// Creates a new surface for the specified window and dimensions.
    pub async fn create_surface<'w>(
        &mut self,
        window: impl Into<SurfaceTarget<'w>>,
        width: u32,
        height: u32,
        present_mode: wgpu::PresentMode,
    ) -> Result<RenderSurface<'w>, VelloError> {
        // Create a surface from the window handle
        let surface = self.instance.create_surface(window.into())?;

        // Find or create a suitable device for rendering to the surface
        let dev_id = self
            .find_or_create_device(Some(&surface))
            .await
            .ok_or(VelloError::NoCompatibleDevice)?;
        let device_handle = self.device_pool[dev_id].clone();

        RenderSurface::new(surface, device_handle, dev_id, width, height, present_mode).await
    }

    /// Finds or creates a compatible device handle id.
    pub async fn find_or_create_device(
        &mut self,
        compatible_surface: Option<&Surface<'_>>,
    ) -> Option<usize> {
        match self.find_existing_device(compatible_surface) {
            Some(device_id) => Some(device_id),
            None => self.create_device(compatible_surface).await,
        }
    }

    /// Finds or creates a compatible device handle id.
    fn find_existing_device(&self, compatible_surface: Option<&Surface<'_>>) -> Option<usize> {
        match compatible_surface {
            Some(s) => self
                .device_pool
                .iter()
                .enumerate()
                .find(|(_, d)| d.adapter.is_surface_supported(s))
                .map(|(i, _)| i),
            None => (!self.device_pool.is_empty()).then_some(0),
        }
    }

    /// Creates a compatible device handle id.
    async fn create_device(&mut self, compatible_surface: Option<&Surface<'_>>) -> Option<usize> {
        let adapter =
            wgpu::util::initialize_adapter_from_env_or_default(&self.instance, compatible_surface)
                .await?;

        // Determine features to request
        // The user may request additional features
        let extra_features = self.extra_features.unwrap_or(Features::empty());
        let vello_features = wgpu::Features::CLEAR_TEXTURE | wgpu::Features::PIPELINE_CACHE;
        let requested_features = vello_features | extra_features;
        let available_features = adapter.features();
        let required_features = requested_features & available_features;

        // Determine limits to request
        // The user may override the limits
        let required_limits = self.override_limits.clone().unwrap_or_default();

        // Create the device and the queue
        let descriptor = wgpu::DeviceDescriptor {
            label: None,
            required_features,
            required_limits,
            memory_hints: MemoryHints::default(),
        };
        let (device, queue) = adapter.request_device(&descriptor, None).await.ok()?;

        // Create the device handle and store in the pool
        let device_handle = DeviceHandle {
            adapter,
            device,
            queue,
        };
        self.device_pool.push(device_handle);

        // Return the ID
        Some(self.device_pool.len() - 1)
    }
}

/// Vello uses a compute shader to render to the provided texture, which means that it can't bind the surface
/// texture in most cases.
///
/// Because of this, we need to create an "intermediate" texture which we render to, and then blit to the surface.
fn create_intermediate_texture(width: u32, height: u32, device: &Device) -> (Texture, TextureView) {
    let target_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        format: TextureFormat::Rgba8Unorm,
        view_formats: &[],
    });
    let target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());
    (target_texture, target_view)
}

/// Combination of surface and its configuration.
pub struct RenderSurface<'s> {
    // The device and queue for rendering to the surface
    pub dev_id: usize,
    pub device_handle: DeviceHandle,

    // The surface, it's configuration and format
    pub surface: Surface<'s>,
    pub config: SurfaceConfiguration,
    pub format: TextureFormat,

    // Intermediate Texture which we render to because compute shaders cannot always render directly
    // to surfaces, and associated TextureView.
    pub target_texture: Texture,
    pub target_view: TextureView,

    // Blitter for blitting from the intermediate texture to the surface.
    pub blitter: TextureBlitter,
}

impl std::fmt::Debug for RenderSurface<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderSurface")
            .field("surface", &self.surface)
            .field("config", &self.config)
            .field("device_handle", &self.device_handle)
            .field("format", &self.format)
            .field("target_texture", &self.target_texture)
            .field("target_view", &self.target_view)
            .field("blitter", &"(Not Debug)")
            .finish()
    }
}

impl<'s> RenderSurface<'s> {
    /// Creates a new render surface for the specified window and dimensions.
    pub async fn new<'w>(
        surface: Surface<'w>,
        device_handle: DeviceHandle,
        dev_id: usize,
        width: u32,
        height: u32,
        present_mode: wgpu::PresentMode,
    ) -> Result<RenderSurface<'w>, VelloError> {
        let capabilities = surface.get_capabilities(&device_handle.adapter);
        let format = capabilities
            .formats
            .into_iter()
            .find(|it| matches!(it, TextureFormat::Rgba8Unorm | TextureFormat::Bgra8Unorm))
            .ok_or(VelloError::UnsupportedSurfaceFormat)?;

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
        let (target_texture, target_view) =
            create_intermediate_texture(width, height, &device_handle.device);
        let blitter = TextureBlitter::new(&device_handle.device, format);
        let surface = RenderSurface {
            dev_id,
            device_handle,
            surface,
            config,
            format,
            target_texture,
            target_view,
            blitter,
        };
        surface.configure();
        Ok(surface)
    }

    /// Resizes the surface to the new dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (texture, view) =
            create_intermediate_texture(width, height, &self.device_handle.device);
        // TODO: Use clever resize semantics to avoid thrashing the memory allocator during a resize
        // especially important on metal.
        self.target_texture = texture;
        self.target_view = view;
        self.config.width = width;
        self.config.height = height;
        self.configure();
    }

    pub fn set_present_mode(&mut self, present_mode: wgpu::PresentMode) {
        self.config.present_mode = present_mode;
        self.configure();
    }

    fn configure(&self) {
        self.surface
            .configure(&self.device_handle.device, &self.config);
    }
}

/// Block on a future, polling the device as needed.
///
/// This will deadlock if the future is awaiting anything other than GPU progress.
#[cfg_attr(docsrs, doc(hidden))]
pub fn block_on_wgpu<F: Future>(device: &Device, fut: F) -> F::Output {
    if cfg!(target_arch = "wasm32") {
        panic!("WGPU is inherently async on WASM, so blocking doesn't work.");
    }

    // Dummy waker
    struct NullWake;
    impl std::task::Wake for NullWake {
        fn wake(self: std::sync::Arc<Self>) {}
    }

    // Create context to poll future with
    let waker = std::task::Waker::from(std::sync::Arc::new(NullWake));
    let mut context = std::task::Context::from_waker(&waker);

    // Same logic as `pin_mut!` macro from `pin_utils`.
    let mut fut = std::pin::pin!(fut);
    loop {
        match fut.as_mut().poll(&mut context) {
            std::task::Poll::Pending => {
                device.poll(wgpu::Maintain::Wait);
            }
            std::task::Poll::Ready(item) => break item,
        }
    }
}
