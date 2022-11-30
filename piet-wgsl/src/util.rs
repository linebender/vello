// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

//! Simple helpers for managing wgpu state and surfaces.

use super::Result;

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use wgpu::{Device, Instance, Limits, Queue, Surface, SurfaceConfiguration};

/// Simple render context that maintains wgpu state for rendering the pipeline.
pub struct RenderContext {
    pub instance: Instance,
    pub device: Device,
    pub queue: Queue,
}

impl RenderContext {
    pub async fn new() -> Result<Self> {
        let instance = Instance::new(wgpu::Backends::PRIMARY);
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let features = adapter.features();
        let mut limits = Limits::default();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: features
                        & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::CLEAR_TEXTURE),
                    limits,
                },
                None,
            )
            .await?;
        Ok(Self {
            instance,
            device,
            queue,
        })
    }

    /// Creates a new surface for the specified window and dimensions.
    pub fn create_surface<W>(&self, window: &W, width: u32, height: u32) -> RenderSurface
    where
        W: HasRawWindowHandle + HasRawDisplayHandle,
    {
        let surface = unsafe { self.instance.create_surface(window) };
        let format = wgpu::TextureFormat::Bgra8Unorm;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
        };
        surface.configure(&self.device, &config);
        RenderSurface { surface, config }
    }

    /// Resizes the surface to the new dimensions.
    pub fn resize_surface(&self, surface: &mut RenderSurface, width: u32, height: u32) {
        surface.config.width = width;
        surface.config.height = height;
        surface.surface.configure(&self.device, &surface.config);
    }
}

/// Combination of surface and its configuration.
pub struct RenderSurface {
    pub surface: Surface,
    pub config: SurfaceConfiguration,
}
