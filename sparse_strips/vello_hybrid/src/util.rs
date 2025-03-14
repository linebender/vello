// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// This file is a modified version of the vello/src/util.rs file.

//! Simple helpers for managing wgpu state and surfaces.

use std::ops::RangeInclusive;

use wgpu::{
    Adapter, Device, Features, Instance, Limits, MemoryHints, Queue, Surface, SurfaceConfiguration,
    SurfaceTarget, TextureFormat,
};

/// Simple render context that maintains wgpu state for rendering the pipeline.
#[derive(Debug)]
pub struct RenderContext {
    /// The instance of the wgpu instance
    pub instance: Instance,
    /// The devices of the wgpu instance
    pub devices: Vec<DeviceHandle>,
}

/// A handle to a device
#[derive(Debug)]
pub struct DeviceHandle {
    /// The adapter of the device
    pub adapter: Adapter,
    /// The device
    pub device: Device,
    /// The queue of the device
    pub queue: Queue,
}

impl RenderContext {
    /// Creates a new render context
    #[expect(
        clippy::new_without_default,
        reason = "Creating a wgpu Instance is something which should only be done rarely"
    )]
    pub fn new() -> Self {
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
    pub async fn create_surface<'w>(
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
    pub async fn create_render_surface<'w>(
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
    pub fn resize_surface(&self, surface: &mut RenderSurface<'_>, width: u32, height: u32) {
        surface.config.width = width;
        surface.config.height = height;
        self.configure_surface(surface);
    }

    /// Sets the present mode for the surface
    pub fn set_present_mode(
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
    pub async fn device(&mut self, compatible_surface: Option<&Surface<'_>>) -> Option<usize> {
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
pub struct RenderSurface<'s> {
    /// The surface
    pub surface: Surface<'s>,
    /// The configuration of the surface
    pub config: SurfaceConfiguration,
    /// The device id
    pub dev_id: usize,
}

struct NullWake;

impl std::task::Wake for NullWake {
    fn wake(self: std::sync::Arc<Self>) {}
}

/// Represents dimension constraints for surfaces
#[derive(Debug)]
pub struct DimensionConstraints {
    /// The valid range for width, inclusive of min and max values
    pub width_range: RangeInclusive<f64>,
    /// The valid range for height, inclusive of min and max values
    pub height_range: RangeInclusive<f64>,
}

impl DimensionConstraints {
    /// Create new constraints with given min/max dimensions
    pub fn new(min_width: f64, min_height: f64, max_width: f64, max_height: f64) -> Self {
        Self {
            width_range: min_width..=max_width,
            height_range: min_height..=max_height,
        }
    }

    /// Calculate dimensions while preserving aspect ratio within constraints
    ///
    /// Some viewboxes could never fit inside this constraint. For example, if the constraint for both axes
    /// is 100.0..=2000.0, if `original_width` is `2.` and `original_height` is `1000.`, there is clearly
    /// no way for that to fit within the constraints.
    /// In these cases, this method clamps to within the ranges (respecting the constraints but losing the aspect ratio).
    pub fn calculate_dimensions(&self, original_width: f64, original_height: f64) -> (f64, f64) {
        // Ensure we have non-zero input dimensions
        let original_width = original_width.max(1.0);
        let original_height = original_height.max(1.0);

        let min_width = *self.width_range.start();
        let max_width = *self.width_range.end();
        let min_height = *self.height_range.start();
        let max_height = *self.height_range.end();

        let (width, height) = if original_width > max_width || original_height > max_height {
            // Scale down if dimensions exceed maximum limits
            let width_ratio = max_width / original_width;
            let height_ratio = max_height / original_height;
            let ratio = width_ratio.min(height_ratio);

            ((original_width * ratio), (original_height * ratio))
        } else if original_width < min_width || original_height < min_height {
            // Scale up if dimensions are below minimum limits
            let width_ratio = min_width / original_width;
            let height_ratio = min_height / original_height;
            let ratio = width_ratio.max(height_ratio);

            ((original_width * ratio), (original_height * ratio))
        } else {
            (original_width, original_height)
        };
        (
            width.clamp(min_width, max_width),
            height.clamp(min_height, max_height),
        )
    }

    /// Converts a floating point dimension to a u16.
    /// For the [default](DimensionConstraints::default) constraints, if the input value
    /// was returned from [`calculate_dimensions`](Self::calculate_dimensions), this function can't
    /// fail. But note that this does not necessarily apply for custom constraints.
    ///
    /// # Panics
    ///
    /// If `value` is negative or larger than [`u16::MAX`].
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Any truncation will be caught by the (saturating) casts into a wider type"
    )]
    #[track_caller]
    pub fn convert_dimension(value: f64) -> u16 {
        (value.ceil() as i32)
            .try_into()
            .expect("Dimensions are clamped into a reasonable range")
    }
}

impl Default for DimensionConstraints {
    /// Creates default constraints with reasonable values
    fn default() -> Self {
        Self {
            width_range: 100.0..=2000.0,
            height_range: 100.0..=2000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DimensionConstraints;

    #[test]
    fn calculate_dimensions_in_range() {
        let new = DimensionConstraints::new;
        for (test_name, constraints) in [
            ("Default", DimensionConstraints::default()),
            (
                "Max width different to max height",
                new(100., 100., 500., 1000.),
            ),
            ("Max width equal to min width", new(100., 100., 100., 1000.)),
            ("Very loose constraints", new(10., 10., 10_000., 10_000.)),
        ] {
            for [test_width, test_height] in [
                [100., 100.],
                [50., 200.],
                [10., 2_000.],
                [10_000., 10_000.],
                // Larger than `u16::MAX`
                [128_000., 128_000.],
            ] {
                let (width, height) = constraints.calculate_dimensions(test_width, test_height);
                assert!(
                    constraints.width_range.contains(&width),
                    "Constraints in {test_name} should have a width in the supported range.\n\
                    Got {width}x{height} from {test_width}x{test_height} in {constraints:?}"
                );
                assert!(constraints.height_range.contains(&height));
            }
        }
    }
}
