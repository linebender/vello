// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(all(feature = "debug_layers", feature = "wgpu"))]
mod renderer;
#[cfg(all(feature = "debug_layers", feature = "wgpu"))]
mod validate;

#[cfg(all(feature = "debug_layers", feature = "wgpu"))]
pub(crate) use renderer::*;

/// Bitflags for enabled debug operations.
///
/// Currently, all layers additionally require the `debug_layers` feature.
#[derive(Copy, Clone)]
pub struct DebugLayers(u8);

// TODO: Currently all layers require read-back of the BumpAllocators buffer. This isn't strictly
// necessary for layers other than `VALIDATION`. The debug visualizations use the bump buffer only
// to obtain various instance counts for draws and these could instead get written out to an
// indirect draw buffer. OTOH `VALIDATION` should always require readback since we want to be able
// to run the same CPU-side tests for both CPU and GPU shaders.
impl DebugLayers {
    /// Visualize the bounding box of every path.
    /// Requires the `debug_layers` feature.
    pub const BOUNDING_BOXES: DebugLayers = DebugLayers(1 << 0);

    /// Visualize the post-flattening line segments using line primitives.
    /// Requires the `debug_layers` feature.
    pub const LINESOUP_SEGMENTS: DebugLayers = DebugLayers(1 << 1);

    /// Visualize the post-flattening line endpoints.
    /// Requires the `debug_layers` feature.
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
    /// Requires the `debug_layers` feature.
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
        self.0 ^= mask.0;
    }
}

impl std::ops::BitOr for DebugLayers {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
