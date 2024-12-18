// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(all(feature = "debug_layers", feature = "wgpu"))]
mod renderer;
#[cfg(all(feature = "debug_layers", feature = "wgpu"))]
mod validate;

use std::fmt::Debug;

#[cfg(all(feature = "debug_layers", feature = "wgpu"))]
pub(crate) use renderer::*;

/// Bitflags for enabled debug operations.
///
/// Currently, all layers additionally require the `debug_layers` feature.
#[cfg_attr(docsrs, doc(hidden))]
#[derive(Copy, Clone)]
pub struct DebugLayers(u8);

impl Debug for DebugLayers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut tuple = f.debug_tuple("DebugLayers");
        if self.contains(Self::BOUNDING_BOXES) {
            tuple.field(&"BOUNDING_BOXES");
        }
        if self.contains(Self::LINESOUP_SEGMENTS) {
            tuple.field(&"LINESOUP_SEGMENTS");
        }
        if self.contains(Self::LINESOUP_POINTS) {
            tuple.field(&"LINESOUP_POINTS");
        }
        if self.contains(Self::VALIDATION) {
            tuple.field(&"VALIDATION");
        }

        tuple.finish()
    }
}

// TODO: Currently all layers require read-back of the BumpAllocators buffer. This isn't strictly
// necessary for layers other than `VALIDATION`. The debug visualizations use the bump buffer only
// to obtain various instance counts for draws and these could instead get written out to an
// indirect draw buffer. OTOH `VALIDATION` should always require readback since we want to be able
// to run the same CPU-side tests for both CPU and GPU shaders.
impl DebugLayers {
    /// Visualize the bounding box of every path.
    /// Requires the `debug_layers` feature.
    pub const BOUNDING_BOXES: Self = Self(1 << 0);

    /// Visualize the post-flattening line segments using line primitives.
    /// Requires the `debug_layers` feature.
    pub const LINESOUP_SEGMENTS: Self = Self(1 << 1);

    /// Visualize the post-flattening line endpoints.
    /// Requires the `debug_layers` feature.
    pub const LINESOUP_POINTS: Self = Self(1 << 2);

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
    pub const VALIDATION: Self = Self(1 << 3);

    /// Construct a `DebugLayers` from the raw bits.
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Get the raw representation of this value.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// A `DebugLayers` with no layers enabled.
    pub const fn none() -> Self {
        Self(0)
    }

    /// A `DebugLayers` with all layers enabled.
    pub const fn all() -> Self {
        // Custom BitOr is not const, so need to manipulate the inner value here
        Self(
            Self::BOUNDING_BOXES.0
                | Self::LINESOUP_SEGMENTS.0
                | Self::LINESOUP_POINTS.0
                | Self::VALIDATION.0,
        )
    }

    /// True if this `DebugLayers` has no layers enabled.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Determine whether `self` is a superset of `mask`.
    pub const fn contains(self, mask: Self) -> bool {
        self.0 & mask.0 == mask.0
    }

    /// Toggle the value of the layers specified in mask.
    pub fn toggle(&mut self, mask: Self) {
        self.0 ^= mask.0;
    }
}

/// Returns the union of the two input `DebugLayers`.
impl std::ops::BitOr for DebugLayers {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
