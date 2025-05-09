// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Different execution modes for kernels.

#[derive(Copy, Clone, Debug)]
/// The execution mode used for the rendering process.
pub enum ExecutionMode {
    /// Only use scalar execution. This is recommended if you want to have
    /// consistent results across different platforms and want to avoid unsafe code,
    /// and is the only option if you disabled the `simd` feature. Performance will be
    /// worse, though.
    Scalar,
    /// Select the best execution mode according to what is available on the host system.
    /// This is the recommended option for highest performance.
    #[cfg(feature = "simd")]
    Auto,
    /// Force the usage of neon SIMD instructions. This will lead to panics in case
    /// the CPU doesn't support the target feature `neon`.
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    Neon,
    /// Force the usage of AVX2 SIMD instructions. This will lead to panics in case
    /// the CPU doesn't support the target features `avx2` and `fma`.
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    Avx2,
}

#[cfg(feature = "simd")]
impl Default for ExecutionMode {
    fn default() -> Self {
        Self::Auto
    }
}

#[cfg(not(feature = "simd"))]
impl Default for ExecutionMode {
    fn default() -> Self {
        Self::Scalar
    }
}

/// Scalar execution mode.
#[derive(Debug)]
pub struct Scalar;

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
#[derive(Debug)]
/// Execute using NEON intrinsics.
pub struct Neon;

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[derive(Debug)]
/// Execute using AVX2 intrinsics.
pub struct Avx2;
