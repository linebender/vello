// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This is a utility library to help integrate `vello_hybrid` WebGPU wgsl shaders into glsl.

#[cfg(feature = "glsl")]
mod compile;
#[cfg(feature = "glsl")]
mod types;

#[cfg(feature = "glsl")]
use naga_28 as naga;

include!(concat!(env!("OUT_DIR"), "/compiled_shaders.rs"));
