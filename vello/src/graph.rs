// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A render graph for Vello.
//!
//! This enables the use of image filters among other things.

/// When making an image filter graph, we need to know a few things:
///
/// 1) The Scene to draw.
/// 2) The resolution of the filter target (i.e. input image).
/// 3) The resolution of the output image.
///
/// The scene to draw might be a texture from a previous step or externally provided.
/// The resolution of the input might change depending on the resolution of the
/// output, because of scaling/rotation/skew.
pub struct Thinking;
