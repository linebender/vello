// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Port of Skia's `StripProcessorSimd.h`, adapted to 16 samples per pixel and 16×16 tiles.
//! `resolveWindingToAlpha` becomes a resolve to sample masks (see the module docs in `mod.rs`).
//!
//! OWNER: raster worker.
