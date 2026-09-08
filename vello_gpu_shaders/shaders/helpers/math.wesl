// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared mathematical constants.

// Mathematical constants.
const PI: f32 = 3.1415926535897932384626433832795028;
const TWO_PI: f32 = 2.0 * PI;
// Tolerance for nearly zero comparisons (matching vello_cpu implementation).
// Note: This must match SCALAR_NEARLY_ZERO in vello_common/src/math.rs
// @see {@link https://github.com/linebender/vello/blob/748ba4c7a8973f642f778591b09658d8ee6e1132/sparse_strips/vello_common/src/math.rs#L21}
const NEARLY_ZERO_TOLERANCE: f32 = 1.0 / 4096.0;
