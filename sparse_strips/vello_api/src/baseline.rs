// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Versions of the APIs in Vello API which are portable to all rendering APIs.
#![expect(clippy::todo, unused_variables, reason = "This code is incomplete.")]

mod painter;
mod prepared;

pub use self::painter::BaselinePainter;
pub use self::prepared::BaselinePreparePaths;
