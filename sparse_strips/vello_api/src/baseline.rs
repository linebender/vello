// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Versions of the APIs in Vello API which are portable to all rendering APIs.
//!
//! These can be used both to polyfill unimplemented parts of Vello API, as well as for potentially
//! acting like the existing Vello `Scene` in that you need no prerequisites to create it.

mod painter;
mod prepared;

pub use self::painter::BaselinePainter;
pub use self::prepared::BaselinePreparePaths;
