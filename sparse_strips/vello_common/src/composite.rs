// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Configuring how a render target is initialized before drawing.

/// Controls how existing target contents are handled before drawing.
///
/// The clear operation is generic so renderers can expose only the operations they support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeMode<C> {
    /// Preserve the existing target contents.
    Preserve,
    /// Apply the supplied clear operation before drawing.
    Clear(C),
}
