// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Configuring how a render target is initialized before drawing.

/// Controls how existing target contents are handled before drawing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetInit<C> {
    /// Composite rendered content over the existing target contents with src-over blending.
    SrcOver,
    /// Clear the drawing surface with the specified color before drawing.
    Clear(C),
}

impl<C> TargetInit<C> {
    /// Transform the clear value while preserving the target initialization mode.
    pub fn map<T>(self, f: impl FnOnce(C) -> T) -> TargetInit<T> {
        match self {
            Self::SrcOver => TargetInit::SrcOver,
            Self::Clear(clear) => TargetInit::Clear(f(clear)),
        }
    }
}
