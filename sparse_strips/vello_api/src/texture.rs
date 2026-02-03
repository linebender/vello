// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Types and utilities for textures, which are 2d images owned by a renderer.

/// A renderer-specific ID for a texture.
///
/// Note that we don't currently expect these to be constructed,
/// because the.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct TextureId(u64);

impl TextureId {
    /// Get the value stored in this texture id.
    ///
    /// The value this returns has a renderer-specific meaning, and so this method
    /// should generally only be called by renderer implementations.
    pub const fn to_raw(&self) -> u64 {
        self.0
    }

    /// Create a new texture id to refer to an existing value.
    ///
    /// It is not valid to store an exposed pointer address in this type.
    ///
    /// The value this returns has a renderer-specific meaning, and so this method
    /// should generally only be called by renderer implementations.
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
}
