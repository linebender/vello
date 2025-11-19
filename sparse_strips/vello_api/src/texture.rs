// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Texture types used in Vello API.

use bitflags::bitflags;
use core::{any::Any, fmt::Debug};

#[derive(Copy, Clone, Debug)]
pub struct TextureDescriptor {
    // TODO: Maybe a better type is `atomicow::CowArc<'static, str>`
    pub label: Option<&'static str>,
    pub width: u16,
    pub height: u16,
    pub usages: TextureUsages,
    // TODO: Format? Premultiplication? Hdr?
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct TextureId(u64);

impl TextureId {
    pub fn next() -> Self {
        #[cfg(target_has_atomic = "64")]
        {
            use core::sync::atomic::{self, Ordering};

            // Overflow: u64 starting at 0 incremented by 1 at a time, so cannot overflow.
            static DOWNLOAD_IDS: atomic::AtomicU64 = atomic::AtomicU64::new(0);
            Self(DOWNLOAD_IDS.fetch_add(1, Ordering::Relaxed))
        }
        #[cfg(not(target_has_atomic = "64"))]
        {
            // Overflow: We expect running this code on 32-bit targets to be rare enough in practise
            // that we don't handle overflow.
            static DOWNLOAD_IDS: atomic::AtomicU32 = atomic::AtomicU32::new(0);
            Self(DOWNLOAD_IDS.fetch_add(1, Ordering::Relaxed).into())
        }
    }
    pub fn to_raw(&self) -> u64 {
        self.0
    }
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    // TODO: Do we want a `from_raw`?
}

bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct TextureUsages: u32 {
        /// This texture can be used as the target in a `prepare_render` operation.
        const RENDER_TARGET = 1 << 0;
        /// This texture can be the target of an "upload image" operation.
        const UPLOAD_TARGET = 1 << 1;
        /// This texture can be the source for a "download" operation.
        const DOWNLOAD_SRC = 1 << 2;
        /// This texture (or a subset of it) can be used for painting.
        const TEXTURE_BINDING = 1 << 3;
        // TODO: Does this make sense to support/require this?
        // /// A subset of this texture can be rendered to.
        // const PARTIAL_RENDER_TARGET = 1<<4;

        /// The usages for an external texture representing a GPU surface.
        const SURFACE = Self::RENDER_TARGET.bits();
        /// The usages for an uploaded texture.
        const UPLOAD = Self::UPLOAD_TARGET.bits() | Self::TEXTURE_BINDING.bits();
        /// The usages for a texture which we want to queue for download.
        const DOWNLOAD = Self::RENDER_TARGET.bits() | Self::DOWNLOAD_SRC.bits();
    }
}
