// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Texture types used in Vello API.

use alloc::sync::Arc;
use bitflags::bitflags;
use core::{fmt::Debug, hash::Hash};

use crate::Renderer;

#[derive(Copy, Clone, Debug)]
pub struct TextureDescriptor {
    // TODO: Maybe a better type is `atomicow::CowArc<'static, str>`
    pub label: Option<&'static str>,
    pub width: u16,
    pub height: u16,
    pub usages: TextureUsages,
    // TODO: Format? Premultiplication? Hdr?
    // TODO: Explicit atlasing?
}

// TODO: Generational index?
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct TextureId(u64);

impl TextureId {
    pub const fn to_raw(&self) -> u64 {
        self.0
    }
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
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

/// A reference counted handle to a texture owned by a [`Renderer`].
#[derive(Debug, Clone)]
pub struct TextureHandle {
    inner: Arc<TextureInner>,
}

impl TextureHandle {
    pub fn id(&self) -> TextureId {
        self.inner.id
    }

    pub(crate) fn new(renderer: Arc<dyn Renderer>, id: TextureId) -> Self {
        Self {
            inner: Arc::new(TextureInner { renderer, id }),
        }
    }
}

impl Hash for TextureHandle {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}
impl PartialEq for TextureHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::as_ptr(&self.inner) == Arc::as_ptr(&other.inner)
    }
}
impl Eq for TextureHandle {}

struct TextureInner {
    renderer: Arc<dyn Renderer>,
    id: TextureId,
}

impl Drop for TextureInner {
    fn drop(&mut self) {
        // TODO: Maybe just log?
        self.renderer
            .free_texture_raw(self.id)
            .expect("Vello API texture managed as a `Texture` shouldn't have been manually freed.");
    }
}

impl Debug for TextureInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PathGroupInner")
            .field("renderer", &"elided")
            .field("id", &self.id)
            .finish()
    }
}
