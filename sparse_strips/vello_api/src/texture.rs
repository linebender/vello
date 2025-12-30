// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Types and utilities for textures, which are 2d images owned by a [`Renderer`].
//!
//! Most textures in Vello Hybrid are reference counted handles, and stored as [`TextureHandle`].
//! As the data referred to by owned by the `TextureHandle`s are "renderer-specific".
//! This means that it can only be used with [`PaintScene`][crate::PaintScene]s which are
//! associated with that renderer.
//! This is validated automatically at scene construction time.
//!
//! You can create a texture using [`Renderer::create_texture`]; the underlying texture's
//! content/memory region will be owned by the renderer.
//! These textures can be used in a few ways, depending on the [`TextureUsages`] they are created with.
//! These are documented on the associated constants of `TextureUsages`.
//!
//! Certain renderers may also support creating textures from external content.
//! For example, [`VelloHybrid::add_external_texture`][todo] allows creating textures
//! corresponding to a `wgpu::TextureView`.
//! This can be used to composite in external content, or for rendering directly to a `Surface`
//! (i.e. the contents of an operating system window) with Vello Hybrid.
//! These methods can be accessed through downcasting the renderer to the specific implementation.
//! Similar mechanisms are also used to provide access to a texture's current content on the CPU.
//!
//! We expect Vello API to gain support for scheduling "downloads" of a texture's content, but
//! this isn't currently implemented.
//! It is currently not supported in general to transfer textures from one renderer
//! to another renderer.
//! This download mechanism *may* allow that in the future, although this may be impossible
//! for some textures due to their `TextureUsages`.
//!
//! It is possible for advanced users to manually allocate and release textures without creating a handle,
//! by calling to [`Renderer::alloc_untracked_texture`] and [`Renderer::free_untracked_texture`].
//! This is not recommended for most users, as it will likely requires a lot more bookkeeping on your side.
//! Additionally, we don't expect textures to be created regularly enough for the allocation cost
//! of this handle to be significant.
//! The ids returned from `alloc_untracked_texture` are renderer specific, in the same way as `TextureHandle`s are.
//! However, this will not be validated at scene construction time (but incorrect usage might lead to future crashes).

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
    // TODO: Format? Premultiplication? Hdr? Opaqueness?
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
        /// This texture (or a subset of it) can be used for painting.
        const TEXTURE_BINDING = 1 << 2;
        /// This texture can be the source for a "download" operation.
        ///
        /// This isn't currently exposed in the API, but might have
        /// a meaning provided by the backend.
        const DOWNLOAD_SRC = 1 << 3;
        // TODO: Does this make sense to support/require this be supported?
        // /// A subset of this texture can be rendered to.
        // const PARTIAL_RENDER_TARGET = 1<<4;
    }
}

/// A reference counted handle to a texture owned by a [`Renderer`].
#[derive(Debug, Clone)]
pub struct TextureHandle {
    inner: Arc<TextureInner>,
}

impl TextureHandle {
    /// Access the "raw" id of this texture.
    pub fn id(&self) -> TextureId {
        self.inner.id
    }
    // TODO: Consider making not `pub(crate)`. We need to be careful that people don't create
    // multiple of these for the same texture accidentally.
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
            .free_untracked_texture(self.id)
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
