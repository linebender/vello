// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A stub crate.
//! Planned to provide an abstraction between different renderers in the future.
//! This abstraction is not yet designed.

#![forbid(unsafe_code)]
#![no_std]

use core::{
    any::Any,
    sync::atomic::{AtomicU64, Ordering},
};

use alloc::{boxed::Box, sync::Arc};
use bitflags::bitflags;
use peniko::Color;

extern crate alloc;

mod design;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DownloadId(u64);

impl DownloadId {
    pub fn next() -> Self {
        static DOWNLOAD_IDS: AtomicU64 = AtomicU64::new(0);
        // Overflow: u64 starting at 0 incremented by 1 at a time, so cannot overflow.
        Self(DOWNLOAD_IDS.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug)]
pub struct SceneOptions {
    /// The target area within the Texture to render to.
    ///
    /// If `None`, will render to the whole texture.
    /// Format: (x0, y0, width, height).
    pub target: Option<(u16, u16, u16, u16)>,
    /// The color which the texture will be cleared to before drawing.
    ///
    /// If this is `None`, the previous content will be retained.
    /// This is useful for use cases such as the web's `CanvasRenderingContext2D`,
    /// which doesn't have automatic clearing.
    pub clear_color: Option<Color>,
}

// TODO: Maybe?
// pub enum MaskOperation {
//     SvgLuminance,
//     CssLuminance,
//     Alpha,
// }

pub trait Renderer {
    /// The `ScenePainter` is the encoder for rendering commands.
    ///
    /// *Ideally*, we'd allow this to borrow shared resources from
    /// the renderer (not exclusively).
    /// However, the lifetimes of that with `AnyRenderer` get messy fast.
    type ScenePainter: PaintScene + 'static;

    // TODO: Not complete.
    type PathPreparer: 'static;

    /// Create a texture for use in renders with this device.
    ///
    /// Cleanup is handled through `Drop`.
    fn create_texture(descriptor: TextureDescriptor) -> Texture;
    // fn create_mask(descriptor: MaskOperation) -> Mask;
    // fn mask_from_scene(from: &Texture, to: &Scene, MaskDescriptor { subset_rect,  });

    fn create_scene(&mut self, to: &Texture, options: SceneOptions) -> Self::ScenePainter;
    fn queue_render(&mut self, from: Self::ScenePainter);

    fn queue_download(&mut self, texture: &Texture) -> DownloadId;

    fn upload_image(to: &Texture, data: peniko::ImageData) -> Texture;

    /// API for efficient glyph rendering.
    // Needs: Shape, Transform, Bounds, Fill or Stroke
    // To render, needs integer translation, paint information.
    // As this is specialised to glyph drawing, I think that a batched API makes sense,
    // i.e. you start a `PathSet`, add several paths to it, then free them all at once.
    // That strategy has several advantages:
    // You gain support for "batched allocations", without fragmentation.
    // There is no possible error case.
    // We need to think about how we make it practical to actually get the integer translations,
    // because of "composition".
    fn prepare_paths(&mut self) -> Self::PathPreparer;
    fn finalise_prepared_paths(&mut self) -> PreparedPaths;
}

pub trait AnyScenePainter: Any {}
impl<T: PaintScene + Any> AnyScenePainter for T {}

pub trait AnyRenderer: Any {}
impl<T: Renderer + Any> AnyRenderer for T {}

pub trait PaintScene {}
pub struct ExampleImplementation<'a> {
    val: &'a str,
}

impl<'a> ExampleImplementation<'a> {
    pub fn new(/* No Width/Height arguments*/ name: &'a str) -> Self {
        ExampleImplementation { val: name }
    }

    pub fn create_texture(descriptor: TextureDescriptor) -> Texture {
        Texture {
            value: Arc::new(()),
            descriptor,
        }
    }

    /// Prepare to render to a `Texture`.
    pub fn prepare_render(to: &Texture, rect: (u16, u16, u16, u16)) -> Result<(), ()> {
        Ok(())
    }

    /// Queue the previously prepared render to be rendered in the next.
    pub fn queue_render(&mut self, to: &Texture) {}
    pub fn queue_download(&mut self, texture: &Texture) -> DownloadId {
        DownloadId::next()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TextureDescriptor {
    // TODO: Maybe a better type is `atomicow::CowArc<'static, str>`
    pub label: Option<&'static str>,
    pub width: u16,
    pub height: u16,
    pub usages: TextureUsages,
    // TODO: Format? Premultiplication? Hdr? Bitdepth?
}

pub struct Texture {
    value: Arc<dyn Any + Send + Sync>,
    descriptor: TextureDescriptor,
}

pub struct PreparedPaths {
    value: Arc<dyn Any + Send + Sync>,
}

pub trait TextureInnerMarker {}

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
