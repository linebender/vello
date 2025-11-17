// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A stub crate.
//! Planned to provide an abstraction between different renderers in the future.
//! This abstraction is not yet designed.

#![forbid(unsafe_code)]
#![no_std]

use core::{
    any::Any,
    sync::atomic::{self, Ordering},
};

use crate::{
    prepared::{PreparePaths, PreparedPaths},
    texture::{Texture, TextureDescriptor},
};
use alloc::sync::Arc;
use peniko::Color;

extern crate alloc;

mod design;
mod free_list;
pub mod prepared;
pub mod texture;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DownloadId(u64);

impl DownloadId {
    pub fn next() -> Self {
        #[cfg(target_has_atomic = "64")]
        {
            // Overflow: u64 starting at 0 incremented by 1 at a time, so cannot overflow.
            static DOWNLOAD_IDS: atomic::AtomicU64 = atomic::AtomicU64::new(0);
            Self(DOWNLOAD_IDS.fetch_add(1, Ordering::Relaxed))
        }
        #[cfg(not(target_has_atomic = "64"))]
        {
            // Overflow: We expect running this code on 32-bit targets to be rare enough in practise
            // that we don't handle overflow.
            // Overflow could only really happen in practise if you are "racing" two renderers.
            static DOWNLOAD_IDS: atomic::AtomicU32 = atomic::AtomicU32::new(0);
            Self(DOWNLOAD_IDS.fetch_add(1, Ordering::Relaxed).into())
        }
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
    type PathPreparer: PreparePaths;

    /// Create a texture for use in renders with this device.
    ///
    /// Cleanup is handled through `Drop`.
    fn create_texture(descriptor: TextureDescriptor) -> Texture;

    // fn create_mask(descriptor: MaskOperation) -> Mask;
    // fn mask_from_scene(from: &Texture, to: &Scene, MaskDescriptor { subset_rect,  });

    fn create_scene(&mut self, to: &Texture, options: SceneOptions) -> Self::ScenePainter;
    fn queue_render(&mut self, from: Self::ScenePainter);

    fn queue_download(&mut self, texture: &Texture) -> DownloadId;

    fn directly_upload_image(
        data: peniko::ImageData,
        region: Option<(u16, u16, u16, u16)>,
    ) -> Result<Texture, ()>;
    fn upload_image(
        to: &Texture,
        data: peniko::ImageData,
        region: Option<(u16, u16, u16, u16)>,
    ) -> Result<(), ()>;

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
    fn take_prepared_paths(&mut self, from: PreparedPaths) -> Self::PathPreparer;
    fn finalise_prepared_paths(&mut self, from: Self::PathPreparer) -> PreparedPaths;
}

pub trait AnyScenePainter: Any {}
impl<T: PaintScene + Any> AnyScenePainter for T {}

pub trait AnyRenderer: Any {}
impl<T: Renderer + Any> AnyRenderer for T {}

pub trait PaintScene {}
pub struct ExampleImplementation<'a> {
    val: &'a str,
}
