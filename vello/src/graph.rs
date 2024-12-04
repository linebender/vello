// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A render graph for Vello.
//!
//! This enables the use of image filters among other things.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use peniko::Image;

struct Generation(u32);

impl Generation {}

/// A partial render graph.
pub struct Gallery {
    id: u32,
    generation: Generation,
    paintings: HashMap<Painting, PaintingSource>,
}

enum PaintingSource {
    Scene(Scene),
    Image(Image),
    Region {
        painting: Painting,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
}

/// An allocator of [`Painting`]s, associated with a [`Vello`].
pub struct Curator(Arc<CuratorInner>);

struct CuratorInner {
    painting_ids: AtomicU64,
}

impl Curator {
    pub fn painting(&self) -> Painting {
        self.0.painting_ids.fetch_add(1, Ordering::Relaxed);
        Painting {}
    }
}

pub enum OutputSize {
    Fixed { width: u32, height: u32 },
    Inferred,
}

impl Gallery {
    pub fn replace_image(&mut self, _texture: Painting, _image: Image) {}
    pub fn subregion(
        &mut self,
        target: Painting,
        of_: Painting,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) {
    }
    pub fn resample(&mut self, painting: Painting, dimensions: OutputSize) -> Painting {
        Painting {}
    }
}

/// A single Scene
pub struct Scene {}

impl Scene {
    #[doc(alias = "image")]
    pub fn painting(&mut self, drawing: Painting, width: u32, height: u32) {}
}

/// A handle to an image managed by the renderer.
///
/// The corresponding resource is reference counted.
#[derive(Clone)]
pub struct Painting {}

/// When making an image filter graph, we need to know a few things:
///
/// 1) The Scene to draw.
/// 2) The resolution of the filter target (i.e. input image).
/// 3) The resolution of the output image.
///
/// The scene to draw might be a texture from a previous step or externally provided.
/// The resolution of the input might change depending on the resolution of the
/// output, because of scaling/rotation/skew.
pub struct Thinking;

/// What threading model do we want. Requirements:
/// 1) Creating scenes on different threads should be possible.
/// 2) Scenes created on different threads should be able to use filter effects.
/// 3) We should only upload each CPU side image once.
///
///
pub struct Threading;

/// Question: What do we win from backpropogating render sizes?
/// Answer: Image sampling
///
/// Conclusion: Special handling of "automatic" scene sizing to
/// render multiple times if needed.
///
/// Conclusion: Two phase approach, backpropogating from every scene
/// with a defined size?
pub struct ThinkingAgain;

/// Do we want custom fully graph nodes?
/// Answer for now: No?
pub struct Scheduling;
