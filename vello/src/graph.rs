// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A render graph for Vello.
//!
//! This enables the use of image filters among other things.

mod runner;

use std::{
    borrow::Cow,
    collections::HashMap,
    hash::Hash,
    num::Wrapping,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
};

use peniko::Image;

// --- MARK: Public API ---

pub struct Vello {}

/// A partial render graph.
///
/// There is expected to be one Gallery per thread.
pub struct Gallery {
    id: u64,
    generation: Generation,
    incoming_deallocations: Receiver<PaintingId>,
    deallocator: Sender<PaintingId>,
    paintings: HashMap<PaintingId, (PaintingSource, Generation)>,
}

/// A handle to an image managed by the renderer.
///
/// This resource is reference counted, and corresponding resources
/// are freed when a rendering operation occurs.
#[derive(Clone)]
pub struct Painting {
    inner: Arc<PaintingInner>,
}

pub enum OutputSize {
    Fixed { width: u32, height: u32 },
    Inferred,
}

impl Gallery {
    pub fn new() -> Self {
        static GALLERY_IDS: AtomicU64 = AtomicU64::new(1);
        // Overflow handling: u64 incremented so can never overflow
        let id = GALLERY_IDS.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::channel();
        Gallery {
            id,
            generation: Generation::default(),
            paintings: HashMap::default(),
            deallocator: tx,
            incoming_deallocations: rx,
        }
    }
}

impl Default for Gallery {
    fn default() -> Self {
        Self::new()
    }
}

impl Gallery {
    pub fn create_painting(&self, label: impl Into<Cow<'static, str>>) -> Painting {
        Painting {
            inner: Arc::new(PaintingInner {
                label: label.into(),
                deallocator: self.deallocator.clone(),
                id: PaintingId::next(),
                gallery: self.id,
            }),
        }
    }

    pub fn paint(&mut self, painting: &Painting) -> Painter<'_> {
        self.generation.nudge();
        Painter {
            gallery: self,
            painting: painting.inner.id,
        }
    }
}

pub struct Painter<'a> {
    gallery: &'a mut Gallery,
    painting: PaintingId,
}

impl Painter<'_> {
    pub fn as_image(self, image: Image) {
        self.insert(PaintingSource::Image(image));
    }
    pub fn as_subregion(self, from: Painting, x: u32, y: u32, width: u32, height: u32) {
        self.insert(PaintingSource::Region {
            painting: from,
            x,
            y,
            width,
            height,
        });
    }
    pub fn as_resample(self, from: Painting, to_dimensions: OutputSize) {
        self.insert(PaintingSource::Resample(from, to_dimensions));
    }
    pub fn as_scene(self, scene: Scene, of_dimensions: OutputSize) {
        self.insert(PaintingSource::Scene(scene, of_dimensions));
    }

    fn insert(self, new_source: PaintingSource) {
        match self.gallery.paintings.entry(self.painting) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                entry.get_mut().0 = new_source;
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert((new_source, Generation::default()));
            }
        };
    }
}

// --- MARK: Internal types ---

/// The shared elements of a `Painting`.
///
/// A painting's identity is its heap allocation; most of
/// the resources are owned by its [`Gallery`].
/// This only stores peripheral information.
struct PaintingInner {
    label: Cow<'static, str>,
    deallocator: Sender<PaintingId>,
    id: PaintingId,
    gallery: u64,
}

impl Drop for PaintingInner {
    fn drop(&mut self) {
        // Ignore the possibility that the corresponding gallery has already been dropped.
        let _ = self.deallocator.send(self.id);
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct PaintingId(u64);

impl PaintingId {
    fn next() -> Self {
        static PAINTING_IDS: AtomicU64 = AtomicU64::new(0);
        Self(PAINTING_IDS.fetch_add(1, Ordering::Relaxed))
    }
}

enum PaintingSource {
    Image(Image),
    Scene(Scene, OutputSize),
    Resample(Painting, OutputSize /* Algorithm */),
    Region {
        painting: Painting,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
}

#[derive(Default)]
struct Generation(Wrapping<u32>);

impl Generation {
    fn nudge(&mut self) {
        self.0 += 1;
    }
}

// --- MARK: Model ---
// A model of the rest of Vello.

/// A single Scene
pub struct Scene {}

impl Scene {
    #[doc(alias = "image")]
    pub fn painting(&mut self, painting: Painting, width: u32, height: u32) {}
}

// --- MARK: Musings ---

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
