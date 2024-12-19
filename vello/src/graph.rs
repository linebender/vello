// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A render graph for Vello.
//!
//! This enables the use of image filters among other things.

#![warn(
    missing_debug_implementations,
    elided_lifetimes_in_paths,
    single_use_lifetimes,
    unnameable_types,
    unreachable_pub,
    clippy::return_self_not_must_use,
    clippy::cast_possible_truncation,
    clippy::missing_assert_message,
    clippy::shadow_unrelated,
    clippy::missing_panics_doc,
    clippy::print_stderr,
    clippy::use_self,
    clippy::match_same_arms,
    clippy::missing_errors_doc,
    clippy::todo,
    clippy::partial_pub_fields,
    reason = "Lint set, currently allowed crate-wide"
)]

mod runner;

use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::Debug,
    hash::Hash,
    num::Wrapping,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc, LazyLock,
    },
};

use peniko::{kurbo::Affine, Blob, Brush, Extend, Image, ImageFormat, ImageQuality};

use crate::Scene;

// --- MARK: Public API ---

#[derive(Debug)]
pub struct Vello {}

/// A partial render graph.
///
/// There is expected to be one Gallery per thread.
pub struct Gallery {
    id: GalleryId,
    label: Cow<'static, str>,
    generation: Generation,
    incoming_deallocations: Receiver<PaintingId>,
    deallocator: Sender<PaintingId>,
    paintings: HashMap<PaintingId, (PaintingSource, Generation)>,
}

impl Debug for Gallery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}({})", self.id, self.label))
    }
}

/// A handle to an image managed by the renderer.
///
/// This resource is reference counted, and corresponding resources
/// are freed when a rendering operation occurs.
#[derive(Clone)]
pub struct Painting {
    inner: Arc<PaintingInner>,
}

impl Debug for Painting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}({})", self.inner.id, self.inner.label))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OutputSize {
    // Is u16 here reasonable?
    pub width: u16,
    pub height: u16,
}

impl Gallery {
    pub fn new(label: impl Into<Cow<'static, str>>) -> Self {
        let id = GalleryId::next();
        Self::new_inner(id, label.into())
    }
    pub fn new_anonymous(prefix: &'static str) -> Self {
        let id = GalleryId::next();
        let label = format!("{prefix}-{id:02}", id = id.0);
        Self::new_inner(id, label.into())
    }
    pub fn gc(&mut self) {
        let mut made_change = false;
        loop {
            let try_recv = self.incoming_deallocations.try_recv();
            let dealloc = match try_recv {
                Ok(dealloc) => dealloc,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    unreachable!("We store a sender alongside the receiver")
                }
            };
            self.paintings.remove(&dealloc);
            made_change = true;
        }
        if made_change {
            self.generation.nudge();
        }
    }
    fn new_inner(id: GalleryId, label: Cow<'static, str>) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            id,
            label,
            generation: Generation::default(),
            paintings: HashMap::default(),
            deallocator: tx,
            incoming_deallocations: rx,
        }
    }
}

impl Gallery {
    pub fn create_painting(&mut self, label: impl Into<Cow<'static, str>>) -> Painting {
        Painting {
            inner: Arc::new(PaintingInner {
                label: label.into(),
                deallocator: self.deallocator.clone(),
                id: PaintingId::next(),
                gallery_id: self.id,
            }),
        }
    }

    /// The painting must have [been created for](Self::create_painting) this gallery.
    ///
    /// This restriction ensures that work does.
    pub fn paint(&mut self, painting: &Painting) -> Option<Painter<'_>> {
        if painting.inner.gallery_id == self.id {
            // TODO: Return error about mismatched Gallery.
            return None;
        }
        self.generation.nudge();
        Some(Painter {
            gallery: self,
            painting: painting.inner.id,
        })
    }
}

/// Defines how a [`Painting`] will be drawn.
#[derive(Debug)]
pub struct Painter<'a> {
    gallery: &'a mut Gallery,
    painting: PaintingId,
}

impl Painter<'_> {
    pub fn as_image(self, image: Image) {
        self.insert(PaintingSource::Image(image));
    }
    pub fn as_subregion(self, from: Painting, x: u16, y: u16, width: u16, height: u16) {
        self.insert(PaintingSource::Region {
            painting: from,
            x,
            y,
            size: OutputSize { width, height },
        });
    }
    pub fn as_resample(self, from: Painting, to_dimensions: OutputSize) {
        self.insert(PaintingSource::Resample(from, to_dimensions));
    }
    pub fn as_scene(self, scene: Canvas, of_dimensions: OutputSize) {
        self.insert(PaintingSource::Canvas(scene, of_dimensions));
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
    id: PaintingId,
    deallocator: Sender<PaintingId>,
    label: Cow<'static, str>,
    gallery_id: GalleryId,
}

impl Drop for PaintingInner {
    fn drop(&mut self) {
        // Ignore the possibility that the corresponding gallery has already been dropped.
        let _ = self.deallocator.send(self.id);
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct PaintingId(u64);

impl Debug for PaintingId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("#{}", self.0))
    }
}

impl PaintingId {
    fn next() -> Self {
        static PAINTING_IDS: AtomicU64 = AtomicU64::new(0);
        Self(PAINTING_IDS.fetch_add(1, Ordering::Relaxed))
    }
}

/// The id of a gallery.
///
/// The debug label is provided for error messaging when
/// a painting is used with the wrong gallery.
#[derive(Clone, Copy, PartialEq, Eq)]
struct GalleryId(u64);

impl Debug for GalleryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("#{}", self.0))
    }
}

impl GalleryId {
    fn next() -> Self {
        static GALLERY_IDS: AtomicU64 = AtomicU64::new(1);
        // Overflow handling: u64 incremented so can never overflow
        let id = GALLERY_IDS.fetch_add(1, Ordering::Relaxed);
        Self(id)
    }
}

#[derive(Debug)]
enum PaintingSource {
    Image(Image),
    Canvas(Canvas, OutputSize),
    Resample(Painting, OutputSize /* Algorithm */),
    Region {
        painting: Painting,
        x: u16,
        y: u16,
        size: OutputSize,
    },
}

#[derive(Default, Debug, PartialEq, Eq, Clone)]
// Not copy because the identity is important; don't want to modify an accidental copy
struct Generation(Wrapping<u32>);

impl Generation {
    fn nudge(&mut self) {
        self.0 += 1;
    }
}

// --- MARK: Model ---
// A model of the rest of Vello.

/// A single Scene, potentially containing paintings.
pub struct Canvas {
    scene: Box<Scene>,
    paintings: HashMap<u64, Painting>,
}

#[derive(Debug)]
/// Created using [`Canvas::new_image`].
pub struct PaintingConfig {
    image: Image,
}

impl PaintingConfig {
    fn new(width: u16, height: u16) -> Self {
        // Create a fake Image, with an empty Blob. We can re-use the allocation between these.
        static EMPTY_ARC: LazyLock<Arc<[u8; 0]>> = LazyLock::new(|| Arc::new([]));
        let data = Blob::new(EMPTY_ARC.clone());
        let image = Image::new(data, ImageFormat::Rgba8, width.into(), height.into());
        Self { image }
    }
    pub fn brush(self) -> Brush {
        Brush::Image(self.image)
    }
    pub fn image(&self) -> &Image {
        &self.image
    }
    /// Builder method for setting the image [extend mode](Extend) in both
    /// directions.
    #[must_use]
    pub fn with_extend(self, mode: Extend) -> Self {
        Self {
            image: self.image.with_extend(mode),
        }
    }

    /// Builder method for setting the image [extend mode](Extend) in the
    /// horizontal direction.
    #[must_use]
    pub fn with_x_extend(self, mode: Extend) -> Self {
        Self {
            image: self.image.with_x_extend(mode),
        }
    }

    /// Builder method for setting the image [extend mode](Extend) in the
    /// vertical direction.
    #[must_use]
    pub fn with_y_extend(self, mode: Extend) -> Self {
        Self {
            image: self.image.with_y_extend(mode),
        }
    }

    /// Builder method for setting a hint for the desired image [quality](ImageQuality)
    /// when rendering.
    #[must_use]
    pub fn with_quality(self, quality: ImageQuality) -> Self {
        Self {
            image: self.image.with_quality(quality),
        }
    }

    /// Returns the image with the alpha multiplier set to `alpha`.
    #[must_use]
    #[track_caller]
    pub fn with_alpha(self, alpha: f32) -> Self {
        Self {
            image: self.image.with_alpha(alpha),
        }
    }

    /// Returns the image with the alpha multiplier multiplied again by `alpha`.
    /// The behaviour of this transformation is undefined if `alpha` is negative.
    #[must_use]
    #[track_caller]
    pub fn multiply_alpha(self, alpha: f32) -> Self {
        Self {
            image: self.image.multiply_alpha(alpha),
        }
    }
}

impl From<Scene> for Canvas {
    fn from(value: Scene) -> Self {
        Self::from_scene(Box::new(value))
    }
}

impl Default for Canvas {
    fn default() -> Self {
        Self::new()
    }
}

impl Canvas {
    pub fn new() -> Self {
        Self::from_scene(Box::<Scene>::default())
    }
    pub fn from_scene(scene: Box<Scene>) -> Self {
        Self {
            scene,
            paintings: HashMap::default(),
        }
    }
    pub fn new_image(&mut self, painting: Painting, width: u16, height: u16) -> PaintingConfig {
        let config = PaintingConfig::new(width, height);
        self.override_image(&config.image, painting);
        config
    }

    #[doc(alias = "image")]
    pub fn draw_painting(
        &mut self,
        painting: Painting,
        width: u16,
        height: u16,
        transform: Affine,
    ) {
        let image = self.new_image(painting, width, height);
        self.scene.draw_image(&image.image, transform);
    }

    #[deprecated(note = "Prefer `draw_painting` for greater efficiency")]
    pub fn draw_image(&mut self, image: &Image, transform: Affine) {
        self.scene.draw_image(image, transform);
    }

    pub fn override_image(&mut self, image: &Image, painting: Painting) {
        self.paintings.insert(image.data.id(), painting);
    }
}

impl Deref for Canvas {
    type Target = Scene;

    fn deref(&self) -> &Self::Target {
        &self.scene
    }
}
impl DerefMut for Canvas {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.scene
    }
}

impl Debug for Canvas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Canvas")
            .field("scene", &"elided")
            .field("paintings", &self.paintings)
            .finish()
    }
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
#[derive(Debug)]
pub struct Thinking;

/// What threading model do we want. Requirements:
/// 1) Creating scenes on different threads should be possible.
/// 2) Scenes created on different threads should be able to use filter effects.
/// 3) We should only upload each CPU side image once.
#[derive(Debug)]
pub struct Threading;

/// Question: What do we win from backpropogating render sizes?
/// Answer: Image sampling
///
/// Conclusion: Special handling of "automatic" scene sizing to
/// render multiple times if needed.
///
/// Conclusion: Two phase approach, backpropogating from every scene
/// with a defined size?
#[derive(Debug)]
pub struct ThinkingAgain;

/// Do we want custom fully graph nodes?
/// Answer for now: No?
#[derive(Debug)]
pub struct Scheduling;
