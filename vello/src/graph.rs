// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello's Render Graph.
//!
//! The core technology of Vello is a vector graphics rasteriser, which converts from a [scene description][crate::Scene] to a rendered texture.
//! This by itself does not support many advanced visual effects, such as blurs, as they are incompatible with the parallelism it exploits.
//! These are instead built on top of this core pipeline, which schedules blurs and other visual effects,
//! alongside the core vector graphics rendering.
//!
// Is this true?: //! Most users of Vello should expect to use this more capable API.
//! If you have your own render graph or otherwise need greater control, the [rasteriser][Renderer] can be used standalone.
//!
//! ## Core Concepts
//!
//! The render graph consists of a few primary types:
//! - [`Vello`] is the core renderer type.
//!   Your application should generally have one of these per GPU.
//! - The [`Gallery`] is the actual core render graph type.
//!   These are also associated with a specific `wgpu` device (and therefore a `Vello`).
//!   This allows for optimisations such as dropping CPU-side textures once they have been uploaded to the GPU.
//! - A [`Painting`] is a reference counted handle for a render graph node in a `Gallery`.
//!   They can be created with [`Gallery::create_painting`], .
//!   These `Painting`s can configure their
//!
//! ## Test
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

mod canvas;
mod filters;
mod runner;

use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::Debug,
    hash::Hash,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{channel, Receiver, Sender},
        Arc, Weak,
    },
};

use crate::Renderer;
use filters::BlurPipeline;
use log::warn;
use peniko::Image;
use runner::RenderOrder;
use wgpu::{Device, Texture, TextureView};

pub use canvas::{Canvas, PaintingConfig};
pub use runner::RenderDetails;

// --- MARK: Public API ---

/// A context for running a render graph.
///
/// You should have one of these per wgpu `Device`.
pub struct Vello {
    vector_renderer: Renderer,
    blur: BlurPipeline,
    device: Device,
}

impl Vello {
    /// Create a new render graph runner.
    ///
    /// # Errors
    ///
    /// Primarily, if the device can't support Vello.
    pub fn new(device: Device, options: crate::RendererOptions) -> crate::Result<Self> {
        let vector_renderer = Renderer::new(&device, options)?;
        let blur = BlurPipeline::new(&device);
        Ok(Self {
            device,
            vector_renderer,
            blur,
        })
    }
}

impl Debug for Vello {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vello")
            .field("renderer", &"elided")
            .field("blur", &self.blur)
            .finish_non_exhaustive()
    }
}

/// A render graph.
///
/// A render graph allows for rendering operations which themselves depend on other rendering operations.
///
/// You should have one of these per wgpu `Device`.
/// This type is reference counted.
pub struct Gallery {
    id: GalleryToken,
    sender: Sender<PaintingAction>,
    receiver: Receiver<PaintingAction>,
    device: Device,
    paintings: HashMap<PaintingId, PaintInner>,
    paint_order: Vec<RenderOrder>,
}

impl Debug for Gallery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // TODO: Improve
        f.debug_tuple("Gallery").field(&self.id).finish()
    }
}

impl Gallery {
    pub fn new(device: Device, label: &'static str) -> Self {
        let (sender, receiver) = channel::<PaintingAction>();
        Self {
            device,
            paintings: Default::default(),
            id: GalleryToken::new(label),
            sender,
            receiver,
            paint_order: vec![],
        }
    }

    pub fn update(&mut self) {
        while let Ok(action) = self.receiver.try_recv() {
            match action {
                PaintingAction::InsertSource(painting_id, source) => {
                    let Some(paint) = self.paintings.get_mut(&painting_id) else {
                        continue;
                    };
                    paint.source = Some(source);
                    paint.source_dirty = true;
                }
                PaintingAction::Drop(painting_id) => {
                    if let Some(paint) = self.paintings.remove(&painting_id) {
                        debug_assert!(
                            paint.shared.upgrade().is_none(),
                            "Dropped a painting for which strong handles still exist"
                        );
                    } else {
                        #[cfg(debug_assertions)]
                        unreachable!("Dropped painting #{painting_id:?} more than once");
                    }
                }
            }
        }
    }
}

/// An editing handle to a render graph node.
///
/// These handles are reference counted, so that a `Painting`
/// which is a dependency of another node is retained while it
/// is still needed.
#[derive(Clone)]
pub struct Painting {
    inner: Arc<PaintingShared>,
}

impl Debug for Painting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Painting({:?})", self.inner))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone)]
/// A description of a new painting, used in [`Gallery::create_painting`].
#[derive(Debug)]
pub struct PaintingDescriptor {
    pub label: Cow<'static, str>,
    pub usages: wgpu::TextureUsages,
    // pub mipmaps: ...
    // pub atlas: ?
}

impl Gallery {
    #[must_use]
    pub fn create_painting(
        &mut self,
        // Not &PaintingDescriptor because `label` might be owned
        desc: PaintingDescriptor,
    ) -> Painting {
        let PaintingDescriptor { label, usages } = desc;
        let id = PaintingId::next();
        let shared = PaintingShared {
            id,
            channel: self.sender.clone(),
            gallery: self.id,
            label,
        };
        let shared = Arc::new(shared);
        let new_inner = PaintInner {
            shared: Arc::downgrade(&shared),
            usages,
            // Default to "uninit" black/purple checkboard source?
            source: None,
            // TODO: Means that cache can be used? Can cache be used
            source_dirty: false,

            paint_index: usize::MAX,
            resolving: false,
            dimensions: OutputSize {
                height: u32::MAX,
                width: u32::MAX,
            },

            texture: None,
            view: None,
        };
        self.paintings.insert(id, new_inner);

        Painting { inner: shared }
    }
}

/// These methods take an internal lock, and so should not happen at the same time as
/// a render operation [is being scheduled](Vello::prepare_render).
impl Painting {
    pub fn paint_image(self, image: Image) {
        self.insert(PaintingSource::Image(image));
    }
    #[expect(
        clippy::missing_panics_doc,
        reason = "Deferred until the rest of the methods also have this"
    )]
    pub fn paint_scene(&self, scene: Canvas, of_dimensions: OutputSize) {
        if let Some(gallery) = scene.gallery.as_ref() {
            // TODO: Use same logic as `assert_same_gallery` for better debug printing.
            assert_eq!(
                gallery, &self.inner.gallery,
                "A painting operation on {self:?} must only operate with paintings from the same gallery."
            );
        }

        self.insert(PaintingSource::Canvas(scene, of_dimensions));
    }

    pub fn paint_blur(&self, from: Self) {
        self.assert_same_gallery(&from);
        self.insert(PaintingSource::Blur(from));
    }

    fn insert(&self, new_source: PaintingSource) {
        match self
            .inner
            .channel
            .send(PaintingAction::InsertSource(self.inner.id, new_source))
        {
            Ok(()) => (),
            Err(_) => {
                // TODO: Is this reasonable to only warn? should we return an error?
                log::warn!(
                    "Tried to paint ({:?}) to dropped Gallery {:?}. Will have no effect",
                    self.inner,
                    self.inner.gallery
                );
            }
        };
    }
    #[track_caller]
    fn assert_same_gallery(&self, other: &Self) {
        assert!(
            self.inner.gallery == other.inner.gallery,
            "A painting operation must only operate with paintings from the same gallery. Found {:?} from {:?}, expected {:?} from {:?}",
            self.inner,
            self.inner.gallery,
            other.inner,
            other.inner.gallery
        );
    }
}

// --- MARK: Internal types ---

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
/// An Id for a Painting.
// TODO: This should be a `Peniko` type: https://github.com/linebender/vello/issues/664
struct PaintingId(u64);

impl Debug for PaintingId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("#{}", self.0))
    }
}

impl PaintingId {
    fn next() -> Self {
        static PAINTING_IDS: AtomicU64 = AtomicU64::new(0);
        // Overflow: u64 so cannot overflow.
        Self(PAINTING_IDS.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Clone, Copy, Eq)]
/// An identifier for a Gallery.
///
/// Used to validate that paintings are from the same gallery.
struct GalleryToken(u64, &'static str);

impl Debug for GalleryToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{} (#{})", self.1, self.0))
    }
}

impl GalleryToken {
    fn new(name: &'static str) -> Self {
        static GALLERY_IDS: AtomicU64 = AtomicU64::new(0);
        // Overflow: u64 so cannot overflow.
        Self(GALLERY_IDS.fetch_add(1, Ordering::Relaxed), name)
    }
}

impl PartialEq for GalleryToken {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

#[derive(Debug)]
enum PaintingSource {
    Image(Image),
    Canvas(Canvas, OutputSize),
    Blur(Painting),
    // WithMipMaps(Painting),
    // Region {
    //     painting: Painting,
    //     x: u32,
    //     y: u32,
    //     size: OutputSize,
    // },
}

enum PaintingAction {
    InsertSource(PaintingId, PaintingSource),
    Drop(PaintingId),
}

struct PaintInner {
    // Immutable fields:
    shared: Weak<PaintingShared>,
    usages: wgpu::TextureUsages,

    // Controlled by the user
    source: Option<PaintingSource>,
    source_dirty: bool,

    // TODO: Some way to handle texture atlasing at this level?

    // Controlled by the runner
    /// The index within the order of painting operations.
    /// This is used *only* to cheaply check if this was scheduled in the current painting operation.
    // TODO: Maybe just use a u32 generation?
    paint_index: usize,
    resolving: bool,
    dimensions: OutputSize,

    texture: Option<Texture>,
    view: Option<TextureView>,
}

struct PaintingShared {
    id: PaintingId,
    label: Cow<'static, str>,
    channel: Sender<PaintingAction>,

    gallery: GalleryToken,
}

impl Drop for PaintingShared {
    fn drop(&mut self) {
        // If this would error, that's expected.
        // It just means that the gallery was dropped before the painting.
        drop(self.channel.send(PaintingAction::Drop(self.id)));
    }
}

impl Debug for PaintingShared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{} ({:?})", self.label, self.id))
    }
}
