// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello's Render Graph.
//!
//! The core technology of Vello is a vector graphics rasteriser, which converts from a [scene description][Scene] to a rendered texture.
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
//! - `Vello` is the core renderer type. Your application should generally only ever have one of these.
//! - A [`Painting`] is a persistent reference counted handle to a texture on the GPU.
//! - The `Gallery`
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
    num::Wrapping,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, PoisonError, Weak,
    },
};

use filters::BlurPipeline;
use peniko::Image;
use wgpu::{Device, Texture, TextureView};

use crate::Renderer;
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
            .finish()
    }
}

struct PaintInner {
    // Immutable fields:
    label: Cow<'static, str>,
    usages: wgpu::TextureUsages,

    // Controlled by the user
    source: Option<PaintingSource>,
    source_dirty: bool,

    // TODO: Some way to handle texture atlasing at this level?

    // Controlled by the runner
    texture: Option<Texture>,
    view: Option<TextureView>,
}

pub struct Gallery2Inner {
    device: Device,
    paintings: Mutex<HashMap<PaintingId, PaintInner>>,
    label: Cow<'static, str>,
}

/// A render graph.
///
/// A render graph allows for rendering operations which themselves depend on other rendering operations.
///
/// You should have one of these per wgpu `Device`.
/// This type is reference counted.
pub struct Gallery2 {
    inner: Arc<Gallery2Inner>,
}

impl Gallery2 {
    pub fn new(device: Device, label: Cow<'static, str>) -> Self {
        let inner = Gallery2Inner {
            device,
            paintings: Default::default(),
            label,
        };
        Self {
            inner: Arc::new(inner),
        }
    }
}

struct Painting2Shared {
    id: PaintingId,
    gallery: Weak<Gallery2Inner>,
}

impl Painting2Shared {
    /// Access the [`PaintInner`].
    ///
    /// The function is called with [`None`] if the painting is dangling (i.e. the corresponding
    /// gallery has been dropped).
    fn lock<R>(&self, f: impl FnOnce(Option<&mut PaintInner>) -> R) -> R {
        match self.gallery.upgrade() {
            Some(it) => {
                let mut lock = it.paintings.lock().unwrap_or_else(PoisonError::into_inner);
                let paint = lock
                    .get_mut(&self.id)
                    .expect("PaintingShared exists, so corresponding entry in Gallery should too");
                f(Some(paint))
            }
            None => f(None),
        }
    }
}

impl Debug for Painting2Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.lock(|paint| match paint {
            Some(paint) => f.write_fmt(format_args!("{} ({:?})", paint.label, self.id)),
            None => f.write_fmt(format_args!("Dangling Painting ({:?})", self.id)),
        })
    }
}

/// An editing handle to a render graph node.mi
///
/// These handles are reference counted, so that a `Painting`
/// which is a dependency of another node is retained when needed.
/// However, this type does not implement [`Clone`], because the intention .
pub struct Painting2 {
    inner: Arc<Painting2Shared>,
}

impl Debug for Painting2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Painting({:?})", self.inner))
    }
}

impl Debug for PaintingRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("PaintingRef({:?})", self.inner))
    }
}

impl Painting2 {
    /// Create an immutable version of this painting.
    ///
    /// Useful for passing to argument.
    ///
    /// Note that this sealing is not semantic, i.e. this painting
    /// can still be modified through `self` and/or its [clones](Self::edit_copy).
    pub fn sealed(&self) -> PaintingRef {
        PaintingRef {
            inner: self.inner.clone(),
        }
    }

    /// Make a copy of `Self` which edits the same underlying painting.
    ///
    /// Semantically, this is similar to `Clone::clone` for a reference counted type.
    ///
    /// However, this type does not implement `Clone` as a hint that most users
    /// should prefer [`sealed`](Self::sealed) to get new copies.
    pub fn edit_copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

#[derive(Clone)]
/// A reference-counted view for a [`Painting2`] which can't be modified.
///
/// This is entirely for code clarity reasons.
pub struct PaintingRef {
    inner: Arc<Painting2Shared>,
}

impl From<&'_ Painting2> for PaintingRef {
    fn from(value: &Painting2) -> Self {
        value.sealed()
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

impl Gallery2 {
    #[must_use]
    pub fn create_painting(
        &mut self,
        // Not &PaintingDescriptor because `label` might be owned
        desc: PaintingDescriptor,
    ) -> Painting2 {
        let PaintingDescriptor { label, usages } = desc;
        let id = PaintingId::next();
        let new_inner = PaintInner {
            label,
            usages,
            // Default to "uninit" black/purple checkboard source?
            source: None,
            // TODO: Means that cache can be used? Can cache be used
            source_dirty: false,
            texture: None,
            view: None,
        };
        {
            let mut lock = self
                .inner
                .paintings
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            lock.insert(id, new_inner);
        }
        let shared = Painting2Shared {
            id,
            gallery: Arc::downgrade(&self.inner),
        };
        Painting2 {
            inner: Arc::new(shared),
        }
    }
}

impl Painting2 {
    pub fn paint_image(self, image: Image) {
        self.insert(PaintingSource::Image(image));
    }
    // /// From must have the `COPY_SRC` usage.
    // pub fn as_subregion(self, from: Painting, x: u32, y: u32, width: u32, height: u32) {
    //     self.insert(PaintingSource::Region {
    //         painting: from,
    //         x,
    //         y,
    //         size: OutputSize { width, height },
    //     });
    // }
    // pub fn with_mipmaps(self, from: Painting) {
    //     self.insert(PaintingSource::WithMipMaps(from));
    // }
    pub fn paint_scene(self, scene: Canvas, of_dimensions: OutputSize) {
        if let Some(gallery) = scene.gallery.as_ref() {
            // TODO: Use same logic as `assert_same_gallery` for better debug printing.
            assert!(
                gallery.ptr_eq(&self.inner.gallery),
                "A painting operation must only operate with paintings from the same gallery."
            )
        }

        self.insert(PaintingSource::Canvas(scene, of_dimensions));
    }

    pub fn paint_blur(self, from: Painting2) {
        self.assert_same_gallery(&from);
        self.insert(PaintingSource::Blur(from));
    }

    fn insert(self, new_source: PaintingSource) {
        self.inner.lock(|paint| match paint {
            Some(paint) => {
                paint.source = Some(new_source);
                paint.source_dirty = true;
            }
            None => {
                // TODO: Is this reasonable to only warn? should we return an error?
                log::warn!("Tried to paint to dropped Gallery. Will have no effect")
            }
        });
    }
    #[track_caller]
    fn assert_same_gallery(&self, other: &Self) {
        // TODO: Show four things:
        // 1) This painting's debug
        // 2) Other painting's debug
        // 3) Other gallery's label
        // 4) This gallery's debug
        assert!(
            Arc::ptr_eq(&self.inner, &other.inner),
            "A painting operation must only operate with paintings from the same gallery."
        )
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
        Self(PAINTING_IDS.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug)]
enum PaintingSource {
    Image(Image),
    Canvas(canvas::Canvas, OutputSize),
    Blur(Painting2),
    // WithMipMaps(Painting),
    // Region {
    //     painting: Painting,
    //     x: u32,
    //     y: u32,
    //     size: OutputSize,
    // },
}

#[derive(Default, Debug, PartialEq, Eq, Clone)]
// Not copy because the identity is important; don't want to modify an accidental copy
struct Generation(Wrapping<u32>);

impl Generation {
    fn nudge(&mut self) {
        self.0 += 1;
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
