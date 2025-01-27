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
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, PoisonError, Weak,
    },
    usize,
};

use crate::Renderer;
use filters::BlurPipeline;
use peniko::Image;
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
    paint_order: Vec<PaintingId>,
}

impl Vello {
    pub fn new(device: Device, options: crate::RendererOptions) -> crate::Result<Self> {
        let vector_renderer = Renderer::new(&device, options)?;
        let blur = BlurPipeline::new(&device);
        Ok(Self {
            device,
            vector_renderer,
            blur,
            paint_order: vec![],
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

/// A render graph.
///
/// A render graph allows for rendering operations which themselves depend on other rendering operations.
///
/// You should have one of these per wgpu `Device`.
/// This type is reference counted.
pub struct Gallery {
    inner: Arc<GalleryInner>,
}

impl Gallery {
    pub fn new(device: Device, label: Cow<'static, str>) -> Self {
        let inner = GalleryInner {
            device,
            paintings: Default::default(),
            label,
        };
        Self {
            inner: Arc::new(inner),
        }
    }
}

/// An editing handle to a render graph node.
///
/// These handles are reference counted, so that a `Painting`
/// which is a dependency of another node is retained while it
/// is still needed.
pub struct Painting {
    inner: Arc<PaintingShared>,
}

impl Debug for Painting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Painting({:?})", self.inner))
    }
}

impl Painting {
    /// Make a copy of `Self` which edits the same underlying painting.
    ///
    /// Semantically, this is similar to `Clone::clone` for a reference counted type.
    ///
    /// However, this type does not implement `Clone` as a hint that most users
    /// should prefer [`sealed`](Self::sealed) to get new copies.
    pub fn clone_handle(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
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
        let new_inner = PaintInner {
            label,
            usages,
            // Default to "uninit" black/purple checkboard source?
            source: None,
            // TODO: Means that cache can be used? Can cache be used
            source_dirty: false,

            paint_index: usize::MAX,
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
        let shared = PaintingShared {
            id,
            gallery: Arc::downgrade(&self.inner),
        };
        Painting {
            inner: Arc::new(shared),
        }
    }
}

/// These methods take an internal lock, and so should not happen at the same time as
/// a render operation [is being scheduled](Vello::prepare_render).
impl Painting {
    pub fn paint_image(self, image: Image) {
        self.insert(PaintingSource::Image(image));
    }
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

    pub fn paint_blur(self, from: Painting) {
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

impl GalleryInner {
    // TODO: Logging if we're poisoned?
    fn lock_paintings(&self) -> std::sync::MutexGuard<'_, HashMap<PaintingId, PaintInner>> {
        self.paintings
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
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
    /// The index within the order of painting operations.
    /// This is used *only* to cheaply check if this was scheduled in the current painting operation.
    // TODO: Maybe just use a u32 generation?
    paint_index: usize,
    resolved: bool,
    texture: Option<Texture>,
    view: Option<TextureView>,
}

struct GalleryInner {
    device: Device,
    paintings: Mutex<HashMap<PaintingId, PaintInner>>,
    label: Cow<'static, str>,
}

struct PaintingShared {
    id: PaintingId,
    gallery: Weak<GalleryInner>,
}

impl PaintingShared {
    /// Access the [`PaintInner`].
    ///
    /// The function is called with [`None`] if the painting is dangling (i.e. the corresponding
    /// gallery has been dropped).
    fn lock<R>(&self, f: impl FnOnce(Option<&mut PaintInner>) -> R) -> R {
        match self.gallery.upgrade() {
            Some(gallery) => {
                let mut paintings = gallery.lock_paintings();
                let paint = paintings
                    .get_mut(&self.id)
                    .expect("PaintingShared exists, so corresponding entry in Gallery should too");
                f(Some(paint))
            }
            None => f(None),
        }
    }
}

impl Debug for PaintingShared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.lock(|paint| match paint {
            Some(paint) => f.write_fmt(format_args!("{} ({:?})", paint.label, self.id)),
            None => f.write_fmt(format_args!("Dangling Painting ({:?})", self.id)),
        })
    }
}
