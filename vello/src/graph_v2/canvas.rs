// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! An interim version of Scene which has render graph compatibility, for use whilst Peniko doesn't know about image ids.

use super::GalleryInner;
use super::Painting;
use crate::Scene;
use peniko::kurbo::Affine;
use peniko::Blob;
use peniko::Brush;
use peniko::Extend;
use peniko::Image;
use peniko::ImageFormat;
use peniko::ImageQuality;
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Weak;

/// A single Scene, potentially containing paintings.
///
/// This type is required because the base `Scene` type from Vello
/// currently doesn't know about the render graph.
/// This is an interim API until that can be resolved.
pub struct Canvas {
    /// The gallery which all paintings in `paintings` is a part of.
    pub(crate) gallery: Option<Weak<GalleryInner>>,
    pub(crate) scene: Box<Scene>,
    pub(crate) paintings: HashMap<u64, Painting>,
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

impl From<Scene> for Canvas {
    fn from(value: Scene) -> Self {
        Self::from_scene(Box::new(value))
    }
}

impl Canvas {
    pub fn new() -> Self {
        Self::from_scene(Box::<Scene>::default())
    }
    pub fn from_scene(scene: Box<Scene>) -> Self {
        Self {
            gallery: None,
            scene,
            paintings: HashMap::default(),
        }
    }
    pub fn new_image(&mut self, painting: Painting, width: u16, height: u16) -> PaintingConfig {
        match self.gallery.as_ref() {
            Some(gallery) => assert!(gallery.ptr_eq(&painting.inner.gallery)),
            None => self.gallery = Some(painting.inner.gallery.clone()),
        }
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

impl Debug for Canvas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Canvas")
            .field("scene", &"elided")
            .field("paintings", &self.paintings)
            .finish()
    }
}

impl Default for Canvas {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
/// Created using [`Canvas::new_image`].
pub struct PaintingConfig {
    pub(crate) image: Image,
}

impl PaintingConfig {
    pub(crate) fn new(width: u16, height: u16) -> Self {
        // Create a fake Image, with an empty Blob. We can re-use the allocation between these.
        pub(crate) static EMPTY_ARC: LazyLock<Arc<[u8; 0]>> = LazyLock::new(|| Arc::new([]));
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
