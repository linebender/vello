// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "will add them later")]
#![allow(missing_debug_implementations, reason = "prototyping")]
#![allow(clippy::todo, reason = "still a prototype")]

use std::{
    num::NonZeroU64,
    sync::{atomic::AtomicU64, Arc},
};

pub use peniko;

use peniko::{
    kurbo::{Affine, BezPath, Rect, Stroke},
    BrushRef, Font, StyleRef,
};

mod any;
mod generic_record;

pub use any::{AnyImage, AnyRecord, AnyRenderCtx, AnyResourceCtx, BoxedAnyRecord, BoxedRenderCtx};
pub use generic_record::{GenericRecorder, GenericResources};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(NonZeroU64);

// TODO: think this through
pub type Error = Box<dyn std::error::Error>;

#[derive(Clone)]
pub struct Path {
    pub id: Id,
    pub path: BezPath,
    // TODO: Vello encoding. kurbo BezPath can be used in interim
    // Question: probably want to special-case rect, line, ellipse at least
    // Probably also rounded-rect (incl varying corner radii)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ImageFormat {
    Grayscale,
    Rgb,
    RgbaSeparate,
    RgbaPremul,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum InterpolationMode {
    NearestNeighbor,
    Bilinear,
    // TODO: want to add cubic etc
}

/// Positioned glyph. This type matches Vello.
pub struct Glyph {
    pub id: u32,
    pub x: f32,
    pub y: f32,
}

pub trait RenderCtx {
    type Resource: ResourceCtx;

    fn playback(&mut self, recording: &Arc<<Self::Resource as ResourceCtx>::Recording>);

    // should even-odd be an arg or another method?
    fn fill(&mut self, path: &Path, brush: BrushRef<'_>);

    fn stroke(&mut self, path: &Path, stroke: &Stroke, brush: BrushRef<'_>);

    // TODO: clamp/extend/mirror
    fn draw_image(
        &mut self,
        image: &<Self::Resource as ResourceCtx>::Image,
        dst_rect: Rect,
        interp: InterpolationMode,
    );

    fn clip(&mut self, path: &Path);

    fn save(&mut self);

    fn restore(&mut self);

    fn transform(&mut self, affine: Affine);

    /// Start a glyph drawing operation
    ///
    /// The glyph drawing operation ends with [`RenderCtx::end_draw_glyphs`]
    fn begin_draw_glyphs(&mut self, font: &Font);

    // Following methods are borrowed from Vello's DrawGlyph
    fn font_size(&mut self, size: f32);

    fn hint(&mut self, hint: bool);

    fn glyph_brush(&mut self, brush: BrushRef<'_>);

    fn draw_glyphs(&mut self, style: StyleRef<'_>, glyphs: &dyn Iterator<Item = Glyph>);

    fn end_draw_glyphs(&mut self);
}

pub trait Record: RenderCtx {
    // It should be possible to take self by move, but that triggers E0161
    fn finish(&mut self) -> Arc<<Self::Resource as ResourceCtx>::Recording>;
}

pub trait ResourceCtx {
    type Image: Clone + Send;

    type Recording: Send + ?Sized;

    type Record: Record + Send;

    fn record(&mut self) -> Self::Record;

    fn make_image_with_stride(
        &mut self,
        width: usize,
        height: usize,
        stride: usize,
        buf: &[u8],
        format: ImageFormat,
    ) -> Result<Self::Image, Error>;
}

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

impl Id {
    pub fn get() -> Self {
        let n = ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(x) = n.checked_add(1) {
            Self(NonZeroU64::new(x).unwrap())
        } else {
            panic!("wow, overflow of u64, congratulations")
        }
    }
}

impl From<BezPath> for Path {
    fn from(path: BezPath) -> Self {
        let id = Id::get();
        Self { id, path }
    }
}
