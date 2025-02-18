// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(unused, reason = "prototyping")]

use std::{any::Any, sync::Arc};

use peniko::{kurbo::Affine, BrushRef};

use crate::{Id, Path, Record, RenderCtx, ResourceCtx};

#[derive(Clone)]
pub struct AnyImage {
    // TODO: move id into trait
    id: Id,
    body: Arc<dyn Any + Send + Sync>,
}

pub trait AnyRecord: Send {
    fn as_any(&mut self) -> &mut dyn std::any::Any;

    fn dyn_finish(&mut self) -> Arc<dyn Any + Send>;
}

impl<R: Record + Send + 'static> AnyRecord for R
where
    <<R as RenderCtx>::Resource as ResourceCtx>::Recording: Sync,
{
    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn dyn_finish(&mut self) -> Arc<dyn Any + Send> {
        let recording = self.finish();
        Arc::new(recording)
    }
}

pub trait AnyRenderCtx {
    fn as_any(&mut self) -> &mut dyn std::any::Any;

    fn dyn_playback(&mut self, recording: &Arc<dyn Any + Send>);

    fn dyn_fill(&mut self, path: &Path, brush: BrushRef<'_>);
}

impl<RC: RenderCtx + 'static> AnyRenderCtx for RC {
    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn dyn_playback(&mut self, recording: &Arc<dyn Any + Send>) {
        if let Some(recording) = recording.downcast_ref() {
            self.playback(recording);
        } else {
            panic!("downcast error on playback");
        }
    }

    fn dyn_fill(&mut self, path: &Path, brush: BrushRef<'_>) {
        self.fill(path, brush);
    }
}

pub type BoxedRenderCtx = Box<dyn AnyRenderCtx>;

impl RenderCtx for BoxedRenderCtx {
    type Resource = Box<dyn AnyResourceCtx>;

    fn playback(&mut self, recording: &Arc<<Self::Resource as ResourceCtx>::Recording>) {
        self.dyn_playback(recording);
    }

    fn fill(&mut self, path: &Path, brush: BrushRef<'_>) {
        self.dyn_fill(path, brush);
    }

    fn stroke(&mut self, path: &Path, stroke: &peniko::kurbo::Stroke, brush: BrushRef<'_>) {
        todo!()
    }

    fn draw_image(
        &mut self,
        image: &<Self::Resource as ResourceCtx>::Image,
        dst_rect: peniko::kurbo::Rect,
        interp: crate::InterpolationMode,
    ) {
        todo!()
    }

    fn clip(&mut self, path: &Path) {
        todo!()
    }

    fn save(&mut self) {
        todo!()
    }

    fn restore(&mut self) {
        todo!()
    }

    fn transform(&mut self, affine: Affine) {
        todo!()
    }

    fn begin_draw_glyphs(&mut self, font: &peniko::Font) {
        todo!()
    }

    fn font_size(&mut self, size: f32) {
        todo!()
    }

    fn hint(&mut self, hint: bool) {
        todo!()
    }

    fn glyph_brush(&mut self, brush: BrushRef<'_>) {
        todo!()
    }

    fn draw_glyphs(
        &mut self,
        style: peniko::StyleRef<'_>,
        glyphs: &dyn Iterator<Item = crate::Glyph>,
    ) {
        todo!()
    }

    fn end_draw_glyphs(&mut self) {
        todo!()
    }
}

pub trait AnyResourceCtx {
    fn as_any(&mut self) -> &mut dyn std::any::Any;

    fn dyn_record(&mut self) -> Box<dyn AnyRecord + Send>;

    fn dyn_make_image_with_stride(
        &mut self,
        width: usize,
        height: usize,
        stride: usize,
        buf: &[u8],
        format: crate::ImageFormat,
    ) -> Result<AnyImage, crate::Error>;
}

impl ResourceCtx for Box<dyn AnyResourceCtx> {
    type Image = AnyImage;

    type Recording = dyn Any + Send;

    type Record = Box<dyn AnyRecord>;

    fn record(&mut self) -> Self::Record {
        self.dyn_record()
    }

    fn make_image_with_stride(
        &mut self,
        width: usize,
        height: usize,
        stride: usize,
        buf: &[u8],
        format: crate::ImageFormat,
    ) -> Result<Self::Image, crate::Error> {
        let image = self.dyn_make_image_with_stride(width, height, stride, buf, format)?;
        let id = Id::get();
        Ok(AnyImage {
            id,
            body: Arc::new(image),
        })
    }
}

pub struct BoxedAnyRecord(Option<Box<dyn AnyRecord>>);

impl RenderCtx for Box<dyn AnyRecord> {
    type Resource = Box<dyn AnyResourceCtx>;

    fn playback(&mut self, recording: &Arc<<Self::Resource as ResourceCtx>::Recording>) {
        self.dyn_playback(recording);
    }

    fn fill(&mut self, path: &Path, brush: BrushRef<'_>) {
        self.dyn_fill(path, brush);
    }

    fn stroke(&mut self, path: &Path, stroke: &peniko::kurbo::Stroke, brush: BrushRef<'_>) {
        todo!()
    }

    fn draw_image(
        &mut self,
        image: &<Self::Resource as ResourceCtx>::Image,
        dst_rect: peniko::kurbo::Rect,
        interp: crate::InterpolationMode,
    ) {
        todo!()
    }

    fn clip(&mut self, path: &Path) {
        todo!()
    }

    fn save(&mut self) {
        todo!()
    }

    fn restore(&mut self) {
        todo!()
    }

    fn transform(&mut self, affine: Affine) {
        todo!()
    }

    fn begin_draw_glyphs(&mut self, font: &peniko::Font) {
        todo!()
    }

    fn font_size(&mut self, size: f32) {
        todo!()
    }

    fn hint(&mut self, hint: bool) {
        todo!()
    }

    fn glyph_brush(&mut self, brush: BrushRef<'_>) {
        todo!()
    }

    fn draw_glyphs(
        &mut self,
        style: peniko::StyleRef<'_>,
        glyphs: &dyn Iterator<Item = crate::Glyph>,
    ) {
        todo!()
    }

    fn end_draw_glyphs(&mut self) {
        todo!()
    }
}

impl Record for Box<dyn AnyRecord> {
    fn finish(&mut self) -> Arc<<Self::Resource as ResourceCtx>::Recording> {
        self.dyn_finish()
    }
}
