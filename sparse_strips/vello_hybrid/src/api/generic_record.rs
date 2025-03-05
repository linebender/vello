// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(unused, reason = "prototyping")]

use std::sync::Arc;

use peniko::{kurbo::Rect, Brush};

use crate::api::{InterpolationMode, Path, Record, RenderCtx, ResourceCtx};

pub struct GenericRecorder<RC: RenderCtx> {
    cmds: Vec<Cmd<RC>>,
}

pub struct GenericResources<RC: RenderCtx> {
    inner: RC::Resource,
}

enum Cmd<RC: RenderCtx> {
    Fill(Path, Brush),
    Image(
        <RC::Resource as ResourceCtx>::Image,
        Rect,
        InterpolationMode,
    ),
}

impl<RC: RenderCtx> GenericRecorder<RC> {
    #[allow(
        clippy::new_without_default,
        reason = "didn't we get rid of this lint? it's so annoying"
    )]
    pub fn new() -> Self {
        let cmds = Vec::new();
        Self { cmds }
    }

    pub fn play(&self, ctx: &mut RC) {
        for cmd in &self.cmds {
            match cmd {
                Cmd::Fill(path, brush) => ctx.fill(path, brush.into()),
                Cmd::Image(image, rect, interp) => ctx.draw_image(image, *rect, *interp),
            }
        }
    }
}

impl<RC: RenderCtx> RenderCtx for GenericRecorder<RC> {
    type Resource = GenericResources<RC>;

    fn playback(&mut self, recording: &std::sync::Arc<<Self::Resource as ResourceCtx>::Recording>) {
        todo!()
    }

    fn fill(&mut self, path: &Path, brush: peniko::BrushRef<'_>) {
        self.cmds.push(Cmd::Fill(path.clone(), brush.to_owned()));
    }

    fn stroke(&mut self, path: &Path, stroke: &peniko::kurbo::Stroke, brush: peniko::BrushRef<'_>) {
        todo!()
    }

    fn draw_image(
        &mut self,
        image: &<Self::Resource as ResourceCtx>::Image,
        dst_rect: peniko::kurbo::Rect,
        interp: crate::api::InterpolationMode,
    ) {
        let image = image.clone();
        self.cmds.push(Cmd::Image(image, dst_rect, interp));
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

    fn transform(&mut self, affine: peniko::kurbo::Affine) {
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

    fn glyph_brush(&mut self, brush: peniko::BrushRef<'_>) {
        todo!()
    }

    fn draw_glyphs(
        &mut self,
        style: peniko::StyleRef<'_>,
        glyphs: &dyn Iterator<Item = crate::api::Glyph>,
    ) {
        todo!()
    }

    fn end_draw_glyphs(&mut self) {
        todo!()
    }
}

impl<RC: RenderCtx> ResourceCtx for GenericResources<RC> {
    type Image = <RC::Resource as ResourceCtx>::Image;

    type Recording = GenericRecorder<RC>;

    type Record = GenericRecorder<RC>;

    fn record(&mut self) -> Self::Record {
        GenericRecorder::new()
    }

    fn make_image_with_stride(
        &mut self,
        width: usize,
        height: usize,
        stride: usize,
        buf: &[u8],
        format: crate::api::ImageFormat,
    ) -> Result<Self::Image, crate::api::Error> {
        self.inner
            .make_image_with_stride(width, height, stride, buf, format)
    }
}

impl<RC: RenderCtx> Record for GenericRecorder<RC> {
    fn finish(&mut self) -> Arc<<Self::Resource as ResourceCtx>::Recording> {
        let cmds = std::mem::take(&mut self.cmds);
        Arc::new(Self { cmds })
    }
}
