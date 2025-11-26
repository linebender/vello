// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(missing_docs, reason = "This code is incomplete.")]
#![expect(clippy::result_unit_err, reason = "This code is incomplete.")]

use alloc::sync::Arc;

use hashbrown::HashMap;
use vello_api::{
    PaintScene, Renderer,
    baseline::{BaselinePainter, BaselinePreparePaths},
    texture::{self, TextureId, TextureUsages},
};
use vello_common::{
    encode::{EncodedImage, EncodedPaint},
    kurbo::{self, Affine, Rect, Shape},
    paint::{ImageId, ImageSource},
    peniko::{BlendMode, Brush, Color, Fill, ImageBrush, ImageData},
    pixmap::Pixmap,
};

use crate::{RenderContext, RenderSettings};

#[derive(Debug)]
pub struct VelloCPU {
    default_render_settings: RenderSettings,
    // TODO: Evaluate whether to use generational(?) indexing instead of a HashMap
    texture_id_source: u64,
    textures: HashMap<TextureId, StoredTexture>,
}

impl VelloCPU {
    pub fn new(default_render_settings: RenderSettings) -> Self {
        Self {
            default_render_settings,
            textures: HashMap::new(),
            texture_id_source: 0,
        }
    }

    pub fn read_texture(&self, texture: &TextureId) -> Result<&Pixmap, ()> {
        Ok(&self.textures.get(texture).ok_or(())?.pixmap)
    }
}

impl Renderer for VelloCPU {
    type ScenePainter = CPUScenePainter;
    type PathPreparer = BaselinePreparePaths;
    type Recording = BaselinePainter<BaselinePreparePaths>;
    type TransformedRecording = BaselinePainter<BaselinePreparePaths>;

    fn create_texture(&mut self, descriptor: texture::TextureDescriptor) -> TextureId {
        // TODO: What do we need to do with the usages?

        // let download_src = descriptor.usages.contains(TextureUsages::DOWNLOAD_SRC);
        // let upload_target = descriptor.usages.contains(TextureUsages::UPLOAD_TARGET);
        // let texture_binding = descriptor.usages.contains(TextureUsages::TEXTURE_BINDING);
        // let render_target = descriptor.usages.contains(TextureUsages::RENDER_TARGET);
        let texture = StoredTexture {
            // Can we avoid filling this pixmap immediately in some cases, e.g. for textures
            // we optimistically think will be uploaded to?
            pixmap: Arc::new(Pixmap::new(descriptor.width, descriptor.height)),
            descriptor,
        };

        let id = TextureId::from_raw(self.texture_id_source);
        self.texture_id_source += 1;
        self.textures.insert(id, texture);
        id
    }

    fn free_texture(&mut self, texture: TextureId) -> Result<(), ()> {
        self.textures.remove(&texture).ok_or(())?;
        Ok(())
    }

    fn create_scene(
        &mut self,
        to: &TextureId,
        options: vello_api::SceneOptions,
    ) -> Result<Self::ScenePainter, ()> {
        let target_texture = self.textures.get(to).ok_or(())?;
        let (width, height) = if let Some(size) = options.size() {
            size
        } else {
            (
                target_texture.descriptor.width,
                target_texture.descriptor.height,
            )
        };

        // TODO: Cache the contexts internally, so that we don't reallocate here?
        let context = RenderContext::new_with(width, height, self.default_render_settings);
        let mut painter = CPUScenePainter {
            render_context: context,
            target: *to,
        };
        if let Some(clear_color) = options.clear_color {
            painter.set_solid_brush(clear_color);
            painter.fill_path(
                Affine::IDENTITY,
                Fill::EvenOdd,
                Rect::new(0., 0., width as f64, height as f64),
            );
        }
        Ok(painter)
    }

    fn queue_render(&mut self, mut from: Self::ScenePainter) {
        for encoded_paint in &mut from.render_context.encoded_paints {
            if let EncodedPaint::Image(EncodedImage {
                source: ref mut source @ ImageSource::OpaqueId(id),
                ..
            }) = *encoded_paint
            {
                // Back-associate each texture with the actual pixmap.
                let idx = u64::from(id.as_u32());
                let pixmap = self
                    .textures
                    .get_mut(&TextureId::from_raw(idx))
                    .expect("todo: handle this case, where the texture passed to 'set_brush' isn't from this renderer.");
                *source = ImageSource::Pixmap(pixmap.pixmap.clone());
            }
        }

        // TODO: We should possibly delay this flush even further, i.e. truly "queue" this render
        // to happen in a multi-threaded way.
        from.render_context.flush();
        // Note that this will panic if the target is used in the rendering.
        // The exact place that error is handled is tbd.
        let texture = self.textures.get_mut(&from.target).unwrap();
        let pixmap = Arc::get_mut(&mut texture.pixmap).unwrap();
        from.render_context.render_to_pixmap(pixmap);
        // We will drop `from`, which includes dropping the pixmap references.
        // TODO: We almost certainly want to keep the render context around.
    }

    fn upload_image(&mut self, to: &TextureId, data: &ImageData) -> Result<(), ()> {
        let source = self.textures.get_mut(to).ok_or(())?;
        if data.height != u32::from(source.descriptor.height)
            || data.width != u32::from(source.descriptor.width)
        {
            return Err(());
        }
        if !source
            .descriptor
            .usages
            .contains(TextureUsages::UPLOAD_TARGET)
        {
            return Err(());
        }
        let target = Arc::get_mut(&mut source.pixmap)
            .expect("Pixmap shouldn't be shared except whilst referenced.");
        // TODO: Reuse the internal allocation of `buf` in `target`
        let data = Pixmap::from_peniko_image_data(data);
        *target = data;
        Ok(())
    }

    fn create_path_cache(&mut self) -> Self::PathPreparer {
        BaselinePreparePaths::new()
    }

    fn create_offset_recording(
        &mut self,
        width: u16,
        height: u16,
        origin_x_offset: i32,
        origin_y_offset: i32,
    ) -> Self::Recording {
        let mut rec = BaselinePainter::default();
        rec.set_dimensions(width, height);
        rec.set_origin_offset(origin_x_offset, origin_y_offset);
        rec
    }
    fn create_transformed_recording(
        &mut self,
        width: u16,
        height: u16,
        origin_x_offset: i32,
        origin_y_offset: i32,
    ) -> Self::TransformedRecording {
        let mut rec = BaselinePainter::default();
        rec.set_dimensions(width, height);
        rec.set_origin_offset(origin_x_offset, origin_y_offset);
        rec
    }
}

#[derive(Debug)]
pub struct CPUScenePainter {
    render_context: RenderContext,
    target: TextureId,
}

impl PaintScene for CPUScenePainter {
    fn width(&self) -> u16 {
        self.render_context.width()
    }

    fn height(&self) -> u16 {
        self.render_context.height()
    }

    fn fill_path(&mut self, transform: Affine, fill_rule: Fill, path: impl Shape) {
        self.render_context.set_transform(transform);
        self.render_context.set_fill_rule(fill_rule);
        // TODO: Tweak inner `fill_path` API to either take a `Shape` or an &[PathEl]
        self.render_context.fill_path(&path.to_path(0.1));
    }

    fn stroke_path(&mut self, transform: Affine, stroke_params: &kurbo::Stroke, path: impl Shape) {
        self.render_context.set_transform(transform);
        self.render_context.set_stroke(stroke_params.clone());
        self.render_context.stroke_path(&path.to_path(0.1));
    }

    fn set_brush(
        &mut self,
        brush: impl Into<Brush<ImageBrush<TextureId>>>,
        paint_transform: Affine,
    ) {
        self.render_context.set_paint_transform(paint_transform);
        let brush = match brush.into() {
            Brush::Solid(alpha_color) => Brush::Solid(alpha_color),
            Brush::Gradient(gradient) => Brush::Gradient(gradient),
            Brush::Image(brush) => {
                // TODO: Make this read more easily.
                let image_index = brush.image.to_raw().try_into().expect("Handle this.");
                Brush::Image(ImageBrush {
                    image: ImageSource::OpaqueId(ImageId::new(image_index)),
                    sampler: brush.sampler,
                })
            }
        };
        self.render_context.set_paint(brush);
    }

    fn set_blurred_rounded_rect_brush(
        &mut self,
        _paint_transform: Affine,
        _color: Color,
        _rect: &kurbo::Rect,
        _radius: f32,
        _std_dev: f32,
    ) {
        // This is "trivially" fixable.
        unimplemented!(
            "Vello CPU doesn't expose drawing a blurred rounded rectangle in custom shapes yet."
        )
    }

    fn fill_blurred_rounded_rect(
        &mut self,
        transform: Affine,
        color: Color,
        rect: &kurbo::Rect,
        radius: f32,
        std_dev: f32,
    ) {
        self.render_context.set_paint(color);
        self.render_context.set_transform(transform);
        self.render_context
            .fill_blurred_rounded_rect(rect, radius, std_dev);
    }

    fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        self.render_context.set_blend_mode(blend_mode);
    }

    fn push_layer(
        &mut self,
        clip_transform: Affine,
        clip_path: Option<impl Shape>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        // mask: Option<Mask>,
    ) {
        self.render_context.set_transform(clip_transform);
        self.render_context.push_layer(
            clip_path.map(|it| it.to_path(0.1)).as_ref(),
            blend_mode,
            opacity,
            None,
            None,
        );
    }

    fn push_clip_layer(&mut self, clip_transform: Affine, path: impl Shape) {
        self.render_context.set_transform(clip_transform);
        self.render_context.push_clip_layer(
            // TODO: Not allocate
            &path.to_path(0.1),
        );
    }

    fn pop_layer(&mut self) {
        self.render_context.pop_layer();
    }

    fn push_clip_path(&mut self, path: &kurbo::BezPath) {
        self.render_context.push_clip_path(path);
    }

    fn pop_clip_path(&mut self) {
        self.render_context.pop_clip_path();
    }
}

#[derive(Debug)]
struct StoredTexture {
    /// This `Arc`s is very carefully managed to be used with [`Arc::get_mut`]
    /// when rendering to the texture.
    ///
    /// As such, to avoid panicks we *cannot* let them be accessed externally.
    pixmap: Arc<Pixmap>,
    descriptor: texture::TextureDescriptor,
}
