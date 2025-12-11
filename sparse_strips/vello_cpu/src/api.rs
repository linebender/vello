// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(missing_docs, reason = "This code is incomplete.")]

use alloc::{
    sync::{Arc, Weak},
    vec::Vec,
};
use core::ptr;

use hashbrown::HashMap;
use vello_api::{
    PaintScene, Renderer, Scene,
    paths::Operation,
    scene::{RenderCommand, extract_integer_translation},
    sync::Lock,
    texture::{self, TextureHandle, TextureId, TextureUsages},
};
use vello_common::{
    encode::{EncodedImage, EncodedPaint},
    kurbo::{self, Affine, BezPath, Rect, Shape},
    paint::{ImageId, ImageSource},
    peniko::{BlendMode, Brush, Color, Fill, ImageBrush, ImageData},
    pixmap::Pixmap,
};

use crate::{RenderContext, RenderSettings};

#[derive(Debug)]
struct VelloCPUContents {
    default_render_settings: RenderSettings,
    // TODO: Evaluate whether to use generational(?) indexing instead of a HashMap
    texture_id_source: u64,
    textures: HashMap<TextureId, StoredTexture>,
}

#[derive(Debug)]
pub struct VelloCPU {
    inner: Lock<VelloCPUContents>,
    this: Weak<VelloCPU>,
}

impl VelloCPU {
    pub fn new(default_render_settings: RenderSettings) -> Arc<Self> {
        Arc::new_cyclic(|this| Self {
            inner: Lock::new(VelloCPUContents {
                default_render_settings,
                texture_id_source: 0,
                textures: HashMap::new(),
            }),
            this: this.clone(),
        })
    }

    // TODO: Find a signature which works here?
    // pub fn read_texture(&self, texture: &TextureId) -> Result<&Pixmap, ()> {
    //     Ok(&self.textures.get(texture).ok_or(())?.pixmap)
    // }
}

impl Renderer for VelloCPU {
    type ScenePainter = CPUScenePainter;

    fn create_scene(
        &self,
        to: &TextureId,
        options: vello_api::SceneOptions,
    ) -> Result<Self::ScenePainter, ()> {
        let renderer = self.as_arc();
        let this = self.inner.lock();
        let target_texture = this.textures.get(to).ok_or(())?;
        let (width, height) = if let Some(size) = options.size() {
            size
        } else {
            (
                target_texture.descriptor.width,
                target_texture.descriptor.height,
            )
        };

        // TODO: Cache the contexts internally, so that we don't reallocate here?
        let context = RenderContext::new_with(width, height, this.default_render_settings);
        drop(this);
        let mut painter = CPUScenePainter {
            render_context: context,
            target: *to,
            renderer,
            textures: Vec::new(),
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

    fn queue_render(&self, mut from: Self::ScenePainter) {
        let mut this = self.inner.lock();
        for encoded_paint in &mut from.render_context.encoded_paints {
            if let EncodedPaint::Image(EncodedImage {
                source: ref mut source @ ImageSource::OpaqueId(id),
                ..
            }) = *encoded_paint
            {
                // Back-associate each texture with the actual pixmap.
                let idx = u64::from(id.as_u32());
                let pixmap = this
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
        let texture = this.textures.get_mut(&from.target).unwrap();
        let pixmap = Arc::get_mut(&mut texture.pixmap).unwrap();
        from.render_context.render_to_pixmap(pixmap);
        // We will drop `from`, which includes dropping the pixmap references.
        // TODO: We almost certainly want to keep the render context around.
    }

    fn upload_image(&self, to: &TextureId, data: &ImageData) -> Result<(), ()> {
        let mut this = self.inner.lock();
        let source = this.textures.get_mut(to).ok_or(())?;
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

    fn alloc_untracked_texture(&self, descriptor: texture::TextureDescriptor) -> TextureId {
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

        let mut this = self.inner.lock();
        let id = TextureId::from_raw(this.texture_id_source);
        this.texture_id_source += 1;
        this.textures.insert(id, texture);
        id
    }

    fn free_untracked_texture(&self, texture: TextureId) -> Result<(), ()> {
        let mut this = self.inner.lock();
        this.textures.remove(&texture).ok_or(())?;
        Ok(())
    }

    fn as_arc(&self) -> Arc<Self>
    where
        Self: Sized,
    {
        self.this
            .upgrade()
            .expect("self still exists, so 'this' should still exist.")
    }

    fn as_dyn_arc(&self) -> Arc<dyn Renderer> {
        self.as_arc()
    }
}

#[derive(Debug)]
pub struct CPUScenePainter {
    render_context: RenderContext,
    target: TextureId,
    renderer: Arc<VelloCPU>,
    textures: Vec<TextureHandle>,
}

impl PaintScene for CPUScenePainter {
    fn append(
        &mut self,
        mut scene_transform: Affine,
        Scene {
            // Make sure we consider all the fields of Scene by destructuring
            paths: input_paths,
            commands: input_commands,
            renderer: input_renderer,
            hinted: input_hinted,
            textures: input_textures,
        }: &Scene,
    ) -> Result<(), ()> {
        // Ideally, we'd use `ptr_eq`, but self.renderer
        if !ptr::addr_eq(Arc::as_ptr(input_renderer), Arc::as_ptr(&self.renderer)) {
            // Mismatched Renderers
            return Err(());
        }

        if *input_hinted {
            if let Some((dx, dy)) = extract_integer_translation(scene_transform) {
                // Update the transform to be a pure integer translation.
                // This is valid as the scene is hinted, so we know it won't be later scaled.
                // As such, a displacement of up to 1/100 of a pixel is inperceptible, but it
                // makes our reasoning about this easier.
                scene_transform = Affine::translate((dx, dy));
            } else {
                // Translation not hinting compatible.
                return Err(());
            }
        }
        for command in input_commands {
            match command {
                RenderCommand::DrawPath(affine, path_id) => {
                    self.render_context.set_transform(scene_transform * *affine);
                    let path = &input_paths.meta[usize::try_from(path_id.0).unwrap()];
                    let path_end = &input_paths
                        .meta
                        .get(usize::try_from(path_id.0).unwrap() + 1)
                        .map_or(input_paths.elements.len(), |it| it.start_index);
                    let segments = &input_paths.elements[path.start_index..*path_end];
                    // Obviously, ideally we'd not be allocating here. This is forced by the current public API of Vello CPU.
                    let bezpath = BezPath::from_iter(segments.iter().cloned());
                    match &path.operation {
                        Operation::Stroke(stroke) => {
                            self.render_context.set_stroke(stroke.clone());
                            self.render_context.stroke_path(&bezpath);
                        }
                        Operation::Fill(fill) => {
                            self.render_context.set_fill_rule(*fill);
                            self.render_context.fill_path(&bezpath);
                        }
                    }
                }
                RenderCommand::PushLayer(push_layer_command) => {
                    self.render_context
                        .set_transform(push_layer_command.clip_transform);
                    let clip_path = if let Some(path_id) = push_layer_command.clip_path {
                        let path = &input_paths.meta[usize::try_from(path_id.0).unwrap()];
                        let path_end = &input_paths
                            .meta
                            .get(usize::try_from(path_id.0).unwrap() + 1)
                            .map_or(input_paths.elements.len(), |it| it.start_index);
                        let segments = &input_paths.elements[path.start_index..*path_end];
                        // Obviously, ideally we'd not be allocating here. This is forced by the current public API of Vello CPU.
                        let bezpath = BezPath::from_iter(segments.iter().cloned());
                        Some(bezpath)
                    } else {
                        None
                    };
                    self.render_context.push_layer(
                        clip_path.as_ref(),
                        push_layer_command.blend_mode,
                        push_layer_command.opacity,
                        None,
                        None,
                    );
                }
                RenderCommand::PopLayer => self.render_context.pop_layer(),
                RenderCommand::SetPaint(paint_transform, brush) => {
                    self.render_context.set_paint_transform(*paint_transform);
                    let brush = match brush {
                        Brush::Solid(alpha_color) => Brush::Solid(*alpha_color),
                        Brush::Gradient(gradient) => Brush::Gradient(gradient.clone()),
                        Brush::Image(brush) => {
                            let image_index =
                                brush.image.to_raw().try_into().expect("Handle this.");
                            Brush::Image(ImageBrush {
                                image: ImageSource::OpaqueId(ImageId::new(image_index)),
                                sampler: brush.sampler,
                            })
                        }
                    };
                    self.render_context.set_paint(brush);
                }
                RenderCommand::BlurredRoundedRectPaint(_) => {
                    unimplemented!(
                        "Vello CPU doesn't expose drawing a blurred rounded rectangle in custom shapes yet."
                    )
                }
            }
        }

        // We avoid duplicating stored handles.
        // It is likely that there's a better data structure for this (a `HashSet`?)
        // but there isn't one provided in core/alloc.
        // We expect the number of textures to be relatively small, so this being
        // O(N^2) isn't an immediate optimisation target.
        for texture in input_textures {
            if !self.textures.contains(texture) {
                self.textures.push(texture.clone());
            }
        }
        Ok(())
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
        brush: impl Into<Brush<ImageBrush<TextureHandle>>>,
        paint_transform: Affine,
    ) {
        self.render_context.set_paint_transform(paint_transform);
        let brush = match brush.into() {
            Brush::Solid(alpha_color) => Brush::Solid(alpha_color),
            Brush::Gradient(gradient) => Brush::Gradient(gradient),
            Brush::Image(brush) => {
                // TODO: Make this read more easily.
                let image_index = brush.image.id().to_raw().try_into().expect("Handle this.");
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
