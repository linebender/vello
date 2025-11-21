// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(missing_docs, reason = "This code is incomplete.")]
#![expect(clippy::result_unit_err, reason = "This code is incomplete.")]

use hashbrown::HashMap;
use vello_api::{
    PaintScene, Renderer,
    baseline::{BaselinePainter, BaselinePreparePaths},
    texture::{self, TextureDescriptor, TextureId, TextureUsages},
};
use vello_common::{
    encode::{EncodedImage, EncodedPaint},
    kurbo::{self, Affine, Shape},
    paint::{ImageId, ImageSource},
    peniko::{BlendMode, Brush, Color, Fill, ImageBrush, ImageData},
    pixmap::Pixmap,
};
use wgpu::{
    Device, Extent3d, Features, Origin3d, Queue, TexelCopyTextureInfo, TextureView,
    wgt::{CommandEncoderDescriptor, TextureViewDescriptor},
};

use crate::{RenderSettings, RenderTargetConfig, Scene};

#[derive(Debug)]
pub struct VelloHybrid {
    default_render_settings: RenderSettings,
    device: Device,
    queue: Queue,
    renderer: crate::Renderer,
    // TODO: Evaluate whether to use generational(?) indexing instead of a HashMap
    texture_id_source: u64,
    textures: HashMap<TextureId, StoredTexture>,
    format: wgpu::TextureFormat,
}

impl VelloHybrid {
    pub const REQUIRED_DEVICE_FEATURES: Features = Features::empty();
    pub fn new(
        device: &Device,
        queue: &Queue,
        render_target_config: &RenderTargetConfig,
        default_render_settings: RenderSettings,
    ) -> Self {
        Self {
            default_render_settings,
            textures: HashMap::new(),
            renderer: crate::Renderer::new(device, render_target_config),
            texture_id_source: 0,
            device: device.clone(),
            queue: queue.clone(),
            // TODO: We should support different internal and external formats/
            // rendering to the "native" format of the screen
            format: render_target_config.format,
        }
    }
    // Minimal API to access the image data. This is not intended for long-term use.
    pub fn wgpu_texture(&self, texture: &TextureId) -> Result<&wgpu::Texture, ()> {
        Ok(self
            .textures
            .get(texture)
            // TODO: Correct error type.
            .ok_or(())?
            .texture
            .as_ref()
            .ok_or(())?
            .view
            .texture())
    }
    // For use as a `Surface`, or similar.
    pub fn add_external_texture(
        &mut self,
        view: &wgpu::TextureView,
        usages: TextureUsages,
    ) -> Result<TextureId, ()> {
        let required_usages = translate_usages(usages);
        if !view.texture().usage().contains(required_usages) {
            // TODO: Better error kind.
            return Err(());
        }

        let texture = Some(RenderTargetTexture {
            view: view.clone(),
            atlas_up_to_date: false,
        });
        let texture = StoredTexture {
            // Can we avoid filling this pixmap immediately in some cases, e.g. for textures
            // we optimistically think will be uploaded to?
            image_id: None,
            texture,
            descriptor: TextureDescriptor {
                label: Some("ExternalRenderTarget"),
                width: view.texture().width().try_into().expect("Absurd."),
                height: view.texture().height().try_into().expect("Absurd."),
                usages,
            },
        };

        let id = TextureId::from_raw(self.texture_id_source);
        self.texture_id_source += 1;
        self.textures.insert(id, texture);
        Ok(id)
    }
}

impl Renderer for VelloHybrid {
    type ScenePainter = HybridScenePainter;
    // TODO: Obviously, a sparse strip native path caching would be ideal.
    type PathPreparer = BaselinePreparePaths;
    type Recording = BaselinePainter<BaselinePreparePaths>;
    type TransformedRecording = BaselinePainter<BaselinePreparePaths>;

    fn create_texture(&mut self, descriptor: texture::TextureDescriptor) -> TextureId {
        // TODO: What do we need to do with the usages?

        // We need an explicit texture if we expect to.
        let texture = if descriptor.usages.contains(TextureUsages::RENDER_TARGET)
            || descriptor.usages.contains(TextureUsages::DOWNLOAD_SRC)
        {
            // TODO: Warn if this is "download"-only, or "render_target" only?
            // For the former, there's no reason to actually perform the rendering, and for the latter,
            // you need to use the `add_external_texture`.
            let wgpu_usages = translate_usages(descriptor.usages);
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: descriptor.label,
                size: Extent3d {
                    width: descriptor.width.into(),
                    height: descriptor.height.into(),
                    depth_or_array_layers: 1,
                },
                // TODO: Support mipmaps?
                mip_level_count: 1,
                // Interaction with multisampling?
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: wgpu_usages,
                view_formats: &[],
            });
            Some(RenderTargetTexture {
                view: texture.create_view(&TextureViewDescriptor {
                    label: descriptor.label,
                    ..Default::default()
                }),
                atlas_up_to_date: false,
            })
        } else {
            None
        };
        let texture = StoredTexture {
            // Can we avoid filling this pixmap immediately in some cases, e.g. for textures
            // we optimistically think will be uploaded to?
            image_id: None,
            texture,
            descriptor,
        };

        let id = TextureId::from_raw(self.texture_id_source);
        self.texture_id_source += 1;
        self.textures.insert(id, texture);
        id
    }

    fn free_texture(&mut self, texture: TextureId) -> Result<(), ()> {
        let val = self.textures.remove(&texture).ok_or(())?;
        if let Some(image_id) = val.image_id {
            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Vello Hybrid Dealloc"),
                });
            self.renderer
                .destroy_image(&self.device, &self.queue, &mut encoder, image_id);
            self.queue.submit([encoder.finish()]);
        }
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

        // TODO: Handle options.clear_color (i.e. by encoding a texture write)
        // TODO: Cache the contexts internally, so that we don't reallocate here?
        let context = Scene::new_with(width, height, self.default_render_settings);
        Ok(HybridScenePainter {
            scene: context,
            target: *to,
        })
    }

    fn queue_render(&mut self, mut from: Self::ScenePainter) {
        // Ideally, we'd put all of the renders into a single encoder (or at least make it somehow configurable
        // - e.g. if we know that the previous submission has finished, there's value in submitting semi-eagerly)
        // However, we need to submit immediately because `render` internally uses `write_buffer`. Those operations
        // occur at the start of any submit calls, rather than being explicitly added to an encoder.
        // Therefore, if we  didn't submit eagerly, "newer" renders (or buffer uploads) would be able to overwrite the
        // rendering of "past" renders.
        // TODO: Fix that, i.e. by avoiding `write_buffer`/manually mapping.
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Vello Hybrid Render"),
            });
        for encoded_paint in &mut from.scene.encoded_paints {
            if let EncodedPaint::Image(EncodedImage {
                source: ref mut source @ ImageSource::OpaqueId(id),
                ..
            }) = *encoded_paint
            {
                // Back-associate each texture with the actual pixmap.
                let idx = u64::from(id.as_u32());
                let wgpu_texture = self
                    .textures
                    .get_mut(&TextureId::from_raw(idx))
                    .expect("todo: handle this case, where the texture passed to 'set_brush' isn't from this renderer.");
                if let Some(texture) = wgpu_texture.texture.as_ref()
                    && !texture.atlas_up_to_date
                {
                    // Luckily, the same thing doesn't apply as for `upload_image`
                    // as this method doesn't use `write_texture` to the atlas textures.
                    // (As instead it's a `render_pass` operation)
                    if let Some(old_id) = wgpu_texture.image_id {
                        self.renderer.destroy_image(
                            &self.device,
                            &self.queue,
                            &mut encoder,
                            old_id,
                        );
                    }
                    // TODO: If there were an old slot (the common case), we should be re-using that (or, of course, just rendering from
                    // the extant `wgpu::Texture` directly)
                    let image_id = self.renderer.upload_image(
                        &self.device,
                        &self.queue,
                        &mut encoder,
                        texture.view.texture(),
                    );
                    wgpu_texture.image_id = Some(image_id);
                }
                let atlas_id = wgpu_texture.image_id.expect(
                    "Texture should have been a render target or uploaded to before using it.",
                );
                *source = ImageSource::OpaqueId(atlas_id);
            }
        }

        let texture = self.textures.get_mut(&from.target).unwrap();
        #[expect(clippy::todo, reason = "Still applies.")]
        if !texture
            .descriptor
            .usages
            .contains(TextureUsages::RENDER_TARGET)
        {
            todo!("Return an error");
        }
        let wgpu_texture = texture
            .texture
            .as_mut()
            .expect("A `RENDER_TARGET` is always created with a corresponding wgpu texture.");

        // TODO: This size isn't right, we need to use the size in the `SceneOptions` (if we decide to keep that...)
        self.renderer
            .render(
                &from.scene,
                &self.device,
                &self.queue,
                &mut encoder,
                &crate::RenderSize {
                    width: texture.descriptor.width as u32,
                    height: texture.descriptor.height as u32,
                },
                &wgpu_texture.view,
            )
            .expect("Better error handling.");
        // If this texture will be used for render.
        wgpu_texture.atlas_up_to_date = false;
        // TODO: Do we need to reset the scene here?
        // That is, should we be taking ownership in this function?
        from.scene.reset();
        // See comment on `encoder` for detail of why we create a temporary encoder here.
        self.queue.submit([encoder.finish()]);

        // TODO: We almost certainly want to keep the scene around.
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
        // TODO: Determine whether we can/should do the BGRA->RGBA swizzle/premultiplication on the GPU
        // as we copy from staging memory.
        let pixmap = Pixmap::from_peniko_image_data(data);
        // TODO: Again (see `queue_render`), we should actually be maintaining a persistent encoder here.
        // However, we submit immediately because `upload_image` uses `Queue::write_texture`, which doesn't
        // operate correctly in the encoder.
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Vello Hybrid Internal Upload"),
            });
        if let Some(texture) = source.texture.as_mut() {
            // TODO: This also should use an explicit staging buffer/copy command.
            self.queue.write_texture(
                TexelCopyTextureInfo {
                    aspect: wgpu::TextureAspect::All,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    texture: texture.view.texture(),
                },
                bytemuck::cast_slice(pixmap.data()),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    // 4 bytes per RGBA8 pixel
                    bytes_per_row: Some(data.width << 2),
                    rows_per_image: Some(data.height),
                },
                Extent3d {
                    width: data.width,
                    height: data.height,
                    depth_or_array_layers: 1,
                },
            );
            texture.atlas_up_to_date = false;
        } else {
            if let Some(old_id) = source.image_id {
                let mut destroy_encoder =
                    self.device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Vello Hybrid Internal Upload"),
                        });
                self.renderer.destroy_image(
                    &self.device,
                    &self.queue,
                    &mut destroy_encoder,
                    old_id,
                );
                // We need to make sure that the slot clearing which this scheduled is executed before
                // the `write_texture`s which `upload_image` does.
                // Unfortunately, the only way to achieve this is to submit this in yet another submission.
                // See comment at the start of queue_render for more context.
                // We expect to change this in the future (by avoiding `queue.write_texture` and friends)
                self.queue.submit([destroy_encoder.finish()]);
            }
            let image_id =
                self.renderer
                    .upload_image(&self.device, &self.queue, &mut encoder, &pixmap);
            source.image_id = Some(image_id);
        }
        self.queue.submit([encoder.finish()]);
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

fn translate_usages(usages: TextureUsages) -> wgpu::TextureUsages {
    let download_src = usages.contains(TextureUsages::DOWNLOAD_SRC);
    let upload_target = usages.contains(TextureUsages::UPLOAD_TARGET);
    let texture_binding = usages.contains(TextureUsages::TEXTURE_BINDING);
    let render_target = usages.contains(TextureUsages::RENDER_TARGET);
    let mut wgpu_usages = wgpu::TextureUsages::empty();
    if render_target {
        wgpu_usages |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    if texture_binding {
        // Currently, all of our rendering happens in the atlas.
        // Ideally, we'd just bind this texture directly, so we need to copy it into the atlas in just-in-time manner.
        // However, currently we don't do that, so we need to copy this into the atlas.
        wgpu_usages |= wgpu::TextureUsages::COPY_SRC;
    }
    if upload_target {
        wgpu_usages |= wgpu::TextureUsages::COPY_DST;
    }
    if download_src {
        wgpu_usages |= wgpu::TextureUsages::COPY_SRC;
    }
    wgpu_usages
}

#[derive(Debug)]
pub struct HybridScenePainter {
    scene: Scene,
    target: TextureId,
}

impl PaintScene for HybridScenePainter {
    fn width(&self) -> u16 {
        self.scene.width()
    }

    fn height(&self) -> u16 {
        self.scene.height()
    }

    fn fill_path_new(&mut self, transform: Affine, fill_rule: Fill, path: impl Shape) {
        self.scene.set_transform(transform);
        self.scene.set_fill_rule(fill_rule);
        // TODO: Tweak inner `fill_path` API to either take a `Shape` or an &[PathEl]
        self.scene.fill_path(&path.to_path(0.1));
    }

    fn stroke_path_new(
        &mut self,
        transform: Affine,
        stroke_params: &kurbo::Stroke,
        path: impl Shape,
    ) {
        self.scene.set_transform(transform);
        self.scene.set_stroke(stroke_params.clone());
        self.scene.stroke_path(&path.to_path(0.1));
    }

    fn set_brush(
        &mut self,
        brush: impl Into<Brush<ImageBrush<TextureId>>>,
        // The transform should be the same as for the following path.
        _: Affine,
        paint_transform: Affine,
    ) {
        // self.scene.set_transform(transform);
        self.scene.set_paint_transform(paint_transform);
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
        self.scene.set_paint(brush);
    }

    fn set_blurred_rounded_rect_brush(
        &mut self,
        _transform: Affine,
        _paint_transform: Affine,
        _color: Color,
        _rect: &kurbo::Rect,
        _radius: f32,
        _std_dev: f32,
    ) {
        unimplemented!("Vello Hybrid doesn't expose drawing blurred rounded rectangles yet.")
    }

    fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        self.scene.set_blend_mode(blend_mode);
    }

    fn push_layer_new(
        &mut self,
        clip_transform: Affine,
        clip_path: Option<impl Shape>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        // mask: Option<Mask>,
    ) {
        self.scene.set_transform(clip_transform);
        self.scene.push_layer(
            clip_path.map(|it| it.to_path(0.1)).as_ref(),
            blend_mode,
            opacity,
            None,
        );
    }

    fn push_clip_layer_new(&mut self, clip_transform: Affine, path: impl Shape) {
        self.scene.set_transform(clip_transform);
        self.scene.push_clip_layer(
            // TODO: Not allocate
            &path.to_path(0.1),
        );
    }

    fn pop_layer(&mut self) {
        self.scene.pop_layer();
    }

    fn read_stateful_transform(&self) -> Affine {
        self.scene.transform
    }

    fn read_stateful_paint_transform(&self) -> Affine {
        self.scene.paint_transform
    }

    fn read_stateful_fill_rule(&self) -> Fill {
        self.scene.fill_rule
    }

    fn read_stateful_stroke(&self) -> kurbo::Stroke {
        self.scene.stroke.clone()
    }

    fn set_stroke(&mut self, stroke: kurbo::Stroke) {
        self.scene.set_stroke(stroke);
    }

    fn set_paint_transform(&mut self, affine: Affine) {
        self.scene.set_paint_transform(affine);
    }

    fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.scene.set_fill_rule(fill_rule);
    }

    fn set_transform(&mut self, transform: Affine) {
        self.scene.set_transform(transform);
    }

    fn push_clip_path(&mut self, path: &kurbo::BezPath) {
        self.scene.push_clip_path(path);
    }

    fn pop_clip_path(&mut self) {
        self.scene.pop_clip_path();
    }

    fn fill_blurred_rounded_rect(&mut self, _rect: &kurbo::Rect, _radius: f32, _std_dev: f32) {
        unimplemented!()
    }
}

#[derive(Debug)]
struct RenderTargetTexture {
    // We store only the `TextureView`, as it's trivial to get a `Texture`
    // from the view.
    view: TextureView,
    atlas_up_to_date: bool,
}

#[derive(Debug)]
struct StoredTexture {
    // TODO: Reason much more carefully about this.
    // In particular, this isn't right for atlased textures.
    texture: Option<RenderTargetTexture>,
    descriptor: texture::TextureDescriptor,
    image_id: Option<ImageId>,
}
