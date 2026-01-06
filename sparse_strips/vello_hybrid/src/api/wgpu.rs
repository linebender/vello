// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::{
    sync::{Arc, Weak},
    vec::Vec,
};
use hashbrown::HashMap;
use vello_api::{
    PaintScene, Renderer,
    peniko::Fill,
    sync::Lock,
    texture::{self, TextureDescriptor, TextureId, TextureUsages},
};
use vello_common::{
    encode::{EncodedImage, EncodedPaint},
    kurbo::{Affine, Rect},
    paint::{ImageId, ImageSource},
    peniko::ImageData,
    pixmap::Pixmap,
};
use wgpu::{
    Device, Extent3d, Features, Origin3d, Queue, TexelCopyTextureInfo, TextureView,
    wgt::{CommandEncoderDescriptor, TextureViewDescriptor},
};

use crate::{RenderSettings, RenderTargetConfig, Scene, api::HybridScenePainter};
#[derive(Debug)]
struct VelloHybridInner {
    renderer: crate::Renderer,
    // TODO: Evaluate whether to use generational(?) indexing instead of a HashMap
    texture_id_source: u64,
    textures: HashMap<TextureId, StoredTexture>,
}

#[derive(Debug)]
pub struct VelloHybrid {
    inner: Lock<VelloHybridInner>,
    device: Device,
    queue: Queue,
    format: wgpu::TextureFormat,
    default_render_settings: RenderSettings,
    this: Weak<Self>,
}

impl VelloHybrid {
    pub const REQUIRED_DEVICE_FEATURES: Features = Features::empty();
    pub fn new(
        device: &Device,
        queue: &Queue,
        render_target_config: &RenderTargetConfig,
        default_render_settings: RenderSettings,
    ) -> Arc<Self> {
        let inner = VelloHybridInner {
            textures: HashMap::new(),
            renderer: crate::Renderer::new(device, render_target_config),
            texture_id_source: 0,
        };
        Arc::new_cyclic(|this| {
            Self {
                default_render_settings,
                inner: Lock::new(inner),
                device: device.clone(),
                queue: queue.clone(),
                // TODO: We should support different internal and external formats/
                // rendering to the "native" format of the screen
                format: render_target_config.format,
                this: this.clone(),
            }
        })
    }
    // Minimal API to access the image data. This API has not been carefully designed, and warrants re-examination.
    pub fn wgpu_texture(&self, texture: TextureId) -> Result<wgpu::Texture, ()> {
        Ok(self
            .inner
            .lock()
            .textures
            .get(&texture)
            // TODO: Correct error type.
            .ok_or(())?
            .texture
            .as_ref()
            .ok_or(())?
            .view
            .texture()
            .clone())
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

        let mut this = self.inner.lock();
        let id = TextureId::from_raw(this.texture_id_source);
        this.texture_id_source += 1;
        this.textures.insert(id, texture);
        Ok(id)
    }
}

impl Renderer for VelloHybrid {
    type ScenePainter = HybridScenePainter;

    fn alloc_untracked_texture(&self, descriptor: texture::TextureDescriptor) -> TextureId {
        // We need an explicit texture if we expect to download from it or render to it.
        let texture = if descriptor.usages.contains(TextureUsages::RENDER_TARGET)
        // || descriptor.usages.contains(TextureUsages::DOWNLOAD_SRC)
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
            image_id: None,
            texture,
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
        let val = this.textures.remove(&texture).ok_or(())?;
        if let Some(image_id) = val.image_id {
            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Vello Hybrid Dealloc"),
                });
            this.renderer
                .destroy_image(&self.device, &self.queue, &mut encoder, image_id);
            self.queue.submit([encoder.finish()]);
        }
        Ok(())
    }

    fn create_scene(
        &self,
        to: &TextureId,
        options: vello_api::SceneOptions,
    ) -> Result<Self::ScenePainter, ()> {
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

        // TODO: Handle options.clear_color more efficiently (i.e. by encoding a load colour in the final render pass)
        // TODO: Cache the contexts internally, so that we don't reallocate here?
        let scene = Scene::new_with(width, height, self.default_render_settings);
        let mut painter = HybridScenePainter {
            scene,
            target: *to,
            renderer: self.as_dyn_arc(),
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
        let this = &mut *this;
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
                let wgpu_texture = this
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
                        this.renderer.destroy_image(
                            &self.device,
                            &self.queue,
                            &mut encoder,
                            old_id,
                        );
                    }
                    // TODO: If there were an old slot (the common case), we should be re-using that (or, of course, just rendering from
                    // the extant `wgpu::Texture` directly)
                    let image_id = this.renderer.upload_image(
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

        let texture = this.textures.get_mut(&from.target).unwrap();
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
        this.renderer
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

    fn upload_image(&self, to: &TextureId, data: &ImageData) -> Result<(), ()> {
        let mut this = self.inner.lock();
        let this = &mut *this;
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
                this.renderer.destroy_image(
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
                this.renderer
                    .upload_image(&self.device, &self.queue, &mut encoder, &pixmap);
            source.image_id = Some(image_id);
        }
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    fn as_arc(&self) -> alloc::sync::Arc<Self>
    where
        Self: Sized,
    {
        self.this
            .upgrade()
            .expect("self still exists, so 'this' should still exist.")
    }

    fn as_dyn_arc(&self) -> alloc::sync::Arc<dyn Renderer> {
        self.as_arc()
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
struct RenderTargetTexture {
    // We store only the `TextureView`, as it's trivial to get a `Texture`
    // from the view.
    view: TextureView,
    // TODO: Because of external textures, this is actually a three-way condition; yes, no, never
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
