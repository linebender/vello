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
    texture::{self, TextureId, TextureUsages},
};
use vello_common::{
    encode::{EncodedImage, EncodedPaint},
    kurbo::{Affine, Rect},
    paint::{ImageId, ImageSource},
    peniko::ImageData,
    pixmap::Pixmap,
};
use web_sys::WebGl2RenderingContext;

use crate::{RenderSettings, Scene, WebGlRenderer, api::HybridScenePainter};

#[derive(Debug)]
struct VelloHybridWebglInner {
    renderer: crate::WebGlRenderer,
    texture_id_source: u64,
    textures: HashMap<TextureId, StoredTexture>,
    default_render_settings: RenderSettings,
    width: u16,
    height: u16,
}

#[derive(Debug)]
pub struct VelloHybridWebgl {
    inner: Lock<VelloHybridWebglInner>,
    this: Weak<VelloHybridWebgl>,
}

impl VelloHybridWebgl {
    pub const CANVAS_TEXTURE_ID: TextureId = TextureId::from_raw(0);

    pub fn new(canvas: &web_sys::HtmlCanvasElement, settings: RenderSettings) -> Arc<Self> {
        let renderer = WebGlRenderer::new_with(canvas, settings);
        let inner = VelloHybridWebglInner {
            renderer,
            // Start at 1 because id 0 is reserved for the canvas.
            texture_id_source: 1,
            textures: HashMap::new(),
            default_render_settings: settings,
            width: canvas.width().try_into().unwrap(),
            height: canvas.height().try_into().unwrap(),
        };
        Arc::new_cyclic(|this| Self {
            inner: Lock::new(inner),
            this: this.clone(),
        })
    }
    // Ideally, we'd support "registering" a second canvas here, so that you can render to multiple canvases in the scene.
    // However, it's not possible to do so in any kind of nice way, because WebGL doesn't support sharing resources between `WebGl2RenderingContext`s,
    // and the workarounds for that are not performant on Firefox - see https://bugzilla.mozilla.org/show_bug.cgi?id=1788206.

    // Minimal API to allow access to the image data. This API has not been carefully designed.
    pub fn gl_context(&self) -> WebGl2RenderingContext {
        self.inner.lock().renderer.gl_context().clone()
    }

    pub fn resize(&mut self, width: u16, height: u16) {
        let mut this = self.inner.lock();
        this.width = width;
        this.height = height;
    }
}

impl Renderer for VelloHybridWebgl {
    type ScenePainter = HybridScenePainter;

    fn alloc_untracked_texture(&self, descriptor: texture::TextureDescriptor) -> TextureId {
        if descriptor.usages.contains(TextureUsages::RENDER_TARGET) {
            // This is *not* a fundamental property of the backend, just an implementation detail.
            // Therefore: TODO: Allow creating and targeting a custom `WebGlFramebuffer`.
            // Obviously you can only actually "read-back"
            unimplemented!(
                "Vello Hybrid's WebGL renderer cannot currently make secondary textures which are rendered to."
            );
        }
        // TODO: If this is not a texture binding & UPLOAD_DST, it's currently completely useless

        let texture = StoredTexture {
            image_id: None,
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
            this.renderer.destroy_image(image_id);
        }
        Ok(())
    }

    fn create_scene(
        &self,
        to: &TextureId,
        options: vello_api::SceneOptions,
    ) -> Result<Self::ScenePainter, ()> {
        assert_eq!(
            *to,
            Self::CANVAS_TEXTURE_ID,
            "Can only render to `CANVAS_TEXTURE_ID` in the WebGL renderer."
        );
        let mut this = self.inner.lock();
        let (width, height) = if let Some(size) = options.size() {
            size
        } else {
            (this.width, this.height)
        };
        this.width = width;
        this.height = height;

        // TODO: Handle options.clear_color (i.e. by encoding a texture write)
        // TODO: Cache the contexts internally, so that we don't reallocate here?
        let scene = Scene::new_with(width, height, this.default_render_settings);
        drop(this);
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
        for encoded_paint in &mut from.scene.encoded_paints {
            if let EncodedPaint::Image(EncodedImage {
                source: ref mut source @ ImageSource::OpaqueId(id),
                ..
            }) = *encoded_paint
            {
                // Back-associate each texture with the actual pixmap.
                let idx = u64::from(id.as_u32());
                let stored_texture = this
                .textures
                .get_mut(&TextureId::from_raw(idx))
                .expect("todo: handle this case, where the texture passed to 'set_brush' isn't from this renderer.");

                let atlas_id = stored_texture
                    .image_id
                    .expect("Texture should have been uploaded to before using it.");
                *source = ImageSource::OpaqueId(atlas_id);
            }
        }

        this.renderer
            .render(
                &from.scene,
                &crate::RenderSize {
                    width: this.width as u32,
                    height: this.height as u32,
                },
            )
            .expect("Better error handling.");
        // TODO: Do we need to reset the scene here?
        // That is, should we be taking ownership in this function?
        from.scene.reset();

        // TODO: We almost certainly want to keep the scene around (but with care about the opaque ids).
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

        // TODO: We really should be reusing the prior atlas slot here, but that isn't currently exposed.
        if let Some(old_id) = source.image_id {
            this.renderer.destroy_image(old_id);
        }
        // TODO: Determine whether we can/should do the BGRA->RGBA swizzle/premultiplication on the GPU
        // as we copy from staging memory.
        let pixmap = Pixmap::from_peniko_image_data(data);
        let image_id = this.renderer.upload_image(&pixmap);
        source.image_id = Some(image_id);

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

#[derive(Debug)]
struct StoredTexture {
    // TODO: Reason much more carefully about this.
    // In particular, this isn't right for atlased textures.
    descriptor: texture::TextureDescriptor,
    image_id: Option<ImageId>,
}
