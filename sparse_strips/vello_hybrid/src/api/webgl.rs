// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use hashbrown::HashMap;
use vello_api::{
    PaintScene, Renderer,
    baseline::{BaselinePainter, BaselinePreparePaths},
    peniko::Fill,
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
pub struct VelloHybridWebgl {
    renderer: crate::WebGlRenderer,
    texture_id_source: u64,
    textures: HashMap<TextureId, StoredTexture>,
    default_render_settings: RenderSettings,
    width: u16,
    height: u16,
}

impl VelloHybridWebgl {
    pub const CANVAS_TEXTURE_ID: TextureId = TextureId::from_raw(0);

    pub fn new(canvas: &web_sys::HtmlCanvasElement, settings: RenderSettings) -> Self {
        let renderer = WebGlRenderer::new_with(canvas, settings);
        Self {
            renderer,
            // Start at 1 because id 0 is reserved for the canvas.
            texture_id_source: 1,
            textures: HashMap::new(),
            default_render_settings: settings,
            width: canvas.width().try_into().unwrap(),
            height: canvas.height().try_into().unwrap(),
        }
    }
    // Ideally, we'd support "registering" a second canvas here, so that you can render to multiple canvases in the scene.
    // However, it's not possible to do so in any kind of nice way, because WebGL doesn't support sharing resources between `WebGl2RenderingContext`s,
    // and the workarounds for that are not performant on Firefox - see https://bugzilla.mozilla.org/show_bug.cgi?id=1788206.

    // Minimal API to allow access to the image data. This API has not been carefully designed.
    pub fn gl_context(&self) -> &WebGl2RenderingContext {
        self.renderer.gl_context()
    }

    pub fn resize(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
    }
}

impl Renderer for VelloHybridWebgl {
    type ScenePainter = HybridScenePainter;
    // TODO: Obviously, a sparse strip native path caching would be ideal.
    type PathPreparer = BaselinePreparePaths;
    type Recording = BaselinePainter<BaselinePreparePaths>;
    type TransformedRecording = BaselinePainter<BaselinePreparePaths>;

    fn create_texture(&mut self, descriptor: texture::TextureDescriptor) -> TextureId {
        if descriptor.usages.contains(TextureUsages::RENDER_TARGET) {
            // This is *not* a fundamental property of the backend, just an implementation detail.
            // Therefore: TODO: Allow creating and targeting a custom `WebGlFramebuffer`.
            // The fact that this is functionally useless shall not be examined.
            unimplemented!(
                "Vello Hybrid's WebGL renderer cannot currently make secondary textures which are rendered to."
            );
        }
        // TODO: If this is not a texture binding & UPLOAD_DST, it's currently completely useless

        let texture = StoredTexture {
            image_id: None,
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
            self.renderer.destroy_image(image_id);
        }
        Ok(())
    }

    fn create_scene(
        &mut self,
        to: &TextureId,
        options: vello_api::SceneOptions,
    ) -> Result<Self::ScenePainter, ()> {
        assert_eq!(
            *to,
            Self::CANVAS_TEXTURE_ID,
            "Can only render to `CANVAS_TEXTURE_ID` in the WebGL renderer."
        );
        let (width, height) = if let Some(size) = options.size() {
            size
        } else {
            (self.width, self.height)
        };
        self.width = width;
        self.height = height;

        // TODO: Handle options.clear_color (i.e. by encoding a texture write)
        // TODO: Cache the contexts internally, so that we don't reallocate here?
        let scene = Scene::new_with(width, height, self.default_render_settings);
        let mut painter = HybridScenePainter { scene, target: *to };
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
        for encoded_paint in &mut from.scene.encoded_paints {
            if let EncodedPaint::Image(EncodedImage {
                source: ref mut source @ ImageSource::OpaqueId(id),
                ..
            }) = *encoded_paint
            {
                // Back-associate each texture with the actual pixmap.
                let idx = u64::from(id.as_u32());
                let stored_texture = self
                .textures
                .get_mut(&TextureId::from_raw(idx))
                .expect("todo: handle this case, where the texture passed to 'set_brush' isn't from this renderer.");

                let atlas_id = stored_texture
                    .image_id
                    .expect("Texture should have been uploaded to before using it.");
                *source = ImageSource::OpaqueId(atlas_id);
            }
        }

        // TODO: This size isn't right, we need to use the size in the `SceneOptions` (if we decide to keep that...)
        self.renderer
            .render(
                &from.scene,
                &crate::RenderSize {
                    width: self.width as u32,
                    height: self.height as u32,
                },
            )
            .expect("Better error handling.");
        // TODO: Do we need to reset the scene here?
        // That is, should we be taking ownership in this function?
        from.scene.reset();

        // TODO: We almost certainly want to keep the scene around (but with care about the opaque ids).
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

        // TODO: We really should be reusing the prior atlas slot here, but that isn't currently exposed.
        if let Some(old_id) = source.image_id {
            self.renderer.destroy_image(old_id);
        }
        // TODO: Determine whether we can/should do the BGRA->RGBA swizzle/premultiplication on the GPU
        // as we copy from staging memory.
        let pixmap = Pixmap::from_peniko_image_data(data);
        let image_id = self.renderer.upload_image(&pixmap);
        source.image_id = Some(image_id);

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
struct StoredTexture {
    // TODO: Reason much more carefully about this.
    // In particular, this isn't right for atlased textures.
    descriptor: texture::TextureDescriptor,
    image_id: Option<ImageId>,
}
