// Copyright 2022 The piet-gpu authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

use piet_gpu::{PixelFormat, RenderConfig};
use piet_gpu_hal::{QueryPool, Session};
use piet_scene::glyph::pinot::{types::Tag, FontDataRef};
use piet_scene::glyph::{GlyphContext, GlyphProvider};
use piet_scene::{Affine, Rect, Scene, SceneFragment};

/// State and resources for rendering a scene.
pub struct PgpuRenderer {
    session: Session,
    pgpu_renderer: Option<piet_gpu::Renderer>,
    query_pool: QueryPool,
    width: u32,
    height: u32,
    is_color: bool,
}

impl PgpuRenderer {
    #[cfg(all(
        not(target_arch = "wasm32"),
        any(target_os = "ios", target_os = "macos")
    ))]
    pub fn new(device: &metal::DeviceRef, queue: &metal::CommandQueueRef) -> Self {
        let piet_device = piet_gpu_hal::Device::new_from_raw_mtl(device, &queue);
        let session = Session::new(piet_device);
        let query_pool = session.create_query_pool(12).unwrap();
        Self {
            session,
            pgpu_renderer: None,
            query_pool,
            width: 0,
            height: 0,
            is_color: false,
        }
    }

    #[cfg(all(
        not(target_arch = "wasm32"),
        any(target_os = "ios", target_os = "macos")
    ))]
    pub fn render(
        &mut self,
        scene: &PgpuScene,
        cmdbuf: &metal::CommandBufferRef,
        target: &metal::TextureRef,
    ) -> u32 {
        let is_color = target.pixel_format() != metal::MTLPixelFormat::R8Unorm;
        let width = target.width() as u32;
        let height = target.height() as u32;
        if self.pgpu_renderer.is_none()
            || self.width != width
            || self.height != height
            || self.is_color != is_color
        {
            self.width = width;
            self.height = height;
            self.is_color = is_color;
            let format = if is_color {
                PixelFormat::Rgba8
            } else {
                PixelFormat::A8
            };
            let config = RenderConfig::new(width as usize, height as usize).pixel_format(format);
            unsafe {
                self.pgpu_renderer =
                    piet_gpu::Renderer::new_from_config(&self.session, config, 1).ok();
            }
        }
        unsafe {
            let mut cmd_buf = self.session.cmd_buf_from_raw_mtl(cmdbuf);
            let dst_image = self
                .session
                .image_from_raw_mtl(target, self.width, self.height);
            if let Some(renderer) = &mut self.pgpu_renderer {
                renderer.upload_scene(&scene.encoded_scene(), 0).unwrap();
                renderer.record(&mut cmd_buf, &self.query_pool, 0);
                // TODO later: we can bind the destination image and avoid the copy.
                cmd_buf.blit_image(&renderer.image_dev, &dst_image);
                cmd_buf.flush();
            }
        }
        0
    }

    pub fn release(&mut self, _id: u32) {
        // TODO: worry about freeing resources / managing overlapping submits
    }
}

/// Encoded streams and resources describing a vector graphics scene.
pub struct PgpuScene(pub Scene);

impl PgpuScene {
    pub fn new() -> Self {
        Self(Scene::default())
    }

    pub fn builder(&mut self) -> PgpuSceneBuilder {
        PgpuSceneBuilder {
            builder: piet_scene::SceneBuilder::for_scene(&mut self.0),
            transform: Affine::IDENTITY,
        }
    }
}

/// Encoded streams and resources describing a vector graphics scene fragment.
pub struct PgpuSceneFragment(pub SceneFragment);

impl PgpuSceneFragment {
    pub fn new() -> Self {
        Self(SceneFragment::default())
    }

    pub fn builder(&mut self) -> PgpuSceneBuilder {
        PgpuSceneBuilder {
            builder: piet_scene::SceneBuilder::for_fragment(&mut self.0),
            transform: Affine::IDENTITY,
        }
    }
}

/// Builder for constructing an encoded scene.
pub struct PgpuSceneBuilder<'a> {
    pub builder: piet_scene::SceneBuilder<'a>,
    pub transform: Affine,
}

impl<'a> PgpuSceneBuilder<'a> {
    pub fn add_glyph(&mut self, glyph: &PgpuGlyph, transform: &piet_scene::Affine) {
        self.builder.append(&glyph.fragment, Some(*transform));
    }

    pub fn finish(self) {
        self.builder.finish();
    }
}

/// Tag and value for a font variation axis.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct PgpuFontVariation {
    /// Tag that specifies the axis.
    pub tag: u32,
    /// Requested setting for the axis.
    pub value: f32,
}

/// Context for loading and scaling glyphs.
pub struct PgpuGlyphContext(GlyphContext);

impl PgpuGlyphContext {
    pub fn new() -> Self {
        Self(GlyphContext::new())
    }

    pub fn new_provider<'a>(
        &'a mut self,
        font_data: &'a [u8],
        font_index: u32,
        font_id: u64,
        ppem: f32,
        hint: bool,
        variations: &[PgpuFontVariation],
    ) -> Option<PgpuGlyphProvider> {
        let font = FontDataRef::new(font_data).and_then(|f| f.get(font_index))?;
        Some(PgpuGlyphProvider(
            self.0.new_provider(
                &font,
                Some(font_id),
                ppem,
                hint,
                variations
                    .iter()
                    .map(|variation| (Tag(variation.tag), variation.value)),
            ),
        ))
    }
}

/// Context for loading a scaling glyphs from a specific font.
pub struct PgpuGlyphProvider<'a>(GlyphProvider<'a>);

impl<'a> PgpuGlyphProvider<'a> {
    pub fn get(&mut self, gid: u16) -> Option<PgpuGlyph> {
        let fragment = self.0.get(gid, None)?;
        Some(PgpuGlyph { fragment })
    }

    pub fn get_color(&mut self, palette_index: u16, gid: u16) -> Option<PgpuGlyph> {
        let fragment = self.0.get_color(palette_index, gid)?;
        Some(PgpuGlyph { fragment })
    }
}

/// Encoded (possibly color) outline for a glyph.
pub struct PgpuGlyph {
    fragment: SceneFragment,
}

impl PgpuGlyph {
    pub fn bbox(&self, transform: Option<Affine>) -> Rect {
        if let Some(transform) = &transform {
            Rect::from_points(
                self.fragment
                    .points()
                    .iter()
                    .map(|p| p.transform(transform)),
            )
        } else {
            Rect::from_points(self.fragment.points())
        }
    }
}
