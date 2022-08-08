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

// We only really have implementations for IOS targets so far
// Note that this is the same cfg that wgpu uses for metal support
#![cfg_attr(
    not(all(
        not(target_arch = "wasm32"),
        any(target_os = "ios", target_os = "macos")
    )),
    allow(unused)
)]

mod render;

use piet_scene::brush::{Brush, Color};
use piet_scene::path::Element;
use piet_scene::scene::Fill;
use render::*;
use std::ffi::c_void;
use std::mem::transmute;

/// Creates a new piet-gpu renderer for the specified Metal device and
/// command queue.
///
/// device: MTLDevice*
/// queue: MTLCommandQueue*
#[no_mangle]
#[cfg(all(
    not(target_arch = "wasm32"),
    any(target_os = "ios", target_os = "macos")
))]
pub unsafe extern "C" fn pgpu_renderer_new(
    device: *mut c_void,
    queue: *mut c_void,
) -> *mut PgpuRenderer {
    let device: &metal::DeviceRef = transmute(device);
    let queue: &metal::CommandQueueRef = transmute(queue);
    Box::into_raw(Box::new(PgpuRenderer::new(device, queue)))
}

/// Renders a prepared scene into a texture target. Commands for rendering are
/// recorded into the specified command buffer. Returns an id representing
/// resources that may have been allocated during this process. After the
/// command buffer has been retired, call `pgpu_renderer_release` with this id
/// to drop any associated resources.
///
/// target: MTLTexture*
/// cmdbuf: MTLCommandBuffer*
#[no_mangle]
#[cfg(all(
    not(target_arch = "wasm32"),
    any(target_os = "ios", target_os = "macos")
))]
pub unsafe extern "C" fn pgpu_renderer_render(
    renderer: *mut PgpuRenderer,
    scene: *const PgpuScene,
    target: *mut c_void,
    cmdbuf: *mut c_void,
) -> u32 {
    let cmdbuf: &metal::CommandBufferRef = transmute(cmdbuf);
    let target: &metal::TextureRef = transmute(target);
    (*renderer).render(&*scene, cmdbuf, target)
}

/// Releases the internal resources associated with the specified id from a
/// previous render operation.
#[no_mangle]
pub unsafe extern "C" fn pgpu_renderer_release(renderer: *mut PgpuRenderer, id: u32) {
    (*renderer).release(id);
}

/// Destroys the piet-gpu renderer.
#[no_mangle]
pub unsafe extern "C" fn pgpu_renderer_destroy(renderer: *mut PgpuRenderer) {
    Box::from_raw(renderer);
}

/// Creates a new, empty piet-gpu scene.
#[no_mangle]
pub unsafe extern "C" fn pgpu_scene_new() -> *mut PgpuScene {
    Box::into_raw(Box::new(PgpuScene::new()))
}

/// Destroys the piet-gpu scene.
#[no_mangle]
pub unsafe extern "C" fn pgpu_scene_destroy(scene: *mut PgpuScene) {
    Box::from_raw(scene);
}

/// Creates a new, empty piet-gpu scene fragment.
#[no_mangle]
pub unsafe extern "C" fn pgpu_scene_fragment_new() -> *mut PgpuSceneFragment {
    Box::into_raw(Box::new(PgpuSceneFragment::new()))
}

/// Destroys the piet-gpu scene fragment.
#[no_mangle]
pub unsafe extern "C" fn pgpu_scene_fragment_destroy(fragment: *mut PgpuSceneFragment) {
    Box::from_raw(fragment);
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum PgpuPathVerb {
    MoveTo = 0,
    LineTo = 1,
    QuadTo = 2,
    CurveTo = 3,
    Close = 4,
}

#[derive(Copy, Clone, Default, Debug)]
#[repr(C)]
pub struct PgpuPoint {
    pub x: f32,
    pub y: f32,
}

/// Rectangle defined by minimum and maximum points.
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct PgpuRect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct PgpuPathElement {
    pub verb: PgpuPathVerb,
    pub points: [PgpuPoint; 3],
}

#[derive(Clone)]
#[repr(C)]
pub struct PgpuPathIter {
    pub context: *mut c_void,
    pub next_element: extern "C" fn(*mut c_void, *mut PgpuPathElement) -> bool,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum PgpuFill {
    NonZero = 0,
    EvenOdd = 1,
}

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub enum PgpuBrushKind {
    Solid = 0,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct PgpuColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[repr(C)]
pub union PgpuBrushData {
    pub solid: PgpuColor,
}

#[repr(C)]
pub struct PgpuBrush {
    pub kind: PgpuBrushKind,
    pub data: PgpuBrushData,
}

/// Affine transformation matrix.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct PgpuTransform {
    pub xx: f32,
    pub yx: f32,
    pub xy: f32,
    pub yy: f32,
    pub dx: f32,
    pub dy: f32,
}

impl From<PgpuTransform> for PgpuAffine {
    fn from(xform: PgpuTransform) -> Self {
        Self {
            xx: xform.xx,
            yx: xform.yx,
            xy: xform.xy,
            yy: xform.yy,
            dx: xform.dx,
            dy: xform.dy,
        }
    }
}

pub type PgpuAffine = piet_scene::geometry::Affine;

/// Creates a new builder for filling a piet-gpu scene. The specified scene
/// should not be accessed while the builder is live.
#[no_mangle]
pub unsafe extern "C" fn pgpu_scene_builder_for_scene(
    scene: *mut PgpuScene,
) -> *mut PgpuSceneBuilder<'static> {
    Box::into_raw(Box::new((*scene).builder()))
}

/// Creates a new builder for filling a piet-gpu scene fragment. The specified
/// scene fragment should not be accessed while the builder is live.
#[no_mangle]
pub unsafe extern "C" fn pgpu_scene_builder_for_fragment(
    fragment: *mut PgpuSceneFragment,
) -> *mut PgpuSceneBuilder<'static> {
    Box::into_raw(Box::new((*fragment).builder()))
}

/// Adds a glyph with the specified transform to the underlying scene.
#[no_mangle]
pub unsafe extern "C" fn pgpu_scene_builder_add_glyph(
    builder: *mut PgpuSceneBuilder<'static>,
    glyph: *const PgpuGlyph,
    transform: *const PgpuTransform,
) {
    (*builder).add_glyph(&*glyph, &(*transform).into());
}

impl Iterator for PgpuPathIter {
    type Item = piet_scene::path::Element;

    fn next(&mut self) -> Option<Self::Item> {
        let mut el = PgpuPathElement {
            verb: PgpuPathVerb::MoveTo,
            points: [PgpuPoint::default(); 3],
        };
        if (self.next_element)(self.context, &mut el as _) {
            let p = &el.points;
            Some(match el.verb {
                PgpuPathVerb::MoveTo => Element::MoveTo((p[0].x, p[0].y).into()),
                PgpuPathVerb::LineTo => Element::LineTo((p[0].x, p[0].y).into()),
                PgpuPathVerb::QuadTo => {
                    Element::QuadTo((p[0].x, p[0].y).into(), (p[1].x, p[1].y).into())
                }
                PgpuPathVerb::CurveTo => Element::CurveTo(
                    (p[0].x, p[0].y).into(),
                    (p[1].x, p[1].y).into(),
                    (p[2].x, p[2].y).into(),
                ),
                PgpuPathVerb::Close => Element::Close,
            })
        } else {
            None
        }
    }
}

/// Sets the current absolute transform for the scene builder.
#[no_mangle]
pub unsafe extern "C" fn pgpu_scene_builder_transform(
    builder: *mut PgpuSceneBuilder<'static>,
    transform: *const PgpuTransform,
) {
    if !transform.is_null() {
        (*builder).0.transform((*transform).into())
    }
}

/// Fills a path using the specified fill style and brush. If the brush
/// parameter is nullptr, a solid color white brush will be used. The
/// brush_transform may be nullptr.
#[no_mangle]
pub unsafe extern "C" fn pgpu_scene_builder_fill_path(
    builder: *mut PgpuSceneBuilder<'static>,
    fill: PgpuFill,
    brush: *const PgpuBrush,
    brush_transform: *const PgpuTransform,
    path: *mut PgpuPathIter,
) {
    let fill = match fill {
        PgpuFill::NonZero => Fill::NonZero,
        PgpuFill::EvenOdd => Fill::EvenOdd,
    };
    let brush = if brush.is_null() {
        Brush::Solid(Color::rgb8(255, 255, 255))
    } else {
        match (*brush).kind {
            PgpuBrushKind::Solid => {
                let color = &(*brush).data.solid;
                Brush::Solid(Color::rgba8(color.r, color.g, color.b, color.a))
            }
        }
    };
    let brush_transform = if brush_transform.is_null() {
        None
    } else {
        Some((*brush_transform).into())
    };
    (*builder)
        .0
        .fill(fill, &brush, brush_transform, (*path).clone());
}

/// Appends a scene fragment to the underlying scene or fragment. The
/// transform parameter represents an absolute transform to apply to
/// the fragment. If it is nullptr, the fragment will be appended to
/// the scene with an assumed identity transform regardless of the
/// current transform state.
#[no_mangle]
pub unsafe extern "C" fn pgpu_scene_builder_append_fragment(
    builder: *mut PgpuSceneBuilder<'static>,
    fragment: *const PgpuSceneFragment,
    transform: *const PgpuTransform,
) {
    let transform = if transform.is_null() {
        None
    } else {
        Some((*transform).into())
    };
    (*builder).0.append(&(*fragment).0, transform);
}

/// Finalizes the scene builder, making the underlying scene ready for
/// rendering. This takes ownership and consumes the builder.
#[no_mangle]
pub unsafe extern "C" fn pgpu_scene_builder_finish(builder: *mut PgpuSceneBuilder<'static>) {
    let builder = Box::from_raw(builder);
    builder.finish();
}

/// Creates a new context for loading glyph outlines.
#[no_mangle]
pub unsafe extern "C" fn pgpu_glyph_context_new() -> *mut PgpuGlyphContext {
    Box::into_raw(Box::new(PgpuGlyphContext::new()))
}

/// Destroys the glyph context.
#[no_mangle]
pub unsafe extern "C" fn pgpu_glyph_context_destroy(gcx: *mut PgpuGlyphContext) {
    Box::from_raw(gcx);
}

/// Description of a font.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct PgpuFontDesc {
    /// Pointer to the context of the font file.
    data: *const u8,
    /// Size of the font file data in bytes.
    data_len: usize,
    /// Index of the requested font in the font file.
    index: u32,
    /// Unique identifier for the font.
    unique_id: u64,
    /// Requested size in pixels per em unit. Set to 0.0 for
    /// unscaled outlines.
    ppem: f32,
    /// Pointer to array of font variation settings.
    variations: *const PgpuFontVariation,
    /// Number of font variation settings.
    variations_len: usize,
}

/// Creates a new glyph provider for the specified glyph context and font
/// descriptor. May return nullptr if the font data is invalid. Only one glyph
/// provider may be live for a glyph context.
#[no_mangle]
pub unsafe extern "C" fn pgpu_glyph_provider_new(
    gcx: *mut PgpuGlyphContext,
    font: *const PgpuFontDesc,
) -> *mut PgpuGlyphProvider<'static> {
    let font = &*font;
    let font_data = std::slice::from_raw_parts(font.data, font.data_len);
    let variations = std::slice::from_raw_parts(font.variations, font.variations_len);
    if let Some(provider) = (*gcx).new_provider(
        font_data,
        font.index,
        font.unique_id,
        font.ppem,
        false,
        variations,
    ) {
        Box::into_raw(Box::new(provider))
    } else {
        std::ptr::null_mut()
    }
}

/// Returns an encoded outline for the specified glyph provider and glyph id.
/// May return nullptr if the requested glyph is not available.
#[no_mangle]
pub unsafe extern "C" fn pgpu_glyph_provider_get(
    provider: *mut PgpuGlyphProvider,
    gid: u16,
) -> *mut PgpuGlyph {
    if let Some(glyph) = (*provider).get(gid) {
        Box::into_raw(Box::new(glyph))
    } else {
        std::ptr::null_mut()
    }
}

/// Returns an encoded color outline for the specified glyph provider, color
/// palette index and glyph id. May return nullptr if the requested glyph is
/// not available.
#[no_mangle]
pub unsafe extern "C" fn pgpu_glyph_provider_get_color(
    provider: *mut PgpuGlyphProvider,
    palette_index: u16,
    gid: u16,
) -> *mut PgpuGlyph {
    if let Some(glyph) = (*provider).get_color(palette_index, gid) {
        Box::into_raw(Box::new(glyph))
    } else {
        std::ptr::null_mut()
    }
}

/// Destroys the glyph provider.
#[no_mangle]
pub unsafe extern "C" fn pgpu_glyph_provider_destroy(provider: *mut PgpuGlyphProvider) {
    Box::from_raw(provider);
}

/// Computes the bounding box for the glyph after applying the specified
/// transform.
#[no_mangle]
pub unsafe extern "C" fn pgpu_glyph_bbox(
    glyph: *const PgpuGlyph,
    transform: &[f32; 6],
) -> PgpuRect {
    let transform = piet_scene::geometry::Affine::new(transform);
    let rect = (*glyph).bbox(Some(transform));
    PgpuRect {
        x0: rect.min.x,
        y0: rect.min.y,
        x1: rect.max.x,
        y1: rect.max.y,
    }
}

/// Destroys the glyph.
#[no_mangle]
pub unsafe extern "C" fn pgpu_glyph_destroy(glyph: *mut PgpuGlyph) {
    Box::from_raw(glyph);
}
