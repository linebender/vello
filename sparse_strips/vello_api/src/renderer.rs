// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::any::Any;

use crate::{
    PaintScene, SceneOptions,
    prepared::PreparePathsDirect,
    recording::{RecordScene, TransformedRecording},
    texture::{TextureDescriptor, TextureId},
};

// This API has been designed so that it can be `Arc<Mutex>`ed.
//
// The `fill_path_cache` API could potentially stall some of those,
// but otherwise the critical sections are reasonably small.

// TODO: Maybe?
// pub enum MaskOperation {
//     SvgLuminance,
//     CssLuminance,
//     Alpha,
// }

// A 2d renderer, which can be used to schedule the renders of multiple scenes.
pub trait Renderer: Any {
    /// The `ScenePainter` is the encoder for rendering commands.
    ///
    /// *Ideally*, we'd allow this to borrow shared resources from
    /// the renderer (not exclusively).
    /// However, the lifetimes of that with `AnyRenderer` get messy fast.
    type ScenePainter: PaintScene;

    // TODO: Not complete.
    type PathPreparer: PreparePathsDirect<Self::ScenePainter>;

    type Recording: RecordScene<Self::ScenePainter>;

    type TransformedRecording: TransformedRecording<Self::ScenePainter>;

    /// Create a texture for use in renders with this device.
    fn create_texture(&mut self, descriptor: TextureDescriptor) -> TextureId;

    // Error if the texture was already freed/not associated with this renderer.
    fn free_texture(&mut self, texture: TextureId) -> Result<(), ()>;
    // TODO: Texture resizing? Reasonable reasons to not do that include cannot resize wgpu textures
    // Also what does that mean for existing content, etc.

    // fn create_mask(descriptor: MaskOperation) -> Mask;
    // fn mask_from_scene(from: &Texture, to: &Scene, MaskDescriptor { subset_rect,  });

    fn create_scene(
        &mut self,
        to: &TextureId,
        options: SceneOptions,
    ) -> Result<Self::ScenePainter, ()>;
    fn queue_render(&mut self, from: Self::ScenePainter);

    // TODO: Reason about how we want downloads to work.
    // fn queue_download(&mut self, of: &TextureId) -> DownloadId;

    // TODO: Better error kinds.
    fn upload_image(&mut self, to: &TextureId, data: &peniko::ImageData) -> Result<(), ()>;

    /// API for efficient glyph rendering.
    // Needs: Shape, Transform, Bounds, Fill or Stroke
    // To render, needs integer translation, paint information.
    // As this is specialised to glyph drawing, I think that a batched API makes sense,
    // i.e. you start a `PathSet`, add several paths to it, then free them all at once.
    // That strategy has several advantages:
    // You gain support for "batched allocations", without fragmentation.
    // There is no possible error case.
    // We need to think about how we make it practical to actually get the integer translations,
    // because of "composition".
    fn create_path_cache(&mut self) -> Self::PathPreparer;

    fn create_offset_recording(
        &mut self,
        width: u16,
        height: u16,
        origin_x_offset: i32,
        origin_y_offset: i32,
    ) -> Self::Recording;

    fn create_transformed_recording(
        &mut self,
        width: u16,
        height: u16,
        x_offset: i32,
        y_offset: i32,
    ) -> Self::TransformedRecording;
}
