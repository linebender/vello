// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
// This API has been designed so that it can be `Arc<Mutex>`ed.
//
// The `fill_path_cache` API could potentially stall some of those,
// but otherwise the critical sections are small.

use crate::{
    DownloadId, PaintScene, SceneOptions,
    prepared::{PreparePaths, PreparedPaths},
    texture::{Texture, TextureDescriptor},
};

// TODO: Maybe?
// pub enum MaskOperation {
//     SvgLuminance,
//     CssLuminance,
//     Alpha,
// }

// A 2d renderer, which can be used to schedule the renders of multiple scenes.
pub trait Renderer: Send {
    /// The `ScenePainter` is the encoder for rendering commands.
    ///
    /// *Ideally*, we'd allow this to borrow shared resources from
    /// the renderer (not exclusively).
    /// However, the lifetimes of that with `AnyRenderer` get messy fast.
    type ScenePainter: PaintScene;

    // TODO: Not complete.
    type PathPreparer: PreparePaths;

    /// Create a texture for use in renders with this device.
    ///
    /// Cleanup is handled through `Drop`.
    fn create_texture(descriptor: TextureDescriptor) -> Texture;

    // fn create_mask(descriptor: MaskOperation) -> Mask;
    // fn mask_from_scene(from: &Texture, to: &Scene, MaskDescriptor { subset_rect,  });

    fn create_scene(&mut self, to: &Texture, options: SceneOptions) -> Self::ScenePainter;
    fn queue_render(&mut self, from: Self::ScenePainter);

    fn queue_download(&mut self, texture: &Texture) -> DownloadId;

    fn upload_image(
        to: &Texture,
        data: peniko::ImageData,
        region: Option<(u16, u16, u16, u16)>,
    ) -> Result<(), ()>;

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
    ///
    /// Cleanup is handled through `Drop`
    fn create_path_cache(&mut self) -> PreparedPaths;
    // Errors if the cache isn't from this renderer.
    fn fill_path_cache<R>(
        &mut self,
        cache: &PreparedPaths,
        f: impl FnOnce(&mut Self::PathPreparer) -> R,
    ) -> Result<R, ()>;
}
