// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The [`Renderer`] trait, which manages resources for 2d rendering.

use core::any::Any;

use crate::{
    PaintScene, SceneOptions,
    prepared::PreparePathsDirect,
    recording::{RecordScene, TransformedRecording},
    texture::{TextureDescriptor, TextureId},
};

// TODO: Maybe?
// pub enum MaskOperation {
//     SvgLuminance,
//     CssLuminance,
//     Alpha,
// }

/// A 2d renderer, which can be used to schedule the renders of multiple 2d scenes.
///
/// Types which implement this trait are the main empty point into Vello API.
/// Once you have a value of this type, you can schedule scenes.
///
/// # Usage
///
/// - Create textures using [`create_texture`](Renderer::create_texture).
///   - Specify correct usages.
/// - If that texture is from an image, use [`upload_image`](Renderer::upload_image).
/// - Create a "scene" for a texture using [`create_scene`](Renderer::create_scene).
/// - Draw into that scene, using the methods on `PaintScene`
/// - Maybe use a recording within that, see [`create_offset_recording`](Renderer::create_offset_recording).
/// - Queue a render of that scene (to the texture) using [`queue_render`](Renderer::queue_render).
/// - Cleanup any no longer needed textures, see [`free_texture`](Renderer::free_texture).
///
/// # Storage
///
/// This API has been designed with the idea that most applications should expect to have a single
/// `Renderer` across all threads (i.e. shared between those threads).
/// For multi-threading aware applications, it is reasonable for your use of this
/// trait to be `Arc<Mutex<dyn AnyRenderer + Send>>`.
/// See the section on multi-threading thoughts for more.
/// This allows scheduling renders from any thread, whilst performing per-thread
/// painting operations without locking.
/// The order in which renders occurs will be determined by the order in which
/// [`queue_render`](Renderer::queue_render) is called on the renderer.
///
/// # External Resources
///
/// The `Renderer` API supports importing texture contents from CPU memory,
/// through [`upload_image`](Renderer::upload_image).
/// However, it does not contain an abstraction for creating textures from values already on the GPU (to avoid readbacks and re-uploads).
/// Instead, that is achieved through renderer-specific APIs such as `VelloHybrid::add_external_texture`.
/// For Vello CPU, there is not a meaningful distinction, so there is no corresponding API beyond `upload_image`.
///
/// # Rendering to surfaces
///
/// The renderer trait does not directly support rendering to an surface
/// (i.e. a window or a canvas on the web).
/// Instead, the resources created using this API are entirely headless.
/// The connection to external surfaces are managed in a renderer-specific way; for
/// Vello Hybrid, this is `VelloHybrid::add_external_texture`.
/// In Vello CPU, this would be achieved by presenting the surface using a crate like pixels or softbuffer.
/// (We have not yet validated the suitability of either of those crates.)
///
/// # Unfiltered multi-threading thoughts
///
/// TODO: Maybe we make all the methods `&self`? The cases are:
/// 1. a "true" `no_std` environment, where there is only one thread so we can use `RefCell`.
/// 2. the web, which is more complicated. But we're likely to manage with just single threaded for the foreseeable
///   - the current multi threaded dispatcher, uses mutexes, so can't run on the web.
///   - Internally, we might want to queue things through e.g. an async channel anyway...
/// 3. a normal process, where we can just use `Mutex` internally. A reminder that the critical sections are
///    expected to be short, i.e. no/very little actual rendering work happens on the main thread.
///
/// The reason to do things that way is to make `Arc<dyn AnyRenderer>` clearly first-class (even if you're internally downcasting to convert into generics?)
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
