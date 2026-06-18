// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! New scheduler implementation for `vello_hybrid`.
//!
//! This first slice supports root-level draws and regular property-less layers.

mod builder;
mod round;

use self::builder::Scheduler;
use self::round::Schedule;
use crate::{GpuStrip, RenderError, Scene};
use alloc::vec::Vec;
use vello_common::TextureId;
use vello_common::encode::EncodedPaint;
use vello_common::geometry::RectU16;
use vello_common::render_graph::LayerId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RootRenderTarget {
    /// The root render target is the user-provided surface.
    UserSurface,
    /// The root render target is an atlas layer.
    AtlasLayer,
}

/// Specifies the target for a strip render pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum StripPassRenderTarget {
    /// Render to the root output target.
    Root(RootRenderTarget),
    /// Render to an allocated region in one of the layer atlas textures.
    Layer(LayerTextureRegion),
    /// Render to a layer in the filter atlas.
    FilterLayer(LayerId),
    /// Render to one of the slot textures used for clipping/blending.
    SlotTexture(u8),
}

/// A rectangular region in one of the layer atlas textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerTextureRegion {
    /// Layer texture index, currently `0` or `1`.
    pub(crate) texture_index: usize,
    /// X coordinate in the layer texture.
    pub(crate) x: u32,
    /// Y coordinate in the layer texture.
    pub(crate) y: u32,
    /// Width of the region.
    pub(crate) width: u32,
    /// Height of the region.
    pub(crate) height: u32,
    /// Bounds of this layer in viewport coordinates.
    pub(crate) scene_bbox: RectU16,
}

/// Specifies a run of strips inside a draw that can be drawn with the same external texture
/// binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExternalTextureRun {
    pub(crate) texture_id: TextureId,
    /// Start index of the strip range for this run. The end is implicitly the start of the next
    /// run, or, for the last run, the total number of strips.
    pub(crate) strips_start: usize,
}

#[allow(dead_code)]
pub(crate) trait RendererBackend {
    /// Return the dimensions of each layer atlas texture.
    fn layer_texture_size(&self) -> (u32, u32);

    /// Clear rectangular regions in layer atlas textures to transparent black.
    fn clear_layer_regions(&mut self, regions: &[LayerTextureRegion]);

    /// Clear specific slots in a texture.
    fn clear_slots(&mut self, texture_index: usize, slots: &[u32]);

    /// Execute a render pass for strips, split into opaque and alpha passes.
    fn render_strips(
        &mut self,
        opaque_strips: &[GpuStrip],
        alpha_strips: &[GpuStrip],
        external_texture_runs: &[ExternalTextureRun],
        target: StripPassRenderTarget,
        load_op: LoadOp,
    );

    /// Apply filter effects for the given layer after its content has been rendered.
    fn apply_filter(&mut self, layer_id: LayerId);
}

/// Backend agnostic enum that specifies the operation to perform to the output attachment at the
/// start of a render pass.
#[derive(Debug, PartialEq)]
#[allow(dead_code)]
pub(crate) enum LoadOp {
    Load,
    Clear,
}

/// Render the supported subset of a scene through the new recorder-based scheduler.
pub(crate) fn render_scene<R: RendererBackend>(
    renderer: &mut R,
    scene: &Scene,
    root_output_target: RootRenderTarget,
    paint_idxs: &[u32],
    encoded_paints: &[EncodedPaint],
) -> Result<(), RenderError> {
    let strip_storage = scene.strip_storage.borrow();
    let mut builder = Scheduler::new(
        scene,
        &strip_storage,
        root_output_target,
        paint_idxs,
        encoded_paints,
        renderer.layer_texture_size(),
    );
    let schedule = builder.build()?;
    execute_schedule(renderer, &schedule);
    Ok(())
}

fn execute_schedule<R: RendererBackend>(renderer: &mut R, schedule: &Schedule) {
    for round in &schedule.rounds {
        for pass in &round.passes {
            renderer.render_strips(
                &pass.draw.opaque,
                &pass.draw.alpha,
                &pass.draw.external_texture_runs,
                pass.target.strip_pass_target(),
                LoadOp::Load,
            );
        }
        renderer.clear_layer_regions(&round.clear_layer_regions);
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum RenderTarget {
    Root(RootRenderTarget),
    Layer(LayerTextureRegion),
}

impl RenderTarget {
    fn strip_pass_target(self) -> StripPassRenderTarget {
        match self {
            Self::Root(root) => StripPassRenderTarget::Root(root),
            Self::Layer(layer) => StripPassRenderTarget::Layer(layer),
        }
    }

    fn allows_opaque_pass(self) -> bool {
        // TODO: Allow opaque strips for intermediate targets?
        matches!(self, Self::Root(RootRenderTarget::UserSurface))
    }

    fn geometry_offset(self) -> (i32, i32) {
        match self {
            Self::Root(_) => (0, 0),
            Self::Layer(region) => (
                region.x as i32 - i32::from(region.scene_bbox.x0),
                region.y as i32 - i32::from(region.scene_bbox.y0),
            ),
        }
    }
}

#[derive(Debug, Default)]
struct Draw {
    opaque: Vec<GpuStrip>,
    alpha: Vec<GpuStrip>,
    external_texture_runs: Vec<ExternalTextureRun>,
}

impl Draw {
    #[inline(always)]
    fn push_opaque(&mut self, gpu_strip: GpuStrip) {
        self.opaque.push(gpu_strip);
    }

    #[inline(always)]
    fn push_alpha(&mut self, gpu_strip: GpuStrip, external_texture_id: Option<TextureId>) {
        if let Some(texture_id) = external_texture_id {
            let needs_new_run = self
                .external_texture_runs
                .last()
                .is_none_or(|run| run.texture_id != texture_id);
            if needs_new_run {
                let strips_start = if self.external_texture_runs.is_empty() {
                    0
                } else {
                    self.alpha.len()
                };
                self.external_texture_runs.push(ExternalTextureRun {
                    strips_start,
                    texture_id,
                });
            }
        }

        self.alpha.push(gpu_strip);
    }

    fn is_empty(&self) -> bool {
        self.opaque.is_empty() && self.alpha.is_empty()
    }
}
