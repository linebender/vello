// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! New scheduler implementation for `vello_hybrid`.
//!
//! This first slice supports root-level draws and regular property-less layers.

mod allocate;
mod builder;
mod draw;
mod round;
mod timeline;

use self::builder::ScheduleBuilder;
pub(crate) use self::round::BlendOp;
pub(crate) use self::round::FilterOp;
use self::round::{Rounds, Schedule};
use crate::{GpuStrip, RenderError, Scene};
use alloc::vec::Vec;
use vello_common::TextureId;
use vello_common::encode::EncodedPaint;
use vello_common::geometry::RectU16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RootRenderTarget {
    /// The root render target is the user-provided surface.
    UserSurface,
    /// The root render target is an atlas layer.
    AtlasLayer,
}

/// Specifies the target for a strip render pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StripPassRenderTarget {
    /// Render to the root output target.
    Root(RootRenderTarget),
    /// Render to a whole layer atlas texture.
    LayerAtlas(usize),
}

/// Identifies one of the intermediate textures used by the hybrid renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextureTarget {
    /// First layer atlas texture.
    Layer0,
    /// Second layer atlas texture.
    Layer1,
    /// First scratch texture.
    Scratch0,
    /// Second scratch texture.
    Scratch1,
}

impl TextureTarget {
    pub(crate) fn layer(index: usize) -> Self {
        match index {
            0 => Self::Layer0,
            1 => Self::Layer1,
            _ => panic!("vello_hybrid only supports two layer textures"),
        }
    }

    pub(crate) fn scratch(index: usize) -> Self {
        match index {
            0 => Self::Scratch0,
            1 => Self::Scratch1,
            _ => panic!("vello_hybrid only supports two scratch textures"),
        }
    }

    pub(crate) fn index(self) -> usize {
        match self {
            Self::Layer0 | Self::Scratch0 => 0,
            Self::Layer1 | Self::Scratch1 => 1,
        }
    }
}

/// A rectangular region in one of the intermediate textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TextureRegion {
    /// Texture index, currently `0` or `1`.
    pub(crate) texture_index: usize,
    /// Region in the texture.
    pub(crate) rect: RectU16,
}

/// A layer texture region with its corresponding viewport-space bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerTextureRegion {
    /// Region in the layer texture.
    pub(crate) texture: TextureRegion,
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

pub(crate) trait RendererBackend {
    /// Ensure intermediate layer/scratch textures required by this scene are allocated.
    fn prepare_intermediate_textures(&mut self, requirements: TextureRequirements);

    /// Return the dimensions of each layer atlas texture.
    fn layer_texture_size(&self) -> (u32, u32);

    /// Clear rectangular regions in a texture to transparent black.
    fn clear_rects(&mut self, target: TextureTarget, populate: impl FnOnce(&mut Vec<RectU16>));

    /// Execute a render pass for strips, split into opaque and alpha passes.
    fn render_strips(
        &mut self,
        opaque_strips: &[GpuStrip],
        alpha_strips: &[GpuStrip],
        external_texture_runs: &[ExternalTextureRun],
        target: StripPassRenderTarget,
    );

    /// Apply non-default blend layer operations.
    fn blend(&mut self, blends: &[BlendOp], texture_index: usize);

    /// Apply filter operations to already-rendered layer atlas regions.
    fn apply_filters(&mut self, filters: &[FilterOp], texture_index: usize);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TextureRequirements {
    pub(crate) layer_textures: [bool; 2],
    pub(crate) scratch_textures: [bool; 2],
}

impl TextureRequirements {
    fn for_scene(scene: &Scene) -> Self {
        let mut layer_textures = [false; 2];
        if scene.recorder.root_is_blend_target {
            // When the root is a blend target, it becomes a synthetic direct child layer of the
            // final output root.
            layer_textures[1] = true;
        }
        // The root layer is depth 0. Direct child layers have depth 1 and need one atlas texture;
        // deeper nesting alternates between the two atlas textures. If the root itself is rendered
        // as an intermediate layer, recorded layers are effectively shifted down by one level.
        let depth_offset = usize::from(scene.recorder.root_is_blend_target);
        for depth in 1..=scene.recorder.max_layer_depth.min(2) {
            layer_textures[(depth + depth_offset) & 1] = true;
        }
        let scratch_textures = if scene.recorder.has_filter_layer {
            [true, true]
        } else if scene.recorder.has_non_default_blend {
            [true, false]
        } else {
            [false, false]
        };

        Self {
            layer_textures,
            scratch_textures,
        }
    }
}

/// Render the supported subset of a scene through the new recorder-based scheduler.
pub(crate) fn render_scene<R: RendererBackend>(
    renderer: &mut R,
    scene: &Scene,
    root_output_target: RootRenderTarget,
    paint_idxs: &[u32],
    encoded_paints: &[EncodedPaint],
) -> Result<(), RenderError> {
    renderer.prepare_intermediate_textures(TextureRequirements::for_scene(scene));
    let strip_storage = scene.strip_storage.borrow();
    let mut builder = ScheduleBuilder::new(
        scene,
        &strip_storage,
        root_output_target,
        paint_idxs,
        encoded_paints,
        renderer.layer_texture_size(),
    );
    let schedule = builder.build()?;
    execute_schedule(renderer, &schedule, root_output_target);
    Ok(())
}

fn execute_schedule<R: RendererBackend>(
    renderer: &mut R,
    schedule: &Schedule,
    root_output_target: RootRenderTarget,
) {
    if !schedule.root_opaque.is_empty() {
        renderer.render_strips(
            &schedule.root_opaque,
            &[],
            &[],
            StripPassRenderTarget::Root(RootRenderTarget::UserSurface),
        );
    }

    execute_rounds(renderer, &schedule.rounds, root_output_target);
}

fn execute_rounds<R: RendererBackend>(
    renderer: &mut R,
    rounds: &Rounds,
    root_output_target: RootRenderTarget,
) {
    for round in &rounds.rounds {
        for texture_index in [1, 0] {
            let layer_round = &round.layer_passes[texture_index];
            let draw = &layer_round.draw;
            renderer.render_strips(
                &[],
                &draw.alpha,
                &draw.external_texture_runs,
                StripPassRenderTarget::LayerAtlas(texture_index),
            );

            renderer.apply_filters(&layer_round.filters, texture_index);
        }

        execute_root_pass(renderer, &round.root_pass, root_output_target);

        for texture_index in 0..round.layer_passes.len() {
            renderer.blend(&round.layer_passes[texture_index].blends, texture_index);
        }
        clear_layer_regions(renderer, &round.layer_texture_clears);
        clear_scratch_regions(renderer, &round.scratch_texture_clears);
    }
}

fn clear_layer_regions<R: RendererBackend>(renderer: &mut R, regions: &[LayerTextureRegion]) {
    let Some(max_texture_index) = regions
        .iter()
        .map(|region| region.texture.texture_index)
        .max()
    else {
        return;
    };

    for texture_index in 0..=max_texture_index {
        renderer.clear_rects(TextureTarget::layer(texture_index), |clear_rects| {
            clear_rects.extend(regions.iter().filter_map(|region| {
                if region.texture.texture_index != texture_index {
                    return None;
                }

                (!region.texture.rect.is_empty()).then_some(region.texture.rect)
            }));
        });
    }
}

fn clear_scratch_regions<R: RendererBackend>(renderer: &mut R, regions: &[TextureRegion]) {
    let Some(max_texture_index) = regions.iter().map(|region| region.texture_index).max() else {
        return;
    };

    for texture_index in 0..=max_texture_index {
        renderer.clear_rects(TextureTarget::scratch(texture_index), |clear_rects| {
            clear_rects.extend(regions.iter().filter_map(|region| {
                if region.texture_index != texture_index {
                    return None;
                }

                (!region.rect.is_empty()).then_some(region.rect)
            }));
        });
    }
}

fn execute_root_pass<R: RendererBackend>(
    renderer: &mut R,
    pass: &round::RootPass,
    root_output_target: RootRenderTarget,
) {
    if pass.draw.is_empty() {
        return;
    }

    renderer.render_strips(
        &[],
        &pass.draw.alpha,
        &pass.draw.external_texture_runs,
        StripPassRenderTarget::Root(root_output_target),
    );
}

#[derive(Debug, Clone, Copy)]
enum RenderTarget {
    Root,
    Layer(LayerTextureRegion),
}

impl RenderTarget {
    fn texture_index(self) -> Option<usize> {
        match self {
            Self::Root => None,
            Self::Layer(region) => Some(region.texture.texture_index),
        }
    }

    fn layer_region(self) -> LayerTextureRegion {
        match self {
            Self::Layer(region) => region,
            Self::Root => panic!("root targets do not have layer regions"),
        }
    }

    fn geometry_offset(self) -> (i32, i32) {
        match self {
            Self::Root => (0, 0),
            Self::Layer(region) => (
                region.texture.rect.x0 as i32 - i32::from(region.scene_bbox.x0),
                region.texture.rect.y0 as i32 - i32::from(region.scene_bbox.y0),
            ),
        }
    }
}
