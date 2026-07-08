// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Builds and executes dependency-ordered rendering rounds for `vello_hybrid`.

mod allocate;
pub(crate) mod buffer;
mod builder;
mod cursor;
mod draw;
mod pool;
pub(crate) mod round;

use self::buffer::{Ranges, ScheduleBuffers};
use self::builder::ScheduleBuilder;
use self::pool::Pools;
use self::round::{FilterPasses, Rounds};
use crate::filter::{FilterContext, FilterPlanScratch};
use crate::paint::PaintResolver;
use crate::schedule::draw::{Draw, OpaqueStrips};
use crate::{GpuStrip, RenderError, Scene};
use alloc::vec::Vec;
use vello_common::TextureId;
use vello_common::encode::EncodedPaint;
use vello_common::geometry::RectU16;
use vello_common::util::RectExt;

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

impl LayerTextureRegion {
    fn empty_for_blend(bbox: RectU16) -> Self {
        Self {
            texture: TextureRegion {
                texture_index: 0,
                rect: RectU16::ZERO,
            },
            scene_bbox: RectU16::new(bbox.x0, bbox.y0, bbox.x0, bbox.y0),
        }
    }

    fn blend_scratch_clear_rect(self, blend_bbox: RectU16) -> RectU16 {
        let x0 = self.texture.rect.x0 + (blend_bbox.x0 - self.scene_bbox.x0);
        let y0 = self.texture.rect.y0 + (blend_bbox.y0 - self.scene_bbox.y0);
        RectU16::new(x0, y0, x0 + blend_bbox.width(), y0 + blend_bbox.height())
    }
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

#[derive(Debug, Default)]
struct Schedule {
    opaque_strips: OpaqueStrips,
    rounds: Rounds,
}

/// Persistent buffers and allocation pools used to build schedules across frames.
#[derive(Debug, Default)]
pub(crate) struct ScheduleStorage {
    pools: Pools,
    pub(crate) buffers: ScheduleBuffers,
    filter_plan_scratch: FilterPlanScratch,
}

pub(crate) trait RendererBackend {
    /// Return the persistent storage used while scheduling and executing a scene.
    fn schedule_storage(&mut self) -> &mut ScheduleStorage;

    /// Ensure intermediate layer/scratch textures required by this scene are allocated.
    fn prepare_intermediate_textures(&mut self, requirements: TextureRequirements);

    /// Return the dimensions of each layer atlas texture.
    fn layer_texture_size(&self) -> (u32, u32);

    /// Clear rectangular regions in a texture to transparent black.
    fn clear_rects(&mut self, target: TextureTarget, populate: impl FnOnce(&mut Vec<RectU16>));

    /// Render the global opaque strips to the user-provided root surface.
    fn render_root_opaque(&mut self, strips: &[GpuStrip]);

    /// Render ranged alpha strips to a root or layer target.
    fn render_draw(
        &mut self,
        strips: &Ranges,
        external_texture_runs: &[ExternalTextureRun],
        target: StripPassRenderTarget,
    );

    /// Apply non-default blend layer operations.
    fn blend(&mut self, blends: &Ranges, texture_index: usize);

    /// Apply filter operations to already-rendered layer atlas regions.
    fn apply_filters(&mut self, passes: &FilterPasses, texture_index: usize);
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
    filter_context: &FilterContext,
) -> Result<(), RenderError> {
    renderer.prepare_intermediate_textures(TextureRequirements::for_scene(scene));
    let strip_storage = scene.strip_storage.borrow();
    let paint_resolver = PaintResolver::new(encoded_paints, paint_idxs);
    let layer_texture_size = renderer.layer_texture_size();
    let schedule = {
        let ScheduleStorage {
            pools,
            buffers,
            filter_plan_scratch,
        } = renderer.schedule_storage();
        buffers.clear();
        let mut builder = ScheduleBuilder::new(
            scene,
            &strip_storage,
            root_output_target,
            paint_resolver,
            filter_context,
            layer_texture_size,
            pools,
            buffers,
            filter_plan_scratch,
        );
        builder.build()?
    };
    schedule.execute(renderer, root_output_target);
    let ScheduleStorage { pools, buffers, .. } = renderer.schedule_storage();
    schedule.recycle(pools);
    buffers.clear();
    Ok(())
}

impl Schedule {
    fn execute<R: RendererBackend>(&self, renderer: &mut R, root_output_target: RootRenderTarget) {
        if let Some(strips) = &self.opaque_strips {
            renderer.render_root_opaque(strips);
        }

        self.rounds.execute(renderer, root_output_target);
    }

    fn recycle(self, pools: &mut Pools) {
        pools.submit_opaque_strips(self.opaque_strips);
        self.rounds.recycle(pools);
    }
}

impl Rounds {
    fn execute<R: RendererBackend>(&self, renderer: &mut R, root_output_target: RootRenderTarget) {
        for round in &self.rounds {
            for texture_index in [1, 0] {
                let layer_round = &round.layer_passes[texture_index];
                let draw = &layer_round.draw;
                renderer.render_draw(
                    &draw.strip_ranges,
                    &draw.external_texture_runs,
                    StripPassRenderTarget::LayerAtlas(texture_index),
                );

                renderer.apply_filters(&layer_round.filter_passes, texture_index);
            }

            Self::execute_root_pass(renderer, &round.root_draw, root_output_target);

            for texture_index in 0..round.layer_passes.len() {
                renderer.blend(
                    &round.layer_passes[texture_index].blend_ranges,
                    texture_index,
                );
            }
            Self::clear_regions(renderer, &round.layer_texture_clears, TextureTarget::layer);
            Self::clear_regions(
                renderer,
                &round.scratch_texture_clears,
                TextureTarget::scratch,
            );
        }
    }

    fn clear_regions<R: RendererBackend>(
        renderer: &mut R,
        regions: &[Vec<RectU16>; 2],
        target: impl Fn(usize) -> TextureTarget,
    ) {
        for (texture_index, regions) in regions.iter().enumerate() {
            renderer.clear_rects(target(texture_index), |clear_rects| {
                clear_rects.extend(regions.iter().copied().filter(|region| !region.is_empty()));
            });
        }
    }

    fn execute_root_pass<R: RendererBackend>(
        renderer: &mut R,
        draw: &Draw,
        root_output_target: RootRenderTarget,
    ) {
        if draw.is_empty() {
            return;
        }

        renderer.render_draw(
            &draw.strip_ranges,
            &draw.external_texture_runs,
            StripPassRenderTarget::Root(root_output_target),
        );
    }
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

    fn draw_bounds(self, scene: &Scene) -> RectU16 {
        match self {
            Self::Root => RectU16::new(0, 0, scene.width, scene.height).snap_to_tile_coordinates(),
            Self::Layer(region) => region.scene_bbox,
        }
    }

    fn required_round_for_layer_sample(
        self,
        child_texture_index: usize,
        child_round: usize,
    ) -> usize {
        match self.texture_index() {
            Some(parent_texture_index)
                if parent_texture_index != child_texture_index
                    && Self::layer_texture_order(child_texture_index)
                        < Self::layer_texture_order(parent_texture_index) =>
            {
                child_round
            }
            Some(_) => child_round + 1,
            None => child_round,
        }
    }

    fn layer_texture_order(texture_index: usize) -> usize {
        match texture_index {
            1 => 0,
            0 => 1,
            _ => texture_index,
        }
    }
}
