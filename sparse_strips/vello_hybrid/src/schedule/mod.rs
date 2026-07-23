// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The core scheduler.
//!
//! After recording the render commands provided by the user, we have a DAG-like representation
//! of the scene, with individual nodes that represent a contiguous sequence of batched draws,
//! followed by an optional layer composition. The task of the scheduler is to consume this
//! representation and fixed intermediate texture page dimensions, schedule draw and layer
//! operations in such a way that they can be executed in render passes by the GPU, to achieve the
//! final intended visual result.
//!
//! There are many different ways of finding such a schedule, each with different advantages and
//! disadvantages. Vello Hybrid's scheduling algorithm has the following core properties:
//!
//! - It always finds a valid schedule for any scene whose individual layers fit within the given
//!   texture page dimensions.
//! - It always chooses a schedule that tries to minimize the number of texture allocations, even
//!   if that means increasing the number of render passes and thus sacrificing performance.
//! - If the _already_ allocated resources are abundant enough, it employs batching to reduce
//!   the number of render passes. However, the schedule is not necessarily the globally most
//!   optimal one in terms of render passes.
//!
//! The above is achieved by following the below three principles.
//!
//! ## Bottom-up layer scheduling
//!
//! The scheduler descends into a child layer before scheduling the part of its parent that
//! composes it. Layer depth determines which intermediate texture group is used: even-depth layers
//! use the even group and odd-depth layers use the odd group. Recorded layer depths start at one,
//! so a simple chain labeled from `L0` can render by ping-ponging between one page from each group:
//!
//! ```text
//! Root surface (not an intermediate layer)
//! └── L0 — depth 1, odd page  — rendered 5th
//!     └── L1 — depth 2, even page — rendered 4th
//!         └── L2 — depth 3, odd page  — rendered 3rd
//!             └── L3 — depth 4, even page — rendered 2nd
//!                 └── L4 — depth 5, odd page  — rendered 1st
//! ```
//!
//! `L4` is rendered first into the odd page. `L3` is then rendered into the even page and composes
//! `L4`, after which the `L4` allocation can be released. `L2` reuses the odd page and composes
//! `L3`; after `L3` is released, `L1` reuses the even page and composes `L2`; and finally `L0`
//! reuses the odd page and composes `L1`. This pattern applies arbitrarily deep chains, with
//! at most two adjacent layer allocations live at once and therefore ensuring minimal memory
//! usage.
//!
//! ## Lazy layer allocation
//!
//! Bottom-up traversal only provides this memory behavior because entering a layer does not
//! immediately allocate its target. The target is allocated lazily when the layer first has draws
//! or a completed child to receive. In the chain above, `L0` through `L3` do not occupy atlas space
//! while the scheduler descends to `L4`. If their targets were allocated eagerly, the outer layers
//! would occupy the even and odd pages before the inner layers could reuse them, defeating the
//! two-page ping-pong scheme.
//!
//! Branching can require more than two live allocations. Consider a parent with two children,
//! where the second child itself has a child:
//!
//! ```text
//! Root surface (not an intermediate layer)
//! └── L0 — depth 1, odd page  — rendered 4th
//!     ├── L1 — depth 2, even page — rendered 1st
//!     └── L2 — depth 2, even page — rendered 3rd
//!         └── L3 — depth 3, odd page  — rendered 2nd
//! ```
//!
//! After `L1` is composed, `L0` has an odd allocation containing that result. This allocation must
//! remain live while `L2` is scheduled. Descending into `L2` then reaches `L3`, which also needs an
//! odd allocation. `L0` and `L3` may share an odd atlas page if both regions fit; otherwise `L3`
//! requires a second odd page. Together with the even page used by `L2`, this means the branch can
//! require three intermediate textures. Lazy allocation minimizes the overlap, but cannot remove
//! allocations whose contents are simultaneously live. However, it should become apparent that,
//! even for complexly nested layer graphs, the number of allocations kept alive at the same time
//! is basically as small as possible and never exceeds the maximum layer depth across the whole
//! scene graph.
//!
//! ## Batching into rounds
//!
//! Memory minimization does not imply one layer per render pass. Intermediate textures are atlases,
//! so independent layers can occupy different regions of the same page. The scheduler places work
//! at the earliest round and stage allowed by its dependencies. Operations can batch into the same
//! round when they require the same even and odd pages and their stages are compatible; otherwise
//! scheduling advances until the dependency is satisfied or the required page pair can be bound.
//! The scheduler does **not** allocate additional pages merely to improve batching, ensuring that
//! the memory minimization property is upheld.
//!
//! This process is monotonic and conservative: given that our cursor is currently at some base
//! round, we **never** attempt to backtrack to find a spot for a layer allocation in a previous
//! round, in an attempt to minimize the number of render passes. This is accepted as a downside of
//! the current algorithm in the interest of keeping the already non-trivial logic as simple as
//! possible. The concrete contents and texture bindings of a round are described in the [`round`]
//! module.
//!
//! For example, consider a single 30x30 texture page. We first allocate a 5x5 region in round 0,
//! followed by a 30x30 region that must wait until the first allocation is released and therefore
//! advances the cursor to round 1. If we then request another 5x5 region, the scheduler does not
//! reconsider round 0, even if that region could have fit beside the first 5x5 allocation.
//!
//! ## Filter and blend layers
//!
//! A filter layer follows the same bottom-up scheduling and lazy allocation process as a regular
//! layer. Its allocation covers the filter's expanded bounds. Once the layer contents have been
//! rendered, the scheduler allocates an equally sized temporary region from the opposite texture
//! group. Filter passes alternate their source and destination between these two regions, and the
//! pass plan ensures that the final result is left in the layer's allocation. The temporary region
//! is then cleared and released. Filters that must preserve the unfiltered pixels, such as drop
//! shadows, additionally use the shared scratch texture. The layer is not made available to its
//! parent until filtering is complete.
//!
//! Default source-over composition does not require a separate blend pass. Instead, the parent
//! draw samples the completed child layer directly. Non-default blending must read both the child
//! and the existing parent contents, so it waits until both are ready and is placed in the next
//! compatible blend stage. The round must bind the pages containing both layers. The backend writes
//! the blended result to the shared scratch texture and copies it back into the parent, after which
//! the child can be cleared and released. When a non-default blend targets the root, the root is
//! first rendered into an intermediate layer because the user-provided target cannot be sampled
//! directly. The remaining process is the same as for any other non-default blend.

mod allocate;
mod cursor;
pub(crate) mod execute;
pub(crate) mod round;
#[cfg(test)]
mod schedule_tests;
#[cfg(test)]
mod test_support;

use self::allocate::{Atlases, LayerAllocationRequest};
use self::cursor::Cursor;
pub(crate) use self::execute::{Backend, execute};
use self::round::{
    BlendOp, FilterOp, FilterTextureRegions, Round, RoundStage, Rounds, SchedulePoint,
};
use crate::draw::{Draw, DrawBuffers, DrawBuilder, DrawState, RectU16Ext};
use crate::filter::{FilterContext, FilterPassPlan, PreparedGpuFilter};
use crate::paint::PaintResolver;
use crate::scene::RecordedDraw;
use crate::schedule::allocate::AllocatedTextureRegion;
use crate::target::{
    DrawTarget, LayerTextureRegion, RootTarget, RoundBindings, TextureParity, TextureRegion,
};
use crate::{RenderError, Scene, blend::BlendStrip};
use alloc::vec::Vec;
use vello_common::filter::FilterLayerPlacement;
use vello_common::geometry::{RectU16, SizeU16};
use vello_common::multi_atlas::AtlasError;
use vello_common::peniko::BlendMode;
use vello_common::record::{CommandRecorder, LayerProps, Node, RecordedLayer, RecordedLayerKind};
use vello_common::strip::visit_strip_fill_segments;
use vello_common::strip_generator::StripStorage;
use vello_common::tile::Tile;

const REGULAR_LAYER_KIND: RecordedLayerKind = RecordedLayerKind::Regular;

/// Dependency-ordered rendering rounds and their intermediate texture requirements.
#[derive(Debug)]
pub(crate) struct Schedule {
    /// Rendering rounds in execution order.
    rounds: Rounds,
    /// Intermediate textures required to execute this schedule.
    intermediate_textures: IntermediateTextureRequirements,
}

impl Schedule {
    pub(crate) fn try_new(
        storage: &mut ScheduleStorage,
        scene: &Scene,
        root_output_target: RootTarget,
        paint_resolver: PaintResolver<'_>,
        texture_size: SizeU16,
        backend_allocations: IntermediateTextureAllocations,
        max_textures: Option<usize>,
    ) -> Result<Self, RenderError> {
        storage.clear();

        let strip_storage = scene.strip_storage.borrow();
        let scene_bbox = RectU16::new(
            0,
            0,
            // Scene size is already snapped to tile coordinates.
            scene.recorder.scene_size.width(),
            scene.recorder.scene_size.height(),
        );

        let scheduler = Scheduler::new(
            &scene.recorder,
            scene_bbox,
            &strip_storage,
            root_output_target,
            paint_resolver,
            texture_size,
            storage,
        );

        let schedule = scheduler.build()?;

        // Only return the schedule if it doesn't allocate more than allowed by the user.
        schedule
            .intermediate_textures
            .validate(backend_allocations, max_textures)?;

        Ok(schedule)
    }

    pub(crate) fn intermediate_texture_requirements(&self) -> IntermediateTextureRequirements {
        self.intermediate_textures
    }
}

/// Intermediate textures required to execute a [`Schedule`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IntermediateTextureRequirements {
    /// Dimensions shared by every intermediate texture.
    pub(crate) size: SizeU16,
    /// Intermediate texture allocations required by the schedule.
    pub(crate) allocations: IntermediateTextureAllocations,
}

/// Counts of allocated or required intermediate textures.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IntermediateTextureAllocations {
    /// Number of layer texture pages included for each parity.
    pub(crate) layer_pages: [usize; 2],
    /// Whether the shared scratch texture is included.
    pub(crate) scratch: bool,
}

impl IntermediateTextureAllocations {
    /// Combine the current set of allocations with another one.
    pub(crate) fn combine(self, existing: Self) -> Self {
        Self {
            layer_pages: core::array::from_fn(|index| {
                self.layer_pages[index].max(existing.layer_pages[index])
            }),
            scratch: self.scratch || existing.scratch,
        }
    }

    /// Return the total number of physical textures represented by these allocations.
    fn texture_count(self) -> usize {
        self.layer_pages.into_iter().sum::<usize>() + usize::from(self.scratch)
    }
}

impl IntermediateTextureRequirements {
    /// Validate that satisfying these requirements from the backend's existing allocations stays
    /// within the configured limit.
    pub(crate) fn validate(
        self,
        existing: IntermediateTextureAllocations,
        max_textures: Option<usize>,
    ) -> Result<(), AtlasError> {
        let retained_textures = self.allocations.combine(existing).texture_count();

        if max_textures.is_some_and(|limit| retained_textures > limit) {
            return Err(AtlasError::NoSpaceAvailable);
        }

        Ok(())
    }
}

/// Plans concrete, executable rounds from a recorded scene.
#[derive(Debug)]
struct Scheduler<'a, 'p> {
    /// Recorded scene graph and draws being scheduled.
    recorder: &'a CommandRecorder<RecordedDraw>,
    /// Bounds of the root target in scene coordinates.
    scene_bbox: RectU16,
    /// Strip data referenced by recorded path draws and clips.
    strip_storage: &'a StripStorage,
    /// Destination used for root-level draws.
    root_render_target: RootTarget,
    /// Resolves recorded paints to their GPU representation.
    paint_resolver: PaintResolver<'a>,
    /// Allocation cursor defining the earliest round new work can use.
    cursor: Cursor,
    /// Dimensions shared by every intermediate texture.
    texture_size: SizeU16,
    /// Reusable buffers populated while constructing the schedule.
    storage: &'p mut ScheduleStorage,
}

impl<'a, 'p> Scheduler<'a, 'p> {
    /// Create a scheduler.
    fn new(
        recorder: &'a CommandRecorder<RecordedDraw>,
        scene_bbox: RectU16,
        strip_storage: &'a StripStorage,
        root_render_target: RootTarget,
        paint_resolver: PaintResolver<'a>,
        texture_size: SizeU16,
        storage: &'p mut ScheduleStorage,
    ) -> Self {
        Self {
            recorder,
            scene_bbox,
            strip_storage,
            root_render_target,
            paint_resolver,
            cursor: Cursor::new(Atlases::new(texture_size)),
            texture_size,
            storage,
        }
    }

    /// Build the complete dependency-ordered schedule.
    fn build(mut self) -> Result<Schedule, RenderError> {
        let mut rounds = Rounds::default();
        self.schedule_root(&mut rounds)?;

        // Since the strips should be rendered front-to-back.
        self.storage.buffers.draw_buffers.opaque_strips.reverse();

        #[cfg(any(test, debug_assertions))]
        rounds.validate(&self.storage.buffers);

        let intermediate_textures = IntermediateTextureRequirements {
            size: self.texture_size,
            allocations: IntermediateTextureAllocations {
                layer_pages: rounds.layer_page_counts(),
                scratch: self.cursor.scratch_texture(),
            },
        };
        Ok(Schedule {
            rounds,
            intermediate_textures,
        })
    }

    /// Schedule the root command stream into the configured output target.
    fn schedule_root(&mut self, rounds: &mut Rounds) -> Result<(), RenderError> {
        let target = self.root_render_target;

        let mut state =
            TargetScheduleState::new(target, self.cursor.current_round(), self.scene_bbox);

        if self.recorder.root_is_blend_target {
            // If the layer is a target of a non-default blending operation, we need to be able to
            // sample from it. However, this is not possible if we render directly into the
            // user-provided view. Therefore, we need to simulate a layer push, do all the rendering
            // there and then blit back into the main frame buffer.

            let opened_layer = self.open_root_layer();
            let layer = self.schedule_layer(opened_layer, rounds)?;

            // Schedule the blit back into the main framebuffer.
            let draw_point = rounds.build_draw(
                &mut state,
                &mut self.storage.buffers.draw_buffers,
                Some(&layer),
                |builder| {
                    builder.push_layer_fill(layer.sample_region, 1.0, None, self.strip_storage);
                },
            );

            // Once the blit back is done, we can release the temporary layer.
            self.clear_and_release_allocation(layer.allocation, draw_point.round, rounds);
        } else {
            for cmd in &self.recorder.nodes {
                // Remember: Each command node consists of a sequence of draws + an optional layer invocation.

                // First submit all the draws. Note that unlike for layers, it's fine to submit
                // the draws before scheduling the child node. This is because the root target is
                // _already_ allocated, so we don't need to ensure lazy allocation here.
                self.push_draws(&cmd.draws, &mut state, rounds);

                // Next, we schedule the layer node. This might trigger advances to our current
                // base round.
                let child = self.schedule_child_layer(cmd, state.draw_state.target_bbox, rounds)?;

                // Finally, we also schedule the layer sampling operation. It's guaranteed to be
                // a simple layer, since we know for sure that the root isn't a blend target.
                if let Some(child) = child {
                    self.compose_simple_layer(child.props, child.layer, &mut state, rounds);
                }
            }
        };

        Ok(())
    }

    /// Schedule an open layer bottom-up and return its completed, still-live allocation.
    fn schedule_layer(
        &mut self,
        mut layer: OpenLayer<'a>,
        rounds: &mut Rounds,
    ) -> Result<ScheduledLayer, RenderError> {
        // Overall we follow a similar flow to `schedule_root` here.

        for cmd in layer.cmds {
            // First make sure that the child node is scheduled, in case it exists. Unlike for root
            // layers, we need to make sure to do this _before_ pushing any draws.
            // TODO: Similarly to Vello CPU, flatten this to avoid stack overflows for deep layers
            let child = self.schedule_child_layer(cmd, layer.sample_placement.dest_bbox, rounds)?;

            // Keep this after `schedule_child_layer`: allocating lazily is what makes traversal
            // bottom-up with respect to memory, while still allowing compatible layers to batch.
            let target = self.ensure_layer_target(&mut layer)?;

            // Now schedule the draws.
            self.push_draws(&cmd.draws, &mut target.schedule_state, rounds);

            // And optionally the composition of the child layer node.
            if let Some(child) = child {
                self.compose_layer(child.props, child.layer, &mut target.schedule_state, rounds)?;
            }
        }

        // In case there weren't any commands, still make sure the target exists.
        self.ensure_layer_target(&mut layer)?;

        let mut target = layer.target.take().unwrap();

        let region = target.schedule_state.draw_state.target;

        if let Some(filter) = target.filter {
            let temporary = self.cursor.allocate_layer(LayerAllocationRequest::new(
                region.texture.rect,
                layer.kind,
                region.texture.target.texture_parity.opposite(),
            ))?;
            let textures = FilterTextureRegions::new(region.texture, temporary.allocation.region);

            if filter.data.needs_copy_pass() {
                self.cursor.require_scratch_texture();
            }

            // Now we determine the point at which we can perform the filter operation.
            let mut filter_point =
                // Obviously, we first need to wait until rendering of the underlying layer finished.
                target.schedule_state.ready
                // Then, we must wait until our reserved space in the atlas is available.
                .max(SchedulePoint::start(temporary.round_idx))
                // Finally, wait until we reach the filter stage.
                .after(RoundStage::filter(region.texture.target.texture_parity));

            // Now find the _actual_ schedule point that corresponds to a round that matches our
            // requirements for texture bindings.
            filter_point = rounds.resolve_binding_point(filter_point, textures.round_bindings());

            rounds.ensure_exists(filter_point.round);
            rounds.round_mut(filter_point.round).push_filter_op(
                region.texture.target.texture_parity,
                &mut self.storage.buffers,
                FilterOp {
                    textures,
                    filter_data_offset: filter.data_offset,
                    gpu_filter: filter.data,
                },
            );

            self.clear_and_release_allocation(temporary.allocation, filter_point.round, rounds);

            // Since we need to apply a filter, update the state of the current target.
            target.schedule_state.ready = filter_point;
        }

        let scheduled = ScheduledLayer {
            sample_region: layer.sample_placement.resolve_sample_region(region),
            allocation: target.allocation,
            ready: target.schedule_state.ready,
        };

        Ok(scheduled)
    }

    /// Resolve and schedule the optional child layer referenced by a command node.
    fn schedule_child_layer(
        &mut self,
        cmd: &Node,
        parent_bounds: RectU16,
        rounds: &mut Rounds,
    ) -> Result<Option<PreparedChild<'a>>, RenderError> {
        let Some(layer_id) = cmd.layer else {
            return Ok(None);
        };

        let layer = &self.recorder.layers[layer_id as usize];

        // TODO: Change recording so layers with empty bboxes are not emitted in the first place.
        let mut bbox = if layer.bbox.is_empty() {
            if layer.props.blend_mode.is_destructive() {
                // Unlike in the non-destructive case, empty *destructive* layers are
                // not a no-op, and we need to clear the whole parent region.
                parent_bounds
            } else {
                return Ok(None);
            }
        } else {
            layer.bbox
        };

        // We cannot do this for filter layers, because the filter needs to be applied to the whole
        // region _BEFORE_ applying clips. Instead, the clip is simply applied when sampling from
        // the filter layer. However, for non-filter layers, we can reduce the work to the clip
        // bbox.
        if matches!(layer.kind, RecordedLayerKind::Regular) {
            bbox = layer
                .props
                .clip_path
                .as_ref()
                .map_or(bbox, |clip| bbox.intersect(clip.bbox));
        }

        if bbox.is_empty() {
            return Ok(None);
        }

        let opened_layer = self.open_layer(layer, bbox);
        let scheduled = self.schedule_layer(opened_layer, rounds)?;

        Ok(Some(PreparedChild {
            props: &layer.props,
            layer: scheduled,
        }))
    }

    /// Create an unallocated scheduling view of a recorded layer with the given visible bounds.
    fn open_layer(&self, layer: &'a RecordedLayer, bbox: RectU16) -> OpenLayer<'a> {
        let sample_placement = match &layer.kind {
            RecordedLayerKind::Regular => LayerSamplePlacement::regular(bbox),
            RecordedLayerKind::Filter { placement, .. } => LayerSamplePlacement::filter(*placement),
        };

        OpenLayer {
            cmds: &layer.nodes,
            kind: &layer.kind,
            texture_parity: self.layer_texture_parity(layer.depth),
            bbox,
            sample_placement,
            target: None,
        }
    }

    /// Represent the root command stream as an intermediate layer that can be sampled.
    fn open_root_layer(&self) -> OpenLayer<'a> {
        OpenLayer {
            cmds: &self.recorder.nodes,
            kind: &REGULAR_LAYER_KIND,
            texture_parity: TextureParity::Odd,
            bbox: self.scene_bbox,
            sample_placement: LayerSamplePlacement::regular(self.scene_bbox),
            target: None,
        }
    }

    /// Select the texture group for a recorded layer depth.
    fn layer_texture_parity(&self, layer_depth: usize) -> TextureParity {
        TextureParity::from_parity(layer_depth + usize::from(self.recorder.root_is_blend_target))
    }

    /// Append a recorded draw range to the next compatible draw pass for the target.
    fn push_draws<T: ScheduleTarget>(
        &mut self,
        draws: &core::ops::Range<u32>,
        state: &mut TargetScheduleState<T>,
        rounds: &mut Rounds,
    ) {
        if draws.is_empty() {
            return;
        }

        rounds.build_draw(
            state,
            &mut self.storage.buffers.draw_buffers,
            // Normal draws don't depend on any child layer.
            None,
            |builder| {
                for draw in &self.recorder.draws[draws.start as usize..draws.end as usize] {
                    builder.push_draw(draw, self.strip_storage, self.paint_resolver);
                }
            },
        );
    }

    /// Compose a rendered child into a parent layer.
    fn compose_layer(
        &mut self,
        child_props: &LayerProps,
        child_layer: ScheduledLayer,
        parent_state: &mut TargetScheduleState<LayerTextureRegion>,
        rounds: &mut Rounds,
    ) -> Result<(), RenderError> {
        let blend_mode = child_props.blend_mode;
        let opacity = child_props.opacity;

        if blend_mode == BlendMode::default() {
            self.compose_simple_layer(child_props, child_layer, parent_state, rounds);

            return Ok(());
        }

        // If we don't have default src-over, we need to schedule a blend operation.

        let parent_region = parent_state.draw_state.target;
        // See the comment in `RecordedLayer::bbox`. The child layer bbox is not necessarily
        // contained within the parent layer bbox, but later code assumes that we only sample
        // pixels contained within the parent layer bbox, so constrain it to the parent region.
        let child_region = child_layer.sample_region.crop_to(parent_region.layer_bbox);

        // For non-destructive blend modes, choose the (smaller) child bbox as the
        // affected region. Otherwise, we need to choose the (larger) parent bbox, since
        // any pixel that doesn't lie in the child needs to be cleared.
        let mut blend_bbox = if blend_mode.is_destructive() {
            parent_region.layer_bbox
        } else {
            child_region.layer_bbox
        };

        // Clips associated with the child layer also reduce the affected area of the blend.
        if let Some(clip_path) = &child_props.clip_path {
            blend_bbox = blend_bbox.intersect(clip_path.bbox);
        }

        // If the affected area would be empty, nothing to do, so just release the child and
        // we are done.
        // TODO: As mentioned elsewhere, we should change the recording so that such layers aren't
        // produced in the first place.
        if blend_bbox.is_empty() {
            // This one needs a bit of special-casing compared to the other ones.
            let round =
                // Before deallocating, we need to make sure the child has been rendered
                // in the first place.
                child_layer.ready
                // However, it's possible that in the meanwhile, we've advanced to a new round.
                // We can't deallocate in the past, so we need to max it with the current round.
                .round.max(self.cursor.current_round());

            // If we choose `cursor.current_round`, there is no guarantee that the child layer texture
            // is actually currently bound. Therefore, we need to resolve the binding point anew
            // to find a compatible round.
            let bindings = RoundBindings::new(child_layer.allocation.region.target);
            let point = rounds.resolve_binding_point(SchedulePoint::start(round), bindings);

            self.clear_and_release_allocation(child_layer.allocation, point.round, rounds);

            return Ok(());
        }

        // If there is a clip path associated, blending should be limited to its area.
        let clip_strips = child_props.clip_path.as_ref().map(|clip_path| {
            let start = self.storage.buffers.blend_strips.len();
            let strips = &self.strip_storage.strips[clip_path.strip_range.clone()];
            // As mentioned in the documentation of `ScratchTexture`, in the scratch atlas we use
            // the same space as the allocation of the parent layer.
            let geometry_shift = parent_region.geometry_shift();
            let tile_bounds = blend_bbox.to_tile_bounds();

            visit_strip_fill_segments(
                strips,
                tile_bounds,
                &mut self.storage.buffers.blend_strips,
                |blend_strips, segment| {
                    blend_strips.push(BlendStrip::from_fill_segment(
                        segment.shift(geometry_shift),
                        Some(segment.alpha_idx / u32::from(Tile::HEIGHT)),
                    ));
                },
                |blend_strips, segment| {
                    blend_strips.push(BlendStrip::from_fill_segment(
                        segment.shift(geometry_shift),
                        None,
                    ));
                },
            );

            let end = self.storage.buffers.blend_strips.len();
            u32::try_from(start).unwrap()..u32::try_from(end).unwrap()
        });

        // TODO: We could add an optimization here that early exits early if clip strips
        // is empty.

        let parent_texture_parity = parent_region.texture.target.texture_parity;
        self.cursor.require_scratch_texture();

        let blend_stage = RoundStage::blend(parent_texture_parity);
        let blend_binding = RoundBindings::new(parent_region.texture.target)
            .merge(RoundBindings::new(child_region.texture.target))
            .expect("parent and child layers must have compatible texture parities");
        let mut blend_point =
            // A blend must execute after the parent is ready.
             parent_state.ready
            // The child must also be ready.
            .max(child_layer.ready)
            // And we must have reached the blend stage.
            .after(blend_stage);
        // Then, simply find the next round that matches binds the parent and child texture.
        blend_point = rounds.resolve_binding_point(blend_point, blend_binding);

        rounds.ensure_exists(blend_point.round);
        rounds.round_mut(blend_point.round).push_blend_op(
            parent_texture_parity,
            &mut self.storage.buffers,
            BlendOp {
                parent_region,
                child_region,
                blend_bbox,
                blend_mode,
                opacity,
                clip_strips,
            },
        );

        // And make sure to release the child now that it's been composited into the parent.
        self.clear_and_release_allocation(child_layer.allocation, blend_point.round, rounds);

        // Make sure the layer's ready state accounts for the blend operation.
        parent_state.ready = blend_point;

        Ok(())
    }

    /// Compose a completed child by drawing it directly into its parent using source-over.
    fn compose_simple_layer<T: ScheduleTarget>(
        &mut self,
        props: &LayerProps,
        child_layer: ScheduledLayer,
        parent_state: &mut TargetScheduleState<T>,
        rounds: &mut Rounds,
    ) {
        // Schedule the actual layer fill command.
        let draw_point = rounds.build_draw(
            parent_state,
            &mut self.storage.buffers.draw_buffers,
            Some(&child_layer),
            |builder| {
                builder.push_layer_fill(
                    child_layer.sample_region,
                    props.opacity,
                    props.clip_path.as_ref(),
                    self.strip_storage,
                );
            },
        );

        // Now that the child layer has been composited into the parent, don't forget to release
        // the child layer at the end of this round, since its rendered representation does not
        // need to be retained in the layer texture anymore!
        self.clear_and_release_allocation(child_layer.allocation, draw_point.round, rounds);

        // The parent's ready state is already updated by `build_draw`.
    }

    /// Schedule an allocation to be cleared and released after the given round.
    fn clear_and_release_allocation(
        &mut self,
        allocation: AllocatedTextureRegion,
        round_idx: usize,
        rounds: &mut Rounds,
    ) {
        let clear_region = allocation.clear_region();

        // This is a very defensive check, but a pretty important one, thus worth keeping
        // as a normal assertion instead of debug assertion. It shouldn't be triggered
        // in assuming our scheduler is correct, but if it does, it's better to crash.
        //
        // Callers just provide a schedule point, but in case there is a bug, it could happen
        // the callers supply a wrong round where a different pair of textures is actually
        // bound. In this case, in case two different atlases both have an image with ID
        // 0, we could potentially clear and deallocate the wrong one without noticing.
        // Therefore, ensure that the current round actually binds the correct texture that
        // is intended to be cleared.
        assert_eq!(
            rounds
                .round_mut(round_idx)
                .texture_binding
                .layer_id(clear_region.target.texture_parity),
            Some(clear_region.target),
            "the clear round must bind the allocation's exact texture page"
        );

        rounds.push_layer_clear(
            round_idx,
            clear_region.target.texture_parity,
            clear_region.rect,
        );

        self.cursor.release(allocation, round_idx);
    }

    /// Lazily allocate a target for an open layer and return its scheduling state.
    fn ensure_layer_target<'b>(
        &mut self,
        layer: &'b mut OpenLayer<'a>,
    ) -> Result<&'b mut LayerTarget, RenderError> {
        if layer.target.is_none() {
            let filter = match layer.kind {
                RecordedLayerKind::Filter { filter_data, .. } => {
                    Some(self.storage.filter_context.push(filter_data))
                }
                RecordedLayerKind::Regular => None,
            };

            let request = LayerAllocationRequest::new(layer.bbox, layer.kind, layer.texture_parity);
            // Note: this might advance the base round, in case the atlas is already full,
            // and we therefore need to advance the round cursor until enough space has been
            // freed.
            let allocation = self.cursor.allocate_layer(request)?;
            let round = allocation.round_idx;
            let allocation = allocation.allocation;
            let region = LayerTextureRegion {
                texture: allocation.region,
                layer_bbox: layer.bbox,
            };

            let schedule_state = TargetScheduleState::new_layer(region, round);
            layer.target = Some(LayerTarget {
                allocation,
                filter,
                schedule_state,
            });
        }

        Ok(layer.target.as_mut().unwrap())
    }
}

/// A layer that has been scheduled and can be sampled by its parent.
#[must_use = "scheduled layers must be released"]
#[derive(Debug)]
struct ScheduledLayer {
    /// Atlas allocation retained until the parent finishes sampling this layer.
    allocation: AllocatedTextureRegion,
    /// Texture region and scene-space bounds sampled by the parent.
    sample_region: LayerTextureRegion,
    /// Point after which the rendered layer contents are available.
    ready: SchedulePoint,
}

/// A recorded child node paired with its scheduled layer contents.
#[derive(Debug)]
struct PreparedChild<'a> {
    /// Composition properties recorded on the child invocation.
    props: &'a LayerProps,
    /// Scheduled contents to compose into the parent.
    layer: ScheduledLayer,
}

/// A recorded layer whose intermediate target may not have been allocated yet.
#[derive(Debug)]
struct OpenLayer<'a> {
    /// Recorded command nodes belonging to the layer.
    cmds: &'a [Node],
    /// Layer kind, including filter data when applicable.
    kind: &'a RecordedLayerKind,
    /// Texture group into which this layer must be allocated.
    texture_parity: TextureParity,
    /// Bounds that must be rendered into the layer allocation.
    bbox: RectU16,
    /// Placement used when the completed layer is sampled by its parent.
    sample_placement: LayerSamplePlacement,
    /// Lazily allocated target and its scheduling state.
    target: Option<LayerTarget>,
}

/// Maps a rendered layer allocation to the region sampled by its parent.
#[derive(Debug, Clone, Copy)]
struct LayerSamplePlacement {
    /// Offset within the rendered layer at which the region sampled by the parent begins.
    src_offset: (u16, u16),
    /// Bounds where the sampled region is placed in the parent.
    dest_bbox: RectU16,
}

impl LayerSamplePlacement {
    fn regular(bbox: RectU16) -> Self {
        Self {
            src_offset: (0, 0),
            dest_bbox: bbox,
        }
    }

    fn filter(placement: FilterLayerPlacement) -> Self {
        Self {
            src_offset: (placement.src_x, placement.src_y),
            dest_bbox: placement.dest_bbox,
        }
    }

    fn resolve_sample_region(self, allocation: LayerTextureRegion) -> LayerTextureRegion {
        let x0 = allocation.texture.rect.x0 + self.src_offset.0;
        let y0 = allocation.texture.rect.y0 + self.src_offset.1;

        LayerTextureRegion {
            texture: TextureRegion {
                target: allocation.texture.target,
                rect: RectU16::new(
                    x0,
                    y0,
                    x0 + self.dest_bbox.width(),
                    y0 + self.dest_bbox.height(),
                ),
            },
            layer_bbox: self.dest_bbox,
        }
    }
}

/// Allocated render target and state associated with an open layer.
#[derive(Debug)]
struct LayerTarget {
    /// Atlas allocation backing the layer.
    allocation: AllocatedTextureRegion,
    /// Prepared filter applied after the layer's draws, if any.
    filter: Option<PreparedGpuFilter>,
    /// Dependency state for operations targeting this layer.
    schedule_state: TargetScheduleState<LayerTextureRegion>,
}

impl Rounds {
    fn build_draw<T: ScheduleTarget>(
        &mut self,
        state: &mut TargetScheduleState<T>,
        draw_buffers: &mut DrawBuffers,
        sampled_layer: Option<&ScheduledLayer>,
        f: impl FnOnce(&mut DrawBuilder<'_, T>),
    ) -> SchedulePoint {
        let sampled_layer_round_bindings = sampled_layer
            .map_or_else(RoundBindings::default, |layer| {
                RoundBindings::new(layer.sample_region.texture.target)
            });

        let round_bindings = state
            .draw_state
            .target
            .round_bindings()
            .merge(sampled_layer_round_bindings)
            .expect("draw target and sampled layer must have compatible texture parities");

        // Determine when the draw is safe to execute.
        let mut draw_point = sampled_layer.map_or_else(
            || state.next_draw(),
            |layer| state.next_draw_after(layer.ready),
        );
        // While also ensuring the chosen round has a compatible texture binding.
        draw_point = self.resolve_binding_point(draw_point, round_bindings);
        state.ready = draw_point;
        self.ensure_exists(draw_point.round);
        let target_round = self.round_mut(draw_point.round);
        let target_draw = state.draw_state.target.draw_mut(target_round);

        let mut builder = DrawBuilder::new(target_draw, draw_buffers, &mut state.draw_state);
        f(&mut builder);

        draw_point
    }
}

/// Reusable operation data referenced by ranges in a [`Schedule`].
#[derive(Debug, Default)]
pub(crate) struct ScheduleBuffers {
    /// Strip data and per-draw range metadata.
    pub(crate) draw_buffers: DrawBuffers,
    /// Filter operations referenced by ranges in scheduled rounds.
    pub(crate) filter_ops: Vec<FilterOp>,
    /// Blend operations referenced by ranges in scheduled rounds.
    pub(crate) blend_ops: Vec<BlendOp>,
    /// Clip strips referenced by non-default blend operations.
    pub(crate) blend_strips: Vec<BlendStrip>,
}

impl ScheduleBuffers {
    fn clear(&mut self) {
        self.draw_buffers.clear();
        self.filter_ops.clear();
        self.blend_ops.clear();
        self.blend_strips.clear();
    }
}

/// Persistent buffers used to build schedules across frames.
#[derive(Debug, Default)]
pub(crate) struct ScheduleStorage {
    /// Reusable operation and strip buffers.
    pub(crate) buffers: ScheduleBuffers,
    /// GPU filter data accumulated while scheduling the scene.
    pub(crate) filter_context: FilterContext,
    /// Reusable pass plan populated immediately before filter execution.
    filter_pass_plan: FilterPassPlan,
}

impl ScheduleStorage {
    fn clear(&mut self) {
        self.buffers.clear();
        self.filter_context.clear();
    }
}

/// State for scheduling draws to a specific target.
#[derive(Debug)]
struct TargetScheduleState<T: ScheduleTarget> {
    /// The underlying draw state.
    draw_state: DrawState<T>,
    /// Point after which all currently scheduled contents of this target are available.
    ready: SchedulePoint,
}

impl<T: ScheduleTarget> TargetScheduleState<T> {
    fn new(target: T, start_round: usize, target_bbox: RectU16) -> Self {
        Self {
            draw_state: DrawState::new(target, target_bbox),
            ready: SchedulePoint::start(start_round),
        }
    }

    /// Return the earliest point at which another draw can be appended to this target.
    ///
    /// If [`Self::ready`] is already at this target's draw stage, the new draw can be batched into
    /// the same pass. Otherwise, it must use the next occurrence of the target's draw stage. The
    /// resulting point can therefore be later than [`Self::ready`]. For example, assume we have a
    /// sequence of three nested layers allocated as follows:
    ///
    /// - L0 in odd texture
    /// - L1 in even texture
    /// - L2 in odd texture
    ///
    /// In a round, we first execute even blends, so we do `Blend(L1, L2)`. L1 is now ready for L0
    /// to sample _in that same round_, so [`Self::ready`] of L1 will still have the same round
    /// index. However, if we want to append more draws to L1, we need to wait until the next round,
    /// since all draws in a round happen before blend ops. Since the next draw depends on the
    /// previous blend to have finished, it cannot happen in the same round anymore.
    fn next_draw(&self) -> SchedulePoint {
        self.ready.after_or_at(self.draw_state.target.draw_stage())
    }

    /// Return the earliest draw point that executes after an external dependency.
    fn next_draw_after(&self, dependency: SchedulePoint) -> SchedulePoint {
        dependency
            .after_or_at(self.draw_state.target.draw_stage())
            .max(self.next_draw())
    }
}

impl TargetScheduleState<LayerTextureRegion> {
    fn new_layer(target: LayerTextureRegion, base_round: usize) -> Self {
        Self::new(target, base_round, target.layer_bbox)
    }
}

trait ScheduleTarget: DrawTarget {
    fn draw_mut<'a>(&self, round: &'a mut Round) -> &'a mut Draw;
    fn draw_stage(&self) -> RoundStage;
    fn round_bindings(&self) -> RoundBindings;
}

impl ScheduleTarget for RootTarget {
    fn draw_mut<'a>(&self, round: &'a mut Round) -> &'a mut Draw {
        round.root_draw_mut()
    }

    fn draw_stage(&self) -> RoundStage {
        RoundStage::RootDraw
    }

    fn round_bindings(&self) -> RoundBindings {
        RoundBindings::default()
    }
}

impl ScheduleTarget for LayerTextureRegion {
    fn draw_mut<'a>(&self, round: &'a mut Round) -> &'a mut Draw {
        round.layer_draw_mut(self.texture.target.texture_parity)
    }

    fn draw_stage(&self) -> RoundStage {
        RoundStage::draw(self.texture.target.texture_parity)
    }

    fn round_bindings(&self) -> RoundBindings {
        RoundBindings::new(self.texture.target)
    }
}
