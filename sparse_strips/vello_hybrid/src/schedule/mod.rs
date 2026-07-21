// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The core scheduler.
//!
//! After recording the render commands provided by the user, we have a DAG-like representation
//! of the scene, with individual nodes that represent a contiguous sequence of batched draws,
//! followed by an optional layer composition. The task of the scheduler is to consume this
//! representation and, given the constraints on intermediate resources imposed by [`LayersConfig`],
//! schedule draw and layer operations in such a way that they can be executed by the GPU, to
//! achieve the final intended visual result.
//!
//! There are many different ways of finding such a schedule, each with different advantages and
//! disadvantages. Vello Hybrid's scheduling algorithm has the following core properties:
//!
//! - It always finds a valid schedule for any scene, assuming such a schedule exists given the
//!   resource constraints.
//! - It always chooses a schedule that minimizes the number of texture allocations, even if that
//!   means increasing the number of render passes (and thus sacrificing performance), and even
//!   if the user allows allocating more resources.
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
//! use the even group and odd-depth layers use the odd group. A simple chain can therefore render
//! by ping-ponging between one page from each group:
//!
//! ```text
//! L1 (even)
//! `-- L2 (odd)
//!     `-- L3 (even)
//!         `-- L4 (odd)
//!             `-- L5 (even)
//! ```
//!
//! `L5` is rendered first into the even page. `L4` is then rendered into the odd page and composes
//! `L5`, after which the `L5` allocation can be released. `L3` reuses the even page and composes
//! `L4`; `L2` reuses the odd page and composes `L3`; and finally `L1` reuses the even page and
//! composes `L2`. This pattern continues for arbitrarily deep chains, with at most two adjacent
//! layer allocations live at once.
//!
//! ## Lazy layer allocation
//!
//! Bottom-up traversal only provides this memory behavior because entering a layer does not
//! immediately allocate its target. The target is allocated lazily when the layer first has draws
//! or a completed child to receive. In the chain above, `L1` through `L4` do not occupy atlas space
//! while the scheduler descends to `L5`. If their targets were allocated eagerly, the outer layers
//! would occupy the even and odd pages before the inner layers could reuse them, defeating the
//! two-page ping-pong scheme.
//!
//! Branching can require more than two live allocations. Consider a parent with two children,
//! where the second child itself has a child:
//!
//! ```text
//! L1 (even)
//! |-- L2 (odd)
//! `-- L3 (odd)
//!     `-- L4 (even)
//! ```
//!
//! After `L2` is composed, `L1` has an even allocation containing that result. This allocation must
//! remain live while `L3` is scheduled. Descending into `L3` then reaches `L4`, which also needs an
//! even allocation. `L1` and `L4` may share an even atlas page if both regions fit; otherwise `L4`
//! requires a second even page. Together with the odd page used by `L3`, this means the branch can
//! require three intermediate textures. Lazy allocation minimizes the overlap, but cannot remove
//! allocations whose contents are simultaneously live. However, it should become apparent that,
//! even for complexly nested layer graphs, the number of allocations kept alive at the same time
//! is basically as small as possible.
//!
//! ## Batching into rounds
//!
//! Memory minimization does not imply one layer per render pass. Intermediate textures are atlases,
//! so independent layers can occupy different regions of the same page. The scheduler places work
//! at the earliest round and stage allowed by its dependencies. Operations can batch into the same
//! round when they require the same even and odd pages and their stages are compatible; otherwise
//! scheduling advances until the dependency is satisfied or the required page pair can be bound.
//! The scheduler does not allocate additional pages merely to improve batching.
//!
//! This process is monotonic and conservative: it does not backtrack to find a globally minimal
//! number of render passes. This is accepted as a downside of the current algorithm in the interest
//! of keeping the already non-trivial logic as simple as possible. The concrete contents and
//! texture bindings of a round are described in the [`round`] module.
//!
//! ## Filters and non-default blends
//!
//! A filter layer first follows the same bottom-up and lazy allocation process as a regular layer.
//! Its main allocation covers the filter's expanded bounds. Once the layer draws are ready, the
//! scheduler allocates an equally sized temporary region from the opposite texture group and
//! places the filter in the first compatible filter stage. The main and temporary regions provide
//! the input/output pair for the GPU pass sequence, whose final result is written back to the main
//! allocation. The temporary region is then cleared and released. Filters that must preserve the
//! unfiltered pixels, such as drop shadows, additionally reserve the shared scratch texture. The
//! layer is not made available to its parent until the filter is complete.
//!
//! Default source-over composition needs no separate blend operation: the completed child is
//! sampled by the parent's next draw. A non-default blend instead waits until both the child and
//! the existing parent contents are ready. It is placed in the next compatible blend stage for the
//! parent's texture group and constrains the round to bind both required layer pages. The backend
//! blends through the shared scratch texture and copies the result back into the parent, after
//! which the child can be cleared and released. If a non-default blend targets the root, the root
//! is first rendered into an intermediate layer because the user-provided target cannot be sampled
//! directly. Apart from that, this case is handled the same as any other layer.

mod allocate;
mod cursor;
pub(crate) mod execute;
pub(crate) mod round;
#[cfg(test)]
mod test_support;

use self::allocate::{Atlases, LayerAllocationRequest};
use self::cursor::Cursor;
pub(crate) use self::execute::{Backend, execute};
use self::round::{
    BlendOp, FilterOp, FilterTextureRegions, Round, RoundStage, Rounds, SchedulePoint,
};
use crate::draw::{Draw, DrawBuffers, DrawBuilder, DrawState};
use crate::filter::{FilterContext, FilterPassPlan, PreparedGpuFilter};
use crate::paint::PaintResolver;
use crate::scene::RecordedDraw;
use crate::schedule::allocate::AllocatedTextureRegion;
use crate::target::{
    DrawTarget, LayerTextureRegion, RootTarget, RoundBindings, TextureParity, TextureRegion,
};
use crate::{LayersConfig, RenderError, Scene, blend::BlendStrip};
use alloc::vec::Vec;
use vello_common::filter::FilterLayerPlacement;
use vello_common::geometry::{RectU16, SizeU16};
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
    /// Dimensions shared by every intermediate texture.
    texture_size: SizeU16,
    /// Whether execution requires the shared scratch texture.
    scratch_texture: bool,
}

impl Schedule {
    pub(crate) fn try_new(
        storage: &mut ScheduleStorage,
        scene: &Scene,
        root_output_target: RootTarget,
        paint_resolver: PaintResolver<'_>,
        texture_size: SizeU16,
        layer_config: LayersConfig,
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
            layer_config,
            storage,
        );

        scheduler.build()
    }

    pub(crate) fn layer_page_counts(&self) -> [usize; 2] {
        self.rounds.layer_page_counts()
    }

    pub(crate) fn scratch_texture(&self) -> bool {
        self.scratch_texture
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
    fn new(
        recorder: &'a CommandRecorder<RecordedDraw>,
        scene_bbox: RectU16,
        strip_storage: &'a StripStorage,
        root_render_target: RootTarget,
        paint_resolver: PaintResolver<'a>,
        texture_size: SizeU16,
        layer_config: LayersConfig,
        storage: &'p mut ScheduleStorage,
    ) -> Self {
        Self {
            recorder,
            scene_bbox,
            strip_storage,
            root_render_target,
            paint_resolver,
            cursor: Cursor::new(Atlases::new(texture_size, layer_config)),
            texture_size,
            storage,
        }
    }

    fn build(mut self) -> Result<Schedule, RenderError> {
        let mut rounds = Rounds::default();
        self.schedule_root(&mut rounds)?;

        // Since the strips should be rendered front-to-back.
        self.storage.buffers.draw_buffers.opaque_strips.reverse();

        let scratch_texture = self.cursor.scratch_texture();
        Ok(Schedule {
            rounds,
            texture_size: self.texture_size,
            scratch_texture,
        })
    }

    fn schedule_root(&mut self, rounds: &mut Rounds) -> Result<(), RenderError> {
        let target = self.root_render_target;

        if self.recorder.root_is_blend_target {
            // If the layer is a target of a non-default blending operation, we need to be able to
            // sample from it. However, this is not possible if we render directly into the
            // user-provided view. Therefore, we need to simulate a layer push, do all the rendering
            // there and then blit back into the main frame buffer.

            let opened_layer = self.open_root_layer();
            let layer = self.schedule_layer(opened_layer, rounds)?;
            let mut state = TargetScheduleState::new(target, layer.ready.round, self.scene_bbox);
            state.wait_until(layer.ready);

            let draw_point = rounds.build_draw(
                &mut state,
                &mut self.storage.buffers.draw_buffers,
                RoundBindings::new(layer.sample_region.texture.target),
                |builder| {
                    builder.push_layer_fill(layer.sample_region, 1.0, None, self.strip_storage);
                },
            );

            self.release_layer(layer, draw_point, rounds);
        } else {
            let mut state =
                TargetScheduleState::new(target, self.cursor.current_round(), self.scene_bbox);

            for cmd in &self.recorder.nodes {
                // Remember: Each command node consists of a sequence of draws + an option layer invocation.

                // First, we schedule the layer node. This might trigger advances to our current base round.
                let child = self.prepare_node(cmd, state.draw_state.target_bbox, rounds)?;

                // Then, we just submit all draws to the root output target for whatever round we are
                // currently in.
                self.push_draws(&cmd.draws, &mut state, rounds);

                // Finally, we also schedule the layer sampling operation.
                if let Some(child) = child {
                    self.compose_simple_layer(child.props, child.layer, &mut state, rounds);
                }
            }
        };

        Ok(())
    }

    fn prepare_node(
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
                // not a no-op.
                parent_bounds
            } else {
                return Ok(None);
            }
        } else {
            layer.bbox
        };

        // We cannot do this for filter layers, because the filter needs to be applied to the whole
        // region _BEFORE_ applying clips. Instead, the clip is simply applied when sampling from
        // the filter layer.
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

    fn open_layer(&self, layer: &'a RecordedLayer, bbox: RectU16) -> OpenLayer<'a> {
        let sample = match &layer.kind {
            RecordedLayerKind::Regular => LayerSamplePlacement::regular(bbox),
            RecordedLayerKind::Filter { placement, .. } => LayerSamplePlacement::filter(*placement),
        };

        OpenLayer {
            cmds: &layer.nodes,
            kind: &layer.kind,
            texture_parity: self.layer_texture_parity(layer.depth),
            bbox,
            sample,
            target: None,
        }
    }

    fn open_root_layer(&self) -> OpenLayer<'a> {
        OpenLayer {
            cmds: &self.recorder.nodes,
            kind: &REGULAR_LAYER_KIND,
            texture_parity: TextureParity::Odd,
            bbox: self.scene_bbox,
            sample: LayerSamplePlacement::regular(self.scene_bbox),
            target: None,
        }
    }

    fn layer_texture_parity(&self, layer_depth: usize) -> TextureParity {
        TextureParity::from_parity(layer_depth + usize::from(self.recorder.root_is_blend_target))
    }

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
            RoundBindings::default(),
            |builder| {
                for draw in &self.recorder.draws[draws.start as usize..draws.end as usize] {
                    builder.push_draw(draw, self.strip_storage, self.paint_resolver);
                }
            },
        );
    }

    fn schedule_layer(
        &mut self,
        mut layer: OpenLayer<'a>,
        rounds: &mut Rounds,
    ) -> Result<ScheduledLayer, RenderError> {
        // Overall we follow a similar flow to `schedule_root` here.

        for cmd in layer.cmds {
            // First make sure that the child node is scheduled, in case it exists.
            // TODO: Similarly to Vello CPU, flatten this to avoid stack overflows for deep layers
            let child = self.prepare_node(cmd, layer.sample.bbox, rounds)?;

            // Important: Keep this after `prepare_node`: allocating lazily is what makes traversal
            // bottom-up with respect to memory, while still allowing compatible layers to batch.
            let target = self.ensure_layer_target(&mut layer)?;

            // Now schedule the draws + optionally the composition of the child layer node.
            self.push_draws(&cmd.draws, &mut target.schedule_state, rounds);

            if let Some(child) = child {
                self.compose_layer(child.props, child.layer, &mut target.schedule_state, rounds)?;
            }
        }

        self.ensure_layer_target(&mut layer)?;

        let target = layer.target.take().unwrap();

        let region = target.schedule_state.draw_state.target;
        let mut ready = target.schedule_state.ready;

        if let Some(filter) = target.filter {
            let temporary =
                self.cursor
                    .allocate_layer(LayerAllocationRequest::filter_temporary(
                        region.texture.rect,
                        region.texture.target.texture_parity.opposite(),
                    ))?;
            let textures = FilterTextureRegions::new(region.texture, temporary.allocation.region);

            if filter.data.needs_copy_pass() {
                self.cursor.require_scratch_texture()?;
            }

            let base_point = ready
                // We must wait until our reserved space is available in the atlas.
                .max(SchedulePoint::start(temporary.round_idx))
                // Wait until we reach the filter stage.
                .next(RoundStage::filter(region.texture.target.texture_parity));

            let filter_point = rounds.resolve_binding_point(base_point, textures.texture_binding());

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

            let clear_region = temporary.allocation.clear_region();
            rounds.push_layer_clear(
                filter_point.round,
                clear_region.target.texture_parity,
                clear_region.rect,
            );
            self.cursor
                .release(temporary.allocation, filter_point.round);

            ready = filter_point;
        }

        let scheduled = ScheduledLayer {
            sample_region: layer.sample.resolve(region),
            allocation: target.allocation,
            ready,
        };

        Ok(scheduled)
    }

    /// Schedule a composition operation for a layer.
    fn compose_layer(
        &mut self,
        props: &LayerProps,
        child_layer: ScheduledLayer,
        state: &mut TargetScheduleState<LayerTextureRegion>,
        rounds: &mut Rounds,
    ) -> Result<(), RenderError> {
        let blend_mode = props.blend_mode;
        let opacity = props.opacity;
        if blend_mode == BlendMode::default() {
            self.compose_simple_layer(props, child_layer, state, rounds);

            return Ok(());
        }

        let parent_region = state.draw_state.target;
        let child_region = child_layer.sample_region;

        // For non-destructive blend modes, choose the (smaller) child bbox as the
        // affected region. Otherwise, we need to choose the (larger) parent bbox.
        let mut blend_bbox = if blend_mode.is_destructive() {
            parent_region.layer_bbox
        } else {
            child_region.layer_bbox
        };

        if let Some(clip_path) = &props.clip_path {
            blend_bbox = blend_bbox.intersect(clip_path.bbox);
        }

        if blend_bbox.is_empty() {
            let child_ready = child_layer.ready;
            self.release_layer(child_layer, child_ready, rounds);

            return Ok(());
        }

        let clip_strips = props.clip_path.as_ref().map(|clip_path| {
            let start = self.storage.buffers.blend_strips.len();
            let strips = &self.strip_storage.strips[clip_path.strip_range.clone()];
            let geometry_shift = parent_region.geometry_shift();
            let tile_bounds = RectU16::new(
                blend_bbox.x0 / Tile::WIDTH,
                blend_bbox.y0 / Tile::HEIGHT,
                blend_bbox.x1 / Tile::WIDTH,
                blend_bbox.y1 / Tile::HEIGHT,
            );

            visit_strip_fill_segments(
                strips,
                tile_bounds,
                &mut self.storage.buffers.blend_strips,
                |blend_strips, segment| {
                    blend_strips.push(BlendStrip::from_fill(
                        segment.shift(geometry_shift),
                        Some(segment.alpha_idx / u32::from(Tile::HEIGHT)),
                    ));
                },
                |blend_strips, segment| {
                    blend_strips.push(BlendStrip::from_fill(segment.shift(geometry_shift), None));
                },
            );

            let end = self.storage.buffers.blend_strips.len();
            u32::try_from(start).unwrap()..u32::try_from(end).unwrap()
        });

        let parent_texture_parity = parent_region.texture.target.texture_parity;
        self.cursor.require_scratch_texture()?;

        // A blend must execute after both the parent and child are ready.
        let blend_stage = RoundStage::blend(parent_texture_parity);
        let blend_point = state
            .ready
            .next(blend_stage)
            .max(child_layer.ready.next(blend_stage));
        let blend_binding = RoundBindings::new(parent_region.texture.target)
            .merge(RoundBindings::new(child_region.texture.target))
            .expect("parent and child layers must have compatible texture parities");
        let blend_point = rounds.resolve_binding_point(blend_point, blend_binding);

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
        self.release_layer(child_layer, blend_point, rounds);

        state.ready = blend_point;
        state.next_draw = blend_point.next(state.draw_state.target.draw_stage());

        Ok(())
    }

    /// Schedule a composition operation for a layer using src-over blending.
    fn compose_simple_layer<T: ScheduleTarget>(
        &mut self,
        props: &LayerProps,
        child_layer: ScheduledLayer,
        state: &mut TargetScheduleState<T>,
        rounds: &mut Rounds,
    ) {
        // Layer invocations introduce a dependency barrier. Find the first draw stage on the
        // parent that executes after the child is ready.
        state.wait_until(child_layer.ready);

        // Schedule the actual layer fill command.
        let draw_point = rounds.build_draw(
            state,
            &mut self.storage.buffers.draw_buffers,
            RoundBindings::new(child_layer.sample_region.texture.target),
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
        self.release_layer(child_layer, draw_point, rounds);
    }

    fn release_layer(&mut self, layer: ScheduledLayer, point: SchedulePoint, rounds: &mut Rounds) {
        // When releasing the layer, we need to make sure to deallocate and clear the space in the
        // layer texture.

        assert!(
            point >= layer.ready,
            "layer released before it became ready"
        );
        rounds.ensure_exists(point.round);

        let layer_region = layer.allocation.clear_region();
        rounds.push_layer_clear(
            point.round,
            layer_region.target.texture_parity,
            layer_region.rect,
        );

        self.cursor.release(layer.allocation, point.round);
    }

    /// Lazily allocate space for an open layer.
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

            let request = LayerAllocationRequest::new(layer);
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
    sample: LayerSamplePlacement,
    /// Lazily allocated target and its scheduling state.
    target: Option<LayerTarget>,
}

/// Maps a rendered layer allocation to the region sampled by its parent.
#[derive(Debug, Clone, Copy)]
struct LayerSamplePlacement {
    /// Offset of the sampled contents from the allocation origin.
    src_offset: (u16, u16),
    /// Bounds of the sampled contents in scene coordinates.
    bbox: RectU16,
}

impl LayerSamplePlacement {
    fn regular(bbox: RectU16) -> Self {
        Self {
            src_offset: (0, 0),
            bbox,
        }
    }

    fn filter(placement: FilterLayerPlacement) -> Self {
        Self {
            src_offset: (placement.src_x, placement.src_y),
            bbox: placement.dest_bbox,
        }
    }

    fn resolve(self, allocation: LayerTextureRegion) -> LayerTextureRegion {
        let x0 = allocation.texture.rect.x0 + self.src_offset.0;
        let y0 = allocation.texture.rect.y0 + self.src_offset.1;

        LayerTextureRegion {
            texture: TextureRegion {
                target: allocation.texture.target,
                rect: RectU16::new(x0, y0, x0 + self.bbox.width(), y0 + self.bbox.height()),
            },
            layer_bbox: self.bbox,
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
        sampled: RoundBindings,
        f: impl FnOnce(&mut DrawBuilder<'_, T>),
    ) -> SchedulePoint {
        let requirement = state
            .draw_state
            .target
            .round_bindings()
            .merge(sampled)
            .expect("draw target and sampled layer must have compatible texture parities");

        let point = self.resolve_binding_point(state.next_draw, requirement);
        state.schedule_draw(point);
        self.ensure_exists(point.round);

        let target_draw = state
            .draw_state
            .target
            .draw_mut(self.round_mut(point.round));

        let mut builder = DrawBuilder::new(target_draw, draw_buffers, &mut state.draw_state);
        f(&mut builder);

        point
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
    // This can be later than [`Self::ready`]. For example, assume we have a sequence of three
    // nested layers allocated as follows:
    // - L0 in odd texture
    // - L1 in even texture
    // - L2 in odd texture
    //
    // In a round, we first execute even blends, so we do Blend(L1, L2). That same layer is now
    // ready for L0 to sample _in that same round_, so `Self::ready` will still have the same round
    // index. However, if we want to append more draws we need to wait until the next round, since
    // all draws in a round happen before blend ops.
    //
    // In other cases, this is often the same as `Self::ready`.
    /// Earliest point at which another draw can be appended to this target.
    next_draw: SchedulePoint,
    /// Point after which all currently scheduled contents of this target are available.
    ready: SchedulePoint,
}

impl<T: ScheduleTarget> TargetScheduleState<T> {
    fn new(target: T, start_round: usize, target_bbox: RectU16) -> Self {
        let ready = SchedulePoint::start(start_round);
        let next_draw = ready.next(target.draw_stage());

        Self {
            draw_state: DrawState::new(target, target_bbox),
            next_draw,
            ready,
        }
    }

    fn wait_until(&mut self, dependency: SchedulePoint) {
        self.next_draw = self
            .next_draw
            .max(dependency.next(self.draw_state.target.draw_stage()));
    }

    fn schedule_draw(&mut self, point: SchedulePoint) {
        debug_assert!(
            point >= self.next_draw,
            "draw schedule points must be monotonically increasing"
        );
        self.next_draw = point;
        self.ready = point;
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

#[cfg(test)]
mod tests {
    use super::ScheduleStorage;
    use super::test_support::{SceneCase, ScheduledCase};
    use crate::filter::FILTER_ATLAS_PADDING;
    use crate::target::{RootTarget, TextureParity};
    use vello_common::filter_effects::{Filter, FilterPrimitive};
    use vello_common::geometry::SizeU16;
    use vello_common::kurbo::Rect;
    use vello_common::peniko::{BlendMode, Compose, Mix};

    fn blend_case() -> ScheduledCase {
        let mut case = SceneCase::new(32, 8);
        case.layer(|case| {
            case.draw(Rect::new(0.0, 0.0, 32.0, 8.0), 0.5);
            case.layer_with(
                None,
                Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
                None,
                |case| case.draw(Rect::new(8.0, 0.0, 16.0, 8.0), 0.5),
            );
        });
        case.schedule_root()
    }

    fn root_blend_case() -> SceneCase {
        let mut case = SceneCase::new(16, 8);
        case.layer_with(
            None,
            Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
            None,
            |case| case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5),
        );
        case
    }

    fn add_chain(case: &mut SceneCase, depth: usize) {
        case.layer(|case| {
            if depth == 1 {
                case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5);
            } else {
                add_chain(case, depth - 1);
            }
        });
    }

    fn add_blend_chain(case: &mut SceneCase, depth: usize) {
        case.layer_with(
            None,
            Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
            None,
            |case| {
                if depth == 1 {
                    case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5);
                } else {
                    add_blend_chain(case, depth - 1);
                }
            },
        );
    }

    fn add_tree(case: &mut SceneCase, depth: usize, children: usize) {
        case.layer(|case| {
            if depth == 1 {
                case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5);
            } else {
                for _ in 0..children {
                    add_tree(case, depth - 1, children);
                }
            }
        });
    }

    fn chain_case(depth: usize) -> ScheduledCase {
        let mut case = SceneCase::new(8, 8);

        add_chain(&mut case, depth);
        case.schedule_root()
    }

    fn binding_case() -> ScheduledCase {
        let mut case = SceneCase::new(64, 64);
        case.layer(|case| case.draw(Rect::new(0.0, 0.0, 60.0, 60.0), 0.5));
        case.layer_with(
            None,
            Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
            Some(Filter::from_primitive(FilterPrimitive::Offset {
                dx: 0.0,
                dy: 0.0,
            })),
            |case| case.draw(Rect::new(0.0, 0.0, 60.0, 60.0), 0.5),
        );
        case.schedule(RootTarget::UserSurface, SizeU16::new(80), 8)
            .unwrap()
    }

    fn sibling_case(count: u16) -> SceneCase {
        let mut case = SceneCase::new(count * 4, 8);
        for index in 0..count {
            case.layer(|case| case.draw_at(f64::from(index * 4), 0.5));
        }
        case
    }

    fn offset_filter() -> Filter {
        Filter::from_primitive(FilterPrimitive::Offset { dx: 0.0, dy: 0.0 })
    }

    fn filter_page_size() -> SizeU16 {
        SizeU16::new(8 + 2 * FILTER_ATLAS_PADDING)
    }

    #[test]
    fn empty_scene() {
        let scheduled = SceneCase::new(32, 32).schedule_root();

        assert!(scheduled.views().is_empty());
        assert_eq!(scheduled.page_counts(), [0, 0]);
        assert!(!scheduled.scratch_texture());
    }

    #[test]
    fn root_draws() {
        let mut case = SceneCase::new(32, 8);
        for x in [0.0, 8.0, 16.0] {
            case.draw_at(x, 0.5);
        }

        let scheduled = case.schedule_root();
        let rounds_view = scheduled.views();

        assert_eq!(rounds_view.len(), 1);
        assert_eq!(rounds_view[0].root.x, [0, 8, 16]);
        assert!(!rounds_view[0].root.has_child_layer);
        assert_eq!(scheduled.page_counts(), [0, 0]);
    }

    #[test]
    fn draw_order() {
        let mut case = SceneCase::new(32, 8);
        case.draw_at(0.0, 0.5);
        case.layer(|case| case.draw_at(8.0, 0.5));
        case.draw_at(16.0, 0.5);

        let scheduled = case.schedule_root();
        let rounds_view = scheduled.views();

        assert_eq!(rounds_view.len(), 1);
        assert_eq!(rounds_view[0].root.x, [0, 8, 16]);
        assert!(rounds_view[0].root.has_child_layer);
    }

    #[test]
    fn opaque_root() {
        let mut case = SceneCase::new(32, 8);
        for x in [0.0, 8.0, 16.0] {
            case.draw_at(x, 1.0);
        }
        case.draw_at(24.0, 0.5);

        let user = case.schedule_root();
        assert_eq!(user.opaque_x(), [16, 8, 0]);
        assert_eq!(user.views()[0].root.x, [24]);

        let atlas = case
            .schedule(RootTarget::AtlasLayer, SizeU16::new(64), 8)
            .unwrap();
        assert!(atlas.opaque_x().is_empty());
        assert_eq!(atlas.views()[0].root.x, [0, 8, 16, 24]);
    }

    #[test]
    fn simple_layer() {
        let mut case = SceneCase::new(32, 8);
        case.layer(|case| case.draw_at(8.0, 0.5));

        let scheduled = case.schedule_root();
        let rounds_view = scheduled.views();
        let round = &rounds_view[0];

        assert_eq!(scheduled.page_counts(), [0, 1]);
        assert_eq!(rounds_view.len(), 1);
        assert_eq!(round.odd.x.len(), 1);
        assert_eq!(round.root.x, [8]);
        assert!(round.root.has_child_layer);
        assert_eq!(round.clears[TextureParity::Odd.get_parity()].len(), 1);
    }

    #[test]
    fn default_blend() {
        let mut case = SceneCase::new(16, 8);
        case.layer(|case| case.draw_at(4.0, 0.5));

        let scheduled = case.schedule_root();

        assert!(scheduled.views()[0].root.has_child_layer);
        assert!(scheduled.storage.buffers.blend_ops.is_empty());
    }

    #[test]
    fn non_default_blend_into_layer() {
        let scheduled = blend_case();
        let rounds_view = scheduled.views();

        assert_eq!(rounds_view.len(), 1);
        assert_eq!(rounds_view[0].blend_passes, [0, 1]);
    }

    #[test]
    fn clipped_away_blend() {
        let mut case = SceneCase::new(32, 8);
        case.layer(|case| {
            case.draw(Rect::new(0.0, 0.0, 32.0, 8.0), 0.5);
            case.layer_with(
                Some(Rect::new(24.0, 0.0, 32.0, 8.0)),
                Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
                Some(Filter::from_primitive(FilterPrimitive::Offset {
                    dx: 0.0,
                    dy: 0.0,
                })),
                |case| case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5),
            );
        });

        let scheduled = case.schedule_root();

        assert_eq!(scheduled.storage.buffers.filter_ops.len(), 1);
        assert!(scheduled.storage.buffers.blend_ops.is_empty());
        assert_eq!(scheduled.total_clears(), 3);
    }

    #[test]
    fn blend_release() {
        let scheduled = blend_case();
        let rounds_view = scheduled.views();
        let blend_round = rounds_view
            .iter()
            .find(|round| round.blend_passes[TextureParity::Odd.get_parity()] == 1)
            .unwrap();

        assert!(scheduled.scratch_texture());
        assert_eq!(
            blend_round.clears[TextureParity::Even.get_parity()].len(),
            1
        );
    }

    #[test]
    fn root_blend_resources() {
        let scheduled = root_blend_case()
            .schedule(RootTarget::UserSurface, SizeU16::new(16), 3)
            .unwrap();

        // Root lands in the first odd layer, its child in the even one.
        assert_eq!(scheduled.page_counts(), [1, 1]);
        assert!(scheduled.scratch_texture());
    }

    #[test]
    fn root_release() {
        let scheduled = root_blend_case()
            .schedule(RootTarget::UserSurface, SizeU16::new(16), 3)
            .unwrap();
        let rounds_view = scheduled.views();
        let root_round = rounds_view
            .iter()
            .find(|round| round.root.has_child_layer)
            .unwrap();

        assert_eq!(root_round.clears[TextureParity::Odd.get_parity()].len(), 1);
    }

    #[test]
    fn root_blend_budget() {
        let case = root_blend_case();

        assert!(
            case.schedule(RootTarget::UserSurface, SizeU16::new(16), 3,)
                .is_ok()
        );
        assert!(matches!(
            case.schedule(RootTarget::UserSurface, SizeU16::new(16), 2,),
            Err(crate::RenderError::AtlasError(_))
        ));
    }

    #[test]
    fn nested_parity() {
        let mut case = SceneCase::new(16, 8);
        case.layer(|case| {
            case.layer(|case| case.draw_at(4.0, 0.5));
        });

        let scheduled = case.schedule_root();
        let rounds_view = scheduled.views();
        let round = &rounds_view[0];

        assert_eq!(scheduled.page_counts(), [1, 1]);
        assert_eq!(rounds_view.len(), 1);
        assert_eq!(round.even.x.len(), 1);
        assert_eq!(round.odd.x.len(), 1);
        assert!(round.odd.has_child_layer);
        assert_eq!(round.root.x.len(), 1);
        assert!(round.root.has_child_layer);
        assert_eq!(round.clears[TextureParity::Even.get_parity()].len(), 1);
        assert_eq!(round.clears[TextureParity::Odd.get_parity()].len(), 1);
    }

    #[test]
    fn even_child() {
        let scheduled = chain_case(2);
        let rounds_view = scheduled.views();

        assert_eq!(rounds_view.len(), 1);
        assert_eq!(rounds_view[0].even.x.len(), 1);
        assert!(rounds_view[0].odd.has_child_layer);
    }

    #[test]
    fn odd_child() {
        let scheduled = chain_case(3);
        let rounds_view = scheduled.views();

        assert_eq!(rounds_view.len(), 2);
        assert_eq!(rounds_view[0].odd.x.len(), 1);
        assert!(rounds_view[0].even.x.is_empty());
        assert!(rounds_view[1].even.has_child_layer);
    }

    #[test]
    fn draw_after_blend() {
        let mut case = SceneCase::new(32, 8);
        case.layer(|case| {
            case.draw_at(0.0, 0.5);
            case.layer_with(
                None,
                Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
                None,
                |case| case.draw_at(8.0, 0.5),
            );
            case.draw_at(24.0, 0.5);
        });

        let scheduled = case.schedule_root();
        let rounds_view = scheduled.views();

        assert_eq!(rounds_view[0].odd.x.len(), 1);
        assert_eq!(
            rounds_view[0].blend_passes[TextureParity::Odd.get_parity()],
            1
        );
        assert_eq!(rounds_view[1].odd.x.len(), 1);
    }

    #[test]
    fn sibling_batch() {
        let mut case = SceneCase::new(32, 8);
        for x in [0.0, 8.0, 16.0] {
            case.layer(|case| case.draw_at(x, 0.5));
        }

        let scheduled = case
            .schedule(RootTarget::UserSurface, SizeU16::from_wh(16, 8), 1)
            .unwrap();
        let rounds_view = scheduled.views();
        let round = &rounds_view[0];

        assert_eq!(scheduled.page_counts(), [0, 1]);
        assert_eq!(rounds_view.len(), 1);
        assert_eq!(round.odd.x.len(), 3);
        assert_eq!(round.root.x, [0, 8, 16]);
        assert_eq!(round.clears[TextureParity::Odd.get_parity()].len(), 3);
    }

    #[test]
    fn incompatible_siblings() {
        let scheduled = binding_case();
        let rounds_view = scheduled.views();

        // Root is blend target, so it lands in odd texture page 1.
        // Sibling layer lands in even texture page 1, and since it has
        // a filter we need a new page allocation in odd, so texthre page 2.
        assert_eq!(scheduled.page_counts(), [1, 2]);
        assert_eq!(rounds_view.len(), 3);
        // Round 0 binds the texture where the root is.
        assert_eq!(
            rounds_view[0].binding[TextureParity::Odd.get_parity()],
            Some(0)
        );
        // Round 1 binds the texture where the filter layer is.
        assert_eq!(
            rounds_view[1].binding[TextureParity::Odd.get_parity()],
            Some(1)
        );
        // Round 0 again binds to the root.
        assert_eq!(
            rounds_view[2].binding[TextureParity::Odd.get_parity()],
            Some(0)
        );
    }

    #[test]
    fn filter_binding_conflict() {
        let scheduled = binding_case();
        let rounds_view = scheduled.views();
        let filter = scheduled.storage.buffers.filter_ops[0];

        assert_eq!(rounds_view[0].filter_passes, [0, 0]);
        assert_eq!(rounds_view[1].filter_passes, [1, 0]);
        for region in [filter.textures.original, filter.textures.temporary] {
            let target = region.target;
            assert_eq!(
                rounds_view[1].binding[target.texture_parity.get_parity()],
                Some(target.page_index)
            );
        }
    }

    #[test]
    fn deep_reuse() {
        const DEPTH: usize = 12;
        let mut case = SceneCase::new(8, 8);
        add_chain(&mut case, DEPTH);

        let scheduled = case
            .schedule(RootTarget::UserSurface, SizeU16::new(8), 2)
            .unwrap();
        let rounds_view = scheduled.views();
        let layer_draws = rounds_view
            .iter()
            .map(|round| round.even.x.len() + round.odd.x.len())
            .sum::<usize>();

        assert!(rounds_view.len() > 1);
        assert_eq!(scheduled.page_counts(), [1, 1]);
        assert_eq!(layer_draws, DEPTH);
        assert_eq!(scheduled.total_clears(), DEPTH);
    }

    // For the next 3 cases: They show that our scheduler has the ability
    // to render arbitrarily deeply nested layers as well as arbitrarily many
    // sibling layers using just two textures, as long as they have
    // at most one child.

    #[test]
    fn deeply_nested_layers() {
        for depth in 1..=32 {
            let mut case = SceneCase::new(8, 8);
            add_chain(&mut case, depth);

            let scheduled = case
                .schedule(RootTarget::UserSurface, SizeU16::new(8), 2)
                .unwrap();
            let textures = scheduled.page_counts().into_iter().sum::<usize>();

            assert!(textures <= 2, "depth {depth} used {textures} textures");
        }
    }

    #[test]
    fn deeply_nested_blend_layers() {
        for depth in 1..=32 {
            let mut case = SceneCase::new(8, 8);
            add_blend_chain(&mut case, depth);

            let scheduled = case
                .schedule(RootTarget::UserSurface, SizeU16::new(8), 3)
                .unwrap();

            assert_eq!(
                (scheduled.scratch_texture(), scheduled.page_counts()),
                (true, [1, 1]),
                "unexpected resources at depth {depth}"
            );
        }
    }

    #[test]
    fn wide_layers() {
        for count in 1..=32 {
            let scheduled = sibling_case(count)
                .schedule(RootTarget::UserSurface, SizeU16::from_wh(64, 8), 1)
                .unwrap();
            let rounds_view = scheduled.views();

            // Many sibling layers are batched into a single round, if atlas space
            // permits it.
            assert_eq!(rounds_view.len(), 1, "failed at width {count}");
            assert_eq!(
                rounds_view[0].odd.x.len(),
                usize::from(count),
                "missing child at width {count}"
            );
        }
    }

    #[test]
    fn nested_children() {
        const CHILDREN: usize = 3;

        // If we have enough atlas space, even deeply and widely nested layer graphs can
        // be batched efficiently. The expected round count will be `layer_depth` / 2.
        // No additional pages need to be created as all layers fit.
        for (depth, expected_rounds) in (2..=6).zip([1, 2, 2, 3, 3]) {
            let mut case = SceneCase::new(8, 8);
            add_tree(&mut case, depth, CHILDREN);
            let scheduled = case
                .schedule(RootTarget::UserSurface, SizeU16::new(256), 2)
                .unwrap();
            let layers = (CHILDREN.pow(depth.try_into().expect("test depth fits in u32")) - 1)
                / (CHILDREN - 1);

            assert_eq!(scheduled.page_counts(), [1, 1], "failed at depth {depth}");
            assert_eq!(
                scheduled.views().len(),
                expected_rounds,
                "failed at depth {depth}"
            );
            assert_eq!(scheduled.total_clears(), layers);
        }
    }

    #[test]
    fn nested_children_spilled() {
        const CHILDREN: usize = 3;

        // If our atlas dimensions are constrained, the scheduler will still find
        // a valid schedule and keep the number of allocated pages to a minimum.
        // However, this is at the cost of a larger round count.
        for (depth, expected_rounds) in (3..=6).zip([21, 39, 201, 363]) {
            let mut case = SceneCase::new(8, 8);
            add_tree(&mut case, depth, CHILDREN);
            let scheduled = case
                .schedule(RootTarget::UserSurface, SizeU16::new(8), 16)
                .unwrap();
            let layers = (CHILDREN.pow(depth.try_into().expect("test depth fits in u32")) - 1)
                / (CHILDREN - 1);

            assert_eq!(
                scheduled.page_counts(),
                [depth / 2, depth.div_ceil(2)],
                "failed at depth {depth}"
            );
            assert_eq!(
                scheduled.views().len(),
                expected_rounds,
                "failed at depth {depth}"
            );
            assert_eq!(scheduled.total_clears(), layers);
        }
    }

    #[test]
    fn empty_layer() {
        let mut case = SceneCase::new(16, 8);
        case.layer(|_| {});

        let scheduled = case.schedule_root();

        assert!(scheduled.views().is_empty());
        assert_eq!(scheduled.page_counts(), [0, 0]);
    }

    #[test]
    fn destructive_empty() {
        let mut case = SceneCase::new(32, 8);
        case.layer(|case| {
            case.draw(Rect::new(0.0, 0.0, 16.0, 8.0), 0.5);
            case.layer_with(
                None,
                Some(BlendMode::new(Mix::Normal, Compose::Clear)),
                None,
                |_| {},
            );
        });

        let scheduled = case.schedule_root();

        assert_eq!(scheduled.page_counts(), [1, 1]);
        assert!(scheduled.scratch_texture());
        assert_eq!(scheduled.storage.buffers.blend_ops.len(), 1);
    }

    #[test]
    fn layer_clip() {
        let clip = Rect::new(8.0, 0.0, 16.0, 8.0);
        let mut clipped = SceneCase::new(32, 8);
        clipped.layer_with(Some(clip), None, None, |case| {
            case.draw(Rect::new(0.0, 0.0, 24.0, 8.0), 0.5);
        });
        let scheduled = clipped.schedule_root();
        let rounds_view = scheduled.views();
        assert_eq!(scheduled.page_counts(), [0, 1]);
        assert_eq!(rounds_view.len(), 1);
        assert_eq!(rounds_view[0].odd.x.len(), 1);
        assert!(rounds_view[0].root.has_child_layer);
        assert_eq!(
            rounds_view[0].clears[TextureParity::Odd.get_parity()].len(),
            1
        );

        let mut disjoint = SceneCase::new(32, 8);
        disjoint.layer_with(Some(Rect::new(24.0, 0.0, 32.0, 8.0)), None, None, |case| {
            case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5);
        });
        let scheduled = disjoint.schedule_root();
        assert!(scheduled.views().is_empty());
        assert_eq!(scheduled.page_counts(), [0, 0]);
    }

    #[test]
    fn filter_layer() {
        let mut case = SceneCase::new(64, 16);
        let clip = Rect::new(16.0, 0.0, 24.0, 8.0);
        let filter = Filter::from_primitive(FilterPrimitive::Offset { dx: 8.0, dy: 0.0 });
        case.layer_with(Some(clip), None, Some(filter), |case| {
            case.draw(Rect::new(8.0, 0.0, 16.0, 8.0), 0.5);
        });

        let scheduled = case
            .schedule(RootTarget::UserSurface, SizeU16::new(128), 2)
            .unwrap();
        let rounds_view = scheduled.views();

        assert_eq!(scheduled.page_counts(), [1, 1]);
        assert_eq!(rounds_view.len(), 1);
        assert_eq!(rounds_view[0].odd.x.len(), 1);
        assert_eq!(rounds_view[0].filter_passes, [0, 1]);
        assert!(rounds_view[0].root.has_child_layer);
        assert_eq!(scheduled.total_clears(), 2);
    }

    #[test]
    fn filter_round_resolving() {
        let mut case = SceneCase::new(8, 8);

        case.layer_with(None, None, Some(offset_filter()), |case| {
            case.layer(|case| case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5));
        });

        let scheduled = case
            .schedule(RootTarget::UserSurface, filter_page_size(), 2)
            .unwrap();
        let rounds_view = scheduled.views();
        let filter = scheduled.storage.buffers.filter_ops[0];

        // The filter source and its ping-pong temporary use opposite parities.
        assert_eq!(
            (
                filter.textures.original.target.texture_parity,
                filter.textures.temporary.target.texture_parity,
            ),
            (TextureParity::Odd, TextureParity::Even)
        );
        // No additional pages are created; the even page is reused after the child release.
        assert_eq!(scheduled.page_counts(), [1, 1]);
        assert_eq!(rounds_view.len(), 2);
        // The child is first drawn into an even texture in round 0.
        // The child is composed into the odd filter source **in round 0**. This is
        // possible because within a round, we first handle draws to even pages, then
        // odd pages.
        assert_eq!(rounds_view[0].odd.x.len(), 1);
        // The child allocation is released from the even page at the end of round 0.
        assert_eq!(
            rounds_view[0].clears[TextureParity::Even.get_parity()].len(),
            1
        );

        // The temporary is not available soon enough to filter in round 0.
        assert_eq!(rounds_view[0].filter_passes, [0, 0]);
        // After the even page becomes reusable, the filter runs in round 1.
        assert_eq!(rounds_view[1].filter_passes, [0, 1]);
    }

    #[test]
    fn filter_siblings() {
        let mut case = SceneCase::new(8, 8);

        for _ in 0..2 {
            case.layer_with(None, None, Some(offset_filter()), |case| {
                case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5);
            });
        }

        let scheduled = case
            .schedule(RootTarget::UserSurface, filter_page_size(), 2)
            .unwrap();

        // Sibling filter layers should also reuse pages.
        assert_eq!(scheduled.page_counts(), [1, 1]);
        assert_eq!(scheduled.storage.buffers.filter_ops.len(), 2);

        // Each filter clears its source allocation and its temporary allocation.
        assert_eq!(scheduled.total_clears(), 4);
    }

    #[test]
    fn storage_reuse() {
        let mut first = SceneCase::new(64, 16);
        first.layer(|case| {
            case.draw(Rect::new(0.0, 0.0, 32.0, 8.0), 0.5);
            case.layer_with(
                Some(Rect::new(8.0, 0.0, 24.0, 8.0)),
                Some(BlendMode::new(Mix::Normal, Compose::Clear)),
                None,
                |case| case.draw(Rect::new(8.0, 0.0, 24.0, 8.0), 0.5),
            );
            case.layer_with(
                None,
                None,
                Some(Filter::from_primitive(FilterPrimitive::Offset {
                    dx: 4.0,
                    dy: 0.0,
                })),
                |case| case.draw(Rect::new(32.0, 0.0, 40.0, 8.0), 0.5),
            );
        });

        let mut storage = ScheduleStorage::default();
        let first_schedule = first
            .schedule_into(&mut storage, RootTarget::UserSurface, SizeU16::new(128), 4)
            .unwrap();

        assert!(!storage.buffers.draw_buffers.strips.is_empty());
        assert!(!storage.buffers.blend_ops.is_empty());
        assert!(!storage.buffers.blend_strips.is_empty());
        assert!(!storage.buffers.filter_ops.is_empty());
        assert!(!storage.filter_context.is_empty());
        assert!(!first_schedule.rounds.rounds.is_empty());

        let mut second = SceneCase::new(64, 16);
        second.draw_at(48.0, 0.5);
        let second_schedule = second
            .schedule_into(&mut storage, RootTarget::UserSurface, SizeU16::new(128), 4)
            .unwrap();

        assert_eq!(storage.buffers.draw_buffers.strips.len(), 1);
        assert!(storage.buffers.draw_buffers.opaque_strips.is_empty());
        assert!(storage.buffers.blend_ops.is_empty());
        assert!(storage.buffers.blend_strips.is_empty());
        assert!(storage.buffers.filter_ops.is_empty());
        assert!(storage.filter_context.is_empty());
        assert_eq!(second_schedule.rounds.rounds.len(), 1);
        let round = &second_schedule.rounds.rounds[0];
        let root = round
            .root_draw_pass(&storage.buffers, RootTarget::UserSurface)
            .unwrap();
        assert_eq!(root.strips.len(), 1);
        assert!(root.external_texture_runs.is_empty());
        assert_eq!(round.layer_passes(&storage.buffers).count(), 0);
    }
}
