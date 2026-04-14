// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! # Scheduling
//!
//! - Draw commands are either issued to the final target or slots in a clip texture.
//! - Rounds represent a draw in up to 3 render targets (two clip textures and a final target).
//! - The clip texture stores slots for many clip depths. Once our clip textures are full,
//!   we flush rounds (i.e. execute render passes) to free up space. Note that a slot refers
//!   to 1 wide tile's worth of pixels in the clip texture.
//! - The `free` vector contains the indices of the slots that are available for use in the two clip textures.
//!
//! ## Example
//!
//! Consider the following scene of drawing a single wide tile with three overlapping rectangles with
//! decreasing width clipping regions.
//!
//! ```rs
//! const WIDTH: f64 = 100.0;
//! const HEIGHT: f64 = Tile::HEIGHT as f64;
//! const OFFSET: f64 = WIDTH / 3.0;
//!
//! let colors = [RED, GREEN, BLUE];
//!
//! for i in 0..3 {
//!     let clip_rect = Rect::new((i as f64) * OFFSET, 0.0, 100, HEIGHT);
//!     ctx.push_clip_layer(&clip_rect.to_path(0.1));
//!     ctx.set_paint(colors[i]);
//!     ctx.fill_rect(&Rect::new(0.0, 0.0, WIDTH, HEIGHT));
//! }
//! for _ in 0..3 {
//!     ctx.pop_layer();
//! }
//! ```
//!
//! This single wide tile scene should produce the below rendering:
//!
//! ┌────────────────────────────┌────────────────────────────┌─────────────────────────────
//! │      ──              ───   │       /     /       /     /│        ──────────────      │
//! │  ────            ────      │      /     /       /     / │────────                    │
//! │──           ─────          │     /     /       /     /  │                            │
//! │        ───Red              │    /     /Green  /     /   │           Blue             │
//! │    ────                ──  │   /     /       /     /    │                     ───────│
//! │ ───                ────    │  /     /       /     /     │       ──────────────       │
//! │                  ──        │ /     /       /     /      │───────                     │
//! └────────────────────────────└────────────────────────────└────────────────────────────┘
//!                                                                                         
//! How the scene is scheduled into rounds and draw calls are shown below:
//!
//! ### Round 0
//!
//! In this round, we don't have any preserved slots or slots that we need to sample from. Simply,
//! draw unclipped primitives.
//!
//! ### Draw to texture 0:
//!
//! In Slot N - 1 of texture 0, draw the unclipped green rectangle.
//!
//! Slot N - 1:
//! ┌──────────────────────────────────────────────────────────────────────────────────────┐
//! │       /     /       /     /        /     /       /     /       /     /       /     / │
//! │      /     /       /     /        /     /       /     /       /     /       /     /  │
//! │     /     /       /     /        /     /       /     /       /     /       /     /   │
//! │    /     /       /     /        /     / Green /     /       /     /       /     /    │
//! │   /     /       /     /        /     /       /     /       /     /       /     /     │
//! │  /     /       /     /        /     /       /     /       /     /       /     /      │
//! │ /     /       /     /        /     /       /     /       /     /       /     /       │
//! └──────────────────────────────────────────────────────────────────────────────────────┘
//!
//! ### Draw to texture 1:
//!
//! In Slot N - 2 of texture 1, draw unclipped red rectangle and, in slot N - 1, draw the unclipped
//! blue rectangle.
//!
//! Slot N - 2:
//! ┌──────────────────────────────────────────────────────────────────────────────────────┐
//! │      ──              ───                            ──              ───              │
//! │  ────            ────               ──          ────            ────               ──│
//! │──           ─────               ────          ──           ─────               ────  │
//! │        ─────                ────        Red           ─────                ────      │
//! │    ────                 ────                      ────                 ────          │
//! │ ───                 ────                       ───                 ────              │
//! │                  ───                                            ───                  │
//! └──────────────────────────────────────────────────────────────────────────────────────┘
//! Slot N - 1:
//! ┌──────────────────────────────────────────────────────────────────────────────────────┐
//! │                                           ────────────────────────────────────────── │
//! │───────────────────────────────────────────                                           │
//! │                                                                                      │
//! │                                         Blue                          ───────────────│
//! │                                           ────────────────────────────               │
//! │               ────────────────────────────                                           │
//! │───────────────                                                                       │
//! └──────────────────────────────────────────────────────────────────────────────────────┘
//!
//! ### Round 1
//!
//! At this point, we have three slots that contain our unclipped rectangles. In this round,
//! we start to sample those pixels to apply clipping (texture 1 samples from texture 0 and
//! the render target view samples from texture 1).
//!
//! ### Draw to texture 0:
//!
//! Slot N - 1 of texture 0 contains our unclipped green rectangle. In this draw, we sample
//! the pixels from slot N - 2 from texture 1 to draw the blue rectangle into this slot.
//!
//! Slot N - 1:
//! ┌─────────────────────────────────────────────────────────┌─────────────────────────────
//! │        /     /       /     /       /     /       /     /│        ──────────────      │
//! │       /     /       /     /       /     /       /     / │────────                    │
//! │      /     /       /     /       /     /       /     /  │                            │
//! │     /     /       /  Green      /     /       /     /   │           Blue             │
//! │    /     /       /     /       /     /       /     /    │                     ───────│
//! │   /     /       /     /       /     /       /     /     │       ──────────────       │
//! │  /     /       /     /       /     /       /     /      │───────                     │
//! └─────────────────────────────────────────────────────────└────────────────────────────┘
//!
//! ### Draw to texture 1:
//!
//! Then, into Slot N - 2 of texture 1, which contains our red rectangle, we sample the pixels
//! from slot N - 1 of texture 0 which contain our green and blue rectangles.
//!
//! ┌────────────────────────────┌────────────────────────────┌─────────────────────────────
//! │      ──              ───   │       /     /       /     /│        ──────────────      │
//! │  ────            ────      │      /     /       /     / │────────                    │
//! │──           ─────          │     /     /       /     /  │                            │
//! │        ───Red              │    /     /Green  /     /   │           Blue             │
//! │    ────                ──  │   /     /       /     /    │                     ───────│
//! │ ───                ────    │  /     /       /     /     │       ──────────────       │
//! │                  ──        │ /     /       /     /      │───────                     │
//! └────────────────────────────└────────────────────────────└────────────────────────────┘
//!
//! ### Draw to render target
//!
//! At this point, we can sample the pixels from slot N - 1 of texture 1 to draw the final
//! result.
//!
//! ## Nuances
//!
//! - When there are no clip/blend regions, we can render directly to the final target.
//! - The above example provides an intuitive explanation for how rounds after 3 clip depths
//!   are scheduled. At clip depths 1 and 2, we can draw directly to the final target within a
//!   single round.
//! - Before drawing into any slot, we need to clear it. If all slots can be cleared or are free,
//!   we can use a `LoadOp::Clear` operation. Otherwise, we need to clear the dirty slots using
//!   a fine grained render pass.
//!
//! ## Clip Depths, Textures, and Rendering
//!
//! The relationship between clip depths, textures, and rendering is as follows:
//!
//! 1. For `clip_depth` 1 (conceptually no clipping):
//!    - Direct rendering to the final target (ix=2)
//!
//! 2. For `clip_depth` 2 (conceptually first level of clipping):
//!    - Draw to the odd texture (ix=1)
//!    - Final drawing samples from odd texture to the final target
//!
//! 3. For `clip_depth` 3+ (conceptually second level of clipping and beyond):
//!    - For odd clip depths:
//!      - Draw initially to the odd texture (ix=1)
//!    - For even clip depths:
//!      - Draw initially to the even texture (ix=0)
//!    - Sampling occurs similarly to the above example.
//!
//! Note: The code implementation uses a 1-indexed system where `clip_depth` starts at 1
//! even when there is conceptually no clipping.
//!
//! For more information about this algorithm, see this [Zulip thread].
//!
//! [Zulip thread]: https://xi.zulipchat.com/#narrow/channel/197075-vello/topic/Spatiotemporal.20allocation.20.28hybrid.29/near/513442829

#![expect(
    clippy::cast_possible_truncation,
    reason = "We temporarily ignore those because the casts\
only break in edge cases, and some of them are also only related to conversions from f64 to f32."
)]

use crate::filter::FilterContext;
use crate::scene::{FastPathRect, FastStripCommand, FastStripsPath, StripPathMode};
use crate::{GpuStrip, RenderError, Scene};
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::ops::Range;
use vello_common::coarse::{
    CmdAlphaFill, CmdClipAlphaFill, CmdClipFill, CmdFill, CommandAttrs, LayerKind, MODE_HYBRID,
    Wide, WideTilesBbox,
};
use vello_common::peniko::BlendMode;
use vello_common::render_graph::{LayerId, RenderNodeKind};
use vello_common::strip_generator::StripStorage;
use vello_common::{
    coarse::{Cmd, WideTile},
    encode::EncodedPaint,
    paint::{ImageSource, Paint},
    tile::Tile,
};

// Constants used for bit packing, matching `render_strips.wgsl`
const COLOR_SOURCE_PAYLOAD: u32 = 0;
const COLOR_SOURCE_SLOT: u32 = 1;
const COLOR_SOURCE_BLEND: u32 = 2;

const PAINT_TYPE_SOLID: u32 = 0;
const PAINT_TYPE_IMAGE: u32 = 1;
const PAINT_TYPE_LINEAR_GRADIENT: u32 = 2;
const PAINT_TYPE_RADIAL_GRADIENT: u32 = 3;
const PAINT_TYPE_SWEEP_GRADIENT: u32 = 4;

/// Bit 31 of [`GpuStrip::paint_and_rect_flag`] signals that the strip
/// represents a full rectangle.
const RECT_STRIP_FLAG: u32 = 1 << 31;
/// The threshold of the rectangle size after which a rectangle should be split up
/// into multiple smaller ones.
const LARGE_RECT_SPLIT_THRESHOLD: u16 = 32;

// The sentinel tile index representing the surface.
const SENTINEL_SLOT_IDX: usize = usize::MAX;

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
    /// Render to a layer in the filter atlas.
    FilterLayer(LayerId),
    /// Render to one of the slot textures used for clipping/blending.
    SlotTexture(u8),
}

/// Trait for abstracting the renderer backend from the scheduler.
pub(crate) trait RendererBackend {
    /// Clear specific slots in a texture.
    fn clear_slots(&mut self, texture_index: usize, slots: &[u32]);

    /// Execute a render pass for strips.
    fn render_strips(
        &mut self,
        strips: &[GpuStrip],
        target: StripPassRenderTarget,
        load_op: LoadOp,
    );

    /// Apply filter effects for the given layer after its content has been rendered.
    fn apply_filter(&mut self, layer_id: LayerId);
}

/// Backend agnostic enum that specifies the operation to perform to the output attachment at the
/// start of a render pass:
///  - `LoadOp::Load` is equivalent to `wgpu::LoadOp::Load`
///  - `LoadOp::Clear` is equivalent `wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)`
#[derive(Debug, PartialEq)]
pub(crate) enum LoadOp {
    Load,
    Clear,
}

#[derive(Debug)]
pub(crate) struct Scheduler {
    /// Index of the current round
    round: usize,
    /// Per-tile command offsets.
    cmd_offsets: Vec<usize>,
    /// The total number of slots in each slot texture.
    total_slots: usize,
    /// The slots that are free to use in each slot texture.
    free: [Vec<usize>; 2],
    /// Rounds are enqueued on push clip commands and dequeued on flush.
    rounds_queue: VecDeque<Round>,
    /// A pool of `Round` objects that can be reused, so that we can reduce
    /// the number of allocations.
    round_pool: RoundPool,
    /// The output target for the main rendering operations.
    output_target: StripPassRenderTarget,
}

#[derive(Debug, Default)]
struct RoundPool(Vec<Round>);

impl RoundPool {
    #[inline]
    fn return_to_pool(&mut self, mut round: Round) {
        const MAX_ELEMENTS: usize = 10;

        // Avoid caching too many objects in adversarial scenarios.
        if self.0.len() < MAX_ELEMENTS {
            // Make sure the round is reset if we reuse it in the future.
            round.clear();

            self.0.push(round);
        }
    }

    #[inline]
    fn take_from_pool(&mut self) -> Round {
        self.0.pop().unwrap_or_default()
    }
}

/// A "round" is a coarse scheduling quantum.
///
/// It represents draws in up to three render targets; two for intermediate clip/blend buffers, and
/// the third for the actual render target.
#[derive(Debug, Default)]
struct Round {
    /// Draw calls scheduled into the two slot textures (0, 1) and the final target (2).
    draws: [Draw; 3],
    /// Slots that will be freed after drawing into the two slot textures [0, 1].
    free: [Vec<usize>; 2],
    /// Slots that will be cleared in the two slot textures (0, 1) before drawing this round.
    clear: [Vec<u32>; 2],
}

impl Round {
    #[inline]
    fn clear(&mut self) {
        for draw in &mut self.draws {
            draw.clear();
        }

        for free in &mut self.free {
            free.clear();
        }

        for clear in &mut self.clear {
            clear.clear();
        }
    }
}

/// Reusable state used by the scheduler. We are holding this separately instead of integrating
/// it into `Scheduler` because it avoids some borrowing issues.
#[derive(Debug, Default)]
pub(crate) struct SchedulerState {
    /// The state of the current wide tile that is being processed.
    tile_state: TileState,
    /// The maximum round that has been allocated for any drawing operation
    /// in any wide tile for the current filter/root layer.
    ///
    /// This value should be reset to the base round every time a new layer is processed.
    max_round: usize,
}

/// State for a single wide tile.
#[derive(Debug, Default)]
struct TileState {
    stack: Vec<TileEl>,
}

impl TileState {
    fn clear(&mut self) {
        self.stack.clear();
    }
}

/// A claimed slot into one of the two slot textures (0, 1).
#[derive(Debug, Copy, Clone)]
enum ClaimedSlot {
    Texture0(usize),
    Texture1(usize),
}

impl ClaimedSlot {
    fn get_idx(&self) -> usize {
        match self {
            Self::Texture0(idx) => *idx,
            Self::Texture1(idx) => *idx,
        }
    }

    fn get_texture(&self) -> usize {
        match self {
            Self::Texture0(_) => 0,
            Self::Texture1(_) => 1,
        }
    }
}

/// `TemporarySlot` is a container that tracks whether a tile's temporary slot is live/invalid, or
/// whether a temporary slot is unnecessary. A temporary slot is an owned slot that resides in the
/// slot texture for the next layer.
#[derive(Clone, Copy, Debug)]
enum TemporarySlot {
    None,
    Valid(ClaimedSlot),
    Invalid(ClaimedSlot),
}

impl TemporarySlot {
    fn invalidate(&mut self) {
        if let Self::Valid(slot) = self {
            *self = Self::Invalid(*slot);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TileEl {
    /// `dest_slot` represents the final location of where the contents of this tile will end up.
    /// This slot will be claimed on the texture at `layer depth % 2`.
    dest_slot: ClaimedSlot,
    /// Temporary slot that is only required if the layer above this tile will be blended into this
    /// tile. This slot has the invariant that it must reside on the other slot texture to
    /// `dest_slot`. Thus when the layers are blended together `temporary_slot` can be read from
    /// to read the pixel data for the destination. The blend result writes to `dest_slot`.
    temporary_slot: TemporarySlot,
    round: usize,
    opacity: f32,
}

impl TileEl {
    fn get_draw_texture(&self, buffer_depth: usize) -> usize {
        if buffer_depth > 1 {
            if let TemporarySlot::Valid(temp_slot) = self.temporary_slot {
                temp_slot.get_texture()
            } else {
                self.dest_slot.get_texture()
            }
        } else {
            // Surface texture identifier
            2
        }
    }
}

#[derive(Debug, Default)]
struct Draw(Vec<GpuStrip>);

impl Draw {
    #[inline(always)]
    fn push(&mut self, gpu_strip: GpuStrip) {
        self.0.push(gpu_strip);
    }

    fn clear(&mut self) {
        self.0.clear();
    }
}

impl Scheduler {
    pub(crate) fn new(total_slots: usize) -> Self {
        let free0: Vec<_> = (0..total_slots).collect();
        let free1 = free0.clone();
        let free: [Vec<usize>; 2] = [free0, free1];
        Self {
            round: 0,
            cmd_offsets: Vec::new(),
            total_slots,
            free,
            rounds_queue: VecDeque::new(),
            round_pool: RoundPool::default(),
            output_target: StripPassRenderTarget::Root(RootRenderTarget::UserSurface),
        }
    }

    fn claim_free_slot<R: RendererBackend>(
        &mut self,
        texture: usize,
        renderer: &mut R,
    ) -> Result<ClaimedSlot, RenderError> {
        while self.free[texture].is_empty() {
            if self.rounds_queue.is_empty() {
                return Err(RenderError::SlotsExhausted);
            }
            self.flush(renderer);
        }

        let slot_ix = self.free[texture].pop().unwrap();

        let slot = match texture {
            0 => ClaimedSlot::Texture0(slot_ix),
            1 => ClaimedSlot::Texture1(slot_ix),
            _ => panic!("invalid slot texture"),
        };

        // Since the slot was claimed, it needs to be cleared in the given round.
        let round = self.get_round(self.round);
        round.clear[slot.get_texture()].push(slot.get_idx() as u32);

        Ok(slot)
    }

    // Note: This is roughly equivalent to `rasterize_with_filters` in vello_cpu.
    // However, unlike `vello_cpu` we have one combined method that handles both, the
    // filter and no-filter case.
    /// Schedule and render the scene.
    pub(crate) fn do_scene<R: RendererBackend>(
        &mut self,
        state: &mut SchedulerState,
        renderer: &mut R,
        scene: &Scene,
        root_output_target: RootRenderTarget,
        paint_idxs: &[u32],
        filter_context: &FilterContext,
        encoded_paints: &[EncodedPaint],
    ) -> Result<(), RenderError> {
        for node_id in scene.render_graph.execution_order() {
            let node = &scene.render_graph.nodes[node_id];

            if node.is_empty() {
                continue;
            }

            match &node.kind {
                RenderNodeKind::FilterLayer {
                    layer_id,
                    wtile_bbox,
                    ..
                } => {
                    self.output_target = StripPassRenderTarget::FilterLayer(*layer_id);
                    self.process_filter_node(
                        state,
                        renderer,
                        scene,
                        *layer_id,
                        *wtile_bbox,
                        paint_idxs,
                        filter_context,
                        encoded_paints,
                    )?;
                }
                RenderNodeKind::RootLayer { .. } => {
                    self.output_target = StripPassRenderTarget::Root(root_output_target);
                    self.process_root_node(
                        state,
                        renderer,
                        scene,
                        paint_idxs,
                        filter_context,
                        encoded_paints,
                    )?;
                }
            }

            while !self.rounds_queue.is_empty() {
                self.flush(renderer);
            }

            // This will actually apply the filter and store the filtered texture in the image
            // atlas
            if let StripPassRenderTarget::FilterLayer(layer_id) = self.output_target {
                renderer.apply_filter(layer_id);
            }
        }

        // Restore state to reuse allocations.
        self.round = 0;
        #[cfg(debug_assertions)]
        {
            for i in 0..self.total_slots {
                debug_assert!(self.free[0].contains(&i), "free[0] is missing slot {i}");
                debug_assert!(self.free[1].contains(&i), "free[1] is missing slot {i}");
            }
        }
        debug_assert!(self.rounds_queue.is_empty(), "rounds_queue is not empty");

        Ok(())
    }

    /// Process the root layer node of the render graph.
    ///
    /// Depending on [`StripPathMode`], the scene is processed in one of three ways:
    ///
    /// - **`FastOnly`**: All paths were rendered directly into the fast strips buffer
    ///   (no layers were ever pushed). We draw the whole scene using a single draw call
    ///   by uploading all strips directly to the GPU.
    ///
    /// - **`CoarseOnly`**: All paths went through coarse rasterization (non-default
    ///   blending was requested, so in case there was a `push_layer` call the fast path
    ///   was flushed retroactively). We use the normal scheduling approach for all commands
    ///   in the wide tile.
    ///
    /// - **`Interleaved`**: The scene mixes fast path strips with coarse-rasterized
    ///   layers. We still need to go through the usual scheduling process, but fast path
    ///   strips can be processed much faster now since they haven't gone through coarse
    ///   rasterization.
    fn process_root_node<R: RendererBackend>(
        &mut self,
        state: &mut SchedulerState,
        renderer: &mut R,
        scene: &Scene,
        paint_idxs: &[u32],
        filter_context: &FilterContext,
        encoded_paints: &[EncodedPaint],
    ) -> Result<(), RenderError> {
        let wide = &scene.wide;
        let rows = wide.height_tiles();
        let cols = wide.width_tiles();
        let num_tiles = (rows * cols) as usize;
        // If we processed any filter layers previously, their maximum round should not leak
        // into the root node.
        state.max_round = self.round;

        // A bit hacky, but we need this since we still need mutable access to
        // self when processing everything.
        let mut cmd_offsets = core::mem::take(&mut self.cmd_offsets);
        cmd_offsets.clear();
        cmd_offsets.resize(num_tiles, 0);

        match scene.strip_path_mode {
            StripPathMode::FastOnly => {
                // We only have strips.
                self.push_direct_strips(
                    scene,
                    0..scene.fast_strips_buffer.commands.len(),
                    state.max_round,
                    paint_idxs,
                    encoded_paints,
                );
            }
            StripPathMode::CoarseOnly => {
                // We only have coarse-rasterized paths.
                self.process_coarse_batch(
                    state,
                    renderer,
                    wide,
                    rows,
                    cols,
                    &mut cmd_offsets,
                    paint_idxs,
                    filter_context,
                    encoded_paints,
                )?;
            }
            StripPathMode::Interleaved => {
                // Alternate fast strip batches with coarse-rasterized layer batches.
                let mut prev_split = 0;

                for &split in &scene.coarse_batch_splits {
                    // First process any direct strips.
                    if prev_split < split {
                        self.push_direct_strips(
                            scene,
                            prev_split..split,
                            state.max_round,
                            paint_idxs,
                            encoded_paints,
                        );
                    }

                    // Then process the coarse batch.
                    self.process_coarse_batch(
                        state,
                        renderer,
                        wide,
                        rows,
                        cols,
                        &mut cmd_offsets,
                        paint_idxs,
                        filter_context,
                        encoded_paints,
                    )?;

                    prev_split = split;
                }

                // Handle the last batch of fast strips, which isn't explicitly delimited in the
                // scene.
                let tail_end = scene.fast_strips_buffer.commands.len();
                if prev_split < tail_end {
                    self.push_direct_strips(
                        scene,
                        prev_split..tail_end,
                        state.max_round,
                        paint_idxs,
                        encoded_paints,
                    );
                }
            }
        }

        // Put the allocation back.
        self.cmd_offsets = cmd_offsets;

        Ok(())
    }

    /// Process a filter node in the render graph.
    fn process_filter_node<R: RendererBackend>(
        &mut self,
        state: &mut SchedulerState,
        renderer: &mut R,
        scene: &Scene,
        layer_id: LayerId,
        wtile_bbox: WideTilesBbox,
        paint_idxs: &[u32],
        filter_context: &FilterContext,
        encoded_paints: &[EncodedPaint],
    ) -> Result<(), RenderError> {
        // The maximum layer of other filter nodes should not leak into new filter nodes, hence
        // we need to reset it.
        state.max_round = self.round;

        for y in wtile_bbox.y0()..wtile_bbox.y1() {
            for x in wtile_bbox.x0()..wtile_bbox.x1() {
                let wide_tile = scene.wide.get(x, y);
                let wide_tile_x = x * WideTile::WIDTH;
                let wide_tile_y = y * Tile::HEIGHT;

                state.tile_state.clear();
                self.initialize_tile_state(
                    &mut state.tile_state,
                    wide_tile,
                    wide_tile_x,
                    wide_tile_y,
                    self.round,
                    encoded_paints,
                    paint_idxs,
                    // Background is only ever applied in the root layer.
                    false,
                );

                let Some(ranges) = wide_tile.layer_cmd_ranges.get(&layer_id) else {
                    continue;
                };

                // TODO: Use enum instead of `wrap_surface`?
                let wrap_surface = matches!(
                    wide_tile.cmds[ranges.full_range.start],
                    Cmd::PushBuf(_, true)
                );

                self.do_tile(
                    state,
                    renderer,
                    encoded_paints,
                    wide_tile_x,
                    wide_tile_y,
                    wide_tile,
                    ranges.render_range.clone(),
                    wrap_surface,
                    paint_idxs,
                    &scene.wide.attrs,
                    filter_context,
                )?;
            }
        }

        Ok(())
    }

    /// Generate `GpuStrips` for a range of direct commands and append them
    /// directly into the current round's surface draw array.
    fn push_direct_strips(
        &mut self,
        scene: &Scene,
        range: Range<usize>,
        round: usize,
        paint_idxs: &[u32],
        encoded_paints: &[EncodedPaint],
    ) {
        let strip_storage = scene.strip_storage.borrow();
        // Always choose the draw of the final surface, since direct strips are only ever
        // rendered to the final surface.
        let draw = self.draw_mut(round, 2);

        for cmd in &scene.fast_strips_buffer.commands[range] {
            match cmd {
                FastStripCommand::Path(path) => {
                    generate_gpu_strips_for_fast_path(
                        path,
                        &strip_storage,
                        scene,
                        encoded_paints,
                        paint_idxs,
                        &mut draw.0,
                    );
                }
                FastStripCommand::Rect(r) => {
                    pack_rectangle_into_gpu(r, encoded_paints, paint_idxs, &mut draw.0);
                }
            }
        }
    }

    /// Process one batch of coarse-rasterized wide tile commands.
    ///
    /// Iterates over all wide tiles, processing commands from each tile's
    /// current offset up to the next `BatchEnd` marker (or the end of the
    /// command list).
    fn process_coarse_batch<R: RendererBackend>(
        &mut self,
        state: &mut SchedulerState,
        renderer: &mut R,
        wide: &Wide<MODE_HYBRID>,
        rows: u16,
        cols: u16,
        cmd_offsets: &mut [usize],
        paint_idxs: &[u32],
        filter_context: &FilterContext,
        encoded_paints: &[EncodedPaint],
    ) -> Result<(), RenderError> {
        for row in 0..rows {
            for col in 0..cols {
                let idx = (row * cols + col) as usize;
                let tile = wide.get(col, row);
                let start_offset = cmd_offsets[idx];

                // Note that we are explicitly checking > instead of >=.
                // The reason is that it can happen the tile has no commands but still has a background,
                // in which case we still need to do the painting of the background
                if start_offset > tile.cmds.len() {
                    continue;
                }

                let tile_x = col * WideTile::WIDTH;
                let tile_y = row * Tile::HEIGHT;

                // We only must paint the background if we are processing the wide tile for the
                // first time (i.e. the start offset is 0).
                let paint_bg = start_offset == 0;

                state.tile_state.clear();

                // This will also paint the background, if necessary.
                self.initialize_tile_state(
                    &mut state.tile_state,
                    tile,
                    tile_x,
                    tile_y,
                    state.max_round,
                    encoded_paints,
                    paint_idxs,
                    paint_bg,
                );
                let end = self.do_tile(
                    state,
                    renderer,
                    encoded_paints,
                    tile_x,
                    tile_y,
                    tile,
                    start_offset..tile.cmds.len(),
                    tile.surface_is_blend_target(),
                    paint_idxs,
                    &wide.attrs,
                    filter_context,
                )?;

                // Advance past the `BatchEnd` marker (if present).
                cmd_offsets[idx] = (end + 1).min(tile.cmds.len());

                state.max_round = state.max_round.max(state.tile_state.stack[0].round);
            }
        }

        Ok(())
    }

    /// Flush one round.
    ///
    /// The rounds queue must not be empty.
    fn flush<R: RendererBackend>(&mut self, renderer: &mut R) {
        let round = self.rounds_queue.pop_front().unwrap();
        for (i, draw) in round.draws.iter().enumerate() {
            #[cfg(debug_assertions)]
            {
                // This is an expensive O(n²) debug only check that enforces that there are no
                // duplicate slots in the clear list. Duplicates signal an inefficiency in
                // scheduling.
                if i != 2 {
                    for (idx, &slot) in round.clear[i].iter().enumerate() {
                        assert!(
                            !round.clear[i][..idx].contains(&slot),
                            "Duplicate slot {slot} found in round.clear[{i}]",
                        );
                    }
                }
            }
            let target = if i == 2 {
                self.output_target
            } else {
                StripPassRenderTarget::SlotTexture(i as u8)
            };

            let load = {
                if i == 2 {
                    // We're rendering to the view, don't clear.
                    LoadOp::Load
                } else if round.clear[i].len() + self.free[i].len() == self.total_slots {
                    // All slots are either unoccupied or need to be cleared. Simply clear the slots
                    // via a load operation.
                    LoadOp::Clear
                } else {
                    // Some slots need to be preserved, so only clear the dirty slots.
                    renderer.clear_slots(i, round.clear[i].as_slice());
                    LoadOp::Load
                }
            };

            if draw.0.is_empty() {
                if load == LoadOp::Clear {
                    // There are no strips to render, so `render_strips` will not run and won't clear
                    // the texture. We still have slots to clear this round so explicitly clear
                    // them.
                    renderer.clear_slots(i, round.clear[i].as_slice());
                }
                continue;
            }

            renderer.render_strips(&draw.0, target, load);
        }
        for i in 0..2 {
            self.free[i].extend(&round.free[i]);
        }
        self.round += 1;

        self.round_pool.return_to_pool(round);
    }

    // Find the appropriate draw call for rendering.
    #[inline(always)]
    fn draw_mut(&mut self, el_round: usize, texture_idx: usize) -> &mut Draw {
        &mut self.get_round(el_round).draws[texture_idx]
    }

    // Find the appropriate round for rendering.
    #[inline(always)]
    fn get_round(&mut self, el_round: usize) -> &mut Round {
        let rel_round = el_round.saturating_sub(self.round);
        if self.rounds_queue.len() == rel_round {
            let round = self.round_pool.take_from_pool();
            self.rounds_queue.push_back(round);
        }
        &mut self.rounds_queue[rel_round]
    }

    /// Render the tile's background color (set by overdraw elimination) to the
    /// surface. Returns `true` if a strip was emitted.
    fn paint_tile_bg(
        &mut self,
        tile: &WideTile<MODE_HYBRID>,
        wide_tile_x: u16,
        wide_tile_y: u16,
        encoded_paints: &[EncodedPaint],
        idxs: &[u32],
    ) {
        let bg = tile.bg.as_premul_rgba8().to_u32();

        if has_non_zero_alpha(bg) {
            let (payload, paint) = Self::process_paint(
                &Paint::Solid(tile.bg),
                encoded_paints,
                (wide_tile_x, wide_tile_y),
                idxs,
            );

            let draw = self.draw_mut(self.round, 2);
            draw.push(
                GpuStripBuilder::at_surface(wide_tile_x, wide_tile_y, WideTile::WIDTH)
                    .paint(payload, paint),
            );
        }
    }

    fn initialize_tile_state(
        &mut self,
        tile_state: &mut TileState,
        tile: &WideTile<MODE_HYBRID>,
        wide_tile_x: u16,
        wide_tile_y: u16,
        initial_round: usize,
        encoded_paints: &[EncodedPaint],
        idxs: &[u32],
        paint_bg: bool,
    ) {
        // Sentinel `TileEl` to indicate the end of the stack where we draw all
        // commands to the final target.
        tile_state.stack.push(TileEl {
            // Note that the sentinel state doesn't actually do any rendering to texture 0,
            // we just need to put _something_ there.
            dest_slot: ClaimedSlot::Texture0(SENTINEL_SLOT_IDX),
            temporary_slot: TemporarySlot::None,
            round: initial_round,
            opacity: 1.,
        });

        if paint_bg {
            self.paint_tile_bg(tile, wide_tile_x, wide_tile_y, encoded_paints, idxs);
        }
    }

    /// Iterates over wide tile commands in `cmd_range` and schedules them for rendering.
    ///
    /// Returns the absolute command index where processing stopped (either triggered by
    /// hitting a batch end, or the end of the current filter layer).
    fn do_tile<R: RendererBackend>(
        &mut self,
        state: &mut SchedulerState,
        renderer: &mut R,
        encoded_paints: &[EncodedPaint],
        wide_tile_x: u16,
        wide_tile_y: u16,
        tile: &WideTile<MODE_HYBRID>,
        cmd_range: Range<usize>,
        surface_is_blend_target: bool,
        paint_idxs: &[u32],
        attrs: &CommandAttrs,
        filter_context: &FilterContext,
    ) -> Result<usize, RenderError> {
        // What is going on with the `surface_is_blend_target` and `is_blend_target` variables in
        // `PushBuf`?
        // For blending of two layers (with a non-default blend mode) to work in vello_hybrid,
        // the two slots being blended must be on the same texture, since we need to be able to
        // read from the destination to do blending, while the target we write the composited result
        // into lives in the other texture. Therefore, we create an additional temporary
        // slot in the other texture than the main destination slot.
        // See https://xi.zulipchat.com/#narrow/channel/197075-vello/topic/Hybrid.20Blending/with/536597802
        // and https://github.com/linebender/vello/pull/1155 for some information on
        // how blending works.
        //
        // Note however that all of this is not necessary when we have layers (for example for
        // clip paths or opacity layers) with normal source-over blending, since the GPU automatically
        // takes care of doing blending in this case. Determining which layers are the target of
        // a blend operation has already happened during coarse rasterization, so here we just
        // need to pass on the boolean flags.
        //
        // `surface_is_blend_target` is needed because the main target surface can obviously also
        // end up being the destination of a blending operation. Since that surface is provided
        // by the user, there is no way we can read from it, which is required for blending. Therefore,
        // in this case we need to "wrap" _everything_ into a push/pop layer operation.

        if surface_is_blend_target {
            self.do_push_buf(state, renderer, true)?;
        }

        let mut cmd_idx = cmd_range.start;
        while cmd_idx < cmd_range.end {
            let cmd = &tile.cmds[cmd_idx];
            match cmd {
                Cmd::Fill(fill) => {
                    self.do_fill(
                        state,
                        encoded_paints,
                        fill,
                        paint_idxs,
                        wide_tile_x,
                        wide_tile_y,
                        attrs,
                    );
                }
                Cmd::AlphaFill(alpha_fill) => {
                    self.do_alpha_fill(
                        state,
                        encoded_paints,
                        alpha_fill,
                        paint_idxs,
                        wide_tile_x,
                        wide_tile_y,
                        attrs,
                    );
                }
                // This is roughly equivalent to `process_layer_tile` in vello_cpu.
                Cmd::PushBuf(LayerKind::Filtered(child_layer_id), _) => {
                    let filtered_ranges = tile.layer_cmd_ranges.get(child_layer_id).unwrap();
                    // If the filter layer was zero-sized, no texture was allocated.
                    // Skip its entire range.
                    let Some(filter_textures) = filter_context.filter_textures.get(child_layer_id)
                    else {
                        cmd_idx = filtered_ranges.full_range.end;
                        continue;
                    };

                    // TODO: This push buf is wasteful (if no blend-mode is associated
                    // with the layer), as the filter layer
                    // itself is already rendered into a new texture, so we are essentially
                    // pushing twice, meaning that we will always render the result
                    // into slot textures instead of the final canvas, even though
                    // this should only be necessary if blending is enabled. However,
                    // skipping while not breaking the `pop_buf` logic is not trivial,
                    // therefore we leave it this way for now. Removing this should
                    // give us a good speed boost, but probably only worth revisiting
                    // once we have made coarse rasterization simpler.
                    // Important: If we change this behavior, we have to update
                    // `wrap_surface` for filter nodes.
                    self.do_push_buf(state, renderer, false)?;

                    let copy_from_filter_layer =
                        |scheduler: &mut Self, state: &mut SchedulerState| {
                            let cmd = CmdFill {
                                x: 0,
                                width: WideTile::WIDTH,
                                attrs_idx: 0,
                            };
                            let encoded_paint = encoded_paints
                                .get(filter_textures.paint_idx as usize)
                                .expect("filter paint not found");
                            let paint_tex_idx = paint_idxs[filter_textures.paint_idx as usize];
                            let (payload, paint) = Self::process_encoded_paint(
                                encoded_paint,
                                paint_tex_idx,
                                wide_tile_x,
                                wide_tile_y,
                            );
                            scheduler.do_fill_with(
                                state,
                                &cmd,
                                wide_tile_x,
                                wide_tile_y,
                                payload,
                                paint,
                            );
                        };

                    // Check what comes after the filtered layer push to determine clipping state
                    match tile.cmds.get(cmd_idx + 1) {
                        // Zero-clip region: tile is completely outside the clip path.
                        // The layer was already rendered for filtering, but we skip compositing
                        // since this tile is entirely clipped out.
                        // (PushZeroClip only appears for clipped filter layers)
                        // See https://github.com/linebender/vello/pull/1541/ for why we
                        // add the ID check.
                        Some(Cmd::PushZeroClip(id)) if *id == *child_layer_id => {
                            // If we have a zero-clip, it means that the whole layer should not be drawn.
                            // Therefore, we want to skip to the very end so that only `PopBuf` will
                            // be run. Therefore, we jump to `filtered_ranges.full_range.end - 1`.
                            cmd_idx = filtered_ranges.full_range.end - 1;
                            continue;
                        }
                        // Partial clip: push the clip buffer, then composite the filtered layer
                        Some(Cmd::PushBuf(LayerKind::Clip(id), is_blend_dest))
                            if *id == *child_layer_id =>
                        {
                            self.do_push_buf(state, renderer, *is_blend_dest)?;
                            cmd_idx += 1;
                            copy_from_filter_layer(self, state);
                        }
                        // No clip or fully inside clip: composite the filtered layer directly
                        _ => {
                            copy_from_filter_layer(self, state);
                        }
                    }

                    cmd_idx = filtered_ranges.render_range.end.max(cmd_idx + 1);
                    continue;
                }
                Cmd::PushBuf(_, is_blend_target) => {
                    self.do_push_buf(state, renderer, *is_blend_target)?;
                }
                Cmd::PopBuf => {
                    self.do_pop_buf(state);
                }
                Cmd::ClipFill(clip_fill) => {
                    self.do_clip_fill(state, wide_tile_x, wide_tile_y, clip_fill);
                }
                Cmd::ClipStrip(clip_alpha_fill) => {
                    self.do_clip_strip(state, wide_tile_x, wide_tile_y, clip_alpha_fill, attrs);
                }
                Cmd::Opacity(opacity) => {
                    self.do_opacity(state, *opacity);
                }
                Cmd::Blend(mode) => {
                    self.do_blend(state, wide_tile_x, wide_tile_y, mode);
                }
                Cmd::Filter(_, _) => {}
                Cmd::BatchEnd => {
                    return Ok(cmd_idx);
                }
                _ => unreachable!(),
            }

            cmd_idx += 1;
        }

        if surface_is_blend_target {
            // Simple source-over compositing into the final render target.
            self.do_blend(state, wide_tile_x, wide_tile_y, &BlendMode::default());
            self.do_pop_buf(state);
        }

        Ok(cmd_range.end)
    }

    #[inline]
    fn do_push_buf<R: RendererBackend>(
        &mut self,
        state: &mut SchedulerState,
        renderer: &mut R,
        needs_temporary_slot: bool,
    ) -> Result<(), RenderError> {
        let depth = state.tile_state.stack.len();

        // `wgpu` does not allow reading/writing from the same slot texture. This means
        // that to represent the binary function `Blend(src_tile, dest_tile)` we need
        // both slots being blended to be on the same texture. This is accomplished as
        // the lower destination tile (`nos`) maintains a proxy slot in the texture
        // above. Blending consumes both the top-of-stack (`tos`) slot, the `src_tile`,
        // and blends it with `nos.temporary_slot` (`dest_tile`). The results of the
        // blend are written to `nos.dest_slot`.
        //
        // This works well _once_. Unfortunately, to blend to `nos` again we need to
        // copy the contents of `dest_slot` back to `temporary_slot` such that it can be
        // composited again. Without this, the tile will not accumulate blends, and
        // instead will have repeated blends clobber the prior blends. Hence, if a
        // buffer is being pushed that will be blended back to `tos`, copy contents from
        // `tos.dest_slot` to `tos.temporary_slot` ready for future blending.
        {
            let tos: &mut TileEl = state.tile_state.stack.last_mut().unwrap();
            if let TemporarySlot::Invalid(temp_slot) = tos.temporary_slot {
                let next_round = depth.is_multiple_of(2);
                let el_round = tos.round + usize::from(next_round);
                let draw = self.draw_mut(el_round, temp_slot.get_texture());
                draw.push(
                    GpuStripBuilder::at_slot(temp_slot.get_idx(), 0, WideTile::WIDTH)
                        .copy_from_slot(tos.dest_slot.get_idx(), 0xFF),
                );

                tos.temporary_slot = TemporarySlot::Valid(temp_slot);
                // Signal when this tile will be ready to use for future blend/pop/clip
                // operations.
                tos.round = el_round + 1;

                // First, we clear the temp slot. THEN, in the same round, we copy the data
                // from dest slot to temp slot. THEN, in the round after that, we clear
                // dest slot, so that in a future blending operation, we can store
                // results into dest slot again.
                let dest_slot = tos.dest_slot;
                let round = self.get_round(el_round);
                round.clear[temp_slot.get_texture()].push(temp_slot.get_idx() as u32);
                debug_assert_ne!(
                    dest_slot.get_idx(),
                    SENTINEL_SLOT_IDX,
                    "surface cannot be read"
                );
                let round1 = self.get_round(tos.round);
                round1.clear[dest_slot.get_texture()].push(dest_slot.get_idx() as u32);
            }
        }

        // Push a new tile.
        let ix = depth % 2;
        let slot = self.claim_free_slot(ix, renderer)?;
        let temporary_slot = if needs_temporary_slot {
            let temp_slot = self.claim_free_slot((ix + 1) % 2, renderer)?;
            debug_assert_ne!(
                slot.get_texture(),
                temp_slot.get_texture(),
                "slot and temporary slot must be on opposite textures."
            );
            TemporarySlot::Valid(temp_slot)
        } else {
            TemporarySlot::None
        };

        state.tile_state.stack.push(TileEl {
            dest_slot: slot,
            temporary_slot,
            round: self.round,
            opacity: 1.,
        });

        Ok(())
    }

    #[inline]
    fn do_pop_buf(&mut self, state: &mut SchedulerState) {
        let depth = state.tile_state.stack.len();

        let tos = state.tile_state.stack.pop().unwrap();
        let nos = state.tile_state.stack.last_mut().unwrap();
        let next_round = depth.is_multiple_of(2) && depth > 2;
        let round = nos.round.max(tos.round + usize::from(next_round));
        // Why do we have to need to change the round here? Let's assume that we are drawing 3
        // nested clip paths. The sequence of commands might look as follows:
        // PushBuf, Fill, PushBuf, Fill, PushBuf, Fill, ClipFill, PopBuf, ClipFill, PopBuf,
        // ClipFill, PopBuf.
        // When executing the first 6 commands, we will happily schedule everything in the
        // first round, since there are no dependencies yet; the contents of each clip
        // layer can be drawn independently. However, upon executing the first ClipFill
        // command, we realize that we need another round: The first fill used texture 1,
        // the second fill used texture 0, and the third fill texture 1 again. Since we
        // cannot read from texture 1 twice within a single round, we need to schedule the
        // `PopBuf` operation for round 1. All subsequent operations that depend on this
        // result must therefore also execute in a later round, and this is achieved by
        // updating `nos` with the new round.
        nos.round = round;
        // free slot after draw
        debug_assert!(round >= self.round, "round must be after current round");
        debug_assert!(
            round - self.round < self.rounds_queue.len(),
            "round must be in queue"
        );
        // Since we pop the buffer, the slot is not needed anymore. Thus, mark it to be
        // freed after the round so that it can be reused in the future.
        self.rounds_queue[round - self.round].free[tos.dest_slot.get_texture()]
            .push(tos.dest_slot.get_idx());
        // If a TileEl was not used for blending the temporary slot may still be in use
        // and needs to be cleared.
        if let TemporarySlot::Valid(temp_slot) | TemporarySlot::Invalid(temp_slot) =
            tos.temporary_slot
        {
            self.rounds_queue[round - self.round].free[temp_slot.get_texture()]
                .push(temp_slot.get_idx());
        }
    }

    #[inline]
    fn do_blend(
        &mut self,
        state: &mut SchedulerState,
        wide_tile_x: u16,
        wide_tile_y: u16,
        mode: &BlendMode,
    ) {
        let depth = state.tile_state.stack.len();
        let tos = state.tile_state.stack.last().unwrap();
        let nos = &state.tile_state.stack[state.tile_state.stack.len() - 2];

        let next_round: bool = depth.is_multiple_of(2) && depth > 2;
        let round = nos.round.max(tos.round + usize::from(next_round));

        let draw = self.draw_mut(
            round,
            if depth <= 2 {
                2
            } else {
                nos.dest_slot.get_texture()
            },
        );

        let gpu_strip_builder = if depth <= 2 {
            GpuStripBuilder::at_surface(wide_tile_x, wide_tile_y, WideTile::WIDTH)
        } else {
            GpuStripBuilder::at_slot(nos.dest_slot.get_idx(), 0, WideTile::WIDTH)
        };

        if let TemporarySlot::Valid(temp_slot) = nos.temporary_slot {
            let opacity_u8 = (tos.opacity * 255.0) as u8;
            let mix_mode = mode.mix as u8;
            let compose_mode = mode.compose as u8;

            draw.push(gpu_strip_builder.blend(
                tos.dest_slot.get_idx(),
                temp_slot.get_idx(),
                opacity_u8,
                mix_mode,
                compose_mode,
            ));
            // Invalidate the temporary slot after use
            let nos_ptr = state.tile_state.stack.len() - 2;
            state.tile_state.stack[nos_ptr].temporary_slot.invalidate();
        } else {
            debug_assert_eq!(
                *mode,
                BlendMode::default(),
                "code path only for default src-over compositing, {mode:?}"
            );

            // Note that despite the slightly misleading name `copy_from_slot`, this will
            // actually perform src-over compositing instead of overriding the colors
            // in the destination (since the render strips pipeline uses
            // `BlendState::PREMULTIPLIED_ALPHA_BLENDING`). This is the whole reason
            // why for default blend modes, we don't need to rely on temporary slots
            // to achieve blending.
            draw.push(
                gpu_strip_builder
                    .copy_from_slot(tos.dest_slot.get_idx(), (tos.opacity * 255.0) as u8),
            );
        }
    }

    #[inline]
    fn do_alpha_fill(
        &mut self,
        state: &mut SchedulerState,
        encoded_paints: &[EncodedPaint],
        cmd: &CmdAlphaFill,
        paint_idxs: &[u32],
        wide_tile_x: u16,
        wide_tile_y: u16,
        attrs: &CommandAttrs,
    ) {
        let depth = state.tile_state.stack.len();

        let el = state.tile_state.stack.last_mut().unwrap();
        let draw = self.draw_mut(el.round, el.get_draw_texture(depth));

        let fill_attrs = &attrs.fill[cmd.attrs_idx as usize];
        let alpha_idx = fill_attrs.alpha_idx(cmd.alpha_offset);
        let col_idx = alpha_idx / u32::from(Tile::HEIGHT);
        let (scene_strip_x, scene_strip_y) = (wide_tile_x + cmd.x, wide_tile_y);
        let (payload, paint) = Self::process_paint(
            &fill_attrs.paint,
            encoded_paints,
            (scene_strip_x, scene_strip_y),
            paint_idxs,
        );

        let gpu_strip_builder = if depth == 1 {
            GpuStripBuilder::at_surface(scene_strip_x, scene_strip_y, cmd.width)
        } else {
            let slot_idx = if let TemporarySlot::Valid(temp_slot) = el.temporary_slot {
                temp_slot.get_idx()
            } else {
                el.dest_slot.get_idx()
            };
            GpuStripBuilder::at_slot(slot_idx, cmd.x, cmd.width)
        };

        draw.push(
            gpu_strip_builder
                .with_sparse(cmd.width, col_idx)
                .paint(payload, paint),
        );
    }

    #[inline]
    fn do_fill(
        &mut self,
        state: &mut SchedulerState,
        encoded_paints: &[EncodedPaint],
        cmd: &CmdFill,
        paint_idxs: &[u32],
        wide_tile_x: u16,
        wide_tile_y: u16,
        attrs: &CommandAttrs,
    ) {
        let fill_attrs = &attrs.fill[cmd.attrs_idx as usize];
        let (scene_strip_x, scene_strip_y) = (wide_tile_x + cmd.x, wide_tile_y);
        let (payload, paint) = Self::process_paint(
            &fill_attrs.paint,
            encoded_paints,
            (scene_strip_x, scene_strip_y),
            paint_idxs,
        );

        self.do_fill_with(state, cmd, scene_strip_x, scene_strip_y, payload, paint);
    }

    #[inline]
    fn do_fill_with(
        &mut self,
        state: &mut SchedulerState,
        cmd: &CmdFill,
        scene_strip_x: u16,
        scene_strip_y: u16,
        payload: u32,
        paint: u32,
    ) {
        let depth = state.tile_state.stack.len();

        let el = state.tile_state.stack.last_mut().unwrap();
        let draw = self.draw_mut(el.round, el.get_draw_texture(depth));

        let gpu_strip_builder = if depth == 1 {
            GpuStripBuilder::at_surface(scene_strip_x, scene_strip_y, cmd.width)
        } else {
            let slot_idx = if let TemporarySlot::Valid(temp_slot) = el.temporary_slot {
                temp_slot.get_idx()
            } else {
                el.dest_slot.get_idx()
            };
            GpuStripBuilder::at_slot(slot_idx, cmd.x, cmd.width)
        };

        draw.push(gpu_strip_builder.paint(payload, paint));
    }

    #[inline]
    fn do_opacity(&self, state: &mut SchedulerState, opacity: f32) {
        state.tile_state.stack.last_mut().unwrap().opacity = opacity;
    }

    fn do_clip_fill(
        &mut self,
        state: &mut SchedulerState,
        wide_tile_x: u16,
        wide_tile_y: u16,
        cmd: &CmdClipFill,
    ) {
        let depth = state.tile_state.stack.len();
        let tos: &TileEl = &state.tile_state.stack[depth - 1];
        let nos = &state.tile_state.stack[depth - 2];

        // Remember that in a single round, we perform the following operations in
        // the following order:
        // 1) Draw to slot in texture 0, potentially read from slot in texture 1.
        // 2) Draw to slot in texture 1, potentially read from slot in texture 0.
        // 3) Draw to final view, potentially read from slot in texture 1.
        // Therefore, for each depth, we can do the following (note that depths
        // are processed inversely, i.e. depth 3 is handled before depth 2, since depth
        // 2 depends on the contents of depth 3):
        // depth = 1 -> do 3).
        // depth = 2 -> do 2).
        // depth = 3 -> do 1).
        // depth = 4 -> We want to do 2) again, but it can't happen in the same round
        // because depth = 3 depends on the result from depth = 4, and there is no write
        // operation to texture 1 that we can allocate in the same round before 1)
        // happens. Therefore, we need to allocate a second round so that we have
        // enough "ping-ponging" to resolve all dependencies.
        let next_round = depth.is_multiple_of(2) && depth > 2;
        let round = nos.round.max(tos.round + usize::from(next_round));
        if let TemporarySlot::Valid(temp_slot) = nos.temporary_slot {
            let draw = self.draw_mut(round, nos.dest_slot.get_texture());
            draw.push(
                GpuStripBuilder::at_slot(nos.dest_slot.get_idx(), 0, WideTile::WIDTH)
                    .copy_from_slot(temp_slot.get_idx(), 0xFF),
            );
        }

        let draw = self.draw_mut(
            round,
            if (depth - 1) <= 1 {
                2
            } else {
                nos.dest_slot.get_texture()
            },
        );
        let gpu_strip_builder = if depth <= 2 {
            GpuStripBuilder::at_surface(wide_tile_x + cmd.x, wide_tile_y, cmd.width)
        } else {
            GpuStripBuilder::at_slot(nos.dest_slot.get_idx(), cmd.x, cmd.width)
        };
        draw.push(gpu_strip_builder.copy_from_slot(tos.dest_slot.get_idx(), 0xFF));

        let nos_ptr = state.tile_state.stack.len() - 2;
        state.tile_state.stack[nos_ptr].temporary_slot.invalidate();
    }

    fn do_clip_strip(
        &mut self,
        state: &mut SchedulerState,
        wide_tile_x: u16,
        wide_tile_y: u16,
        cmd: &CmdClipAlphaFill,
        attrs: &CommandAttrs,
    ) {
        let depth = state.tile_state.stack.len();
        let tos = &state.tile_state.stack[depth - 1];
        let nos = &state.tile_state.stack[depth - 2];

        let next_round = depth.is_multiple_of(2) && depth > 2;
        let round = nos.round.max(tos.round + usize::from(next_round));

        // If nos has a temporary slot, copy it to `dest_slot` first
        if let TemporarySlot::Valid(temp_slot) = nos.temporary_slot {
            let draw = self.draw_mut(round, nos.dest_slot.get_texture());
            draw.push(
                GpuStripBuilder::at_slot(nos.dest_slot.get_idx(), 0, WideTile::WIDTH)
                    .copy_from_slot(temp_slot.get_idx(), 0xFF),
            );
        }

        let draw = self.draw_mut(
            round,
            if (depth - 1) <= 1 {
                2
            } else {
                nos.dest_slot.get_texture()
            },
        );
        let gpu_strip_builder = if depth <= 2 {
            GpuStripBuilder::at_surface(wide_tile_x + cmd.x, wide_tile_y, cmd.width)
        } else {
            GpuStripBuilder::at_slot(nos.dest_slot.get_idx(), cmd.x, cmd.width)
        };

        let clip_attrs = &attrs.clip[cmd.attrs_idx as usize];
        let alpha_idx = clip_attrs.alpha_idx(cmd.alpha_offset);
        let col_idx = alpha_idx / u32::from(Tile::HEIGHT);

        draw.push(
            gpu_strip_builder
                .with_sparse(cmd.width, col_idx)
                .copy_from_slot(tos.dest_slot.get_idx(), 0xFF),
        );
        let nos_ptr = state.tile_state.stack.len() - 2;
        state.tile_state.stack[nos_ptr].temporary_slot.invalidate();
    }

    /// Process a paint and return (`payload`, `paint`)
    #[inline(always)]
    fn process_paint(
        paint: &Paint,
        encoded_paints: &[EncodedPaint],
        (scene_strip_x, scene_strip_y): (u16, u16),
        paint_idxs: &[u32],
    ) -> (u32, u32) {
        match paint {
            Paint::Solid(color) => {
                let rgba = color.as_premul_rgba8().to_u32();
                debug_assert!(
                    has_non_zero_alpha(rgba),
                    "Color fields with 0 alpha are reserved for clipping"
                );
                let paint_packed = (COLOR_SOURCE_PAYLOAD << 30) | (PAINT_TYPE_SOLID << 27);
                (rgba, paint_packed)
            }
            Paint::Indexed(indexed_paint) => {
                let paint_id = indexed_paint.index();
                let paint_idx = paint_idxs.get(paint_id).copied().unwrap();

                match encoded_paints.get(paint_id) {
                    Some(e) => {
                        Self::process_encoded_paint(e, paint_idx, scene_strip_x, scene_strip_y)
                    }
                    None => unimplemented!("Unsupported paint type"),
                }
            }
        }
    }

    fn process_encoded_paint(
        encoded_paint: &EncodedPaint,
        paint_idx: u32,
        scene_strip_x: u16,
        scene_strip_y: u16,
    ) -> (u32, u32) {
        match encoded_paint {
            EncodedPaint::Image(encoded_image) => match &encoded_image.source {
                ImageSource::OpaqueId { .. } => {
                    let paint_packed = (COLOR_SOURCE_PAYLOAD << 29)
                        | (PAINT_TYPE_IMAGE << 26)
                        | (paint_idx & 0x03FF_FFFF);
                    let scene_strip_xy = ((scene_strip_y as u32) << 16) | (scene_strip_x as u32);
                    (scene_strip_xy, paint_packed)
                }
                _ => unimplemented!("Unsupported image source"),
            },
            EncodedPaint::Gradient(gradient) => {
                use vello_common::encode::EncodedKind;
                let gradient_paint_type = match &gradient.kind {
                    EncodedKind::Linear(_) => PAINT_TYPE_LINEAR_GRADIENT,
                    EncodedKind::Radial(_) => PAINT_TYPE_RADIAL_GRADIENT,
                    EncodedKind::Sweep(_) => PAINT_TYPE_SWEEP_GRADIENT,
                };
                let paint_packed = (COLOR_SOURCE_PAYLOAD << 29)
                    | (gradient_paint_type << 26)
                    | (paint_idx & 0x03FF_FFFF);
                let scene_strip_xy = ((scene_strip_y as u32) << 16) | (scene_strip_x as u32);
                (scene_strip_xy, paint_packed)
            }
            _ => unimplemented!("Unsupported paint type"),
        }
    }
}

/// Helper for more semantically constructing `GpuStrip`s.
struct GpuStripBuilder {
    x: u16,
    y: u16,
    width: u16,
    dense_width_or_rect_height: u16,
    col_idx_or_rect_frac: u32,
}

impl GpuStripBuilder {
    /// Position at surface coordinates.
    fn at_surface(x: u16, y: u16, width: u16) -> Self {
        Self {
            x,
            y,
            width,
            dense_width_or_rect_height: 0,
            col_idx_or_rect_frac: 0,
        }
    }

    /// Position within a slot.
    fn at_slot(slot_idx: usize, x_offset: u16, width: u16) -> Self {
        Self {
            x: x_offset,
            y: u16::try_from(slot_idx).unwrap() * Tile::HEIGHT,
            width,
            dense_width_or_rect_height: 0,
            col_idx_or_rect_frac: 0,
        }
    }

    /// Add sparse strip parameters.
    fn with_sparse(mut self, dense_width: u16, col_idx: u32) -> Self {
        self.dense_width_or_rect_height = dense_width;
        self.col_idx_or_rect_frac = col_idx;
        self
    }

    /// Paint into strip.
    fn paint(self, payload: u32, paint: u32) -> GpuStrip {
        GpuStrip {
            x: self.x,
            y: self.y,
            width: self.width,
            dense_width_or_rect_height: self.dense_width_or_rect_height,
            col_idx_or_rect_frac: self.col_idx_or_rect_frac,
            payload,
            paint_and_rect_flag: paint,
        }
    }

    /// Copy from slot.
    fn copy_from_slot(self, from_slot: usize, opacity: u8) -> GpuStrip {
        GpuStrip {
            x: self.x,
            y: self.y,
            width: self.width,
            dense_width_or_rect_height: self.dense_width_or_rect_height,
            col_idx_or_rect_frac: self.col_idx_or_rect_frac,
            payload: u32::try_from(from_slot).unwrap(),
            paint_and_rect_flag: (COLOR_SOURCE_SLOT << 29) | (opacity as u32),
        }
    }

    /// Blend two slots.
    fn blend(
        self,
        src_slot: usize,
        dest_slot: usize,
        opacity: u8,
        mix_mode: u8,
        compose_mode: u8,
    ) -> GpuStrip {
        GpuStrip {
            x: self.x,
            y: self.y,
            width: self.width,
            dense_width_or_rect_height: self.dense_width_or_rect_height,
            col_idx_or_rect_frac: self.col_idx_or_rect_frac,
            payload: (u32::try_from(src_slot).unwrap())
                | ((u32::try_from(dest_slot).unwrap()) << 16),
            paint_and_rect_flag: (COLOR_SOURCE_BLEND << 29)
                | ((opacity as u32) << 16)
                | ((mix_mode as u32) << 8)
                | (compose_mode as u32),
        }
    }
}

#[inline(always)]
fn has_non_zero_alpha(rgba: u32) -> bool {
    rgba >= 0x1_00_00_00
}

fn generate_gpu_strips_for_fast_path(
    path: &FastStripsPath,
    strip_storage: &StripStorage,
    scene: &Scene,
    encoded_paints: &[EncodedPaint],
    paint_idxs: &[u32],
    gpu_strips: &mut Vec<GpuStrip>,
) {
    let strips = &strip_storage.strips[path.strips.clone()];

    if strips.is_empty() {
        return;
    }

    // Note: Some of this logic is similar to current coarse rasterization code, but
    // the coarse rasterization code is more complex due to clip paths and other factors.
    // It might be possible to reuse some code here, but it seems hard.

    for i in 0..strips.len() - 1 {
        let strip = &strips[i];

        if strip.x >= scene.width {
            continue;
        }

        let next_strip = &strips[i + 1];
        let col = strip.alpha_idx() / u32::from(Tile::HEIGHT);
        let next_col = next_strip.alpha_idx() / u32::from(Tile::HEIGHT);
        let strip_width = next_col.saturating_sub(col) as u16;
        let x0 = strip.x;
        let y = strip.y;

        // Alpha fill for the strip's coverage region.
        if strip_width > 0 {
            let (payload, paint) =
                Scheduler::process_paint(&path.paint, encoded_paints, (x0, y), paint_idxs);
            gpu_strips.push(
                GpuStripBuilder::at_surface(x0, y, strip_width)
                    .with_sparse(strip_width, col)
                    .paint(payload, paint),
            );
        }

        // Solid fill for the gap to the next strip.
        if next_strip.fill_gap() && strip.strip_y() == next_strip.strip_y() {
            let x1 = x0.saturating_add(strip_width);
            let x2 = next_strip.x.min(
                scene
                    .width
                    .checked_next_multiple_of(WideTile::WIDTH)
                    .unwrap_or(u16::MAX),
            );
            if x2 > x1 {
                let (payload, paint) =
                    Scheduler::process_paint(&path.paint, encoded_paints, (x1, y), paint_idxs);
                gpu_strips.push(GpuStripBuilder::at_surface(x1, y, x2 - x1).paint(payload, paint));
            }
        }
    }
}

fn pack_rectangle_into_gpu(
    rect: &FastPathRect,
    encoded_paints: &[EncodedPaint],
    paint_idxs: &[u32],
    out: &mut Vec<GpuStrip>,
) {
    let split = split_rect(rect);
    for part in [
        Some(split.main),
        split.top,
        split.bottom,
        split.left,
        split.right,
    ]
    .into_iter()
    .flatten()
    {
        let (payload, paint_packed) =
            Scheduler::process_paint(&rect.paint, encoded_paints, (part.x, part.y), paint_idxs);
        out.push(make_gpu_rect(part, payload, paint_packed));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RectPart {
    x: u16,
    y: u16,
    width: u16,
    height: u16,
    frac: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SplitRect {
    main: RectPart,
    top: Option<RectPart>,
    bottom: Option<RectPart>,
    left: Option<RectPart>,
    right: Option<RectPart>,
}

fn split_rect(rect: &FastPathRect) -> SplitRect {
    let sx0 = rect.x0.floor();
    let sy0 = rect.y0.floor();
    let sx1 = rect.x1.ceil();
    let sy1 = rect.y1.ceil();

    let x = sx0 as u16;
    let y = sy0 as u16;
    // Are guaranteed to be > 0 since we rejected negative rectangles.
    let width = (sx1 - sx0) as u16;
    let height = (sy1 - sy0) as u16;

    // Note that `top_frac` and `left_fract` store the actual coverage, while
    // `right_frac` and `bottom_fract` store one minus the coverage. This is on purpose
    // and handled that way in the shader.
    let left_frac = rect.x0 - sx0;
    let top_frac = rect.y0 - sy0;
    let right_frac = sx1 - rect.x1;
    let bottom_frac = sy1 - rect.y1;

    // There's a balance to strike between reducing work in the fragment shader by splitting
    // out the inner part of the rectangle without anti-aliasing, and additional overhead
    // that arises from rendering 5 rectangles instead of just one. While the exact threshold
    // will obviously depend on the device, some experiments on a low-tier tablet showed that
    // `LARGE_RECT_SPLIT_THRESHOLD` seems to be a a reasonable value.
    if rect.x1 - rect.x0 < f32::from(LARGE_RECT_SPLIT_THRESHOLD)
        || rect.y1 - rect.y0 < f32::from(LARGE_RECT_SPLIT_THRESHOLD)
    {
        return SplitRect {
            main: RectPart {
                x,
                y,
                width,
                height,
                frac: pack_unorm4x8([left_frac, top_frac, right_frac, bottom_frac]),
            },
            top: None,
            bottom: None,
            left: None,
            right: None,
        };
    }

    let has_left_aa = left_frac > 0.0;
    let has_top_aa = top_frac > 0.0;
    let has_right_aa = right_frac > 0.0;
    let has_bottom_aa = bottom_frac > 0.0;
    let has_top_strip = has_top_aa || has_left_aa || has_right_aa;
    let has_bottom_strip = has_bottom_aa || has_left_aa || has_right_aa;
    let left_inset = u16::from(has_left_aa);
    let right_inset = u16::from(has_right_aa);
    let top_inset = u16::from(has_top_strip);
    let bottom_inset = u16::from(has_bottom_strip);
    let inner_x = x + left_inset;
    let inner_y = y + top_inset;
    // Can't underflow because rectangles have at least `LARGE_RECT_SPLIT_THRESHOLD` in each
    // direction, which is larger than 2.
    let inner_width = width - left_inset - right_inset;
    let inner_height = height - top_inset - bottom_inset;

    SplitRect {
        main: RectPart {
            x: inner_x,
            y: inner_y,
            width: inner_width,
            height: inner_height,
            frac: 0,
        },
        top: has_top_strip.then_some(RectPart {
            x,
            y,
            width,
            height: 1,
            frac: pack_unorm4x8([left_frac, top_frac, right_frac, 0.0]),
        }),
        bottom: has_bottom_strip.then_some(RectPart {
            x,
            y: y + height - 1,
            width,
            height: 1,
            frac: pack_unorm4x8([left_frac, 0.0, right_frac, bottom_frac]),
        }),
        left: has_left_aa.then_some(RectPart {
            x,
            y: inner_y,
            width: 1,
            height: inner_height,
            frac: pack_unorm4x8([left_frac, 0.0, 0.0, 0.0]),
        }),
        right: has_right_aa.then_some(RectPart {
            x: x + width - 1,
            y: inner_y,
            width: 1,
            height: inner_height,
            frac: pack_unorm4x8([0.0, 0.0, right_frac, 0.0]),
        }),
    }
}

fn make_gpu_rect(part: RectPart, payload: u32, paint_packed: u32) -> GpuStrip {
    GpuStrip {
        x: part.x,
        y: part.y,
        width: part.width,
        dense_width_or_rect_height: part.height,
        col_idx_or_rect_frac: part.frac,
        payload,
        paint_and_rect_flag: paint_packed | RECT_STRIP_FLAG,
    }
}

fn pack_unorm4x8(v: [f32; 4]) -> u32 {
    let q = |f: f32| -> u8 { (f * 255.0 + 0.5) as u8 };
    u32::from(q(v[0]))
        | (u32::from(q(v[1])) << 8)
        | (u32::from(q(v[2])) << 16)
        | (u32::from(q(v[3])) << 24)
}

#[cfg(test)]
mod tests {
    use super::{
        RECT_STRIP_FLAG, RectPart, SplitRect, pack_rectangle_into_gpu, pack_unorm4x8, split_rect,
    };
    use crate::scene::FastPathRect;
    use alloc::vec;
    use alloc::vec::Vec;
    use vello_common::encode::EncodedImage;
    use vello_common::kurbo::{Affine, Vec2};
    use vello_common::paint::{Color, ImageId, ImageSource, IndexedPaint, Paint};
    use vello_common::peniko::ImageSampler;

    fn solid_rect(x0: f32, y0: f32, x1: f32, y1: f32) -> FastPathRect {
        FastPathRect {
            x0,
            y0,
            x1,
            y1,
            paint: Paint::from(Color::from_rgba8(255, 0, 0, 255)),
        }
    }

    fn part(x: u16, y: u16, width: u16, height: u16, frac: [f32; 4]) -> RectPart {
        RectPart {
            x,
            y,
            width,
            height,
            frac: pack_unorm4x8(frac),
        }
    }

    #[test]
    fn splitter_keeps_small_rect_whole() {
        let rect = solid_rect(10.25, 20.5, 25.75, 35.25);
        let split = split_rect(&rect);

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 16, 16, [0.25, 0.5, 0.25, 0.75]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_subpixel_rect_inside_one_pixel() {
        let rect = solid_rect(10.125, 20.25, 10.875, 20.75);
        let split = split_rect(&rect);

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 1, 1, [0.125, 0.25, 0.125, 0.25]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_subpixel_rect_spanning_two_pixels_in_width() {
        let rect = solid_rect(10.75, 20.125, 11.25, 20.875);
        let split = split_rect(&rect);

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 2, 1, [0.75, 0.125, 0.75, 0.125]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_subpixel_rect_spanning_two_pixels_in_height() {
        let rect = solid_rect(10.125, 20.75, 10.875, 21.25);
        let split = split_rect(&rect);

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 1, 2, [0.125, 0.75, 0.125, 0.75]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_multi_pixel_width_rect_within_one_pixel_height() {
        let rect = solid_rect(10.25, 20.125, 14.75, 20.875);
        let split = split_rect(&rect);

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 5, 1, [0.25, 0.125, 0.25, 0.125]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_multi_pixel_height_rect_within_one_pixel_width() {
        let rect = solid_rect(10.125, 20.25, 10.875, 24.75);
        let split = split_rect(&rect);

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 1, 5, [0.125, 0.25, 0.125, 0.25]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_splits_large_rect_into_five_parts() {
        let rect = solid_rect(10.25, 20.5, 42.75, 52.75);
        let split = split_rect(&rect);

        assert_eq!(
            split,
            SplitRect {
                main: part(11, 21, 31, 31, [0.0, 0.0, 0.0, 0.0]),
                top: Some(part(10, 20, 33, 1, [0.25, 0.5, 0.25, 0.0])),
                bottom: Some(part(10, 52, 33, 1, [0.25, 0.0, 0.25, 0.25])),
                left: Some(part(10, 21, 1, 31, [0.25, 0.0, 0.0, 0.0])),
                right: Some(part(42, 21, 1, 31, [0.0, 0.0, 0.25, 0.0])),
            }
        );
    }

    #[test]
    fn splitter_omits_unneeded_edge_parts() {
        let rect = solid_rect(10.0, 20.5, 42.0, 53.0);
        let split = split_rect(&rect);

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 21, 32, 32, [0.0, 0.0, 0.0, 0.0]),
                top: Some(part(10, 20, 32, 1, [0.0, 0.5, 0.0, 0.0])),
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_handles_large_rect_with_only_vertical_aa() {
        let rect = solid_rect(5.0, 2.25, 37.0, 34.75);
        let split = split_rect(&rect);

        assert_eq!(
            split,
            SplitRect {
                main: part(5, 3, 32, 31, [0.0, 0.0, 0.0, 0.0]),
                top: Some(part(5, 2, 32, 1, [0.0, 0.25, 0.0, 0.0])),
                bottom: Some(part(5, 34, 32, 1, [0.0, 0.0, 0.0, 0.25])),
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_large_aligned_rect_as_single_main_rect() {
        let rect = solid_rect(10.0, 20.0, 42.0, 60.0);
        let split = split_rect(&rect);

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 32, 40, [0.0, 0.0, 0.0, 0.0]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn gpu_upload_emits_main_and_present_optional_parts() {
        let rect = solid_rect(10.0, 20.5, 42.0, 53.0);
        let mut out = Vec::new();

        pack_rectangle_into_gpu(&rect, &[], &[], &mut out);

        assert_eq!(out.len(), 2);
        assert_eq!(
            (
                out[0].x,
                out[0].y,
                out[0].width,
                out[0].dense_width_or_rect_height
            ),
            (10, 21, 32, 32)
        );
        assert_eq!(out[0].col_idx_or_rect_frac, 0);
        assert_eq!(
            (
                out[1].x,
                out[1].y,
                out[1].width,
                out[1].dense_width_or_rect_height
            ),
            (10, 20, 32, 1)
        );
        assert_eq!(
            out[1].col_idx_or_rect_frac,
            pack_unorm4x8([0.0, 0.5, 0.0, 0.0])
        );
        assert!(
            out.iter()
                .all(|strip| strip.paint_and_rect_flag & RECT_STRIP_FLAG != 0)
        );
    }

    #[test]
    fn gpu_upload_updates_payload_for_each_split_part() {
        let rect = FastPathRect {
            x0: 10.25,
            y0: 20.5,
            x1: 42.75,
            y1: 52.75,
            paint: Paint::Indexed(IndexedPaint::new(0)),
        };
        let encoded_paints = vec![vello_common::encode::EncodedPaint::Image(EncodedImage {
            source: ImageSource::opaque_id(ImageId::new(1)),
            sampler: ImageSampler::new(),
            may_have_opacities: false,
            transform: Affine::IDENTITY,
            x_advance: Vec2::new(1.0, 0.0),
            y_advance: Vec2::new(0.0, 1.0),
            tint: None,
        })];
        let mut out = Vec::new();

        pack_rectangle_into_gpu(&rect, &encoded_paints, &[7], &mut out);

        assert_eq!(out.len(), 5);
        assert_eq!(out[0].payload, (21_u32 << 16) | 11_u32);
        assert_eq!(out[1].payload, (20_u32 << 16) | 10_u32);
        assert_eq!(out[2].payload, (52_u32 << 16) | 10_u32);
        assert_eq!(out[3].payload, (21_u32 << 16) | 10_u32);
        assert_eq!(out[4].payload, (21_u32 << 16) | 42_u32);
    }
}
