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

use crate::render::common::GpuEncodedImage;
use crate::{GpuStrip, RenderError, Scene};
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::mem;
use vello_common::peniko::{BlendMode, Compose, Mix};
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

/// Trait for abstracting the renderer backend from the scheduler.
pub(crate) trait RendererBackend {
    /// Clear specific slots in a texture.
    fn clear_slots(&mut self, texture_index: usize, slots: &[u32]);

    /// Execute a render pass for strips.
    fn render_strips(&mut self, strips: &[GpuStrip], target_index: usize, load_op: LoadOp);
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
    /// The total number of slots in each slot texture.
    total_slots: usize,
    /// The slots that are free to use in each slot texture.
    free: [Vec<usize>; 2],
    /// Rounds are enqueued on push clip commands and dequeued on flush.
    rounds_queue: VecDeque<Round>,
    #[cfg(debug_assertions)]
    clear: [Vec<u32>; 2],
}

/// The information required to resume a wide tile draw operation. Note that the suspended tile owns
/// global slot resources, so must be resumed to free the owned slots. Tile drawing must suspend to
/// allow blend layers to be merged.
#[derive(Debug)]
struct PendingTileWork<'a> {
    /// Used to reference the wide tile.
    wide_tile_col: u16,
    /// Used to reference the wide tile.
    wide_tile_row: u16,
    /// On resuming the cmd index to start processing draws from.
    next_cmd_idx: usize,
    /// The suspended tile state stack representing active layers.
    stack: TileState,
    /// Round at which this work was suspended. Used to calculate round offsets.
    suspended_at_round: usize,
    /// The draw commands being iterated to draw the wide tile.
    annotated_cmds: Vec<AnnotatedCmd<'a>>,
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

/// State for a single wide tile.
#[derive(Debug, Default)]
struct TileState {
    stack: Vec<TileEl>,
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

/// Annotated commands lets the scheduler be smarter about what work to do per command by preparing
/// some work upfront in `O(N)` time, where `N` is the length of the wide tile draw commands.
///
/// TODO: In the future these annotations could be optionally enabled in coarse.rs directly avoiding
/// the need to do a linear scan.
#[derive(Debug)]
enum AnnotatedCmd<'a> {
    /// A wrapped command - no semantic meaning added.
    IdentityBorrowed(&'a Cmd),
    /// An owned generated command. Allows adding additional commands.
    Generated(Cmd),
    /// A `Cmd::PushBuf` that will be blended into by the layer above it. Will need a
    /// `temporary_slot` created for it to enable compositing.
    PushBufWithTemporarySlot,
}

impl<'a> AnnotatedCmd<'a> {
    fn unwrap<'b: 'a>(&'b self) -> &'a Cmd {
        match self {
            AnnotatedCmd::IdentityBorrowed(cmd) => cmd,
            AnnotatedCmd::Generated(cmd) => cmd,
            AnnotatedCmd::PushBufWithTemporarySlot => &Cmd::PushBuf,
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
        match self {
            Self::Invalid(_) => {}
            Self::None => {}
            Self::Valid(slot) => {
                *self = Self::Invalid(*slot);
            }
        };
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
            2
        }
    }
}

#[derive(Debug, Default)]
struct Draw(Vec<GpuStrip>);

impl Draw {
    #[inline(always)]
    fn push(&mut self, gpu_strip: GpuStrip) {
        self.0.push(gpu_strip)
    }
}

impl Scheduler {
    pub(crate) fn new(total_slots: usize) -> Self {
        let free0: Vec<_> = (0..total_slots).collect();
        let free1 = free0.clone();
        let free: [Vec<usize>; 2] = [free0, free1];
        Self {
            round: 0,
            total_slots,
            free,
            rounds_queue: Default::default(),
            #[cfg(debug_assertions)]
            clear: [Vec::new(), Vec::new()],
        }
    }

    fn claim_free_slot<R: RendererBackend>(
        &mut self,
        texture: usize,
        renderer: &mut R,
    ) -> Result<ClaimedSlot, RenderError> {
        assert!(matches!(texture, 0 | 1));
        while self.free[texture].is_empty() {
            if self.rounds_queue.is_empty() {
                return Err(RenderError::SlotsExhausted);
            }
            self.flush(renderer);
        }

        let slot_ix = self.free[texture].pop().unwrap();

        match texture {
            0 => Ok(ClaimedSlot::Texture0(slot_ix)),
            1 => Ok(ClaimedSlot::Texture1(slot_ix)),
            _ => panic!("invalid slot texture"),
        }
    }

    pub(crate) fn do_scene<'scene, R: RendererBackend>(
        &mut self,
        renderer: &mut R,
        scene: &'scene Scene,
    ) -> Result<(), RenderError> {
        let wide_tiles_per_row = scene.wide.width_tiles();
        let wide_tiles_per_col = scene.wide.height_tiles();
        let mut pending_work: Vec<PendingTileWork<'scene>> = Default::default();

        // Left to right, top to bottom iteration over wide tiles.
        for wide_tile_row in 0..wide_tiles_per_col {
            for wide_tile_col in 0..wide_tiles_per_row {
                let wide_tile = scene.wide.get(wide_tile_col, wide_tile_row);
                let wide_tile_x = wide_tile_col * WideTile::WIDTH;
                let wide_tile_y = wide_tile_row * Tile::HEIGHT;

                let tile_state = self.initialize_tile_state(wide_tile, wide_tile_x, wide_tile_y);
                let annotated_cmds = prepare_cmds(&wide_tile.cmds);
                pending_work.push(PendingTileWork {
                    wide_tile_col,
                    wide_tile_row,
                    next_cmd_idx: 0,
                    stack: tile_state,
                    suspended_at_round: self.round,
                    annotated_cmds,
                });
            }
        }

        // TODO: Eagerly processing tile-by-tile is expensive as a blend requires suspending until a
        // Round has flushed. Instead of flushing to wait for a blend, we suspend the tile's work,
        // and continue iterating through the tiles. Slots claimed from the slot textures are a
        // shared resource, and it's possible for all the tiles to not be able to "resume" from
        // their "suspended" state due to no slots being available.
        while !self.rounds_queue.is_empty() || !pending_work.is_empty() {
            if !self.rounds_queue.is_empty() {
                self.flush(renderer);
            }

            let pending: Vec<PendingTileWork<'scene>> = mem::take(&mut pending_work);
            for work in pending {
                let wide_tile_col = work.wide_tile_col;
                let wide_tile_row = work.wide_tile_row;
                let wide_tile_x = wide_tile_col * WideTile::WIDTH;
                let wide_tile_y = wide_tile_row * Tile::HEIGHT;
                let mut tile_state = work.stack;

                let round_offset = self.round - (work.suspended_at_round);
                for el in tile_state.stack.iter_mut() {
                    el.round += round_offset;
                }

                if let Some(next_cmd_idx) = self.do_tile(
                    renderer,
                    scene,
                    wide_tile_x,
                    wide_tile_y,
                    &work.annotated_cmds,
                    &mut tile_state,
                    work.next_cmd_idx,
                )? {
                    pending_work.push(PendingTileWork {
                        wide_tile_col,
                        wide_tile_row,
                        next_cmd_idx,
                        stack: tile_state,
                        suspended_at_round: self.round,
                        annotated_cmds: work.annotated_cmds,
                    });
                }
            }
        }

        while !self.rounds_queue.is_empty() {
            self.flush(renderer);
        }

        // Restore state to reuse allocations.
        self.round = 0;
        #[cfg(debug_assertions)]
        {
            debug_assert!(self.clear[0].is_empty(), "clear has not reset");
            debug_assert!(self.clear[1].is_empty(), "clear has not reset");
            for i in 0..self.total_slots {
                debug_assert!(self.free[0].contains(&i), "free[0] is missing slot {i}");
                debug_assert!(self.free[1].contains(&i), "free[1] is missing slot {i}");
            }
        }
        debug_assert!(self.rounds_queue.is_empty(), "rounds_queue is not empty");

        Ok(())
    }

    /// Flush one round.
    ///
    /// The rounds queue must not be empty.
    fn flush<R: RendererBackend>(&mut self, renderer: &mut R) {
        let round = self.rounds_queue.pop_front().unwrap();
        for (i, draw) in round.draws.iter().enumerate() {
            let load = {
                if i == 2 {
                    // We're rendering to the view, don't clear.
                    LoadOp::Load
                } else if round.clear[i].len() + self.free[i].len() == self.total_slots {
                    #[cfg(debug_assertions)]
                    self.clear[i].clear();
                    // All slots are either unoccupied or need to be cleared. Simply clear the slots
                    // via a load operation.
                    LoadOp::Clear
                } else {
                    #[cfg(debug_assertions)]
                    {
                        // This debug_assertion is expensive, but it enforces that duplicates cannot
                        // be cleared. This usually signals a bug in the scheduler.
                        for (idx, &slot) in round.clear[i].iter().enumerate() {
                            debug_assert!(
                                !round.clear[i][..idx].contains(&slot),
                                "Duplicate slot {} found in round.clear[{}]",
                                slot,
                                i
                            );

                            // We can't use `retain` because `self.clear` tracks clearing globally
                            // across rounds. This means that it _can_ contain duplicates if a slot
                            // is scheduled to be cleared in two rounds simultaneously.
                            if let Some(pos) = self.clear[i].iter().position(|&x| x == slot) {
                                self.clear[i].swap_remove(pos);
                            }
                        }
                    }

                    // Some slots need to be preserved, so only clear the dirty slots.
                    renderer.clear_slots(i, round.clear[i].as_slice());
                    LoadOp::Load
                }
            };

            if draw.0.is_empty() {
                if load == LoadOp::Clear {
                    // There are no strips to render, so render_strips will not run and won't clear
                    // the texture. We still have slots to clear this round so explicitly clear
                    // them.
                    renderer.clear_slots(i, round.clear[i].as_slice());
                }
                continue;
            }

            renderer.render_strips(&draw.0, i, load);
        }
        for i in 0..2 {
            self.free[i].extend(&round.free[i]);
        }
        self.round += 1;
    }

    // Find the appropriate draw call for rendering.
    fn draw_mut(&mut self, el_round: usize, texture_idx: usize) -> &mut Draw {
        &mut self.get_round(el_round).draws[texture_idx]
    }

    // Find the appropriate round for rendering.
    fn get_round(&mut self, el_round: usize) -> &mut Round {
        let rel_round = el_round.saturating_sub(self.round);
        if self.rounds_queue.len() == rel_round {
            self.rounds_queue.push_back(Round::default());
        }
        &mut self.rounds_queue[rel_round]
    }

    fn initialize_tile_state(
        &mut self,
        tile: &WideTile,
        wide_tile_x: u16,
        wide_tile_y: u16,
    ) -> TileState {
        let mut state = TileState::default();
        // Sentinel `TileEl` to indicate the end of the stack where we draw all
        // commands to the final target.
        state.stack.push(TileEl {
            dest_slot: ClaimedSlot::Texture0(usize::MAX),
            temporary_slot: TemporarySlot::None,
            round: self.round,
            opacity: 1.,
        });
        {
            // If the background has a non-zero alpha then we need to render it.
            let bg = tile.bg.as_premul_rgba8().to_u32();
            if has_non_zero_alpha(bg) {
                let draw = self.draw_mut(self.round, 2);
                draw.push(GpuStrip {
                    x: wide_tile_x,
                    y: wide_tile_y,
                    width: WideTile::WIDTH,
                    dense_width: 0,
                    col_idx: 0,
                    payload: bg,
                    paint: 0,
                });
            }
        }
        state
    }

    /// Iterates over wide tile commands and schedules them for rendering.
    ///
    /// Returns `Some(command_idx)` if there is more work to be done. Returns `None` if the wide
    /// tile has been fully consumed.
    fn do_tile<'a, R: RendererBackend>(
        &mut self,
        renderer: &mut R,
        scene: &Scene,
        wide_tile_x: u16,
        wide_tile_y: u16,
        cmds: &[AnnotatedCmd<'a>],
        state: &mut TileState,
        start_cmd_idx: usize,
    ) -> Result<Option<usize>, RenderError> {
        let mut has_blended = false;
        for (offset_cmd_idx, annotated_cmd) in cmds[start_cmd_idx..].iter().enumerate() {
            let cmd_idx = start_cmd_idx + offset_cmd_idx;
            // Note: this starts at 1 (for the final target)
            let clip_depth = state.stack.len();
            let cmd = annotated_cmd.unwrap();
            match cmd {
                Cmd::Fill(fill) => {
                    let el = state.stack.last_mut().unwrap();
                    let draw = self.draw_mut(el.round, el.get_draw_texture(clip_depth));

                    let (scene_strip_x, scene_strip_y) = (wide_tile_x + fill.x, wide_tile_y);
                    let (payload, paint) =
                        Self::process_paint(&fill.paint, scene, (scene_strip_x, scene_strip_y));

                    let (x, y) = if clip_depth == 1 {
                        (scene_strip_x, scene_strip_y)
                    } else {
                        let slot_idx = if let TemporarySlot::Valid(temp_slot) = el.temporary_slot {
                            temp_slot.get_idx()
                        } else {
                            el.dest_slot.get_idx()
                        };
                        (fill.x, slot_idx as u16 * Tile::HEIGHT)
                    };

                    draw.push(GpuStrip {
                        x,
                        y,
                        width: fill.width,
                        dense_width: 0,
                        col_idx: 0,
                        payload,
                        paint,
                    });
                }
                Cmd::AlphaFill(alpha_fill) => {
                    let el = state.stack.last_mut().unwrap();
                    let draw = self.draw_mut(el.round, el.get_draw_texture(clip_depth));

                    let col_idx = (alpha_fill.alpha_idx / usize::from(Tile::HEIGHT))
                        .try_into()
                        .expect("Sparse strips are bound to u32 range");

                    let (scene_strip_x, scene_strip_y) = (wide_tile_x + alpha_fill.x, wide_tile_y);
                    let (payload, paint) = Self::process_paint(
                        &alpha_fill.paint,
                        scene,
                        (scene_strip_x, scene_strip_y),
                    );

                    let (x, y) = if clip_depth == 1 {
                        (scene_strip_x, scene_strip_y)
                    } else {
                        let slot_idx = if let TemporarySlot::Valid(temp_slot) = el.temporary_slot {
                            temp_slot.get_idx()
                        } else {
                            el.dest_slot.get_idx()
                        };
                        (alpha_fill.x, slot_idx as u16 * Tile::HEIGHT)
                    };

                    draw.push(GpuStrip {
                        x,
                        y,
                        width: alpha_fill.width,
                        dense_width: alpha_fill.width,
                        col_idx,
                        payload,
                        paint,
                    });
                }
                Cmd::PushBuf => {
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
                        let tos: &mut TileEl = state.stack.last_mut().unwrap();
                        if let TemporarySlot::Invalid(temp_slot) = tos.temporary_slot {
                            if has_blended {
                                // There has been a popped blend in this run of `do_tile`. Suspend
                                // to give the blend command a chance to flush.
                                return Ok(Some(cmd_idx));
                            }

                            let draw = self.draw_mut(tos.round - 1, temp_slot.get_texture());
                            // Generate a full opacity copy.
                            let paint = COLOR_SOURCE_SLOT << 30 | 0xFF;
                            draw.push(GpuStrip {
                                x: 0,
                                y: temp_slot.get_idx() as u16 * Tile::HEIGHT,
                                width: WideTile::WIDTH,
                                dense_width: 0,
                                col_idx: 0,
                                payload: tos.dest_slot.get_idx() as u32,
                                paint,
                            });

                            tos.temporary_slot = TemporarySlot::Valid(temp_slot);

                            // Make sure the destination slot and temporary slot are cleared
                            // appropriately.
                            let dest_slot = tos.dest_slot;
                            let round = self.get_round(tos.round - 1);
                            round.clear[temp_slot.get_texture()].push(temp_slot.get_idx() as u32);
                            #[cfg(debug_assertions)]
                            self.clear[temp_slot.get_texture()].push(temp_slot.get_idx() as u32);
                            if dest_slot.get_idx() as u32 != u32::MAX {
                                let round = self.get_round(tos.round);
                                round.clear[dest_slot.get_texture()]
                                    .push(dest_slot.get_idx() as u32);
                                #[cfg(debug_assertions)]
                                self.clear[dest_slot.get_texture()]
                                    .push(dest_slot.get_idx() as u32);
                            }
                        }
                    }

                    // Suspend if there are no free slots.
                    if self.free[0].is_empty() || self.free[1].is_empty() {
                        return Ok(Some(cmd_idx));
                    }

                    // Push a new tile.
                    let ix = clip_depth % 2;
                    let slot = self.claim_free_slot(ix, renderer)?;
                    {
                        let round = self.get_round(self.round);
                        round.clear[slot.get_texture()].push(slot.get_idx() as u32);
                        #[cfg(debug_assertions)]
                        self.clear[slot.get_texture()].push(slot.get_idx() as u32);
                    }
                    let temporary_slot =
                        if matches!(annotated_cmd, AnnotatedCmd::PushBufWithTemporarySlot) {
                            let temp_slot = self.claim_free_slot((ix + 1) % 2, renderer)?;
                            let round = self.get_round(self.round);
                            round.clear[temp_slot.get_texture()].push(temp_slot.get_idx() as u32);
                            #[cfg(debug_assertions)]
                            self.clear[temp_slot.get_texture()].push(temp_slot.get_idx() as u32);
                            debug_assert_ne!(
                                slot.get_texture(),
                                temp_slot.get_texture(),
                                "slot and temporary slot must be on opposite textures."
                            );
                            TemporarySlot::Valid(temp_slot)
                        } else {
                            TemporarySlot::None
                        };
                    state.stack.push(TileEl {
                        dest_slot: slot,
                        temporary_slot,
                        round: self.round,
                        opacity: 1.,
                    });
                }
                Cmd::PopBuf => {
                    let tos = state.stack.pop().unwrap();
                    let nos = state.stack.last_mut().unwrap();
                    let next_round = clip_depth % 2 == 0 && clip_depth > 2;
                    let round = nos.round.max(tos.round + usize::from(next_round));
                    nos.round = round;
                    // free slot after draw
                    debug_assert!(round >= self.round, "round must be after current round");
                    debug_assert!(
                        round - self.round < self.rounds_queue.len(),
                        "round must be in queue"
                    );

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
                Cmd::ClipFill(clip_fill) => {
                    let tos: &TileEl = &state.stack[clip_depth - 1];
                    let nos = &state.stack[clip_depth - 2];

                    // Basically if we are writing onto the even texture, we need to go up a round
                    // to target it.
                    let next_round = clip_depth % 2 == 0 && clip_depth > 2;
                    let round = nos.round.max(tos.round + usize::from(next_round));
                    if let TemporarySlot::Valid(temp_slot) = nos.temporary_slot {
                        let draw = self.draw_mut(round, nos.dest_slot.get_texture());
                        let paint: u32 = COLOR_SOURCE_SLOT << 30 | 0xFF; // Full opacity copy
                        draw.push(GpuStrip {
                            x: 0,
                            y: nos.dest_slot.get_idx() as u16 * Tile::HEIGHT,
                            width: WideTile::WIDTH,
                            dense_width: 0,
                            col_idx: 0,
                            payload: temp_slot.get_idx() as u32,
                            paint,
                        });
                    }

                    let draw = self.draw_mut(
                        round,
                        if (clip_depth - 1) <= 1 {
                            2
                        } else {
                            nos.dest_slot.get_texture()
                        },
                    );
                    let (x, y) = if clip_depth <= 2 {
                        (wide_tile_x + clip_fill.x as u16, wide_tile_y)
                    } else {
                        (
                            clip_fill.x as u16,
                            nos.dest_slot.get_idx() as u16 * Tile::HEIGHT,
                        )
                    };
                    // Opacity packed into the first 8 bits – pack full opacity (0xFF).
                    let paint = COLOR_SOURCE_SLOT << 30 | 0xFF;
                    draw.push(GpuStrip {
                        x,
                        y,
                        width: clip_fill.width as u16,
                        dense_width: 0,
                        col_idx: 0,
                        payload: tos.dest_slot.get_idx() as u32,
                        paint,
                    });

                    let nos_ptr = state.stack.len() - 2;
                    state.stack[nos_ptr].temporary_slot.invalidate();
                }
                Cmd::ClipStrip(clip_alpha_fill) => {
                    let tos = &state.stack[clip_depth - 1];
                    let nos = &state.stack[clip_depth - 2];

                    let next_round = clip_depth % 2 == 0 && clip_depth > 2;
                    let round = nos.round.max(tos.round + usize::from(next_round));

                    // If nos has a temporary slot, copy it to `dest_slot` first
                    if let TemporarySlot::Valid(temp_slot) = nos.temporary_slot {
                        let draw = self.draw_mut(round, nos.dest_slot.get_texture());
                        let paint = COLOR_SOURCE_SLOT << 30 | 0xFF;
                        draw.push(GpuStrip {
                            x: 0,
                            y: nos.dest_slot.get_idx() as u16 * Tile::HEIGHT,
                            width: WideTile::WIDTH,
                            dense_width: 0,
                            col_idx: 0,
                            payload: temp_slot.get_idx() as u32,
                            paint,
                        });
                    }

                    let draw = self.draw_mut(
                        round,
                        if (clip_depth - 1) <= 1 {
                            2
                        } else {
                            nos.dest_slot.get_texture()
                        },
                    );
                    let (x, y) = if clip_depth <= 2 {
                        (wide_tile_x + clip_alpha_fill.x as u16, wide_tile_y)
                    } else {
                        (
                            clip_alpha_fill.x as u16,
                            nos.dest_slot.get_idx() as u16 * Tile::HEIGHT,
                        )
                    };
                    // Opacity packed into the first 8 bits – pack full opacity (0xFF).
                    let paint = COLOR_SOURCE_SLOT << 30 | 0xFF;
                    draw.push(GpuStrip {
                        x,
                        y,
                        width: clip_alpha_fill.width as u16,
                        dense_width: clip_alpha_fill.width as u16,
                        col_idx: (clip_alpha_fill.alpha_idx / usize::from(Tile::HEIGHT))
                            .try_into()
                            .expect("Sparse strips are bound to u32 range"),
                        payload: tos.dest_slot.get_idx() as u32,
                        paint,
                    });

                    let nos_ptr = state.stack.len() - 2;
                    state.stack[nos_ptr].temporary_slot.invalidate();
                }
                Cmd::Opacity(opacity) => {
                    state.stack.last_mut().unwrap().opacity = *opacity;
                }
                Cmd::Blend(mode) => {
                    assert!(
                        matches!(mode.mix, Mix::Normal),
                        "Only Mix::Normal is supported currently"
                    );
                    #[cfg(all(target_arch = "wasm32", feature = "webgl"))]
                    assert!(
                        matches!(mode.compose, Compose::SrcOver),
                        "webgl backend does not support blend modes yet."
                    );

                    let tos = state.stack.last().unwrap();
                    let nos = &state.stack[state.stack.len() - 2];

                    let next_round: bool = clip_depth % 2 == 0 && clip_depth > 2;
                    let round = nos.round.max(tos.round + usize::from(next_round));

                    if let TemporarySlot::Valid(temp_slot) = nos.temporary_slot {
                        let draw = self.draw_mut(
                            round,
                            if clip_depth <= 2 {
                                2
                            } else {
                                nos.dest_slot.get_texture()
                            },
                        );
                        let (x, y) = if clip_depth <= 2 {
                            (wide_tile_x, wide_tile_y)
                        } else {
                            (0, nos.dest_slot.get_idx() as u16 * Tile::HEIGHT)
                        };

                        let opacity_u8 = (tos.opacity * 255.0) as u32;

                        // src_slot (bits 0-15) | `dest_slot` (bits 16-31)
                        let payload =
                            tos.dest_slot.get_idx() as u32 | ((temp_slot.get_idx() as u32) << 16);

                        // Pack opacity, mix_mode, and compose_mode into paint
                        // Extract the actual mix and compose values from mode
                        let mix_mode = mode.mix as u32; // Assuming Mix has repr(u8) like Compose
                        let compose_mode = mode.compose as u32;

                        let paint = (COLOR_SOURCE_BLEND << 30) | (opacity_u8 << 16)           // opacity (bits 16-23)
                            | (mix_mode << 8)              // mix_mode (bits 8-15)
                            | compose_mode; // compose_mode (bits 0-7)

                        draw.push(GpuStrip {
                            x,
                            y,
                            width: WideTile::WIDTH,
                            dense_width: 0,
                            col_idx: 0,
                            payload,
                            paint,
                        });

                        // Invalidate the temporary slot after use
                        let nos_ptr = state.stack.len() - 2;
                        state.stack[nos_ptr].temporary_slot.invalidate();
                        // Signal to suspend before pushing a new buffer.
                        has_blended = true;
                    } else {
                        let draw = self.draw_mut(round, tos.get_draw_texture(clip_depth - 1));
                        let (x, y) = if (clip_depth - 1) <= 2 {
                            (wide_tile_x, wide_tile_y)
                        } else {
                            (0, nos.dest_slot.get_idx() as u16 * Tile::HEIGHT)
                        };
                        assert_eq!(
                            nos.dest_slot.get_idx(),
                            usize::MAX,
                            "code path only for copying to sentinel slot, {mode:?}"
                        );
                        // The final canvas is write-only, hence we can only copy with opacity.
                        let opacity_u8 = (tos.opacity * 255.0) as u32;
                        let paint = (COLOR_SOURCE_SLOT << 30) | opacity_u8;

                        draw.push(GpuStrip {
                            x,
                            y,
                            width: WideTile::WIDTH,
                            dense_width: 0,
                            col_idx: 0,
                            payload: tos.dest_slot.get_idx() as u32,
                            paint,
                        });
                    }
                }
                _ => unimplemented!(),
            }
        }

        Ok(None)
    }

    /// Process a paint and return (`payload`, `paint`)
    fn process_paint(
        paint: &Paint,
        scene: &Scene,
        (scene_strip_x, scene_strip_y): (u16, u16),
    ) -> (u32, u32) {
        match paint {
            Paint::Solid(color) => {
                let rgba = color.as_premul_rgba8().to_u32();
                debug_assert!(
                    has_non_zero_alpha(rgba),
                    "Color fields with 0 alpha are reserved for clipping"
                );
                let paint_packed = (COLOR_SOURCE_PAYLOAD << 30) | (PAINT_TYPE_SOLID << 29);
                (rgba, paint_packed)
            }
            Paint::Indexed(indexed_paint) => {
                let paint_id = indexed_paint.index();
                // 16 bytes per texel: Rgba32Uint (4 bytes) * 4 (4 texels)
                let paint_tex_id = (paint_id * size_of::<GpuEncodedImage>() / 16) as u32;

                match scene.encoded_paints.get(paint_id) {
                    Some(EncodedPaint::Image(encoded_image)) => match &encoded_image.source {
                        ImageSource::OpaqueId(_) => {
                            let paint_packed = (COLOR_SOURCE_PAYLOAD << 30)
                                | (PAINT_TYPE_IMAGE << 28)
                                | (paint_tex_id & 0x0FFFFFFF);
                            let scene_strip_xy =
                                ((scene_strip_y as u32) << 16) | (scene_strip_x as u32);
                            (scene_strip_xy, paint_packed)
                        }
                        _ => unimplemented!("Unsupported image source"),
                    },
                    _ => unimplemented!("Unsupported paint type"),
                }
            }
        }
    }
}

#[inline(always)]
fn has_non_zero_alpha(rgba: u32) -> bool {
    rgba >= 0x1_00_00_00
}

/// Does a single linear scan over the wide tile commands to prepare them for `do_tile`. Notably
/// this function:
///  - Precomputes the layers that require temporary slots due to blending.
///
/// TODO: Can be triggered via a const generic on coarse draw cmd generation which will avoid
/// linear scans.
fn prepare_cmds<'a>(cmds: &'a [Cmd]) -> Vec<AnnotatedCmd<'a>> {
    let mut annotated_commands: Vec<AnnotatedCmd<'a>> = Default::default();
    // vello_hybrid cannot support non destructive in-place composites. Expand out to explicit blend
    // layers.
    let mut seen_blend_to_canvas = false;
    let mut depth = 1;
    for cmd in cmds {
        depth += match cmd {
            Cmd::PushBuf => 1,
            Cmd::PopBuf => -1,
            _ => 0,
        };
        // A blend command is issued directly to the viewport.
        seen_blend_to_canvas |= matches!(cmd, Cmd::Blend(_)) && depth == 2;
        annotated_commands.push(AnnotatedCmd::IdentityBorrowed(cmd));
    }
    if seen_blend_to_canvas {
        // We need to wrap the draw commands with an extra layer - preventing blending directly into
        // the canvas. Linear time.
        annotated_commands.insert(0, AnnotatedCmd::Generated(Cmd::PushBuf));
        annotated_commands.push(AnnotatedCmd::Generated(Cmd::Blend(BlendMode {
            mix: Mix::Normal,
            compose: Compose::SrcOver,
        })));
        annotated_commands.push(AnnotatedCmd::Generated(Cmd::PopBuf));
    }

    let mut pointer_to_push_buf_stack: Vec<usize> = Default::default();

    // Second pass: Precompute which buffers will need temporary slots for blending.
    for idx in 0..annotated_commands.len() {
        match &annotated_commands[idx].unwrap() {
            Cmd::PushBuf => {
                pointer_to_push_buf_stack.push(idx);
            }
            Cmd::PopBuf => {
                pointer_to_push_buf_stack.pop();
            }
            Cmd::Blend(_) => {
                // For blending of two layers to work in vello_hybrid, the two slots being blended
                // must be on the same texture. Hence, annotate the next-on-stack (nos) tile such
                // that it uses a temporary slot on the same texture as this blend.
                if pointer_to_push_buf_stack.len() >= 2 {
                    let push_buf_idx =
                        pointer_to_push_buf_stack[pointer_to_push_buf_stack.len() - 2];
                    annotated_commands[push_buf_idx] = AnnotatedCmd::PushBufWithTemporarySlot;
                }
            }
            _ => {}
        }
    }

    annotated_commands
}
