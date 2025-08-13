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
    /// Slots that require clearing before subsequent draws for each slot texture.
    clear: [Vec<u32>; 2],
    /// Rounds are enqueued on push clip commands and dequeued on flush.
    rounds_queue: VecDeque<Round>,
}

#[derive(Debug)]
struct PendingTileWork<'a> {
    // Used to reference the wide tile.
    wide_tile_col: u16,
    // Used to reference the wide tile.
    wide_tile_row: u16,
    next_cmd_idx: usize,
    stack: TileState,
    suspended_at_round: usize,
    annotated_cmds: Vec<AnnotatedCmd<'a>>,
}

/// A "round" is a coarse scheduling quantum.
///
/// It represents draws in up to three render targets; two for intermediate
/// clip/blend buffers, and the third for the actual render target.
#[derive(Debug, Default)]
struct Round {
    /// Draw calls scheduled into the two slot textures (0, 1) and the final target (2).
    draws: [Draw; 3],
    /// Slots that will be freed after drawing into the two slot textures [0, 1].
    free: [Vec<usize>; 2],
}

/// State for a single wide tile.
#[derive(Debug, Default)]
struct TileState {
    stack: Vec<TileEl>,
}

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
#[derive(Debug)]
enum AnnotatedCmd<'a> {
    IdentityBorrowed(&'a Cmd),
    Generated(Cmd),
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

/// `TemporarySlot` is a container that tracks whether a given tile's temporary slot is live, or
/// whether the tile never needs the temporary slot.
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
    /// `dest_slot` represents the final location of where the contents of this tile will end up. If
    /// there are layers, then painting happens to `temporary_slot` such that blending can be done
    /// to the `dest_slot` with another slot.
    dest_slot: ClaimedSlot,
    /// Temporary slot only used if we know there is another layer pushed above this tile element.
    /// INVARIANT: This slot and the `dest_slot` are always on opposite textures.
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
    fn push(&mut self, gpu_strip: GpuStrip) {
        println!("{gpu_strip:?}#");
        self.0.push(gpu_strip)
    }
}

impl Scheduler {
    pub(crate) fn new(total_slots: usize) -> Self {
        let free0: Vec<_> = (0..total_slots).collect();
        let free1 = free0.clone();
        let free = [free0, free1];
        let clear = [Vec::new(), Vec::new()];
        Self {
            round: 0,
            total_slots,
            free,
            clear,
            rounds_queue: Default::default(),
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
        self.clear[texture].push(slot_ix as u32);

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

                let tile_state = self.initialize_tile(wide_tile, wide_tile_x, wide_tile_y);
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
        while !self.rounds_queue.is_empty() || !pending_work.is_empty() {
            if !self.rounds_queue.is_empty() {
                self.flush(renderer);
            }

            // Resume pending tiles after flush
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

                println!("do tile");
                match self.do_tile(
                    renderer,
                    scene,
                    wide_tile_x,
                    wide_tile_y,
                    &work.annotated_cmds,
                    &mut tile_state,
                    work.next_cmd_idx,
                )? {
                    Some(next_cmd_idx) => {
                        // More work remains
                        pending_work.push(PendingTileWork {
                            wide_tile_col,
                            wide_tile_row,
                            next_cmd_idx,
                            stack: tile_state,
                            suspended_at_round: self.round,
                            annotated_cmds: work.annotated_cmds,
                        });
                    }
                    None => {
                        // Tile completed
                    }
                };
            }
        }

        // Restore state to reuse allocations.
        self.round = 0;
        debug_assert!(self.clear[0].is_empty(), "clear has not reset");
        debug_assert!(self.clear[1].is_empty(), "clear has not reset");
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

    /// Flush one round.
    ///
    /// The rounds queue must not be empty.
    fn flush<R: RendererBackend>(&mut self, renderer: &mut R) {
        let round = self.rounds_queue.pop_front().unwrap();
        #[cfg(debug_assertions)]
        {
            println!("=== Round {} ===", self.round);

            for (target_idx, draw) in round.draws.iter().enumerate() {
                if draw.0.is_empty() {
                    continue;
                }

                let target = match target_idx {
                    0 => "Texture 0 (even clip buffer)",
                    1 => "Texture 1 (odd clip buffer)",
                    2 => "Final Output",
                    _ => "Unknown",
                };
                let sampling = match target_idx {
                    0 => "Texture 1 (odd clip buffer)",
                    1 => "Texture 0 (even clip buffer)",
                    2 => "Texture 1 (odd clip buffer)",
                    _ => "Unknown",
                };

                println!(
                    "  Rendering to: {} by sampling slots from {}",
                    target, sampling
                );

                for (_, strip) in draw.0.iter().enumerate() {
                    if target_idx < 2 {
                        let slot_idx = strip.y / Tile::HEIGHT;
                        println!(
                            "    S: slot={} x_offset={} width={}",
                            slot_idx,
                            strip.x,
                            if strip.dense_width > 0 {
                                strip.dense_width
                            } else {
                                4
                            }
                        );
                    } else {
                        // Not a slot texture - final canvas.
                        println!(
                            "    S: pos=({}, {}) width={}",
                            strip.x,
                            strip.y,
                            if strip.dense_width > 0 {
                                strip.dense_width
                            } else {
                                4
                            }
                        );
                    }

                    let color_source = (strip.paint >> 30) & 0x3;
                    match color_source {
                        COLOR_SOURCE_PAYLOAD => {
                            let paint_type = (strip.paint >> 28) & 0x3;
                            match paint_type {
                                PAINT_TYPE_SOLID => {
                                    let a = (strip.payload >> 24) & 0xFF;
                                    let b = (strip.payload >> 16) & 0xFF;
                                    let g = (strip.payload >> 8) & 0xFF;
                                    let r = strip.payload & 0xFF;
                                    println!("      -> Solid color: RGBA({},{},{},{})", r, g, b, a);
                                }
                                PAINT_TYPE_IMAGE => {
                                    let scene_x = strip.payload & 0xFFFF;
                                    let scene_y = strip.payload >> 16;
                                    let paint_tex_id = strip.paint & 0x0FFFFFFF;
                                    println!(
                                        "      -> Image: scene_pos=({},{}) texture_id={}",
                                        scene_x, scene_y, paint_tex_id
                                    );
                                }
                                _ => println!("      -> Unknown paint type: {}", paint_type),
                            }
                        }
                        COLOR_SOURCE_SLOT => {
                            let slot = strip.payload;
                            let opacity = strip.paint & 0xFF;
                            let opacity_f = opacity as f32 / 255.0;
                            println!(
                                "      -> Sample from slot {} with opacity {:.2}",
                                slot, opacity_f
                            );
                        }

                        COLOR_SOURCE_BLEND => {
                            // Extract slots from payload
                            let src_slot = strip.payload & 0xFFFF; // bits 0-15
                            let dest_slot = (strip.payload >> 16) & 0xFFFF; // bits 16-31

                            // Extract blend parameters from paint
                            let opacity = (strip.paint >> 16) & 0xFF; // bits 16-23
                            let opacity_f = opacity as f32 / 255.0;
                            let mix_mode = (strip.paint >> 8) & 0xFF; // bits 8-15
                            let compose_mode = strip.paint & 0xFF; // bits 0-7

                            let compose_name = match compose_mode {
                                0 => "Clear",
                                1 => "Copy",
                                2 => "Dest",
                                3 => "SrcOver",
                                4 => "DestOver",
                                5 => "SrcIn",
                                6 => "DestIn",
                                7 => "SrcOut",
                                8 => "DestOut",
                                9 => "SrcAtop",
                                10 => "DestAtop",
                                11 => "Xor",
                                12 => "Plus",
                                13 => "PlusLighter",
                                _ => "Unknown",
                            };

                            let mix_name = match mix_mode {
                                0 => "Normal",
                                _ => "Unknown",
                            };

                            println!(
                                "      -> Blend: src_slot={} dest_slot={} opacity={:.2} mix={} compose={}",
                                src_slot, dest_slot, opacity_f, mix_name, compose_name
                            );
                        }
                        _ => println!("      -> Unknown color source: {}", color_source),
                    }

                    // If it's a sparse strip (dense_width > 0), show alpha column info
                    if strip.dense_width > 0 {
                        println!("      -> Alpha column index: {}", strip.col_idx);
                    }
                }
            }

            if !round.free[0].is_empty() || !round.free[1].is_empty() {
                println!("  Slots freed after rendering:");
                if !round.free[0].is_empty() {
                    println!("    Texture 0: {:?}", round.free[0]);
                }
                if !round.free[1].is_empty() {
                    println!("    Texture 1: {:?}", round.free[1]);
                }
            }

            println!();
        }
        for (i, draw) in round.draws.iter().enumerate() {
            let load = {
                if i == 2 {
                    // We're rendering to the view, don't clear.
                    LoadOp::Load
                } else if self.clear[i].len() + self.free[i].len() == self.total_slots {
                    // All slots are either unoccupied or need to be cleared.
                    // Simply clear the slots via a load operation.
                    self.clear[i].clear();
                    LoadOp::Clear
                } else {
                    // Some slots need to be preserved, so only clear the dirty slots.
                    renderer.clear_slots(i, self.clear[i].as_slice());
                    self.clear[i].clear();
                    LoadOp::Load
                }
            };

            if draw.0.is_empty() {
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
        let rel_round = el_round.saturating_sub(self.round);
        if self.rounds_queue.len() == rel_round {
            self.rounds_queue.push_back(Round::default());
        }
        &mut self.rounds_queue[rel_round].draws[texture_idx]
    }

    fn initialize_tile(
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

    /// Iterates over wide tile commands and schedules them for rendering. Returns
    /// `Some(command_idx)` if there is more work to be done. Returns `None` if the wide tile has
    /// been fully consumed.
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
            println!("{annotated_cmd:?}");
            let cmd = annotated_cmd.unwrap();
            match cmd {
                Cmd::Fill(fill) => {
                    let el = state.stack.last().unwrap();
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
                    let el = state.stack.last().unwrap();
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
                    // When we blend we consume temporary slots, and blend into the destination.
                    // This works well _once_. Unfortunately to blend to the buffer again we need to
                    // copy the destination up to a new temporary slot, and we need to clear the
                    // destination. We pull this off by:
                    //   1. Claiming a new temporary slot.
                    //   2. Creating a GpuStrip to copy from destination to temporary.
                    //   3. Free the stale destination, and claim a new one. Step 3 prevents a blend
                    // into a stale slot which hasn't been cleared yet.
                    {
                        let tos: &mut TileEl = state.stack.last_mut().unwrap();
                        if let TemporarySlot::Invalid(temp_slot) = tos.temporary_slot {
                            if has_blended {
                                // There has been a popped blend in this run of do_tile. Suspend to
                                // give the blend command a chance to flush.
                                println!("SUSPENDING 2");
                                return Ok(Some(cmd_idx));
                            }

                            let draw = self.draw_mut(tos.round - 1, temp_slot.get_texture());
                            let paint = COLOR_SOURCE_SLOT << 30 | 0xFF; // Full opacity copy
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

                            // Make sure the destination slot and temporary slot are cleared before next paint.
                            let dest_slot = tos.dest_slot;
                            self.clear[temp_slot.get_texture()].push(temp_slot.get_idx() as u32);
                            self.clear[dest_slot.get_texture()].push(dest_slot.get_idx() as u32);
                        }
                    }

                    if self.free[0].is_empty() || self.free[1].is_empty() {
                        println!("SUSPENDING");
                        return Ok(Some(cmd_idx));
                    }

                    // Push a new tile.
                    let ix = clip_depth % 2;
                    let slot = self.claim_free_slot(ix, renderer)?;
                    let temporary_slot =
                        if matches!(annotated_cmd, AnnotatedCmd::PushBufWithTemporarySlot) {
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
                    if let TemporarySlot::Valid(temp_slot) = tos.temporary_slot {
                        self.rounds_queue[round - self.round].free[temp_slot.get_texture()]
                            .push(temp_slot.get_idx());
                    }
                    if let TemporarySlot::Invalid(temp_slot) = tos.temporary_slot {
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
                    println!("{state:#?}");

                    let tos = state.stack.last().unwrap();
                    let nos = &state.stack[state.stack.len() - 2];

                    self.clear[nos.dest_slot.get_texture()].push(nos.dest_slot.get_idx() as u32);

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
///  - Removes certain optimizations that `vello_cpu` can leverage.
///  - Precomputes the layers that require temporary slots due to blending.
fn prepare_cmds<'a>(cmds: &'a [Cmd]) -> Vec<AnnotatedCmd<'a>> {
    let mut annotated_commands: Vec<AnnotatedCmd<'a>> = Default::default();
    annotated_commands.push(AnnotatedCmd::Generated(Cmd::PushBuf));
    // vello_hybrid cannot support non destructive in-place composites. Expand out to explicit blend
    // layers.
    for cmd in cmds {
        match cmd {
            Cmd::AlphaFill(fill) if fill.blend_mode.is_some() => {
                let blend_mode = fill.blend_mode.unwrap();
                annotated_commands.push(AnnotatedCmd::Generated(Cmd::PushBuf));
                let mut fill_without_blend = fill.clone();
                fill_without_blend.blend_mode = None;
                annotated_commands
                    .push(AnnotatedCmd::Generated(Cmd::AlphaFill(fill_without_blend)));
                annotated_commands.push(AnnotatedCmd::Generated(Cmd::Blend(blend_mode)));
                annotated_commands.push(AnnotatedCmd::Generated(Cmd::PopBuf));
            }
            Cmd::Fill(fill) if fill.blend_mode.is_some() => {
                let blend_mode = fill.blend_mode.unwrap();
                annotated_commands.push(AnnotatedCmd::Generated(Cmd::PushBuf));
                let mut fill_without_blend = fill.clone();
                fill_without_blend.blend_mode = None;
                annotated_commands.push(AnnotatedCmd::Generated(Cmd::Fill(fill_without_blend)));
                annotated_commands.push(AnnotatedCmd::Generated(Cmd::Blend(blend_mode)));
                annotated_commands.push(AnnotatedCmd::Generated(Cmd::PopBuf));
            }
            _ => {
                annotated_commands.push(AnnotatedCmd::IdentityBorrowed(cmd));
            }
        }
    }

    annotated_commands.push(AnnotatedCmd::Generated(Cmd::Blend(BlendMode {
        mix: Mix::Normal,
        compose: Compose::SrcOver,
    })));
    annotated_commands.push(AnnotatedCmd::Generated(Cmd::PopBuf));

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
