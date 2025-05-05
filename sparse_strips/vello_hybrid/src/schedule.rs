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
//! rendition.
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
//! For more information about this algorithm, see this [Zulip thread].
//!
//! [Zulip thread]: https://xi.zulipchat.com/#narrow/channel/197075-vello/topic/Spatiotemporal.20allocation.20.28hybrid.29/near/513442829

use crate::{GpuStrip, RenderError, Scene, render::RendererJunk};
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::mem;
use vello_common::{
    coarse::{Cmd, WideTile},
    paint::Paint,
    tile::Tile,
};

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
    /// State for a single wide tile.
    tile_state: TileState,
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

// State for a single wide tile.
#[derive(Debug, Default)]
struct TileState {
    stack: Vec<TileEl>,
}

#[derive(Clone, Copy, Debug)]
struct TileEl {
    slot_ix: usize,
    round: usize,
}

#[derive(Debug, Default)]
struct Draw(Vec<GpuStrip>);

impl Scheduler {
    pub(crate) fn new(total_slots: usize) -> Self {
        let free0: Vec<_> = (0..total_slots).collect();
        let free1 = free0.clone();
        let free = [free0, free1];
        let clear = [Vec::new(), Vec::new()];
        let mut rounds_queue = VecDeque::new();
        rounds_queue.push_back(Round::default());
        Self {
            round: 0,
            total_slots,
            free,
            clear,
            rounds_queue,
            tile_state: Default::default(),
        }
    }

    pub(crate) fn do_scene(
        &mut self,
        junk: &mut RendererJunk<'_>,
        scene: &Scene,
    ) -> Result<(), RenderError> {
        let mut tile_state = mem::take(&mut self.tile_state);
        let wide_tiles_per_row = (scene.width).div_ceil(WideTile::WIDTH);
        let wide_tiles_per_col = (scene.height).div_ceil(Tile::HEIGHT);

        // Left to right, top to bottom iteration over wide tiles.
        for wide_tile_row in 0..wide_tiles_per_col {
            for wide_tile_col in 0..wide_tiles_per_row {
                let wide_tile_idx = usize::from(wide_tile_row * wide_tiles_per_row + wide_tile_col);
                let wide_tile = &scene.wide.tiles[wide_tile_idx];
                let wide_tile_x = wide_tile_col * WideTile::WIDTH;
                let wide_tile_y = wide_tile_row * Tile::HEIGHT;
                self.do_tile(junk, wide_tile_x, wide_tile_y, wide_tile, &mut tile_state)?;
            }
        }
        while !self.rounds_queue.is_empty() {
            self.flush(junk);
        }

        // Restore state to reuse allocations.
        self.round = 0;
        self.tile_state = tile_state;
        self.tile_state.stack.clear();
        debug_assert!(self.clear[0].is_empty(), "clear has not reset");
        debug_assert!(self.clear[1].is_empty(), "clear has not reset");
        if cfg!(debug_assertions) {
            for i in 0..self.total_slots {
                debug_assert!(self.free[0].contains(&i), "free[0] is missing slot {}", i);
                debug_assert!(self.free[1].contains(&i), "free[1] is missing slot {}", i);
            }
        }
        debug_assert!(self.rounds_queue.is_empty(), "rounds_queue is not empty");

        Ok(())
    }

    /// Flush one round.
    ///
    /// The rounds queue must not be empty.
    fn flush(&mut self, junk: &mut RendererJunk<'_>) {
        let round = self.rounds_queue.pop_front().unwrap();
        for (i, draw) in round.draws.iter().enumerate() {
            if draw.0.is_empty() {
                continue;
            }

            let load = {
                if i == 2 {
                    // We're rendering to the view, don't clear.
                    wgpu::LoadOp::Load
                } else if self.clear[i].len() + self.free[i].len() == self.total_slots {
                    // All slots are either unoccupied or need to be cleared.
                    // Simply clear the slots via a load operation.
                    self.clear[i].clear();
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                } else {
                    // Some slots need to be preserved, so only clear the dirty slots.
                    junk.do_clear_slots_render_pass(i, self.clear[i].as_slice());
                    self.clear[i].clear();
                    wgpu::LoadOp::Load
                }
            };
            junk.do_strip_render_pass(&draw.0, i, load);
        }
        for i in 0..2 {
            self.free[i].extend(&round.free[i]);
        }
        self.round += 1;
    }

    // Find the appropriate draw call for rendering.
    fn draw_mut(&mut self, el_round: usize, clip_depth: usize) -> &mut Draw {
        let ix = if clip_depth == 1 {
            // We can draw to the final target
            2
        } else {
            1 - clip_depth % 2
        };
        let rel_round = el_round.saturating_sub(self.round);
        if self.rounds_queue.len() == rel_round {
            self.rounds_queue.push_back(Round::default());
        }
        &mut self.rounds_queue[rel_round].draws[ix]
    }

    /// Iterates over wide tile commands and schedules them for rendering.
    fn do_tile(
        &mut self,
        junk: &mut RendererJunk<'_>,
        wide_tile_x: u16,
        wide_tile_y: u16,
        tile: &WideTile,
        state: &mut TileState,
    ) -> Result<(), RenderError> {
        state.stack.clear();
        // Sentinel `TileEl` to indicate the end of the stack where we draw all
        // commands to the final target.
        state.stack.push(TileEl {
            slot_ix: !0,
            round: self.round,
        });
        let bg = tile.bg.as_premul_rgba8().to_u32();
        // If the background has a non-zero alpha then we need to render it.
        if has_non_zero_alpha(bg) {
            let draw = self.draw_mut(self.round, 1);
            draw.0.push(GpuStrip {
                x: wide_tile_x,
                y: wide_tile_y,
                width: WideTile::WIDTH,
                dense_width: 0,
                col: 0,
                rgba: bg,
            });
        }
        for cmd in &tile.cmds {
            // Note: this starts at 1 (for the final target)
            let clip_depth = state.stack.len();
            match cmd {
                Cmd::Fill(fill) => {
                    let el = state.stack.last().unwrap();
                    let draw = self.draw_mut(el.round, clip_depth);
                    let color = match fill.paint {
                        Paint::Solid(color) => color,
                        Paint::Indexed(_) => unimplemented!(),
                    };
                    let rgba = color.as_premul_rgba8().to_u32();
                    debug_assert!(
                        has_non_zero_alpha(rgba),
                        "Color fields with 0 alpha are reserved for clipping"
                    );
                    let (x, y) = if clip_depth == 1 {
                        (wide_tile_x + fill.x, wide_tile_y)
                    } else {
                        (fill.x, el.slot_ix as u16 * Tile::HEIGHT)
                    };
                    draw.0.push(GpuStrip {
                        x,
                        y,
                        width: fill.width,
                        dense_width: 0,
                        col: 0,
                        rgba,
                    });
                }
                Cmd::AlphaFill(alpha_fill) => {
                    let el = state.stack.last().unwrap();
                    let draw = self.draw_mut(el.round, clip_depth);
                    let color = match alpha_fill.paint {
                        Paint::Solid(color) => color,
                        Paint::Indexed(_) => unimplemented!(),
                    };
                    let rgba = color.as_premul_rgba8().to_u32();
                    debug_assert!(
                        has_non_zero_alpha(rgba),
                        "Color fields with 0 alpha are reserved for clipping"
                    );
                    let (x, y) = if clip_depth == 1 {
                        (wide_tile_x + alpha_fill.x, wide_tile_y)
                    } else {
                        (alpha_fill.x, el.slot_ix as u16 * Tile::HEIGHT)
                    };
                    draw.0.push(GpuStrip {
                        x,
                        y,
                        width: alpha_fill.width,
                        dense_width: alpha_fill.width,
                        col: (alpha_fill.alpha_idx / usize::from(Tile::HEIGHT))
                            .try_into()
                            .expect("Sparse strips are bound to u32 range"),
                        rgba,
                    });
                }
                Cmd::PushBuf => {
                    let ix = clip_depth % 2;
                    while self.free[ix].is_empty() {
                        if self.rounds_queue.is_empty() {
                            return Err(RenderError::SlotsExhausted);
                        }
                        self.flush(junk);
                    }
                    let slot_ix = self.free[ix].pop().unwrap();
                    self.clear[ix].push(slot_ix as u32);
                    state.stack.push(TileEl {
                        slot_ix,
                        round: self.round,
                    });
                }
                Cmd::PopBuf => {
                    let tos = state.stack.pop().unwrap();
                    let nos = state.stack.last_mut().unwrap();
                    let next_round = clip_depth % 2 == 0 && clip_depth > 2;
                    let round = nos.round.max(tos.round + next_round as usize);
                    nos.round = round;
                    // free slot after draw
                    debug_assert!(round >= self.round, "round must be after current round");
                    debug_assert!(
                        round - self.round < self.rounds_queue.len(),
                        "round must be in queue"
                    );
                    self.rounds_queue[round - self.round].free[1 - clip_depth % 2]
                        .push(tos.slot_ix);
                }
                Cmd::ClipFill(clip_fill) => {
                    let tos = &state.stack[clip_depth - 1];
                    let nos = &state.stack[clip_depth - 2];
                    let next_round = clip_depth % 2 == 0 && clip_depth > 2;
                    let round = nos.round.max(tos.round + next_round as usize);
                    let draw = self.draw_mut(round, clip_depth - 1);
                    let (x, y) = if clip_depth <= 2 {
                        (wide_tile_x + clip_fill.x as u16, wide_tile_y)
                    } else {
                        (clip_fill.x as u16, nos.slot_ix as u16 * Tile::HEIGHT)
                    };
                    draw.0.push(GpuStrip {
                        x,
                        y,
                        width: clip_fill.width as u16,
                        dense_width: 0,
                        col: 0,
                        rgba: tos.slot_ix as u32,
                    });
                }
                Cmd::ClipStrip(clip_alpha_fill) => {
                    let tos = &state.stack[clip_depth - 1];
                    let nos = &state.stack[clip_depth - 2];
                    let next_round = clip_depth % 2 == 0 && clip_depth > 2;
                    let round = nos.round.max(tos.round + next_round as usize);
                    let draw = self.draw_mut(round, clip_depth - 1);
                    let (x, y) = if clip_depth <= 2 {
                        (wide_tile_x + clip_alpha_fill.x as u16, wide_tile_y)
                    } else {
                        (clip_alpha_fill.x as u16, nos.slot_ix as u16 * Tile::HEIGHT)
                    };
                    draw.0.push(GpuStrip {
                        x,
                        y,
                        width: clip_alpha_fill.width as u16,
                        dense_width: clip_alpha_fill.width as u16,
                        col: (clip_alpha_fill.alpha_idx / usize::from(Tile::HEIGHT))
                            .try_into()
                            .expect("Sparse strips are bound to u32 range"),
                        rgba: tos.slot_ix as u32,
                    });
                }
                _ => unimplemented!(),
            }
        }

        Ok(())
    }
}

#[inline(always)]
fn has_non_zero_alpha(rgba: u32) -> bool {
    rgba >= 0x1_00_00_00
}
