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

use crate::{GpuStrip, RenderError, Scene};
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use vello_common::coarse::{CommandAttrs, MODE_HYBRID};
use vello_common::peniko::{BlendMode, Compose, Mix};
use vello_common::{
    coarse::{Cmd, LayerKind, WideTile},
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

// The sentinel tile index representing the surface.
const SENTINEL_SLOT_IDX: usize = usize::MAX;

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
/// the need to do linear scans.
#[derive(Debug)]
enum AnnotatedCmd<'a> {
    /// A wrapped command - no semantic meaning added.
    IdentityBorrowed(&'a Cmd),
    PushBuf,
    Empty,
    SrcOverNormalBlend,
    PopBuf,
    /// A `Cmd::PushBuf` that will be blended into by the layer above it. Will need a
    /// `temporary_slot` created for it to enable compositing.
    PushBufWithTemporarySlot,
}

impl<'a> AnnotatedCmd<'a> {
    fn as_cmd<'b: 'a>(&'b self) -> Option<&'a Cmd> {
        match self {
            AnnotatedCmd::IdentityBorrowed(cmd) => Some(cmd),
            AnnotatedCmd::PushBufWithTemporarySlot => Some(&Cmd::PushBuf(LayerKind::Regular(0))),
            AnnotatedCmd::PushBuf => Some(&Cmd::PushBuf(LayerKind::Regular(0))),
            AnnotatedCmd::SrcOverNormalBlend => Some(&Cmd::Blend(BlendMode {
                mix: Mix::Normal,
                compose: Compose::SrcOver,
            })),
            AnnotatedCmd::PopBuf => Some(&Cmd::PopBuf),
            AnnotatedCmd::Empty => None,
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
            rounds_queue: VecDeque::new(),
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

        match texture {
            0 => Ok(ClaimedSlot::Texture0(slot_ix)),
            1 => Ok(ClaimedSlot::Texture1(slot_ix)),
            _ => panic!("invalid slot texture"),
        }
    }

    pub(crate) fn do_scene<R: RendererBackend>(
        &mut self,
        renderer: &mut R,
        scene: &Scene,
        paint_idxs: &[u32],
    ) -> Result<(), RenderError> {
        let wide_tiles_per_row = scene.wide.width_tiles();
        let wide_tiles_per_col = scene.wide.height_tiles();

        // Left to right, top to bottom iteration over wide tiles.
        for wide_tile_row in 0..wide_tiles_per_col {
            for wide_tile_col in 0..wide_tiles_per_row {
                let wide_tile = scene.wide.get(wide_tile_col, wide_tile_row);
                let wide_tile_x = wide_tile_col * WideTile::WIDTH;
                let wide_tile_y = wide_tile_row * Tile::HEIGHT;

                let tile_state = self.initialize_tile_state(
                    wide_tile,
                    wide_tile_x,
                    wide_tile_y,
                    scene,
                    paint_idxs,
                );
                let annotated_cmds = prepare_cmds(&wide_tile.cmds);
                self.do_tile(
                    renderer,
                    scene,
                    wide_tile_x,
                    wide_tile_y,
                    &annotated_cmds,
                    tile_state,
                    paint_idxs,
                    &scene.wide.attrs,
                )?;
            }
        }

        while !self.rounds_queue.is_empty() {
            self.flush(renderer);
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
    #[inline(always)]
    fn get_round(&mut self, el_round: usize) -> &mut Round {
        let rel_round = el_round.saturating_sub(self.round);
        if self.rounds_queue.len() == rel_round {
            self.rounds_queue.push_back(Round::default());
        }
        &mut self.rounds_queue[rel_round]
    }

    fn initialize_tile_state(
        &mut self,
        tile: &WideTile<MODE_HYBRID>,
        wide_tile_x: u16,
        wide_tile_y: u16,
        scene: &Scene,
        idxs: &[u32],
    ) -> TileState {
        let mut state = TileState::default();
        // Sentinel `TileEl` to indicate the end of the stack where we draw all
        // commands to the final target.
        state.stack.push(TileEl {
            dest_slot: ClaimedSlot::Texture0(SENTINEL_SLOT_IDX),
            temporary_slot: TemporarySlot::None,
            round: self.round,
            opacity: 1.,
        });
        {
            // If the background has a non-zero alpha then we need to render it.
            let bg = tile.bg.as_premul_rgba8().to_u32();
            if has_non_zero_alpha(bg) {
                let (payload, paint) = Self::process_paint(
                    &Paint::Solid(tile.bg),
                    scene,
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
        cmds: &'a [AnnotatedCmd<'a>],
        mut state: TileState,
        paint_idxs: &[u32],
        attrs: &CommandAttrs,
    ) -> Result<(), RenderError> {
        for annotated_cmd in cmds {
            // Note: this starts at 1 (for the final target)
            let depth = state.stack.len();
            let Some(cmd) = annotated_cmd.as_cmd() else {
                continue;
            };

            match cmd {
                Cmd::Fill(fill) => {
                    let el = state.stack.last_mut().unwrap();
                    let draw = self.draw_mut(el.round, el.get_draw_texture(depth));

                    let fill_attrs = &attrs.fill[fill.attrs_idx as usize];
                    let (scene_strip_x, scene_strip_y) = (wide_tile_x + fill.x, wide_tile_y);
                    let (payload, paint) = Self::process_paint(
                        &fill_attrs.paint,
                        scene,
                        (scene_strip_x, scene_strip_y),
                        paint_idxs,
                    );

                    let gpu_strip_builder = if depth == 1 {
                        GpuStripBuilder::at_surface(scene_strip_x, scene_strip_y, fill.width)
                    } else {
                        let slot_idx = if let TemporarySlot::Valid(temp_slot) = el.temporary_slot {
                            temp_slot.get_idx()
                        } else {
                            el.dest_slot.get_idx()
                        };
                        GpuStripBuilder::at_slot(slot_idx, fill.x, fill.width)
                    };

                    draw.push(gpu_strip_builder.paint(payload, paint));
                }
                Cmd::AlphaFill(alpha_fill) => {
                    let el = state.stack.last_mut().unwrap();
                    let draw = self.draw_mut(el.round, el.get_draw_texture(depth));

                    let fill_attrs = &attrs.fill[alpha_fill.attrs_idx as usize];
                    let alpha_idx = fill_attrs.alpha_idx(alpha_fill.alpha_offset);
                    let col_idx = alpha_idx / u32::from(Tile::HEIGHT);
                    let (scene_strip_x, scene_strip_y) = (wide_tile_x + alpha_fill.x, wide_tile_y);
                    let (payload, paint) = Self::process_paint(
                        &fill_attrs.paint,
                        scene,
                        (scene_strip_x, scene_strip_y),
                        paint_idxs,
                    );

                    let gpu_strip_builder = if depth == 1 {
                        GpuStripBuilder::at_surface(scene_strip_x, scene_strip_y, alpha_fill.width)
                    } else {
                        let slot_idx = if let TemporarySlot::Valid(temp_slot) = el.temporary_slot {
                            temp_slot.get_idx()
                        } else {
                            el.dest_slot.get_idx()
                        };
                        GpuStripBuilder::at_slot(slot_idx, alpha_fill.x, alpha_fill.width)
                    };

                    draw.push(
                        gpu_strip_builder
                            .with_sparse(alpha_fill.width, col_idx)
                            .paint(payload, paint),
                    );
                }
                Cmd::PushBuf(_layer_id) => {
                    // TODO: Handle layer_id for filter effects when implemented.
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

                            // Make sure the destination slot and temporary slot are cleared
                            // appropriately.
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
                    {
                        let round = self.get_round(self.round);
                        round.clear[slot.get_texture()].push(slot.get_idx() as u32);
                    }
                    let temporary_slot =
                        if matches!(annotated_cmd, AnnotatedCmd::PushBufWithTemporarySlot) {
                            let temp_slot = self.claim_free_slot((ix + 1) % 2, renderer)?;
                            let round = self.get_round(self.round);
                            round.clear[temp_slot.get_texture()].push(temp_slot.get_idx() as u32);
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
                    let next_round = depth.is_multiple_of(2) && depth > 2;
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
                    let tos: &TileEl = &state.stack[depth - 1];
                    let nos = &state.stack[depth - 2];

                    // Basically if we are writing onto the even texture, we need to go up a round
                    // to target it.
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
                        GpuStripBuilder::at_surface(
                            wide_tile_x + clip_fill.x,
                            wide_tile_y,
                            clip_fill.width,
                        )
                    } else {
                        GpuStripBuilder::at_slot(
                            nos.dest_slot.get_idx(),
                            clip_fill.x,
                            clip_fill.width,
                        )
                    };
                    draw.push(gpu_strip_builder.copy_from_slot(tos.dest_slot.get_idx(), 0xFF));

                    let nos_ptr = state.stack.len() - 2;
                    state.stack[nos_ptr].temporary_slot.invalidate();
                }
                Cmd::ClipStrip(clip_alpha_fill) => {
                    let tos = &state.stack[depth - 1];
                    let nos = &state.stack[depth - 2];

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
                        GpuStripBuilder::at_surface(
                            wide_tile_x + clip_alpha_fill.x,
                            wide_tile_y,
                            clip_alpha_fill.width,
                        )
                    } else {
                        GpuStripBuilder::at_slot(
                            nos.dest_slot.get_idx(),
                            clip_alpha_fill.x,
                            clip_alpha_fill.width,
                        )
                    };

                    let clip_attrs = &attrs.clip[clip_alpha_fill.attrs_idx as usize];
                    let alpha_idx = clip_attrs.alpha_idx(clip_alpha_fill.alpha_offset);
                    let col_idx = alpha_idx / u32::from(Tile::HEIGHT);

                    draw.push(
                        gpu_strip_builder
                            .with_sparse(clip_alpha_fill.width, col_idx)
                            .copy_from_slot(tos.dest_slot.get_idx(), 0xFF),
                    );
                    let nos_ptr = state.stack.len() - 2;
                    state.stack[nos_ptr].temporary_slot.invalidate();
                }
                Cmd::Opacity(opacity) => {
                    state.stack.last_mut().unwrap().opacity = *opacity;
                }
                Cmd::Blend(mode) => {
                    let tos = state.stack.last().unwrap();
                    let nos = &state.stack[state.stack.len() - 2];

                    let next_round: bool = depth.is_multiple_of(2) && depth > 2;
                    let round = nos.round.max(tos.round + usize::from(next_round));

                    if let TemporarySlot::Valid(temp_slot) = nos.temporary_slot {
                        let draw = self.draw_mut(
                            round,
                            if depth <= 2 {
                                2
                            } else {
                                nos.dest_slot.get_texture()
                            },
                        );
                        let opacity_u8 = (tos.opacity * 255.0) as u8;
                        let mix_mode = mode.mix as u8;
                        let compose_mode = mode.compose as u8;

                        let gpu_strip_builder = if depth <= 2 {
                            GpuStripBuilder::at_surface(wide_tile_x, wide_tile_y, WideTile::WIDTH)
                        } else {
                            GpuStripBuilder::at_slot(nos.dest_slot.get_idx(), 0, WideTile::WIDTH)
                        };

                        draw.push(gpu_strip_builder.blend(
                            tos.dest_slot.get_idx(),
                            temp_slot.get_idx(),
                            opacity_u8,
                            mix_mode,
                            compose_mode,
                        ));
                        // Invalidate the temporary slot after use
                        let nos_ptr = state.stack.len() - 2;
                        state.stack[nos_ptr].temporary_slot.invalidate();
                    } else {
                        assert_eq!(
                            nos.dest_slot.get_idx(),
                            SENTINEL_SLOT_IDX,
                            "code path only for copying to sentinel slot, {mode:?}"
                        );

                        let draw = self.draw_mut(round, tos.get_draw_texture(depth - 1));
                        draw.push(
                            GpuStripBuilder::at_surface(wide_tile_x, wide_tile_y, WideTile::WIDTH)
                                .copy_from_slot(
                                    tos.dest_slot.get_idx(),
                                    (tos.opacity * 255.0) as u8,
                                ),
                        );
                    }
                }
                _ => unimplemented!(),
            }
        }

        Ok(())
    }

    /// Process a paint and return (`payload`, `paint`)
    #[inline(always)]
    fn process_paint(
        paint: &Paint,
        scene: &Scene,
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

                match scene.encoded_paints.get(paint_id) {
                    Some(EncodedPaint::Image(encoded_image)) => match &encoded_image.source {
                        ImageSource::OpaqueId(_) => {
                            let paint_packed = (COLOR_SOURCE_PAYLOAD << 30)
                                | (PAINT_TYPE_IMAGE << 27)
                                | (paint_idx & 0x07FFFFFF);
                            let scene_strip_xy =
                                ((scene_strip_y as u32) << 16) | (scene_strip_x as u32);
                            (scene_strip_xy, paint_packed)
                        }
                        _ => unimplemented!("Unsupported image source"),
                    },
                    Some(EncodedPaint::Gradient(gradient)) => {
                        use vello_common::encode::EncodedKind;
                        let gradient_paint_type = match &gradient.kind {
                            EncodedKind::Linear(_) => PAINT_TYPE_LINEAR_GRADIENT,
                            EncodedKind::Radial(_) => PAINT_TYPE_RADIAL_GRADIENT,
                            EncodedKind::Sweep(_) => PAINT_TYPE_SWEEP_GRADIENT,
                        };
                        let paint_packed = (COLOR_SOURCE_PAYLOAD << 30)
                            | (gradient_paint_type << 27)
                            | (paint_idx & 0x07FFFFFF);
                        let scene_strip_xy =
                            ((scene_strip_y as u32) << 16) | (scene_strip_x as u32);
                        (scene_strip_xy, paint_packed)
                    }

                    _ => unimplemented!("Unsupported paint type"),
                }
            }
        }
    }
}

/// Helper for more semantically constructing `GpuStrip`s.
struct GpuStripBuilder {
    x: u16,
    y: u16,
    width: u16,
    dense_width: u16,
    col_idx: u32,
}

impl GpuStripBuilder {
    /// Position at surface coordinates.
    fn at_surface(x: u16, y: u16, width: u16) -> Self {
        Self {
            x,
            y,
            width,
            dense_width: 0,
            col_idx: 0,
        }
    }

    /// Position within a slot.
    fn at_slot(slot_idx: usize, x_offset: u16, width: u16) -> Self {
        Self {
            x: x_offset,
            y: u16::try_from(slot_idx).unwrap() * Tile::HEIGHT,
            width,
            dense_width: 0,
            col_idx: 0,
        }
    }

    /// Add sparse strip parameters.
    fn with_sparse(mut self, dense_width: u16, col_idx: u32) -> Self {
        self.dense_width = dense_width;
        self.col_idx = col_idx;
        self
    }

    /// Paint into strip.
    fn paint(self, payload: u32, paint: u32) -> GpuStrip {
        GpuStrip {
            x: self.x,
            y: self.y,
            width: self.width,
            dense_width: self.dense_width,
            col_idx: self.col_idx,
            payload,
            paint,
        }
    }

    /// Copy from slot.
    fn copy_from_slot(self, from_slot: usize, opacity: u8) -> GpuStrip {
        GpuStrip {
            x: self.x,
            y: self.y,
            width: self.width,
            dense_width: self.dense_width,
            col_idx: self.col_idx,
            payload: u32::try_from(from_slot).unwrap(),
            paint: (COLOR_SOURCE_SLOT << 30) | (opacity as u32),
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
            dense_width: self.dense_width,
            col_idx: self.col_idx,
            payload: (u32::try_from(src_slot).unwrap())
                | ((u32::try_from(dest_slot).unwrap()) << 16),
            paint: (COLOR_SOURCE_BLEND << 30)
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

/// Does a single linear scan over the wide tile commands to prepare them for `do_tile`. Notably
/// this function:
///  - Precomputes the layers that require temporary slots due to blending.
///
/// TODO: Can be triggered via a const generic on coarse draw cmd generation which will avoid
/// a linear scan.
fn prepare_cmds<'a>(cmds: &'a [Cmd]) -> Vec<AnnotatedCmd<'a>> {
    // Reserve room for three extra items such that we can prevent repeated blends into the surface.
    let mut annotated_commands: Vec<AnnotatedCmd<'a>> = Vec::with_capacity(cmds.len() + 3);
    let mut pointer_to_push_buf_stack: Vec<usize> = Vec::new();
    // We pretend that the surface might be blended into. This will be removed if no blends occur to
    // the surface.
    pointer_to_push_buf_stack.push(0);
    annotated_commands.push(AnnotatedCmd::PushBuf);
    for cmd in cmds {
        match cmd {
            Cmd::PushBuf(_layer_id) => {
                // TODO: Handle layer_id for filter effects when implemented.
                pointer_to_push_buf_stack.push(annotated_commands.len());
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
        };

        annotated_commands.push(AnnotatedCmd::IdentityBorrowed(cmd));
    }

    if matches!(
        annotated_commands[0],
        AnnotatedCmd::PushBufWithTemporarySlot
    ) {
        // We need to wrap the draw commands with an extra layer - preventing blending directly into
        // the surface.
        annotated_commands.push(AnnotatedCmd::SrcOverNormalBlend);
        annotated_commands.push(AnnotatedCmd::PopBuf);
    } else {
        // This extra wrapping can be removed.
        annotated_commands[0] = AnnotatedCmd::Empty;
    }
    annotated_commands
}
