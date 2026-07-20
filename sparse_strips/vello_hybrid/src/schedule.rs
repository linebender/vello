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
    TextureId,
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
const PAINT_TYPE_BLURRED_ROUNDED_RECT: u32 = 5;

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

/// Specifies a run of strips inside [`Draw`] that can be drawn with the same external texture
/// binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExternalTextureRun {
    pub(crate) texture_id: TextureId,

    /// Start index of the strip range for this run. The end is implicitly the start of the next
    /// run, or, for the last run, the total number of strips.
    pub(crate) strips_start: usize,
}

/// Trait for abstracting the renderer backend from the scheduler.
pub(crate) trait RendererBackend {
    /// Clear specific slots in a texture.
    fn clear_slots(&mut self, texture_index: usize, slots: &[u32]);

    /// Execute a render pass for strips, split into opaque and alpha passes.
    ///
    /// For output targets, render the strips in opaque then alpha order with:
    ///
    /// | Pass   | Depth Test       | Depth Write | Blend | Strip Ordering |
    /// | ------ | ---------------- | ----------- | ----- | -------------- |
    /// | Opaque | ON (`LessEqual`) | ON          | OFF   | Front-to-back  |
    /// | Alpha  | ON (`LessEqual`) | OFF         | ON    | Back-to-front  |
    ///
    /// For slot textures, there are no opaque strips, so we only render the alpha strips with:
    ///
    /// | Pass   | Depth Test       | Depth Write | Blend | Strip Ordering |
    /// | ------ | ---------------- | ----------- | ----- | -------------- |
    /// | Alpha  | OFF              | OFF         | ON    | Back-to-front  |
    ///
    // TODO: Consider using opaque passes for non-output targets.
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
/// start of a render pass:
///  - `LoadOp::Load` is equivalent to `wgpu::LoadOp::Load`
///  - `LoadOp::Clear` is equivalent `wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)`
#[derive(Debug, PartialEq)]
pub(crate) enum LoadOp {
    Load,
    Clear,
}

#[derive(Clone, Copy)]
struct ProcessedPaint {
    payload: u32,
    paint: u32,
    external_texture_id: Option<TextureId>,
}

/// Cull bounds for root-destined strips, in device pixels, derived from the
/// root-pass scissor rects ([`crate::RenderConfig::scissors`]). This is
/// the prepare-side complement to those scissors: rather than only clipping in
/// the render pass, strips outside the region are never built or uploaded.
///
/// Every render pass that attaches the root target is scissored to this
/// rectangle, so a strip whose quad lies entirely outside it can never
/// contribute a pixel. The scheduler therefore skips building and uploading
/// such strips ([`generate_gpu_strips_for_fast_path`],
/// [`pack_rectangle_into_gpu`]) and skips whole wide tiles outside the
/// rectangle in root coarse batches ([`Scheduler::process_coarse_batch`]),
/// removing the per-render CPU cost that otherwise scales with the full
/// encoded scene rather than with the damaged region.
///
/// # Why this is winding-safe
///
/// Culling happens strictly *after* winding resolution. Sparse-strip
/// generation (`vello_common::strip::render`) folds a path's winding into
/// per-strip alpha coverage plus a `fill_gap` bit ("the gap between the
/// previous strip in this row and this strip is interior"), and coarse
/// rasterization (`Wide::generate`) folds those into self-contained
/// per-wide-tile command lists. At scheduling time no winding state flows
/// between emitted quads: a fill spanning the scissor boundary is emitted as
/// coverage quads at the path's edges plus interior gap quads, and each quad
/// is culled independently (a gap quad crossing the boundary is kept even
/// when the coverage strips that delimit it lie outside the scissor and are
/// culled). Wide tiles are likewise independent: a tile's commands only draw
/// within the tile's own screen rectangle, and the slot-texture scratch they
/// claim ([`Scheduler::do_push_buf`]) is consumed by that same tile's draws
/// before being freed, so skipping a tile wholesale cannot affect any other
/// tile.
///
/// Content destined for intermediate targets is never culled: filter layers
/// ([`Scheduler::process_filter_node`]) render to the filter atlas in full
/// (the root pass samples them), and slot-texture work only exists inside a
/// wide tile's command list, which is culled all-or-nothing with the tile
/// itself.
#[derive(Debug, Clone)]
pub(crate) struct StripCull {
    /// Bounding box of `rects` — left/top inclusive, right/bottom exclusive,
    /// device pixels. The cheap first test (and the whole test for a single
    /// rect).
    x0: u16,
    y0: u16,
    x1: u16,
    y1: u16,
    /// The individual scissor rects in the same edge form, populated only
    /// when there is more than one (multi-rect damage): a quad inside the
    /// bounding box must still intersect one of these to survive.
    rects: Vec<[u16; 4]>,
}

impl StripCull {
    /// Build cull bounds from a set of `[x, y, width, height]` scissor
    /// rectangles (the per-rect scissored root draws).
    ///
    /// An empty/degenerate set culls everything. Edges beyond `u16::MAX` are
    /// clamped (strip coordinates are `u16`, so nothing representable is
    /// lost).
    fn new(scissors: &[[u32; 4]]) -> Self {
        let clamp = |v: u32| v.min(u32::from(u16::MAX)) as u16;
        let mut rects: Vec<[u16; 4]> = Vec::new();
        let (mut x0, mut y0, mut x1, mut y1) = (u16::MAX, u16::MAX, 0_u16, 0_u16);
        for &[x, y, w, h] in scissors {
            if w == 0 || h == 0 {
                continue;
            }
            let r = [
                clamp(x),
                clamp(y),
                clamp(x.saturating_add(w)),
                clamp(y.saturating_add(h)),
            ];
            x0 = x0.min(r[0]);
            y0 = y0.min(r[1]);
            x1 = x1.max(r[2]);
            y1 = y1.max(r[3]);
            rects.push(r);
        }
        if rects.is_empty() {
            // Everything degenerate: cull everything.
            return Self {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
                rects,
            };
        }
        if rects.len() == 1 {
            // The bounding box IS the one rect — no per-rect pass needed.
            rects.clear();
        }
        Self {
            x0,
            y0,
            x1,
            y1,
            rects,
        }
    }

    /// Whether a quad at `(x, y)` spanning `width × height` pixels lies fully
    /// outside every scissor rect (and therefore cannot produce any pixel in
    /// the per-rect scissored root draws).
    #[inline(always)]
    fn culls_quad(&self, x: u16, y: u16, width: u16, height: u16) -> bool {
        let outside_bounds = x >= self.x1
            || y >= self.y1
            || u32::from(x) + u32::from(width) <= u32::from(self.x0)
            || u32::from(y) + u32::from(height) <= u32::from(self.y0);
        if outside_bounds || self.rects.is_empty() {
            return outside_bounds;
        }
        // Multi-rect: a quad in the gap between rects has no scissored draw.
        !self.rects.iter().any(|&[rx0, ry0, rx1, ry1]| {
            x < rx1
                && y < ry1
                && u32::from(x) + u32::from(width) > u32::from(rx0)
                && u32::from(y) + u32::from(height) > u32::from(ry0)
        })
    }

    /// The wide-tile columns intersecting the scissor bounds, clamped to `cols`.
    fn wtile_cols(&self, cols: u16) -> Range<u16> {
        let c0 = (self.x0 / WideTile::WIDTH).min(cols);
        let c1 = self.x1.div_ceil(WideTile::WIDTH).clamp(c0, cols);
        c0..c1
    }

    /// The wide-tile rows intersecting the scissor bounds, clamped to `rows`.
    fn wtile_rows(&self, rows: u16) -> Range<u16> {
        let r0 = (self.y0 / Tile::HEIGHT).min(rows);
        let r1 = self.y1.div_ceil(Tile::HEIGHT).clamp(r0, rows);
        r0..r1
    }
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
    /// See [`DepthCounter`].
    depth: DepthCounter,
    /// Cull bounds for root-destined strips, derived from the root-pass
    /// scissor for the duration of one [`Scheduler::do_scene`] call. `None`
    /// (the default, and always the value between calls) disables culling.
    /// See [`StripCull`].
    root_cull: Option<StripCull>,
    /// Total number of root-destined strip emissions (fast-path coverage,
    /// gap, and rectangle quads) skipped by scissor culling. Debug counter;
    /// see [`StripCull`].
    culled_strips: u64,
    /// Total number of wide tiles skipped wholesale by scissor culling in
    /// root coarse batches (counted per batch sweep). Debug counter; see
    /// [`StripCull`].
    culled_wide_tiles: u64,
}

/// Assigns z depth indices to GPU strips for early z rejection.
///
/// Because only opaque strips write to the depth buffer, only opaque
/// strips require unique z values (so z must be incremented for each opaque).
/// Alpha strips, however, can re-use whatever was the last z index since the
/// GPU does a LEQUAL test only.
#[derive(Debug, Default)]
struct DepthCounter {
    count: u32,
}

impl DepthCounter {
    #[inline(always)]
    fn reset(&mut self) {
        self.count = 0;
    }

    /// Returns the next depth index for early z rejection.
    #[inline(always)]
    fn next(&mut self, opaque: bool) -> u32 {
        self.count += opaque as u32;
        self.count
    }
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
struct Draw {
    /// Opaque strips: See `RendererBackend::render_strips` documentation.
    opaque: Vec<GpuStrip>,
    /// Alpha strips: See `RendererBackend::render_strips` documentation.
    alpha: Vec<GpuStrip>,
    /// Runs within [`Self::alpha`] that can be drawn with the same external texture binding.
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
            // Runs are consecutive `GpuStrip`s that can be rendered with a single external texture.
            // If the external texture changes, a new run is created. The first external texture
            // encountered creates the first run, which is "retroactively" set to start from the
            // first `GpuStrip`, such that it can be applied within the same draw call.
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

    fn clear(&mut self) {
        self.opaque.clear();
        self.alpha.clear();
        self.external_texture_runs.clear();
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
            depth: DepthCounter::default(),
            root_cull: None,
            culled_strips: 0,
            culled_wide_tiles: 0,
        }
    }

    /// Total number of root-destined strip emissions skipped by scissor
    /// culling across all renders. See [`StripCull`].
    pub(crate) fn culled_strips(&self) -> u64 {
        self.culled_strips
    }

    /// Total number of wide tiles skipped wholesale by scissor culling in
    /// root coarse batches across all renders. See [`StripCull`].
    pub(crate) fn culled_wide_tiles(&self) -> u64 {
        self.culled_wide_tiles
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
    ///
    /// `root_scissors` is the root-pass scissor rect set (N pairwise-disjoint
    /// rects from [`crate::RenderConfig::scissors`]): root-destined strips and
    /// wide tiles outside every rect are culled during scheduling (see
    /// [`StripCull`] for the mechanism and its correctness argument). Pass
    /// `&[]` (full render) to schedule everything — byte-identical to the
    /// pre-culling behavior.
    pub(crate) fn do_scene<R: RendererBackend>(
        &mut self,
        state: &mut SchedulerState,
        renderer: &mut R,
        scene: &Scene,
        root_output_target: RootRenderTarget,
        paint_idxs: &[u32],
        filter_context: &FilterContext,
        encoded_paints: &[EncodedPaint],
        root_scissors: &[[u32; 4]],
    ) -> Result<(), RenderError> {
        self.depth.reset();
        self.root_cull = (!root_scissors.is_empty()).then(|| StripCull::new(root_scissors));
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
        self.root_cull = None;
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
        let mut depth = core::mem::take(&mut self.depth);
        // TODO: Also allow the split when rendering to an atlas layer.
        let allow_opaque_split = self.is_rendering_to_user_surface();
        // Direct strips only ever draw to the (scissored) root target, so
        // quads entirely outside the scissor rects are culled (see
        // `StripCull`). Depth indices are still allocated per command so that
        // the kept strips are packed identically to a full render.
        let cull = self.root_cull.clone();
        let mut culled_strips = 0_u64;
        let draw = self.draw_mut(round, 2);

        for cmd in &scene.fast_strips_buffer.commands[range] {
            match cmd {
                FastStripCommand::Path(path) => {
                    let is_opaque = !path.paint.may_have_transparency(encoded_paints);
                    let depth_index = depth.next(is_opaque && allow_opaque_split);
                    generate_gpu_strips_for_fast_path(
                        path,
                        &strip_storage,
                        scene,
                        encoded_paints,
                        paint_idxs,
                        depth_index,
                        is_opaque && allow_opaque_split,
                        cull.as_ref(),
                        &mut culled_strips,
                        draw,
                    );
                }
                FastStripCommand::Rect(r) => {
                    let is_opaque = !r.paint.may_have_transparency(encoded_paints);
                    let depth_index = depth.next(is_opaque && allow_opaque_split);
                    pack_rectangle_into_gpu(
                        r,
                        encoded_paints,
                        paint_idxs,
                        depth_index,
                        is_opaque && allow_opaque_split,
                        cull.as_ref(),
                        &mut culled_strips,
                        draw,
                    );
                }
            }
        }
        self.depth = depth;
        self.culled_strips += culled_strips;
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
        // Root coarse batches only draw to the root target and to per-tile
        // slot scratch, so wide tiles entirely outside the root scissor are
        // skipped wholesale (see `StripCull`). The scissor is constant for
        // the whole `do_scene`, so a skipped tile is skipped in *every*
        // batch, and its `cmd_offsets` entry — never advanced — is never
        // consumed either.
        let (row_range, col_range) = match &self.root_cull {
            Some(cull) => (cull.wtile_rows(rows), cull.wtile_cols(cols)),
            None => (0..rows, 0..cols),
        };
        self.culled_wide_tiles +=
            u64::from(rows) * u64::from(cols) - row_range.len() as u64 * col_range.len() as u64;
        for row in row_range {
            for col in col_range.clone() {
                // Multi-rect damage: gap tiles are skipped like the
                // out-of-bounds ones (same never-consumed `cmd_offsets`
                // argument).
                if let Some(cull) = &self.root_cull
                    && cull.culls_quad(
                        col * WideTile::WIDTH,
                        row * Tile::HEIGHT,
                        WideTile::WIDTH,
                        Tile::HEIGHT,
                    )
                {
                    self.culled_wide_tiles += 1;
                    continue;
                }
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
        let mut round = self.rounds_queue.pop_front().unwrap();
        for (i, draw) in round.draws.iter_mut().enumerate() {
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

            if draw.is_empty() {
                if load == LoadOp::Clear {
                    // There are no strips to render, so `render_strips` will not run and won't clear
                    // the texture. We still have slots to clear this round so explicitly clear
                    // them.
                    renderer.clear_slots(i, round.clear[i].as_slice());
                }
                continue;
            }

            if i == 2 {
                // Output target: reverse opaque for front-to-back rendering.
                // Alpha strips stay in natural back-to-front order.
                let Draw {
                    opaque,
                    alpha,
                    external_texture_runs,
                } = draw;
                // This leaves `opaque` in a dirty state, but it doesn't matter because we never use it again.
                opaque.reverse();
                renderer.render_strips(opaque, alpha, external_texture_runs, target, load);
            } else {
                // Slot textures: no depth optimization, everything in alpha list.
                assert!(
                    draw.opaque.is_empty(),
                    "opaque pass unsupported for slot textures"
                );
                renderer.render_strips(&[], &draw.alpha, &draw.external_texture_runs, target, load);
            }
        }
        for i in 0..2 {
            self.free[i].extend(&round.free[i]);
        }
        self.round += 1;

        self.round_pool.return_to_pool(round);
    }

    /// Whether the scheduler is currently rendering to the final user surface
    /// (as opposed to a filter layer or slot texture).
    #[inline(always)]
    fn is_rendering_to_user_surface(&self) -> bool {
        matches!(
            self.output_target,
            StripPassRenderTarget::Root(RootRenderTarget::UserSurface)
        )
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
    /// surface.
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
            let processed = Self::process_paint(
                &Paint::Solid(tile.bg),
                encoded_paints,
                (wide_tile_x, wide_tile_y),
                idxs,
            );

            let is_opaque = tile.bg.is_opaque();
            let is_user_surface = self.is_rendering_to_user_surface();
            let bg_depth_index = self.depth.next(is_opaque && is_user_surface);
            let draw = self.draw_mut(self.round, 2);
            let strip = GpuStripBuilder::at_surface(wide_tile_x, wide_tile_y, WideTile::WIDTH)
                .paint(processed.payload, processed.paint, bg_depth_index);
            if is_opaque && is_user_surface {
                draw.push_opaque(strip);
            } else {
                draw.push_alpha(strip, processed.external_texture_id);
            }
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
                            let processed = Self::process_encoded_paint(
                                encoded_paint,
                                paint_tex_idx,
                                wide_tile_x,
                                wide_tile_y,
                            );
                            let depth_index = scheduler.depth.next(false);
                            scheduler.do_fill_with(
                                state,
                                &cmd,
                                wide_tile_x,
                                wide_tile_y,
                                processed,
                                false,
                                depth_index,
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
                let depth_index = self.depth.next(false);
                let draw = self.draw_mut(el_round, temp_slot.get_texture());
                draw.push_alpha(
                    GpuStripBuilder::at_slot(temp_slot.get_idx(), 0, WideTile::WIDTH)
                        .copy_from_slot(tos.dest_slot.get_idx(), 0xFF, depth_index),
                    None,
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
        let depth_index = self.depth.next(false);

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

            draw.push_alpha(
                gpu_strip_builder.blend(
                    tos.dest_slot.get_idx(),
                    temp_slot.get_idx(),
                    opacity_u8,
                    mix_mode,
                    compose_mode,
                    depth_index,
                ),
                None,
            );
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
            draw.push_alpha(
                gpu_strip_builder.copy_from_slot(
                    tos.dest_slot.get_idx(),
                    (tos.opacity * 255.0) as u8,
                    depth_index,
                ),
                None,
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
        let depth_index = self.depth.next(false);

        let el = state.tile_state.stack.last_mut().unwrap();
        let draw = self.draw_mut(el.round, el.get_draw_texture(depth));

        let fill_attrs = &attrs.fill[cmd.attrs_idx as usize];
        let alpha_idx = fill_attrs.alpha_idx(cmd.alpha_offset);
        let col_idx = alpha_idx / u32::from(Tile::HEIGHT);
        let (scene_strip_x, scene_strip_y) = (wide_tile_x + cmd.x, wide_tile_y);
        let processed = Self::process_paint(
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

        draw.push_alpha(
            gpu_strip_builder.with_sparse(cmd.width, col_idx).paint(
                processed.payload,
                processed.paint,
                depth_index,
            ),
            processed.external_texture_id,
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
        let is_opaque = !fill_attrs.paint.may_have_transparency(encoded_paints);
        let stack_depth = state.tile_state.stack.len();
        let is_root_opaque = stack_depth == 1 && is_opaque && self.is_rendering_to_user_surface();
        let depth_index = self.depth.next(
            // We currently only support opaques that are drawn to the user surface.
            // See TODO in `RendererBackend::render_strips`.
            is_root_opaque,
        );
        let (scene_strip_x, scene_strip_y) = (wide_tile_x + cmd.x, wide_tile_y);
        let processed = Self::process_paint(
            &fill_attrs.paint,
            encoded_paints,
            (scene_strip_x, scene_strip_y),
            paint_idxs,
        );

        self.do_fill_with(
            state,
            cmd,
            scene_strip_x,
            scene_strip_y,
            processed,
            is_root_opaque,
            depth_index,
        );
    }

    #[inline]
    fn do_fill_with(
        &mut self,
        state: &mut SchedulerState,
        cmd: &CmdFill,
        scene_strip_x: u16,
        scene_strip_y: u16,
        processed: ProcessedPaint,
        is_root_opaque: bool,
        depth_index: u32,
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

        let strip = gpu_strip_builder.paint(processed.payload, processed.paint, depth_index);
        if is_root_opaque {
            draw.push_opaque(strip);
        } else {
            draw.push_alpha(strip, processed.external_texture_id);
        }
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
        let depth_index = self.depth.next(false);
        if let TemporarySlot::Valid(temp_slot) = nos.temporary_slot {
            let draw = self.draw_mut(round, nos.dest_slot.get_texture());
            draw.push_alpha(
                GpuStripBuilder::at_slot(nos.dest_slot.get_idx(), 0, WideTile::WIDTH)
                    .copy_from_slot(temp_slot.get_idx(), 0xFF, depth_index),
                None,
            );
        }

        let depth_index = self.depth.next(false);
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
        draw.push_alpha(
            gpu_strip_builder.copy_from_slot(tos.dest_slot.get_idx(), 0xFF, depth_index),
            None,
        );

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

        let depth_index = self.depth.next(false);
        // If nos has a temporary slot, copy it to `dest_slot` first
        if let TemporarySlot::Valid(temp_slot) = nos.temporary_slot {
            let draw = self.draw_mut(round, nos.dest_slot.get_texture());
            draw.push_alpha(
                GpuStripBuilder::at_slot(nos.dest_slot.get_idx(), 0, WideTile::WIDTH)
                    .copy_from_slot(temp_slot.get_idx(), 0xFF, depth_index),
                None,
            );
        }

        let depth_index = self.depth.next(false);
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

        draw.push_alpha(
            gpu_strip_builder
                .with_sparse(cmd.width, col_idx)
                .copy_from_slot(tos.dest_slot.get_idx(), 0xFF, depth_index),
            None,
        );
        let nos_ptr = state.tile_state.stack.len() - 2;
        state.tile_state.stack[nos_ptr].temporary_slot.invalidate();
    }

    /// Process a paint and return the packed payload, paint and optional external texture id.
    #[inline(always)]
    fn process_paint(
        paint: &Paint,
        encoded_paints: &[EncodedPaint],
        (scene_strip_x, scene_strip_y): (u16, u16),
        paint_idxs: &[u32],
    ) -> ProcessedPaint {
        match paint {
            Paint::Solid(color) => {
                let rgba = color.as_premul_rgba8().to_u32();
                let paint_packed = (COLOR_SOURCE_PAYLOAD << 30) | (PAINT_TYPE_SOLID << 27);
                ProcessedPaint {
                    payload: rgba,
                    paint: paint_packed,
                    external_texture_id: None,
                }
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
    ) -> ProcessedPaint {
        match encoded_paint {
            EncodedPaint::Image(encoded_image) => match &encoded_image.source {
                ImageSource::OpaqueId { .. } => {
                    let paint_packed = (COLOR_SOURCE_PAYLOAD << 29)
                        | (PAINT_TYPE_IMAGE << 26)
                        | (paint_idx & 0x03FF_FFFF);
                    let scene_strip_xy = ((scene_strip_y as u32) << 16) | (scene_strip_x as u32);
                    ProcessedPaint {
                        payload: scene_strip_xy,
                        paint: paint_packed,
                        external_texture_id: None,
                    }
                }
                _ => unimplemented!("Unsupported image source"),
            },
            EncodedPaint::ExternalTexture(texture) => {
                let paint_packed = (COLOR_SOURCE_PAYLOAD << 29)
                    | (PAINT_TYPE_IMAGE << 26)
                    | (paint_idx & 0x03FF_FFFF);
                let scene_strip_xy = ((scene_strip_y as u32) << 16) | (scene_strip_x as u32);
                ProcessedPaint {
                    payload: scene_strip_xy,
                    paint: paint_packed,
                    external_texture_id: Some(texture.texture_id),
                }
            }
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
                ProcessedPaint {
                    payload: scene_strip_xy,
                    paint: paint_packed,
                    external_texture_id: None,
                }
            }
            EncodedPaint::BlurredRoundedRect(_) => {
                let paint_packed = (COLOR_SOURCE_PAYLOAD << 29)
                    | (PAINT_TYPE_BLURRED_ROUNDED_RECT << 26)
                    | (paint_idx & 0x03FF_FFFF);
                let scene_strip_xy = ((scene_strip_y as u32) << 16) | (scene_strip_x as u32);
                ProcessedPaint {
                    payload: scene_strip_xy,
                    paint: paint_packed,
                    external_texture_id: None,
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
    fn paint(self, payload: u32, paint: u32, depth_index: u32) -> GpuStrip {
        GpuStrip {
            x: self.x,
            y: self.y,
            width: self.width,
            dense_width_or_rect_height: self.dense_width_or_rect_height,
            col_idx_or_rect_frac: self.col_idx_or_rect_frac,
            payload,
            paint_and_rect_flag: paint,
            depth_index,
        }
    }

    /// Copy from slot.
    fn copy_from_slot(self, from_slot: usize, opacity: u8, depth_index: u32) -> GpuStrip {
        GpuStrip {
            x: self.x,
            y: self.y,
            width: self.width,
            dense_width_or_rect_height: self.dense_width_or_rect_height,
            col_idx_or_rect_frac: self.col_idx_or_rect_frac,
            payload: u32::try_from(from_slot).unwrap(),
            paint_and_rect_flag: (COLOR_SOURCE_SLOT << 29) | (opacity as u32),
            depth_index,
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
        depth_index: u32,
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
            depth_index,
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
    depth_index: u32,
    is_opaque: bool,
    cull: Option<&StripCull>,
    culled_strips: &mut u64,
    draw: &mut Draw,
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
        //
        // The coverage quad and the gap quad below are culled independently
        // against the root scissor: winding is already resolved into
        // coverage + `fill_gap` at this point (see `StripCull`), so a gap
        // quad reaching into the scissor is kept even when the coverage
        // strips delimiting it are culled.
        if strip_width > 0 {
            if cull.is_some_and(|c| c.culls_quad(x0, y, strip_width, Tile::HEIGHT)) {
                *culled_strips += 1;
            } else {
                let processed =
                    Scheduler::process_paint(&path.paint, encoded_paints, (x0, y), paint_idxs);
                draw.push_alpha(
                    GpuStripBuilder::at_surface(x0, y, strip_width)
                        .with_sparse(strip_width, col)
                        .paint(processed.payload, processed.paint, depth_index),
                    processed.external_texture_id,
                );
            }
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
                if cull.is_some_and(|c| c.culls_quad(x1, y, x2 - x1, Tile::HEIGHT)) {
                    *culled_strips += 1;
                } else {
                    let processed =
                        Scheduler::process_paint(&path.paint, encoded_paints, (x1, y), paint_idxs);
                    let strip = GpuStripBuilder::at_surface(x1, y, x2 - x1).paint(
                        processed.payload,
                        processed.paint,
                        depth_index,
                    );
                    if is_opaque {
                        draw.push_opaque(strip);
                    } else {
                        draw.push_alpha(strip, processed.external_texture_id);
                    }
                }
            }
        }
    }
}

fn pack_rectangle_into_gpu(
    rect: &FastPathRect,
    encoded_paints: &[EncodedPaint],
    paint_idxs: &[u32],
    depth_index: u32,
    is_opaque: bool,
    cull: Option<&StripCull>,
    culled_strips: &mut u64,
    draw: &mut Draw,
) {
    let split = split_rect(rect);

    let mut is_first = true;
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
        // The main part is always the first entry; remember that before the
        // cull below so a culled main part cannot promote an AA edge part
        // into the opaque pass.
        let is_main = is_first;
        is_first = false;

        // Each split part is an independent quad; parts entirely outside the
        // root scissor are culled (see `StripCull`).
        if cull.is_some_and(|c| c.culls_quad(part.x, part.y, part.width, part.height)) {
            *culled_strips += 1;
            continue;
        }

        let processed =
            Scheduler::process_paint(&rect.paint, encoded_paints, (part.x, part.y), paint_idxs);
        let strip = make_gpu_rect(part, processed.payload, processed.paint, depth_index);
        if is_main && is_opaque && part.frac == 0 {
            draw.push_opaque(strip);
        } else {
            draw.push_alpha(strip, processed.external_texture_id);
        }
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

fn make_gpu_rect(part: RectPart, payload: u32, paint_packed: u32, depth_index: u32) -> GpuStrip {
    GpuStrip {
        x: part.x,
        y: part.y,
        width: part.width,
        dense_width_or_rect_height: part.height,
        col_idx_or_rect_frac: part.frac,
        payload,
        paint_and_rect_flag: paint_packed | RECT_STRIP_FLAG,
        depth_index,
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
        Draw, ExternalTextureRun, GpuStrip, LoadOp, RECT_STRIP_FLAG, RectPart, RendererBackend,
        RootRenderTarget, Scheduler, SchedulerState, SplitRect, StripCull, StripPassRenderTarget,
        TextureId, Tile, generate_gpu_strips_for_fast_path, pack_rectangle_into_gpu, pack_unorm4x8,
        split_rect,
    };
    use crate::Scene;
    use crate::filter::FilterContext;
    use crate::scene::{FastPathRect, FastStripCommand};
    use alloc::vec;
    use alloc::vec::Vec;
    use vello_common::encode::EncodedImage;
    use vello_common::kurbo::{Affine, BezPath, Rect, Vec2};
    use vello_common::multi_atlas::AtlasConfig;
    use vello_common::paint::{Color, ImageId, ImageSource, IndexedPaint, Paint};
    use vello_common::peniko::ImageSampler;
    use vello_common::render_graph::LayerId;

    const DUMMY_STRIP: GpuStrip = GpuStrip {
        x: 0,
        y: 0,
        width: 1,
        dense_width_or_rect_height: 1,
        col_idx_or_rect_frac: 0,
        payload: 0,
        paint_and_rect_flag: 0,
        depth_index: 0,
    };

    #[test]
    fn draw_external_texture_runs() {
        let texture_a = TextureId(1);
        let texture_b = TextureId(2);
        let mut draw = Draw::default();

        draw.push_alpha(DUMMY_STRIP, None);
        assert!(draw.external_texture_runs.is_empty());

        draw.push_alpha(DUMMY_STRIP, Some(texture_a));
        draw.push_alpha(DUMMY_STRIP, Some(texture_a));
        draw.push_alpha(DUMMY_STRIP, None);
        draw.push_alpha(DUMMY_STRIP, Some(texture_b));
        draw.push_alpha(DUMMY_STRIP, None);
        draw.push_alpha(DUMMY_STRIP, Some(texture_b));
        draw.push_alpha(DUMMY_STRIP, Some(texture_a));

        assert_eq!(
            draw.external_texture_runs,
            vec![
                ExternalTextureRun {
                    strips_start: 0,
                    texture_id: texture_a,
                },
                ExternalTextureRun {
                    strips_start: 4,
                    texture_id: texture_b,
                },
                ExternalTextureRun {
                    strips_start: 7,
                    texture_id: texture_a,
                },
            ]
        );
    }

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
        let mut draw = Draw::default();

        pack_rectangle_into_gpu(&rect, &[], &[], 0, true, None, &mut 0, &mut draw);

        let out: Vec<_> = draw.opaque.iter().chain(draw.alpha.iter()).collect();
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
            may_have_transparency: false,
            transform: Affine::IDENTITY,
            x_advance: Vec2::new(1.0, 0.0),
            y_advance: Vec2::new(0.0, 1.0),
            tint: None,
        })];
        let mut draw = Draw::default();

        pack_rectangle_into_gpu(
            &rect,
            &encoded_paints,
            &[7],
            0,
            true,
            None,
            &mut 0,
            &mut draw,
        );

        let out: Vec<_> = draw.opaque.iter().chain(draw.alpha.iter()).collect();
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].payload, (21_u32 << 16) | 11_u32);
        assert_eq!(out[1].payload, (20_u32 << 16) | 10_u32);
        assert_eq!(out[2].payload, (52_u32 << 16) | 10_u32);
        assert_eq!(out[3].payload, (21_u32 << 16) | 10_u32);
        assert_eq!(out[4].payload, (21_u32 << 16) | 42_u32);
    }

    // ------------------------------------------------------------------
    // Prepare-side scissor strip culling (see `StripCull`).
    // ------------------------------------------------------------------

    /// All eight `GpuStrip` fields, for exact comparisons (`GpuStrip` itself
    /// deliberately doesn't derive `PartialEq`).
    type StripKey = (u16, u16, u16, u16, u32, u32, u32, u32);

    fn strip_key(s: &GpuStrip) -> StripKey {
        (
            s.x,
            s.y,
            s.width,
            s.dense_width_or_rect_height,
            s.col_idx_or_rect_frac,
            s.payload,
            s.paint_and_rect_flag,
            s.depth_index,
        )
    }

    #[test]
    fn strip_cull_quad_and_tile_range_semantics() {
        // Scissor x ∈ [32, 80), y ∈ [16, 56).
        let c = StripCull::new(&[[32, 16, 48, 40]]);

        // Inside and 1px overlaps are kept.
        assert!(!c.culls_quad(32, 16, 1, 1));
        assert!(!c.culls_quad(0, 16, 33, 10), "1px x-overlap must be kept");
        assert!(
            !c.culls_quad(79, 55, 100, 100),
            "corner overlap must be kept"
        );

        // Half-open boundaries: touching without overlapping is culled.
        assert!(c.culls_quad(80, 16, 10, 10), "starts at the right edge");
        assert!(c.culls_quad(0, 16, 32, 10), "ends at the left edge");
        assert!(c.culls_quad(32, 56, 10, 10), "starts at the bottom edge");
        assert!(c.culls_quad(32, 0, 10, 16), "ends at the top edge");

        // Wide-tile ranges (256×4 tiles) and their clamping.
        assert_eq!(c.wtile_cols(5), 0..1);
        assert_eq!(c.wtile_rows(240), 4..14);
        assert_eq!(c.wtile_rows(10), 4..10);
        assert_eq!(StripCull::new(&[[600, 0, 100, 4]]).wtile_cols(2), 2..2);

        // An empty scissor culls everything.
        let e = StripCull::new(&[[32, 16, 0, 40]]);
        assert!(e.culls_quad(0, 0, u16::MAX, u16::MAX));
        assert_eq!(e.wtile_cols(5), 0..0);
        assert_eq!(e.wtile_rows(240), 0..0);
    }

    #[test]
    fn strip_cull_multi_rect_culls_the_gap() {
        // Two disjoint rects (opposite corners): quads inside either rect
        // survive, quads in the GAP — inside the bounding box, outside both
        // rects — are culled, and the wide-tile ranges span the bounding box.
        let c = StripCull::new(&[[10, 10, 40, 40], [200, 200, 40, 40]]);
        assert!(!c.culls_quad(12, 12, 8, 8), "inside rect 1");
        assert!(!c.culls_quad(210, 210, 8, 8), "inside rect 2");
        assert!(
            c.culls_quad(100, 100, 8, 8),
            "gap quad (inside bounds, outside both rects)"
        );
        assert!(
            c.culls_quad(45, 100, 8, 8),
            "gap quad under rect 1's columns"
        );
        assert!(
            !c.culls_quad(45, 45, 160, 160),
            "quad straddling into a rect survives"
        );
        assert!(c.culls_quad(300, 10, 8, 8), "outside the bounding box");
        // Bounding-box tile ranges (WideTile::WIDTH = 256, Tile::HEIGHT = 4).
        assert_eq!(c.wtile_cols(4), 0..1);
        assert_eq!(c.wtile_rows(240), 2..60);
        // Degenerate members are dropped; a lone survivor collapses to the
        // single-rect fast path (no per-rect scan).
        let one = StripCull::new(&[[10, 10, 40, 40], [5, 5, 0, 9]]);
        assert!(one.culls_quad(60, 12, 4, 4), "outside the lone survivor");
        assert!(!one.culls_quad(12, 12, 4, 4));
    }

    #[test]
    fn rect_parts_cull_independently_against_scissor() {
        // Large rect with AA on all four edges → 5 split parts.
        let rect = solid_rect(10.25, 20.5, 90.75, 80.25);

        let mut full = Draw::default();
        pack_rectangle_into_gpu(&rect, &[], &[], 3, true, None, &mut 0, &mut full);
        assert_eq!(full.opaque.len() + full.alpha.len(), 5);

        // Scissor x ∈ [0, 11): keeps the left AA column plus the top/bottom
        // edge strips (both start at x = 10); culls the main part (starts at
        // x = 11) and the right AA column.
        let cull = StripCull::new(&[[0, 0, 11, 200]]);
        let mut culled = 0_u64;
        let mut part = Draw::default();
        pack_rectangle_into_gpu(
            &rect,
            &[],
            &[],
            3,
            true,
            Some(&cull),
            &mut culled,
            &mut part,
        );

        assert_eq!(culled, 2, "main + right AA column must be culled");
        assert!(
            part.opaque.is_empty(),
            "the culled main part must not promote an AA edge into the opaque pass"
        );
        let expected: Vec<StripKey> = full
            .opaque
            .iter()
            .chain(full.alpha.iter())
            .filter(|s| s.x < 11)
            .map(strip_key)
            .collect();
        let kept: Vec<StripKey> = part.alpha.iter().map(strip_key).collect();
        assert_eq!(
            kept, expected,
            "kept parts must be byte-identical to the full run's"
        );
    }

    #[test]
    fn fast_path_strips_cull_per_emitted_quad() {
        // A triangle spanning rows 2..48, so both coverage quads and interior
        // gap quads exist on both sides of the scissor band.
        let mut scene = Scene::new(200, 200);
        scene.set_paint(Color::from_rgba8(255, 0, 0, 255));
        let mut path = BezPath::new();
        path.move_to((10.0, 2.0));
        path.line_to((190.0, 25.0));
        path.line_to((10.0, 48.0));
        path.close_path();
        scene.fill_path(&path);

        let storage = scene.strip_storage.borrow();
        let FastStripCommand::Path(path_cmd) = &scene.fast_strips_buffer.commands[0] else {
            panic!("triangle must take the fast strip path");
        };

        let mut full = Draw::default();
        generate_gpu_strips_for_fast_path(
            path_cmd,
            &storage,
            &scene,
            &[],
            &[],
            7,
            true,
            None,
            &mut 0,
            &mut full,
        );
        let full_total = full.opaque.len() + full.alpha.len();
        assert!(full_total > 0, "reference run must emit strips");

        // A strip-row-aligned horizontal band: y ∈ [16, 32).
        let cull = StripCull::new(&[[0, 16, 200, 16]]);
        let in_band = |s: &GpuStrip| s.y < 32 && u32::from(s.y) + u32::from(Tile::HEIGHT) > 16;
        let mut culled = 0_u64;
        let mut part = Draw::default();
        generate_gpu_strips_for_fast_path(
            path_cmd,
            &storage,
            &scene,
            &[],
            &[],
            7,
            true,
            Some(&cull),
            &mut culled,
            &mut part,
        );

        // Kept strips are byte-identical to the full run's in-band strips
        // (same depth index, same packing); everything else is counted.
        let expected_opaque: Vec<StripKey> = full
            .opaque
            .iter()
            .filter(|s| in_band(s))
            .map(strip_key)
            .collect();
        let expected_alpha: Vec<StripKey> = full
            .alpha
            .iter()
            .filter(|s| in_band(s))
            .map(strip_key)
            .collect();
        assert!(
            !expected_opaque.is_empty(),
            "band must contain interior gap fills"
        );
        assert!(
            !expected_alpha.is_empty(),
            "band must contain coverage strips"
        );
        assert_eq!(
            part.opaque.iter().map(strip_key).collect::<Vec<_>>(),
            expected_opaque
        );
        assert_eq!(
            part.alpha.iter().map(strip_key).collect::<Vec<_>>(),
            expected_alpha
        );
        let kept_total = part.opaque.len() + part.alpha.len();
        assert_eq!(culled as usize, full_total - kept_total);

        // An empty scissor culls every emission.
        let empty = StripCull::new(&[[50, 50, 0, 0]]);
        let mut all_culled = 0_u64;
        let mut none = Draw::default();
        generate_gpu_strips_for_fast_path(
            path_cmd,
            &storage,
            &scene,
            &[],
            &[],
            7,
            true,
            Some(&empty),
            &mut all_culled,
            &mut none,
        );
        assert!(none.opaque.is_empty() && none.alpha.is_empty());
        assert_eq!(all_culled as usize, full_total);
    }

    /// Records every strip pass the scheduler issues.
    #[derive(Default)]
    struct RecordingBackend {
        passes: Vec<(Vec<GpuStrip>, Vec<GpuStrip>, StripPassRenderTarget)>,
    }

    impl RendererBackend for RecordingBackend {
        fn clear_slots(&mut self, _texture_index: usize, _slots: &[u32]) {}

        fn render_strips(
            &mut self,
            opaque_strips: &[GpuStrip],
            alpha_strips: &[GpuStrip],
            _external_texture_runs: &[ExternalTextureRun],
            target: StripPassRenderTarget,
            _load_op: LoadOp,
        ) {
            self.passes
                .push((opaque_strips.to_vec(), alpha_strips.to_vec(), target));
        }

        fn apply_filter(&mut self, _layer_id: LayerId) {}
    }

    impl RecordingBackend {
        /// All strips issued against the root target, in issue order.
        fn root_strips(&self) -> Vec<GpuStrip> {
            self.passes
                .iter()
                .filter(|(_, _, target)| matches!(target, StripPassRenderTarget::Root(_)))
                .flat_map(|(opaque, alpha, _)| opaque.iter().chain(alpha.iter()).copied())
                .collect()
        }

        /// Total number of strips issued against the slot textures.
        fn slot_strip_count(&self) -> usize {
            self.passes
                .iter()
                .filter(|(_, _, target)| matches!(target, StripPassRenderTarget::SlotTexture(_)))
                .map(|(opaque, alpha, _)| opaque.len() + alpha.len())
                .sum()
        }
    }

    /// 512×16 scene (2 wide-tile columns × 4 rows) that goes through coarse
    /// rasterization (a clip layer), with content on both sides of the
    /// wide-tile boundary at x = 256.
    fn coarse_scene() -> Scene {
        let mut scene = Scene::new(512, 16);
        scene.set_paint(Color::from_rgba8(255, 0, 0, 255));
        // Root-level rect crossing the wide-tile boundary.
        scene.fill_rect(&Rect::new(200.0, 2.0, 300.0, 10.0));
        // Clip layer spanning both wide-tile columns (forces `CoarseOnly`).
        let mut clip = BezPath::new();
        clip.move_to((100.0, 0.0));
        clip.line_to((400.0, 16.0));
        clip.line_to((100.0, 16.0));
        clip.close_path();
        scene.push_clip_layer(&clip);
        scene.set_paint(Color::from_rgba8(0, 255, 0, 128));
        scene.fill_rect(&Rect::new(0.0, 0.0, 512.0, 16.0));
        scene.pop_layer();
        scene
    }

    fn run_scheduler(scene: &Scene, root_scissors: &[[u32; 4]]) -> (RecordingBackend, u64) {
        let mut scheduler = Scheduler::new(64);
        let mut state = SchedulerState::default();
        let mut backend = RecordingBackend::default();
        let filter_context = FilterContext::new(AtlasConfig::default());
        scheduler
            .do_scene(
                &mut state,
                &mut backend,
                scene,
                RootRenderTarget::UserSurface,
                &[],
                &filter_context,
                &[],
                root_scissors,
            )
            .expect("scheduling must succeed");
        (backend, scheduler.culled_wide_tiles())
    }

    #[test]
    fn coarse_batches_skip_wide_tiles_outside_scissor() {
        let scene = coarse_scene();

        let (full, full_culled_tiles) = run_scheduler(&scene, &[]);
        assert_eq!(full_culled_tiles, 0, "no scissor ⇒ no culled tiles");
        let full_root = full.root_strips();
        assert!(
            full_root.iter().any(|s| s.x < 256) && full_root.iter().any(|s| s.x >= 256),
            "reference run must emit root strips in both wide-tile columns"
        );
        assert!(
            full.slot_strip_count() > 0,
            "clip content must use slot textures"
        );

        // Scissor entirely within wide-tile column 1: x ∈ [320, 384).
        let (part, culled_tiles) = run_scheduler(&scene, &[[320, 0, 64, 16]]);
        assert_eq!(culled_tiles, 4, "the 4 tiles of column 0 must be skipped");
        let part_root = part.root_strips();
        assert!(
            part_root.iter().all(|s| s.x >= 256),
            "no root strip may originate from a culled tile"
        );

        // Kept tiles schedule the same root-target geometry as the full run
        // (slot indices and depth indices may differ once tiles are skipped,
        // so compare geometry only; byte-exact pixel equality inside the
        // scissor is pinned by the GPU tests).
        let geometry = |strips: &[GpuStrip]| {
            let mut v: Vec<(u16, u16, u16, u16)> = strips
                .iter()
                .map(|s| (s.x, s.y, s.width, s.dense_width_or_rect_height))
                .collect();
            v.sort_unstable();
            v
        };
        let expected: Vec<_> = full_root.iter().filter(|s| s.x >= 256).copied().collect();
        assert_eq!(geometry(&part_root), geometry(&expected));
        assert!(
            part.slot_strip_count() < full.slot_strip_count(),
            "clip content of culled tiles must not reach the slot textures"
        );

        // An empty scissor culls every tile and issues no strips at all.
        let (none, all_culled) = run_scheduler(&scene, &[[0, 0, 0, 0]]);
        assert_eq!(all_culled, 8);
        assert!(
            none.passes
                .iter()
                .all(|(o, a, _)| o.is_empty() && a.is_empty())
        );
    }
}
