// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::collections::VecDeque;

use vello_common::{
    coarse::{Cmd, WideTile},
    paint::Paint,
    tile::Tile,
};

use crate::{GpuStrip, Scene, render::RendererJunk};

pub(crate) struct Schedule {
    /// Index of the current round
    round: usize,
    free: [Vec<usize>; 2],
    rounds: VecDeque<Round>,
}

/// A "round" is a coarse scheduling quantum.
///
/// It represents draws in up to three render targets; two for intermediate
/// clip/blend buffers, and the third for the actual render target. The two
/// clip buffers are for even and odd clip depths.
#[derive(Default)]
struct Round {
    draws: [Draw; 3],
    /// Slots that will be freed after the draws
    free: [Vec<usize>; 2],
}

/// State for a single tile.
///
/// Perhaps this should just be a field in the scheduler.
#[derive(Default)]
struct TileState {
    stack: Vec<TileEl>,
}

#[derive(Clone, Copy)]
struct TileEl {
    slot_ix: usize,
    round: usize,
}

#[derive(Default)]
struct Draw(Vec<GpuStrip>);

impl Schedule {
    pub(crate) fn new(n_slots: usize) -> Self {
        let free0: Vec<_> = (0..n_slots).collect();
        let free1 = free0.clone();
        let free = [free0, free1];
        let mut rounds = VecDeque::new();
        rounds.push_back(Round::default());
        Self {
            round: 0,
            free,
            rounds,
        }
    }

    pub(crate) fn do_scene(&mut self, junk: &mut RendererJunk<'_>, scene: &Scene) {
        let mut state = TileState::default();
        let wide_tiles_per_row = (scene.width).div_ceil(WideTile::WIDTH);
        let wide_tiles_per_col = (scene.height).div_ceil(Tile::HEIGHT);
        for wide_tile_row in 0..wide_tiles_per_col {
            for wide_tile_col in 0..wide_tiles_per_row {
                let wide_tile_idx = usize::from(wide_tile_row) * usize::from(wide_tiles_per_row)
                    + usize::from(wide_tile_col);
                let wide_tile = &scene.wide.tiles[wide_tile_idx];
                let wide_tile_x = wide_tile_col * WideTile::WIDTH;
                let wide_tile_y = wide_tile_row * Tile::HEIGHT;
                self.do_tile(junk, wide_tile_x, wide_tile_y, wide_tile, &mut state);
            }
        }
        while !self.rounds.is_empty() {
            self.flush(junk);
        }
    }

    /// Flush one round.
    ///
    /// The rounds queue must not be empty.
    fn flush(&mut self, junk: &mut RendererJunk<'_>) {
        let round = self.rounds.pop_front().unwrap();
        for (i, draw) in round.draws.iter().enumerate() {
            if !draw.0.is_empty() {
                junk.do_render_pass(&draw.0, self.round, i);
            }
        }
        for i in 0..1 {
            self.free[i].extend(&round.free[i]);
        }
        self.round += 1;
    }

    #[allow(clippy::todo, reason = "still working on this")]
    fn do_tile(
        &mut self,
        junk: &mut RendererJunk<'_>,
        wide_tile_x: u16,
        wide_tile_y: u16,
        tile: &WideTile,
        state: &mut TileState,
    ) {
        state.stack.clear();
        state.stack.push(TileEl {
            slot_ix: !0,
            round: self.round,
        });
        let bg = tile.bg.to_u32();
        if bg >= 0x1_00_00_00 {
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
                    let rgba = color.to_u32();
                    // color fields with 0 alpha are reserved for clipping
                    if rgba >= 0x1_00_00_00 {
                        // TODO: x and y base coordinates are from wide_tile if
                        // clip depth is 1, otherwise point to slot ix
                        draw.0.push(GpuStrip {
                            x: wide_tile_x + fill.x,
                            y: wide_tile_y,
                            width: fill.width,
                            dense_width: 0,
                            col: 0,
                            rgba,
                        });
                    }
                }
                Cmd::AlphaFill(alpha_fill) => {
                    let el = state.stack.last().unwrap();
                    let draw = self.draw_mut(el.round, clip_depth);
                    let color = match alpha_fill.paint {
                        Paint::Solid(color) => color,
                        Paint::Indexed(_) => unimplemented!(),
                    };
                    let rgba = color.to_u32();
                    // color fields with 0 alpha are reserved for clipping
                    if rgba >= 0x1_00_00_00 {
                        // msg is a variable here to work around rustfmt failure
                        let msg = "GpuStrip fields use u32 and values are expected to fit within that range";
                        draw.0.push(GpuStrip {
                            x: wide_tile_x + alpha_fill.x,
                            y: wide_tile_y,
                            width: alpha_fill.width,
                            dense_width: alpha_fill.width,
                            col: (alpha_fill.alpha_idx / usize::from(Tile::HEIGHT))
                                .try_into()
                                .expect(msg),
                            rgba,
                        });
                    }
                }
                Cmd::PushClip => {
                    let ix = clip_depth % 2;
                    while self.free[ix].is_empty() {
                        if self.rounds.is_empty() {
                            // Probably should return error here
                            panic!("failed to allocate slot");
                        }
                        self.flush(junk);
                    }
                    let slot_ix = self.free[ix].pop().unwrap();
                    // Note: the allocated slot will need to get cleared before
                    // drawing, maybe add it to a clear list. Of course, if all slots
                    // can be cleared, then do clear with `LoadOp::Clear` instead.
                    state.stack.push(TileEl {
                        slot_ix,
                        round: self.round,
                    });
                }
                Cmd::PopClip => {
                    let tos = state.stack.pop().unwrap();
                    let nos = state.stack.last_mut().unwrap();
                    let next_round = clip_depth % 2 == 0 && clip_depth > 2;
                    let round = nos.round.max(tos.round + next_round as usize);
                    nos.round = round;
                    // free slot after draw
                    // TODO: ensure round exists
                    // TODO: saturating_sub here, or do we have guarantee round >= self.round?
                    self.rounds[round - self.round].free[1 - clip_depth % 2].push(tos.slot_ix);
                }
                Cmd::ClipFill(_cmd_clip_fill) => {
                    let next_round = clip_depth % 2 == 0 && clip_depth > 2;
                    let tos = &state.stack[clip_depth - 1];
                    let nos = &state.stack[clip_depth - 2];
                    let round = nos.round.max(tos.round + next_round as usize);
                    let _draw = self.draw_mut(round, clip_depth - 1);
                    // TODO: push GpuStrip; use `tos.slot_x` for rgba field
                }
                Cmd::ClipStrip(_cmd_clip_alpha_fill) => todo!(),
            }
        }
    }

    // Find the appropriate draw call for rendering.
    fn draw_mut(&mut self, el_round: usize, clip_depth: usize) -> &mut Draw {
        let rel_round = el_round.saturating_sub(self.round);
        let ix = if clip_depth == 1 {
            2
        } else {
            1 - clip_depth % 2
        };
        if self.rounds.len() == rel_round {
            self.rounds.push_back(Round::default());
        }
        &mut self.rounds[rel_round].draws[ix]
    }
}
