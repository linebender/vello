// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Reusable schedule storage shared across rendered frames.

use super::buffer::Ranges;
use super::draw::{Draw, OpaqueStrips, OpaqueStripsExt};
use super::round::{FilterPasses, LayerTexturePass, Round};
use crate::GpuStrip;
use alloc::vec::Vec;
use vello_common::geometry::RectU16;
use vello_common::util::Pool;

#[derive(Debug, Default)]
pub(super) struct Pools {
    opaque_strips: Pool<Vec<GpuStrip>>,
    root_draws: Pool<Draw>,
    layer_draws: Pool<Draw>,
    filter_ops: Pool<Ranges>,
    filter_passes: Pool<FilterPasses>,
    blend_ranges: Pool<Ranges>,
    layer_texture_clears: Pool<Vec<RectU16>>,
    scratch_texture_clears: Pool<Vec<RectU16>>,
}

impl Pools {
    pub(super) fn take_opaque_strips(&mut self, enabled: bool) -> OpaqueStrips {
        if enabled {
            Some(self.opaque_strips.take())
        } else {
            OpaqueStrips::new(false)
        }
    }

    pub(super) fn submit_opaque_strips(&mut self, strips: OpaqueStrips) {
        if let Some(strips) = strips {
            self.opaque_strips.submit(strips);
        }
    }

    fn take_root_draw(&mut self) -> Draw {
        self.root_draws.take()
    }

    fn submit_root_draw(&mut self, draw: Draw) {
        self.root_draws.submit(draw);
    }

    fn take_layer_draw(&mut self) -> Draw {
        self.layer_draws.take()
    }

    fn submit_layer_draw(&mut self, draw: Draw) {
        self.layer_draws.submit(draw);
    }

    fn take_layer_texture_pass(&mut self) -> LayerTexturePass {
        LayerTexturePass {
            draw: self.take_layer_draw(),
            filter_ranges: self.filter_ops.take(),
            filter_passes: self.filter_passes.take(),
            blend_ranges: self.blend_ranges.take(),
        }
    }

    fn submit_layer_texture_pass(&mut self, layer_texture_pass: LayerTexturePass) {
        self.submit_layer_draw(layer_texture_pass.draw);
        self.filter_ops.submit(layer_texture_pass.filter_ranges);
        self.filter_passes.submit(layer_texture_pass.filter_passes);
        self.blend_ranges.submit(layer_texture_pass.blend_ranges);
    }

    fn take_layer_texture_clears(&mut self) -> [Vec<RectU16>; 2] {
        [
            self.layer_texture_clears.take(),
            self.layer_texture_clears.take(),
        ]
    }

    fn submit_layer_texture_clears(&mut self, clears: [Vec<RectU16>; 2]) {
        for rects in clears {
            self.layer_texture_clears.submit(rects);
        }
    }

    fn take_scratch_texture_clears(&mut self) -> [Vec<RectU16>; 2] {
        [
            self.scratch_texture_clears.take(),
            self.scratch_texture_clears.take(),
        ]
    }

    fn submit_scratch_texture_clears(&mut self, clears: [Vec<RectU16>; 2]) {
        for rects in clears {
            self.scratch_texture_clears.submit(rects);
        }
    }

    pub(super) fn take_round(&mut self) -> Round {
        Round {
            root_draw: self.take_root_draw(),
            layer_texture_passes: [
                self.take_layer_texture_pass(),
                self.take_layer_texture_pass(),
            ],
            layer_texture_clears: self.take_layer_texture_clears(),
            scratch_texture_clears: self.take_scratch_texture_clears(),
        }
    }

    pub(super) fn submit_round(&mut self, round: Round) {
        self.submit_root_draw(round.root_draw);
        for layer_texture_pass in round.layer_texture_passes {
            self.submit_layer_texture_pass(layer_texture_pass);
        }
        self.submit_layer_texture_clears(round.layer_texture_clears);
        self.submit_scratch_texture_clears(round.scratch_texture_clears);
    }
}
