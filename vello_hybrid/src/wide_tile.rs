// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_api::peniko::color::{AlphaColor, Srgb};

pub(crate) const WIDE_TILE_WIDTH: usize = 256;
pub(crate) const STRIP_HEIGHT: usize = 4;

pub(crate) struct WideTile {
    pub(crate) bg: AlphaColor<Srgb>,
    pub(crate) cmds: Vec<Cmd>,
}

#[derive(Debug)]
pub(crate) enum Cmd {
    Fill(CmdFill),
    Strip(CmdStrip),
}

#[derive(Debug)]
pub(crate) struct CmdFill {
    pub(crate) x: u32,
    pub(crate) width: u32,
    // TODO: Probably want this pre-packed to u32 to avoid packing cost
    pub(crate) color: AlphaColor<Srgb>,
}

#[derive(Debug)]
pub(crate) struct CmdStrip {
    pub(crate) x: u32,
    pub(crate) width: u32,
    pub(crate) alpha_ix: usize,
    pub(crate) color: AlphaColor<Srgb>,
}

impl Default for WideTile {
    fn default() -> Self {
        Self {
            bg: AlphaColor::TRANSPARENT,
            cmds: vec![],
        }
    }
}

impl WideTile {
    pub(crate) fn fill(&mut self, x: u32, width: u32, color: AlphaColor<Srgb>) {
        if x == 0 && width == WIDE_TILE_WIDTH as u32 && color.components[3] == 1.0 {
            self.cmds.clear();
            self.bg = color;
        } else {
            self.cmds.push(Cmd::Fill(CmdFill { x, width, color }));
        }
    }

    pub(crate) fn push(&mut self, cmd: Cmd) {
        self.cmds.push(cmd);
    }
}
