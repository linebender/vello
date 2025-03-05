// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::api::peniko::color::{AlphaColor, Srgb};

pub(crate) const WIDE_TILE_WIDTH: usize = 256;
pub(crate) const STRIP_HEIGHT: usize = 4;

#[derive(Debug)]
pub(crate) struct WideTile {
    pub(crate) bg: AlphaColor<Srgb>,
    pub(crate) cmds: Vec<Cmd>,
    n_zero_clip: usize,
    n_clip: usize,
}

#[derive(Debug)]
pub(crate) enum Cmd {
    Fill(CmdFill),
    Strip(CmdStrip),
    /// Pushes a new transparent buffer to the clip stack.
    PushClip,
    /// Pops the clip stack.
    PopClip,
    ClipFill(CmdClipFill),
    ClipStrip(CmdClipStrip),
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

/// Same as fill, but copies top of clip stack to next on stack.
#[derive(Debug)]
pub(crate) struct CmdClipFill {
    pub(crate) x: u32,
    pub(crate) width: u32,
    // TODO: this should probably get at least an alpha for group opacity
    // Also, this is where blend modes go.
}

/// Same as strip, but composites top of clip stack to next on stack.
#[derive(Debug)]
pub(crate) struct CmdClipStrip {
    pub(crate) x: u32,
    pub(crate) width: u32,
    pub(crate) alpha_ix: usize,
    // See `CmdClipFill` for blending extension points
}

impl Default for WideTile {
    fn default() -> Self {
        Self {
            bg: AlphaColor::TRANSPARENT,
            cmds: vec![],
            n_zero_clip: 0,
            n_clip: 0,
        }
    }
}

impl WideTile {
    pub(crate) fn fill(&mut self, x: u32, width: u32, color: AlphaColor<Srgb>) {
        if !self.is_zero_clip() {
            // Note that we could be more aggressive in optimizing a whole-tile opaque fill
            // even with a clip stack. It would be valid to elide all drawing commands from
            // the enclosing clip push up to the fill. Further, we could extend the clip
            // push command to include a background color, rather than always starting with
            // a transparent buffer. Lastly, a sequence of push(bg); strip/fill; pop could
            // be replaced with strip/fill with the color (the latter is true even with a
            // non-opaque color).
            //
            // However, the extra cost of tracking such optimizations may outweigh the
            // benefit, especially in hybrid mode with GPU painting.
            if x == 0
                && width == WIDE_TILE_WIDTH as u32
                && color.components[3] == 1.0
                && self.n_clip == 0
            {
                self.cmds.clear();
                self.bg = color;
            } else {
                self.cmds.push(Cmd::Fill(CmdFill { x, width, color }));
            }
        }
    }

    pub(crate) fn strip(&mut self, cmd_strip: CmdStrip) {
        if !self.is_zero_clip() {
            self.cmds.push(Cmd::Strip(cmd_strip));
        }
    }

    pub(crate) fn push(&mut self, cmd: Cmd) {
        self.cmds.push(cmd);
    }

    pub(crate) fn push_clip(&mut self) {
        if !self.is_zero_clip() {
            self.push(Cmd::PushClip);
            self.n_clip += 1;
        }
    }

    pub(crate) fn pop_clip(&mut self) {
        if !self.is_zero_clip() {
            if matches!(self.cmds.last(), Some(Cmd::PushClip)) {
                // Nothing was drawn inside the clip, elide it.
                self.cmds.pop();
            } else {
                self.push(Cmd::PopClip);
            }
            self.n_clip -= 1;
        }
    }

    pub(crate) fn push_zero_clip(&mut self) {
        self.n_zero_clip += 1;
    }

    pub(crate) fn pop_zero_clip(&mut self) {
        self.n_zero_clip -= 1;
    }

    pub(crate) fn is_zero_clip(&mut self) -> bool {
        self.n_zero_clip > 0
    }

    pub(crate) fn clip_strip(&mut self, cmd_clip_strip: CmdClipStrip) {
        if !self.is_zero_clip() && !matches!(self.cmds.last(), Some(Cmd::PushClip)) {
            self.cmds.push(Cmd::ClipStrip(cmd_clip_strip));
        }
    }

    pub(crate) fn clip_fill(&mut self, x: u32, width: u32) {
        if !self.is_zero_clip() && !matches!(self.cmds.last(), Some(Cmd::PushClip)) {
            self.cmds.push(Cmd::ClipFill(CmdClipFill { x, width }));
        }
    }
}
