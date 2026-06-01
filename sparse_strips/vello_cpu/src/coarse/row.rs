// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::cmd::{
    BlendAlphaFillCmd, BlendFillCmd, FillCmd, FineCmd, GeneratedAlphaFill, GeneratedFill,
};
use alloc::vec::Vec;
use vello_common::mask::Mask;
use vello_common::util::Clear;

#[derive(Debug, Default)]
pub(crate) struct RowCommands {
    pub(crate) cmds: Vec<FineCmd>,
    pub(crate) opaque: Vec<FillCmd>,
    bounds: Option<(u16, u16)>,
    opaque_bounds: Option<(u16, u16)>,
    max_opaque_path_id: u32,
    pub(super) layer_depth: usize,
}

impl RowCommands {
    pub(super) fn new() -> Self {
        Self {
            cmds: Vec::new(),
            opaque: Vec::new(),
            bounds: None,
            opaque_bounds: None,
            max_opaque_path_id: 0,
            layer_depth: 0,
        }
    }

    fn clear(&mut self) {
        self.cmds.clear();
        self.opaque.clear();
        self.bounds = None;
        self.opaque_bounds = None;
        self.max_opaque_path_id = 0;
        self.layer_depth = 0;
    }

    pub(super) fn push_cmd(&mut self, cmd: FineCmd, width: u16) {
        if let Some((x, cmd_width)) = cmd.generated_span() {
            self.include_bounds(x, cmd_width, width);
        }
        self.cmds.push(cmd);
    }

    pub(super) fn push_layer(&mut self) {
        self.cmds.push(FineCmd::PushLayer);
        self.layer_depth += 1;
    }

    pub(super) fn pop_layer(
        &mut self,
        x: u16,
        width: u16,
        mask: Option<&Mask>,
        opacity: f32,
        blend_attrs_idx: u32,
    ) {
        if let Some(mask) = mask {
            self.cmds.push(FineCmd::Mask(mask.clone()));
        }
        if opacity != 1.0 {
            self.cmds.push(FineCmd::Opacity(opacity));
        }
        self.cmds.push(FineCmd::BlendFill(BlendFillCmd {
            x,
            width,
            attrs_idx: blend_attrs_idx,
        }));
        self.pop_buf();
    }

    pub(super) fn push_layer_props(&mut self, mask: Option<&Mask>, opacity: f32) {
        if let Some(mask) = mask {
            self.cmds.push(FineCmd::Mask(mask.clone()));
        }
        if opacity != 1.0 {
            self.cmds.push(FineCmd::Opacity(opacity));
        }
    }

    pub(super) fn push_blend_fill(
        &mut self,
        fill: GeneratedFill,
        blend_attrs_idx: u32,
        full_width: u16,
    ) {
        self.push_cmd(
            FineCmd::BlendFill(BlendFillCmd {
                x: fill.x,
                width: fill.width,
                attrs_idx: blend_attrs_idx,
            }),
            full_width,
        );
    }

    pub(super) fn push_blend_alpha_fill(
        &mut self,
        fill: GeneratedAlphaFill,
        blend_attrs_idx: u32,
        full_width: u16,
    ) {
        self.push_cmd(
            FineCmd::BlendAlphaFill(BlendAlphaFillCmd {
                x: fill.x,
                width: fill.width,
                alpha_idx: fill.alpha_idx,
                attrs_idx: blend_attrs_idx,
            }),
            full_width,
        );
    }

    pub(super) fn pop_buf(&mut self) {
        self.cmds.push(FineCmd::PopBuf);
        self.layer_depth -= 1;
    }

    pub(super) fn push_opaque(&mut self, cmd: FillCmd, width: u16, path_id: u32) {
        self.include_bounds(cmd.x, cmd.width, width);
        self.include_opaque_bounds(cmd.x, cmd.width, width);
        self.max_opaque_path_id = self.max_opaque_path_id.max(path_id);
        self.opaque.push(cmd);
    }

    pub(crate) fn bounds(&self) -> Option<(u16, u16)> {
        self.bounds
    }

    pub(crate) fn depth_affects(&self, x: u16, cmd_width: u16, path_id: u32) -> bool {
        if path_id >= self.max_opaque_path_id {
            return false;
        }

        let Some((opaque_start, opaque_end)) = self.opaque_bounds else {
            return false;
        };

        let end = x.saturating_add(cmd_width);
        if x >= opaque_end || end <= opaque_start {
            return false;
        }

        true
    }

    fn include_bounds(&mut self, x: u16, cmd_width: u16, width: u16) {
        let start = x.min(width);
        let end = start.saturating_add(cmd_width).min(width);
        if start >= end {
            return;
        }

        self.bounds = Some(match self.bounds {
            Some((old_start, old_end)) => (old_start.min(start), old_end.max(end)),
            None => (start, end),
        });
    }

    fn include_opaque_bounds(&mut self, x: u16, cmd_width: u16, width: u16) {
        let start = x.min(width);
        let end = start.saturating_add(cmd_width).min(width);
        if start >= end {
            return;
        }

        self.opaque_bounds = Some(match self.opaque_bounds {
            Some((old_start, old_end)) => (old_start.min(start), old_end.max(end)),
            None => (start, end),
        });
    }
}

impl Clear for RowCommands {
    fn clear(&mut self) {
        Self::clear(self);
    }
}
