// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::cmd::{BlendFillCmd, FillCmd, FineCmd, Span};
use super::depth::DepthRowState;
use alloc::vec::Vec;
use vello_common::util::Clear;

#[derive(Debug, Default)]
pub(crate) struct RowCommands {
    pub(crate) cmds: Vec<FineCmd>,
    pub(crate) opaque: Vec<FillCmd>,
    bounds: Option<Span>,
    depth: DepthRowState,
    pub(super) layer_depth: usize,
}

impl RowCommands {
    pub(super) fn new() -> Self {
        Self {
            cmds: Vec::new(),
            opaque: Vec::new(),
            bounds: None,
            depth: DepthRowState::default(),
            layer_depth: 0,
        }
    }

    fn clear(&mut self) {
        self.cmds.clear();
        self.opaque.clear();
        self.bounds = None;
        self.depth.clear();
        self.layer_depth = 0;
    }

    pub(super) fn push_cmd(&mut self, cmd: FineCmd, width: u16) {
        if let Some(span) = cmd.generated_span() {
            self.include_bounds(span, width);
        }
        self.cmds.push(cmd);
    }

    pub(super) fn push_layer(&mut self) {
        self.cmds.push(FineCmd::PushLayer);
        self.layer_depth += 1;
    }

    pub(super) fn pop_layer(
        &mut self,
        span: Span,
        mask_idx: Option<u32>,
        opacity: f32,
        blend_attrs_idx: u32,
    ) {
        if let Some(mask_idx) = mask_idx {
            self.cmds.push(FineCmd::Mask(mask_idx));
        }
        if opacity != 1.0 {
            self.cmds.push(FineCmd::Opacity(opacity));
        }
        self.cmds.push(FineCmd::BlendFill(BlendFillCmd::new(
            span,
            None,
            blend_attrs_idx,
        )));
        self.pop_buf();
    }

    pub(super) fn push_layer_props(&mut self, mask_idx: Option<u32>, opacity: f32) {
        if let Some(mask_idx) = mask_idx {
            self.cmds.push(FineCmd::Mask(mask_idx));
        }
        if opacity != 1.0 {
            self.cmds.push(FineCmd::Opacity(opacity));
        }
    }

    pub(super) fn push_blend_fill(
        &mut self,
        span: Span,
        alpha_idx: Option<u32>,
        blend_attrs_idx: u32,
        full_width: u16,
    ) {
        self.push_cmd(
            FineCmd::BlendFill(BlendFillCmd::new(span, alpha_idx, blend_attrs_idx)),
            full_width,
        );
    }

    pub(super) fn pop_buf(&mut self) {
        self.cmds.push(FineCmd::PopBuf);
        self.layer_depth -= 1;
    }

    pub(super) fn push_opaque(&mut self, cmd: FillCmd, width: u16, draw_id: u32) {
        self.include_bounds(cmd.span, width);
        self.depth.include_opaque(cmd.span, width, draw_id);
        self.opaque.push(cmd);
    }

    pub(crate) fn bounds(&self) -> Option<Span> {
        self.bounds
    }

    pub(crate) fn depth_affects(&self, span: Span, draw_id: u32) -> bool {
        self.depth.affects_later_draw(span, draw_id)
    }

    fn include_bounds(&mut self, span: Span, width: u16) {
        if span.pixel_x() >= width {
            return;
        }

        if let Some(bounds) = &mut self.bounds {
            bounds.extend(span);
        } else {
            self.bounds = Some(span);
        }
    }
}

impl Clear for RowCommands {
    fn clear(&mut self) {
        Self::clear(self);
    }
}
