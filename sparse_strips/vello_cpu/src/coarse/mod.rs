// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

mod bucketer;
mod cmd;
mod layer;
mod row;
mod strip;

pub(crate) const DEPTH_BUCKET_WIDTH: u16 = 128;

pub(crate) use bucketer::CommandBucketer;
pub(crate) use cmd::{FillAttrs, FillCmd, FilterLayerAttrs, FilterLayerCmd, FineCmd, RenderCmd};
pub(crate) use layer::LayerClip;
pub(crate) use row::RowCommands;
