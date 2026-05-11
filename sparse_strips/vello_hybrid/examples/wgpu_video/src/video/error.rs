// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Errors produced by the video pipeline.

use super::frame::FourCC;

#[derive(Debug, thiserror::Error)]
pub(crate) enum AvError {
    #[error("source I/O error: {0}")]
    Source(String),

    #[error("unsupported container: {0}")]
    UnsupportedContainer(&'static str),

    #[error("unsupported codec: {0}")]
    UnsupportedCodec(FourCC),

    #[error("track {0} not found")]
    TrackNotFound(u32),

    #[error("unexpected wgpu backend: expected {expected}, got {got:?}")]
    UnexpectedBackend {
        expected: &'static str,
        got: wgpu::Backend,
    },

    #[error("backend error: {message} (osstatus = {osstatus})")]
    Backend { message: String, osstatus: i32 },
}

impl AvError {
    pub(crate) fn backend(message: impl Into<String>, osstatus: i32) -> Self {
        Self::Backend {
            message: message.into(),
            osstatus,
        }
    }
}
