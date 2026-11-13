// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Options for resetting a render context or scene.

/// Options for resetting a render context or scene.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResetOptions {
    /// The new width and height, or `None` to keep the current size.
    pub size: Option<(u16, u16)>,
    /// Whether to free the memory grown for earlier contents.
    ///
    /// The memory is reallocated for the size after the reset, so resetting to a smaller size
    /// at the same time frees more.
    pub shrink: bool,
}
