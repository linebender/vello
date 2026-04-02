// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! An abstraction for rendering glyphs in a backend-agnostic way.
// TODO: Note that right now, this is intended to be used for vello renderers.
// But it should be possible to repurpose this to be more general. In particular, some methods
// like `render_colr_directly` could probably be implemented here if we implement it to use
// a generic rendering API like `anyrender` or `imaging`.
pub mod vello_renderer;
