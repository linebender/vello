// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Running a render graph has three important steps:
//!
//! 1) Resolving the paintings to be rendered, and importantly their sizes.
//!    Note that this *might* involve splitting the tree, because of [`OutputSize::Inferred`]
//! 2) Creating a graph to find the order in which those are to be rendered.
//!    Note that this doesn't have to dictate the order that their commands
//!    are encoded, only the order in which they are submitted.
//! 3) Running that graph. This involves encoding all the commands, and submitting them
//!    in the order calculated in step 2.

use super::{OutputSize, Vello};

/// For inter-frame caching, we keep the same Vello struct around.
impl Vello {
    pub fn run_render() {}
}

fn resolve_graph() {
    match OutputSize::Inferred {
        OutputSize::Fixed { width, height } => todo!(),
        OutputSize::Inferred => todo!(),
    }
}
