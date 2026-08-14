// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared texture access helpers.

fn flat_index_to_texture_coord(index: u32, width: u32) -> vec2<u32> {
    return vec2<u32>(index % width, index / width);
}
