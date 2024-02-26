// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// This file defines utility functions that interact with host-shareable buffer objects. It should
// be imported once following the resource binding declarations in the shader module that access
// them.

// Reads a draw tag from the scene buffer, defaulting to DRAWTAG_NOP if the given `ix` is beyond the
// range of valid draw objects (e.g this can happen if `ix` is derived from an invocation ID in a
// workgroup that partially spans valid range).
//
// This function depends on the following global declarations:
//    * `scene`: array<u32>
//    * `config`: Config (see config.wgsl)
fn read_draw_tag_from_scene(ix: u32) -> u32 {
    var tag_word: u32;
    if ix < config.n_drawobj {
        let tag_ix = config.drawtag_base + ix;
        tag_word = scene[tag_ix];
    } else {
        tag_word = DRAWTAG_NOP;
    }
    return tag_word;
}
