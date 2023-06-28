// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{BumpAllocators, IndirectCount};

const WG_SIZE: usize = 256;

pub fn path_tiling_setup(bump: &BumpAllocators, indirect: &mut IndirectCount) {
    let segments = bump.seg_counts;
    indirect.count_x = (segments + (WG_SIZE as u32 - 1)) / WG_SIZE as u32;
    indirect.count_y = 1;
    indirect.count_z = 1;
}
