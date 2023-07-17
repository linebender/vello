// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{BumpAllocators, IndirectCount};

use crate::cpu_dispatch::CpuBinding;

const WG_SIZE: usize = 256;

fn path_tiling_setup_main(bump: &BumpAllocators, indirect: &mut IndirectCount) {
    let segments = bump.seg_counts;
    indirect.count_x = (segments + (WG_SIZE as u32 - 1)) / WG_SIZE as u32;
    indirect.count_y = 1;
    indirect.count_z = 1;
}

pub fn path_tiling_setup(_n_wg: u32, resources: &[CpuBinding]) {
    let r0 = resources[0].as_buf();
    let mut r1 = resources[1].as_buf();
    let bump = bytemuck::from_bytes(&r0);
    let indirect = bytemuck::from_bytes_mut(r1.as_mut());
    path_tiling_setup_main(bump, indirect);
}
