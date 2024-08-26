// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use vello_encoding::{BumpAllocators, ConfigUniform, IndirectCount};

use super::CpuBinding;

const WG_SIZE: usize = 256;

fn path_tiling_setup_main(
    // Config is needed to detect allocation failure
    _config: &ConfigUniform,
    bump: &BumpAllocators,
    indirect: &mut IndirectCount,
) {
    let segments = bump.seg_counts;
    indirect.count_x = (segments + (WG_SIZE as u32 - 1)) / WG_SIZE as u32;
    indirect.count_y = 1;
    indirect.count_z = 1;
}

pub fn path_tiling_setup(_n_wg: u32, resources: &[CpuBinding]) {
    let config = resources[0].as_typed();
    let bump = resources[1].as_typed();
    let mut indirect = resources[2].as_typed_mut();
    // binding 2 is ptcl, which we would need if we propagate failure
    path_tiling_setup_main(&config, &bump, &mut indirect);
}
