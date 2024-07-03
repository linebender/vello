// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use vello_encoding::{BumpAllocators, ConfigUniform};

use super::CpuBinding;

fn prepare_main(_config: &ConfigUniform, bump: &mut BumpAllocators) {
    // We don't yet do robust bump handling in the CPU shaders, so do the minimal version for this shader
    bump.binning = 0;
    bump.ptcl = 0;
    bump.tile = 0;
    bump.seg_counts = 0;
    bump.segments = 0;
    bump.blend_spill = 0;
    bump.lines = 0;
    bump.failed = 0;
}

pub fn prepare(_n_wg: u32, resources: &[CpuBinding]) {
    // On the GPU, config is mutable, but our CPU runner doesn't allow accessing uploaded buffers as mutable
    // This is a hack
    let config = resources[0].as_typed();
    let mut bump = resources[1].as_typed_mut();
    prepare_main(&config, &mut bump);
}
