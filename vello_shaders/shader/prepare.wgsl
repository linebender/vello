// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Determine whether the Vello pipeline is likely to fail during this run
// and therefore whether all later stages should be cancelled.

#import config
#import bump

@group(0) @binding(0)
var<storage, read_write> config: Config;

@group(0) @binding(1)
// TODO: Use a non-atomic version of BumpAllocators?
var<storage, read_write> bump: BumpAllocators;

@compute @workgroup_size(1)
fn main() {
    var should_cancel = false;
    let previous_failure = atomicLoad(&bump.failed);
    if previous_failure == PREVIOUS_RUN {
        // Don't early-exit from multiple frames in a row
        // The CPU should be blocking on the frame which failed anyway, so this
        // case should never be reached, but if the CPU side isn't doing that
        // properly, we can try again.
        // (Note that this check is simply an early-exit for this case, as all the 
        // bump values would have been reset to 0 anyway)
        atomicStore(&bump.failed, 0u);
    } else if previous_failure != 0u {
        // If the previous frame failed (for another reason)

        // And we don't have enough memory to have run that previous frame
        if config.lines_size < atomicLoad(&bump.lines) {
            should_cancel = true;
        }
        if config.binning_size < atomicLoad(&bump.binning) {
            should_cancel = true;
        }
        if config.ptcl_size < atomicLoad(&bump.ptcl) {
            should_cancel = true;
        }
        if config.tiles_size < atomicLoad(&bump.tile) {
            should_cancel = true;
        }
        if config.seg_counts_size < atomicLoad(&bump.seg_counts) {
            should_cancel = true;
        }
        if config.segments_size < atomicLoad(&bump.segments) {
            should_cancel = true;
        }
        if config.lines_size < atomicLoad(&bump.lines) {
            should_cancel = true;
        }
        // config.blend_size < atomicLoad(&bump.blend)
        if should_cancel {
            // Then don't run this frame
            config.cancelled = 1u;
            atomicStore(&bump.failed, PREVIOUS_RUN);
        } else {
            atomicStore(&bump.failed, 0u);
        }
    }
    atomicStore(&bump.binning, 0u);
    atomicStore(&bump.ptcl, 0u);
    atomicStore(&bump.tile, 0u);
    atomicStore(&bump.seg_counts, 0u);
    atomicStore(&bump.segments, 0u);
    atomicStore(&bump.blend, 0u);
    atomicStore(&bump.lines, 0u);
}
