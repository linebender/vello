// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// TODO: robust memory (failure flags)
struct BumpAllocators {
    binning: atomic<u32>,
    ptcl: atomic<u32>,
    tile: atomic<u32>,
    segments: atomic<u32>,
}
