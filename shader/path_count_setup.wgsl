// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Set up dispatch size for path count stage.

#import bump

@group(0) @binding(0)
var<storage, read_write> bump: BumpAllocators;

@group(0) @binding(1)
var<storage, read_write> indirect: IndirectCount;

// Partition size for path count stage
let WG_SIZE = 256u;

@compute @workgroup_size(1)
fn main() {
    let lines = atomicLoad(&bump.lines);
    indirect.count_x = (lines + (WG_SIZE - 1u)) / WG_SIZE;
    indirect.count_y = 1u;
    indirect.count_z = 1u;
}
