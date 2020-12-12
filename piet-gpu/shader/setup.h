// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Various constants for the sizes of groups and tiles.

// Much of this will be made dynamic in various ways, but for now it's easiest
// to hardcode and keep all in one place.

// A LG_WG_FACTOR of n scales workgroup sizes by 2^n. Use 0 for a
// maximum workgroup size of 128, or 1 for a maximum size of 256.
#define LG_WG_FACTOR 1
#define WG_FACTOR (1<<LG_WG_FACTOR)

// TODO: compute all these

#define WIDTH_IN_TILES 128
#define HEIGHT_IN_TILES 96
#define TILE_WIDTH_PX 16
#define TILE_HEIGHT_PX 16

#define PTCL_INITIAL_ALLOC 1024

// These should probably be renamed and/or reworked. In the binning
// kernel, they represent the number of bins. Also, the workgroup size
// of that kernel is equal to the number of bins, but should probably
// be more flexible (it's 512 in the K&L paper).
#define N_TILE_X 16
#define N_TILE_Y (8 * WG_FACTOR)
#define N_TILE (N_TILE_X * N_TILE_Y)
#define LG_N_TILE (7 + LG_WG_FACTOR)
#define N_SLICE (N_TILE / 32)
