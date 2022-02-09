// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Various constants for the sizes of groups and tiles.

// Much of this will be made dynamic in various ways, but for now it's easiest
// to hardcode and keep all in one place.

// A LG_WG_FACTOR of n scales workgroup sizes by 2^n. Use 0 for a
// maximum workgroup size of 128, or 1 for a maximum size of 256.
#define LG_WG_FACTOR 1
#define WG_FACTOR (1<<LG_WG_FACTOR)

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

#define GRADIENT_WIDTH 512

#ifdef ERR_MALLOC_FAILED
struct Config {
    uint n_elements; // paths
    uint n_pathseg;
    uint width_in_tiles;
    uint height_in_tiles;
    Alloc tile_alloc;
    Alloc bin_alloc;
    Alloc ptcl_alloc;
    Alloc pathseg_alloc;
    Alloc anno_alloc;
    Alloc trans_alloc;
    // new element pipeline stuff follows

    // Bounding boxes of paths, stored as int (so atomics work)
    Alloc bbox_alloc;
    // Monoid for draw objects
    Alloc drawmonoid_alloc;

    // Number of transforms in scene
    // This is probably not needed.
    uint n_trans;
    // This only counts actual paths, not EndClip.
    uint n_path;
    // Offset (in bytes) of transform stream in scene buffer
    uint trans_offset;
    // Offset (in bytes) of linewidth stream in scene
    uint linewidth_offset;
    // Offset (in bytes) of path tag stream in scene
    uint pathtag_offset;
    // Offset (in bytes) of path segment stream in scene
    uint pathseg_offset;
};
#endif

// Fill modes.
#define MODE_NONZERO 0
#define MODE_STROKE 1

#define TILESEG_HAS_LEFT_EDGE 1
#define TILESEG_NORMAL_NEGATE_X (1 << 1)

// Size of kernel4 clip state, in words.
#define CLIP_STATE_SIZE 2

// fill_mode_from_flags extracts the fill mode from tag flags.
uint fill_mode_from_flags(uint flags) {
    return flags & 0x1;
}
