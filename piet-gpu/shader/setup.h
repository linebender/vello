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

// We allocate this many blend stack entries in registers, and spill
// to memory for the overflow.
#define BLEND_STACK_SPLIT 4

#ifdef MALLOC_FAILED
struct Config {
    uint mem_size; // in bytes
    uint n_elements; // paths
    uint n_pathseg;
    uint width_in_tiles;
    uint height_in_tiles;
    Alloc tile_alloc;
    Alloc bin_alloc;
    Alloc ptcl_alloc;
    Alloc pathseg_alloc;
    Alloc anno_alloc;
    // new element pipeline stuff follows

    // Bounding boxes of paths, stored as int (so atomics work)
    Alloc path_bbox_alloc;
    // Monoid for draw objects
    Alloc drawmonoid_alloc;

    // BeginClip(path_ix) / EndClip
    Alloc clip_alloc;
    // Intermediate bicyclic semigroup
    Alloc clip_bic_alloc;
    // Intermediate stack
    Alloc clip_stack_alloc;
    // Clip processing results (path_ix + bbox)
    Alloc clip_bbox_alloc;
    // Bounding box per draw object
    Alloc draw_bbox_alloc;
    // Info computed in draw stage, per draw object
    Alloc drawinfo_alloc;

    // Number of transforms in scene
    // This is probably not needed.
    uint n_trans;
    // This *should* count only actual paths, but in the current
    // implementation is redundant with n_elements.
    uint n_path;
    // Total number of BeginClip and EndClip draw objects.
    uint n_clip;

    // Note: one of these offsets *could* be hardcoded to zero (as was the
    // original element stream), but for now retain flexibility.

    // Offset (in bytes) of transform stream in scene buffer
    uint trans_offset;
    // Offset (in bytes) of linewidth stream in scene
    uint linewidth_offset;
    // Offset (in bytes) of path tag stream in scene
    uint pathtag_offset;
    // Offset (in bytes) of path segment stream in scene
    uint pathseg_offset;
    // Offset (in bytes) of draw object tag stream in scene; see drawtag.h
    uint drawtag_offset;
    // Offset (in bytes) of draw payload stream in scene
    uint drawdata_offset;
};
#endif

// Fill modes.
#define MODE_NONZERO 0
#define MODE_STROKE 1

// Size of kernel4 clip state, in words.
#define CLIP_STATE_SIZE 1

// fill_mode_from_flags extracts the fill mode from tag flags.
uint fill_mode_from_flags(uint flags) {
    return flags & 0x1;
}
