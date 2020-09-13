// Various constants for the sizes of groups and tiles.

// Much of this will be made dynamic in various ways, but for now it's easiest
// to hardcode and keep all in one place.

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
#define N_TILE_Y 16
#define N_TILE (N_TILE_X * N_TILE_Y)
#define LG_N_TILE 8
#define N_SLICE (N_TILE / 32)
