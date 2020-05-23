// Various constants for the sizes of groups and tiles.

// Much of this will be made dynamic in various ways, but for now it's easiest
// to hardcode and keep all in one place.

// TODO: make the image size dynamic.
#define IMAGE_WIDTH 2048
#define IMAGE_HEIGHT 1536

// TODO: compute this
#define WIDTH_IN_TILEGROUPS 4

#define TILEGROUP_WIDTH_PX 512
#define TILEGROUP_HEIGHT_PX 16

#define TILEGROUP_INITIAL_ALLOC 1024

// Quick note on layout of tilegroups (k1 output): in the base,
// there is a region of size TILEGROUP_STRIDE for each tilegroup.
// At offset 0 are the main instances, encoded with Jump. At offset
// TILEGROUP_STROKE_START are the stroke instances, encoded with
// Head and Link. Similarly for fill.
#define TILEGROUP_STRIDE 2048
#define TILEGROUP_STROKE_START 1024
#define TILEGROUP_FILL_START 1536
#define TILEGROUP_STROKE_ALLOC 1024
#define TILEGROUP_FILL_ALLOC 1024
#define TILEGROUP_INITIAL_STROKE_ALLOC 512
#define TILEGROUP_INITIAL_FILL_ALLOC 512

// TODO: compute all these

#define WIDTH_IN_TILES 128
#define TILEGROUP_WIDTH_TILES 32
#define TILE_WIDTH_PX 16
#define TILE_HEIGHT_PX 16

#define PTCL_INITIAL_ALLOC 1024

// Maximum number of segments in a SegChunk
#define SEG_CHUNK_N 32
#define SEG_CHUNK_ALLOC 512

// Stuff for new algorithm follows; some of the above should get
// deleted.

// These should probably be renamed and/or reworked. In the binning
// kernel, they represent the number of bins. Also, the workgroup size
// of that kernel is equal to the number of bins, but should probably
// be more flexible (it's 512 in the K&L paper).
#define N_TILE_X 16
#define N_TILE_Y 16
#define N_TILE (N_TILE_X * N_TILE_Y)
#define LG_N_TILE 8
#define N_SLICE (N_TILE / 32)
// Number of workgroups for binning kernel
#define N_WG 16

// This is the ratio of the number of elements in a binning workgroup
// over the number of elements in a partition workgroup.
#define ELEMENT_BINNING_RATIO 2

#define BIN_INITIAL_ALLOC 64
#define BIN_ALLOC 256
