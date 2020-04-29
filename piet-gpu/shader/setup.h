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
// Head and Link.
#define TILEGROUP_STRIDE 2048
#define TILEGROUP_STROKE_START 1024
#define TILEGROUP_STROKE_ALLOC 1024

// TODO: compute all these

#define WIDTH_IN_TILES 128
#define TILEGROUP_WIDTH_TILES 32
#define TILE_WIDTH_PX 16
#define TILE_HEIGHT_PX 16

#define PTCL_INITIAL_ALLOC 1024
