// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Tile allocation (and zeroing of tiles)

#import config
#import bump
#import drawtag
#import tile

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> scene: array<u32>;

@group(0) @binding(2)
var<storage> draw_bboxes: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> bump: BumpAllocators;

@group(0) @binding(4)
var<storage, read_write> paths: array<Path>;

@group(0) @binding(5)
var<storage, read_write> tiles: array<Tile>;

let WG_SIZE = 256u;

var<workgroup> sh_tile_count: array<u32, WG_SIZE>;
var<workgroup> sh_tile_offset: u32;
#ifdef have_uniform
var<workgroup> sh_atomic_failed: u32;
#endif

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    // Exit early if prior stages failed, as we can't run this stage.
    // We need to check only prior stages, as if this stage has failed in another workgroup, 
    // we still want to know this workgroup's memory requirement.
#ifdef have_uniform
    if local_id.x == 0u {
        sh_atomic_failed = atomicLoad(&bump.failed);
    }
    let failed = workgroupUniformLoad(&sh_atomic_failed);
#else
    let failed = atomicLoad(&bump.failed);
#endif
    if (failed & STAGE_BINNING) != 0u {
        return;
    }    
    // scale factors useful for converting coordinates to tiles
    // TODO: make into constants
    let SX = 1.0 / f32(TILE_WIDTH);
    let SY = 1.0 / f32(TILE_HEIGHT);

    let drawobj_ix = global_id.x;
    var drawtag = DRAWTAG_NOP;
    if drawobj_ix < config.n_drawobj {
        drawtag = scene[config.drawtag_base + drawobj_ix];
    }
    var x0 = 0;
    var y0 = 0;
    var x1 = 0;
    var y1 = 0;
    if drawtag != DRAWTAG_NOP && drawtag != DRAWTAG_END_CLIP {
        let bbox = draw_bboxes[drawobj_ix];
        x0 = i32(floor(bbox.x * SX));
        y0 = i32(floor(bbox.y * SY));
        x1 = i32(ceil(bbox.z * SX));
        y1 = i32(ceil(bbox.w * SY));
    }
    let ux0 = u32(clamp(x0, 0, i32(config.width_in_tiles)));
    let uy0 = u32(clamp(y0, 0, i32(config.height_in_tiles)));
    let ux1 = u32(clamp(x1, 0, i32(config.width_in_tiles)));
    let uy1 = u32(clamp(y1, 0, i32(config.height_in_tiles)));
    let tile_count = (ux1 - ux0) * (uy1 - uy0);
    var total_tile_count = tile_count;
    sh_tile_count[local_id.x] = tile_count;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            total_tile_count += sh_tile_count[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        sh_tile_count[local_id.x] = total_tile_count;
    }
    if local_id.x == WG_SIZE - 1u {
        let count = sh_tile_count[WG_SIZE - 1u];
        var offset = atomicAdd(&bump.tile, count);
        if offset + count > config.tiles_size {
            offset = 0u;
            atomicOr(&bump.failed, STAGE_TILE_ALLOC);
        }
        paths[drawobj_ix].tiles = offset;
    }    
    // Using storage barriers is a workaround for what appears to be a miscompilation
    // when a normal workgroup-shared variable is used to broadcast the value.
    storageBarrier();
    let tile_offset = paths[drawobj_ix | (WG_SIZE - 1u)].tiles;
    storageBarrier();
    if drawobj_ix < config.n_drawobj {
        let tile_subix = select(0u, sh_tile_count[local_id.x - 1u], local_id.x > 0u);
        let bbox = vec4(ux0, uy0, ux1, uy1);
        let path = Path(bbox, tile_offset + tile_subix);
        paths[drawobj_ix] = path;
    }

    // zero allocated memory
    // Note: if the number of draw objects is small, utilization will be poor.
    // There are two things that can be done to improve that. One would be a
    // separate (indirect) dispatch. Another would be to have each workgroup
    // process fewer draw objects than the number of threads in the wg.
    let total_count = sh_tile_count[WG_SIZE - 1u];
    for (var i = local_id.x; i < total_count; i += WG_SIZE) {
        // Note: could format output buffer as u32 for even better load balancing.
        tiles[tile_offset + i] = Tile(0, 0u);
    }
}
