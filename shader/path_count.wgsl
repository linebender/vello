// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Stage to compute counts of number of segments in each tile

#import bump
#import config
#import segment
#import tile

@group(0) @binding(0)
var<uniform> config: Config;

// TODO: this is cut'n'pasted from path_coarse.
struct AtomicTile {
    backdrop: atomic<i32>,
    // This is "segments" but we're renaming
    count: atomic<u32>,
}

@group(0) @binding(1)
var<storage, read_write> bump: BumpAllocators;

@group(0) @binding(2)
var<storage> lines: array<LineSoup>;

@group(0) @binding(3)
var<storage> paths: array<Path>;

@group(0) @binding(4)
var<storage, read_write> tile: array<AtomicTile>;

@group(0) @binding(5)
var<storage, read_write> seg_counts: array<SegmentCount>;

#ifdef load_balanced
let WG_SIZE = 256u;
var<workgroup> sh_count: array<u32, WG_SIZE>;
var<workgroup> sh_seg_base: u32;
#endif

// number of integer cells spanned by interval defined by a, b
fn span(a: f32, b: f32) -> u32 {
    return u32(max(ceil(max(a, b)) - floor(min(a, b)), 1.0));
}

// Note regarding clipping to bounding box:
//
// We have to do the backdrop bumps for all tiles to the left of the bbox.
// This should probably be a separate loop. This also causes a numerical
// robustness issue.

// This shader is dispatched with one thread for each line.
@compute @workgroup_size(256)
fn main(
#ifdef load_balanced
    @builtin(local_invocation_id) local_id: vec3<u32>,
#endif
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let n_lines = atomicLoad(&bump.lines);
    var count = 0u;
    if global_id.x < n_lines {
        let line = lines[global_id.x];
        // coarse rasterization logic to count number of tiles touched by line
        let is_down = line.p1.y >= line.p0.y;
        let xy0 = select(line.p1, line.p0, is_down);
        let xy1 = select(line.p0, line.p1, is_down);
        let s0 = xy0 * TILE_SCALE;
        let s1 = xy1 * TILE_SCALE;
        // TODO: clip line to bounding box
        count = span(s0.x, s1.x) + span(s0.y, s1.y) - 1u;
        // two possibilities: just loop (which may cause load balancing issues),
        // or do partition wide prefix sum + binary search.
#ifdef load_balanced
    }
    sh_count[local_id.x] = count;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i++) {
        workgroupBarrier();
        if local_id.x >= 1u << i {
            count += sh_count[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        sh_count[local_id.x] = count;
    }
    if local_id.x == WG_SIZE - 1u {
        sh_seg_base = atomicAdd(&bump.seg_counts, count);
    }
    workgroupBarrier();
#ifdef have_uniform
    let total = workgroupUniformLoad(&sh_count[WG_SIZE - 1u]);
#else
    let total = sh_count[WG_SIZE - 1u];
#endif
    for (var i = local_id.x; i < total; i += WG_SIZE) {
        // Following line is inside the loop because of a suspected Metal
        // miscompilation, similar to #199. If GPUs actually worked, then
        // this line would be above the for loop.
        let seg_base = sh_seg_base;
        // binary search to find tile
        var el_ix = 0u;
        for (var j = 0u; j < firstTrailingBit(WG_SIZE); j++) {
            let probe = el_ix + ((WG_SIZE / 2u) >> j);
            if i >= sh_count[probe - 1u] {
                el_ix = probe;
            }
        }
        let subix = i - select(0u, sh_count[el_ix - 1u], el_ix > 0u);

        // load line again
        let line_ix = (global_id.x & (0u - WG_SIZE)) + el_ix;
        let line = lines[line_ix];
        let is_down = line.p1.y >= line.p0.y;
        let xy0 = select(line.p1, line.p0, is_down);
        let xy1 = select(line.p0, line.p1, is_down);
        let s0 = xy0 * TILE_SCALE;
        let s1 = xy1 * TILE_SCALE;
#else
        let line_ix = global_id.x;
        let seg_base = atomicAdd(&bump.seg_counts, count);
#endif

        // A good case can be made that this setup should be in shared
        // memory in the load-balanced case, as we'll be taking multiple
        // iterations.
        let dx = abs(s1.x - s0.x);
        let dy = s1.y - s0.y;
        let idxdy = 1.0 / (dx + dy);
        let a = dx * idxdy;
        let is_positive_slope = s1.x >= s0.x;
        let sign = select(-1.0, 1.0, is_positive_slope);
        let xt0 = floor(s0.x * sign);
        let c = s0.x * sign - xt0;
        let y0i = floor(s0.y);
        let ytop = select(y0i + 1.0, ceil(s0.y), s0.y == s1.y);
        let b = (dy * c + dx * (ytop - s0.y)) * idxdy;
        let x0i = i32(xt0 * sign + 0.5 * (sign - 1.0));

        let path = paths[line.path_ix];
        let bbox = vec4<i32>(path.bbox);
        let stride = bbox.z - bbox.x;
#ifdef load_balanced
        {
#else
        var last_z = 0.0;
        for (var i = 0u; i < count; i++) {
            let subix = i;
#endif
            // coarse rasterization logic
            // Note: we hope fast-math doesn't strength reduce this.
            let zf = a * f32(subix) + b;
            let z = floor(zf);
            // x, y are tile coordinates relative to render target
            let y = i32(y0i + f32(subix) - z);
            // TODO: should be clipped above, not here
            // Dummy output; when we clip properly, we won't have to do this,
            // all outputs will be valid.
            var seg_count = SegmentCount(~0u, 0u);
            if (y >= bbox.y && y < bbox.w) {
                let x = x0i + i32(sign * z);
                let base = i32(path.tiles) + (y - bbox.y) * stride - bbox.x;
#ifdef load_balanced
                // Note: we hope fast-math doesn't rewrite this to
                // floor(zf - a), which is correct for reals but not floats.
                let last_z = floor(a * (f32(subix) - 1.0) + b);
#endif
                let top_edge = select(last_z == z, y0i == s0.y, subix == 0u);
                if top_edge && bbox.x < bbox.z && x + 1 < bbox.z {
                    let x_bump = max(x + 1, bbox.x);
                    let delta = select(1, -1, is_down);
                    atomicAdd(&tile[base + x_bump].backdrop, delta);
                }
                if (x >= bbox.x && x < bbox.z) {
                    let seg_within_slice = atomicAdd(&tile[base + x].count, 1u);
                    // Pack two count values into a single u32
                    let counts = (seg_within_slice << 16u) | subix;
                    seg_count = SegmentCount(line_ix, counts);
                }
            }
#ifndef load_balanced
            // Note: since we're iterating, we have a reliable value for
            // last_z.
            last_z = z;
#endif
            seg_counts[seg_base + i] = seg_count;
        }
    }
}
