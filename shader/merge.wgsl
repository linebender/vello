// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Merge tiles

// TODO: lots of rework here
struct Minitile {
    path_ix: u32,
    x: u32,
    y: u32,
    p0: u32, // packed
    p1: u32, // packed
}

fn unpack_point(p: u32) -> vec2f {
    let x = f32(p & 0xffffu) * (1.0 / 8192.0);
    let y = f32(p >> 16u) * (1.0 / 8192.0);
    return vec2(x, y);
}

@group(0) @binding(0)
var<storage> input: array<Minitile>;

struct MergeMonoid {
    winding: i32,
    n_strips: u32,
    // this probably doesn't belong in reduced
    start: u32,
    start_x: u32,
}

@group(1) @binding(1)
var<storage> reduced_mm: array<MergeMonoid>;

fn combine_merge_monoid(a: MergeMonoid, b: MergeMonoid) -> MergeMonoid {
    var c: MergeMonoid;
    c.winding = a.winding + b.winding;
    c.n_strips = a.n_strips + b.n_strips;
    c.start = max(a.start, b.start);
    return c;
}

fn mt_delta(t: Minitile) -> i32 {
    return i32((t.p1 >> 16u) == 0u) - i32((t.p0 >> 16u) == 0u);
}

fn mt_histogram(t: Minitile) -> u32 {
    let x0 = f32(t.p0 & 0xffffu) * (1.0 / 8192.0);
    let x1 = f32(t.p1 & 0xffffu) * (1.0 / 8192.0);
    let xmin = u32(floor(min(x0, x1)));
    let xmax = u32(ceil(max(x0, x1)));
    let rshift = (4u - (xmax - xmin)) * 8u;
    let lshift = xmin * 8u;
    return (0x01010101u >> rshift) << lshift;
}

fn reduce_histo(histo: u32) -> u32 {
    let tmp = (histo & 0xff00ffu) + ((histo >> 8u) & 0xff00ffu);
    return (tmp >> 16u) + (tmp & 0xffffu);
}

const WG_SIZE = 256u;

var<workgroup> sh_mm: array<MergeMonoid, WG_SIZE>;
var<workgroup> sh_histo: array<u32, WG_SIZE>;
var<workgroup> sh_seg_end: array<u32, WG_SIZE>;
var<workgroup> sh_inclusive_cols: array<u32, WG_SIZE>;
var<workgroup> sh_area: array<atomic<i32>, WG_SIZE>;
var<workgroup> sh_carryover: array<i32, 4>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    // scan merge monoid
    var first = false;
    var first_x = false;
    // predicate? or pad?
    let global_ix = wg_id.x * WG_SIZE + local_id.x;
    let tile = input[global_ix];
    if global_ix != 0u {
        let prev = input[global_ix - 1u];
        first = tile.path_ix != prev.path_ix || tile.y != prev.y;
        first_x = first || tile.x != prev.x;
    }
    let winding = mt_delta(tile);
    let n_strips = u32(first);
    let start = select(0u, local_id.x, first);
    let start_x = select(0u, local_id.x, first_x);
    var agg = MergeMonoid(winding, n_strips, start, start_x);

    let local_histo = mt_histogram(tile);
    var histo = local_histo;
    sh_mm[local_id.x] = agg;
    sh_histo[local_id.x] = histo;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i++) {
        workgroupBarrier();
        if local_id.x >= 1u << i {
            let other = sh_mm[local_id.x - (1u << i)];
            agg = combine_merge_monoid(agg, other);
            histo += sh_histo[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        sh_mm[local_id.x] = agg;
        sh_histo[local_id.x] = histo;
    }
    if wg_id.x > 0u {
        let prefix = reduced_mm[wg_id.x - 1u];
        agg = combine_merge_monoid(prefix, agg);
    }
    workgroupBarrier();
    // subtract off start of scanline winding number

    // This is a workaround for overflow at 256. One alternative approach is to
    // only do 255 items per workgroup.
    sh_inclusive_cols[local_id.x] = reduce_histo(local_histo) + reduce_histo(histo - local_histo);
    let seg_rel_histo = histo - select(0u, sh_histo[agg.start_x - 1u], agg.start_x > 0u);
    workgroupBarrier();
    // sh_histo now contains histograms relative to segment
    sh_histo[local_id.x] = seg_rel_histo;

    let last_x = local_id.x == WG_SIZE - 1u || sh_mm[local_id.x + 1u].start_x != agg.start_x;
    if last_x {
        sh_seg_end[agg.start_x] = local_id.x;
    }

    let total_cols = workgroupUniformLoad(&sh_inclusive_cols[WG_SIZE - 1u]);
    // maybe don't need to fill this; consumers can only read from starts
    if agg.start_x != local_id.x {
        sh_seg_end[local_id.x] = sh_seg_end[agg.start_x];
    }
    let n_blocks = (total_cols + WG_SIZE - 1u) / WG_SIZE;
    for (var block_ix = 0u; block_ix < n_blocks; block_ix++) {
        let ix = block_ix * WG_SIZE + local_id.x;
        // binary search to find work item
        var cols = 0u; // misnamed
        for (var i = 0u; i < firstTrailingBit(WG_SIZE); i++) {
            let probe = cols + ((WG_SIZE / 2u) >> i);
            if ix > sh_inclusive_cols[probe - 1u] {
                cols = probe;
            }
        }
        let seg_start = sh_mm[cols].start_x;
        let prefix_cols = select(0u, sh_inclusive_cols[seg_start - 1u], seg_start > 0u);
        let col_within_segment = ix - prefix_cols;
        // now choose a column; this can fail in the 256 case
        let seg_end = sh_seg_end[seg_start];
        let last_histo = sh_histo[seg_end];
        var tile_within_col = col_within_segment;
        var col = 0u;
        while col < 3u {
            let hist_val = (last_histo >> (col * 8u)) & 0xffu;
            if tile_within_col >= hist_val {
                tile_within_col -= hist_val;
                col++;
            } else {
                break;
            }
        }
        // do binary search to find tile within column
        // (search is in seg_start..=seg_end)
        var lo = seg_start;
        var hi = seg_end + 1u;
        let goal = tile_within_col;
        while hi > lo + 1u {
            let mid = (lo + hi) >> 1u;
            if goal >= ((sh_histo[mid - 1u] >> (col * 8u)) & 0xffu) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // at this point, lo should index our tile
        // TODO: predicate on ix < total_cols?
        let render_tile = input[wg_id.x * WG_SIZE + lo];
        var alphas = 0u;
        for (var y = 0u; y < 4u; y++) {
            if tile_within_col == 0u {
                atomicStore(&sh_area[local_id.x], 0);
            }
            workgroupBarrier();
            var area_init = 0;
            if local_id.x == 0u && block_ix != 0u {
                area_init = sh_carryover[y];
            }
            let area = area_init; // TODO: compute from tile
            atomicAdd(&sh_area[local_id.x - tile_within_col], area);
            workgroupBarrier();
            if tile_within_col == 0u {
                let summed_area = atomicLoad(&sh_area[local_id.x]);
                if seg_end == WG_SIZE - 1u {
                    // TODO: only if last column
                    sh_carryover[y] = summed_area;
                }
                let winding_area = sh_mm[seg_end].winding * 256;
                let alpha_u8 = u32(min(abs(summed_area + winding_area), 255));
                alphas = (alphas >> 8u) + (alpha_u8 << 24u);
            }
        }
        if tile_within_col == 0u {
            // TODO: store alphas
        }
    }
}
