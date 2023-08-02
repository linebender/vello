// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Write path segments

#import bump
#import config
#import segment
#import tile

@group(0) @binding(0)
var<storage> bump: BumpAllocators;

@group(0) @binding(1)
var<storage> seg_counts: array<SegmentCount>;

@group(0) @binding(2)
var<storage> lines: array<LineSoup>;

@group(0) @binding(3)
var<storage> paths: array<Path>;

@group(0) @binding(4)
var<storage> tiles: array<Tile>;

@group(0) @binding(5)
var<storage, read_write> segments: array<Segment>;

fn span(a: f32, b: f32) -> u32 {
    return u32(max(ceil(max(a, b)) - floor(min(a, b)), 1.0));
}

// One invocation for each tile that is to be written.
// Total number of invocations = bump.seg_counts
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let n_segments = atomicLoad(&bump.seg_counts);
    if global_id.x < n_segments {
        let seg_count = seg_counts[global_id.x];
        let line = lines[seg_count.line_ix];
        let counts = seg_count.counts;
        let seg_within_slice = counts >> 16u;
        let seg_within_line = counts & 0xffffu;

        // coarse rasterization logic
        let is_down = line.p1.y >= line.p0.y;
        var xy0 = select(line.p1, line.p0, is_down);
        var xy1 = select(line.p0, line.p1, is_down);
        let s0 = xy0 * TILE_SCALE;
        let s1 = xy1 * TILE_SCALE;
        let count = span(s0.x, s1.x) + span(s0.y, s1.y) - 1u;
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
        let z = floor(a * f32(seg_within_line) + b);
        let x = x0i + i32(sign * z);
        let y = i32(y0i + f32(seg_within_line) - z);

        let path = paths[line.path_ix];
        let bbox = vec4<i32>(path.bbox);
        let stride = bbox.z - bbox.x;
        let tile_ix = i32(path.tiles) + (y - bbox.y) * stride + x - bbox.x;
        let tile = tiles[tile_ix];
        let tile_xy = vec2(f32(x) * f32(TILE_WIDTH), f32(y) * f32(TILE_HEIGHT));
        let tile_xy1 = tile_xy + vec2(f32(TILE_WIDTH), f32(TILE_HEIGHT));

        if seg_within_line > 0u {
            let z_prev = floor(a * (f32(seg_within_line) - 1.0) + b);
            if z == z_prev {
                // Top edge is clipped
                var xt = xy0.x + (xy1.x - xy0.x) * (tile_xy.y - xy0.y) / (xy1.y - xy0.y);
                // TODO: we want to switch to tile-relative coordinates
                xt = clamp(xt, tile_xy.x + 1e-3, tile_xy1.x);
                xy0 = vec2(xt, tile_xy.y);
            } else {
                // If is_positive_slope, left edge is clipped, otherwise right
                let x_clip = select(tile_xy1.x, tile_xy.x, is_positive_slope);
                var yt = xy0.y + (xy1.y - xy0.y) * (x_clip - xy0.x) / (xy1.x - xy0.x);
                yt = clamp(yt, tile_xy.y + 1e-3, tile_xy1.y);
                xy0 = vec2(x_clip, yt);
            }
        }
        if seg_within_line < count - 1u {
            let z_next = floor(a * (f32(seg_within_line) + 1.0) + b);
            if z == z_next {
                // Bottom edge is clipped
                var xt = xy0.x + (xy1.x - xy0.x) * (tile_xy1.y - xy0.y) / (xy1.y - xy0.y);
                xt = clamp(xt, tile_xy.x + 1e-3, tile_xy1.x);
                xy1 = vec2(xt, tile_xy1.y);
            } else {
                // If is_positive_slope, right edge is clipped, otherwise left
                let x_clip = select(tile_xy.x, tile_xy1.x, is_positive_slope);
                var yt = xy0.y + (xy1.y - xy0.y) * (x_clip - xy0.x) / (xy1.x - xy0.x);
                yt = clamp(yt, tile_xy.y + 1e-3, tile_xy1.y);
                xy1 = vec2(x_clip, yt);
            }
        }
        // See comments in CPU version of shader
        var y_edge = 1e9;
        if xy0.x == tile_xy.x {
            y_edge = xy0.y;
        } else if xy1.x == tile_xy.x {
            y_edge = xy1.y;
        }
        if !is_down {
            let tmp = xy0;
            xy0 = xy1;
            xy1 = tmp;
        }
        let segment = Segment(xy0, xy1 - xy0, y_edge);
        segments[tile.segments + seg_within_slice] = segment;
    }
}
