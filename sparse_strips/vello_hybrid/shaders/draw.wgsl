// This shader takes the wide tile commands (and their positions) as vertex
// instance data. The vertex buffer steps per index.

// The shader is not truly generic over the values here, as the layout of and
// indexing into `alpha_masks` is directly influenced by it. They're named as
// constants mostly for clarity.
const TILE_WIDTH: u32 = 4;
const TILE_HEIGHT: u32 = 4;

// A draw command strip.
struct Instance {
    // The draw command origin x-coordinate in pixels.
    @location(0) x: u32,
    // The draw command origin y-coordinate in pixels.
    @location(1) y: u32,
    // The width in pixels of the strip (the height is given by `TILE_HEIGHT`).
    @location(2) width: u32,
    // The starting index into the alpha mask buffer. If this is 0xffff, the
    // draw is a fill.
    @location(3) alpha_idx: u32,
    // The color to draw.
    @location(4) color: u32,
    // An alpha mask vector to be applied to the columns.
    @location(5) column_mask: vec4<f32>,
}

struct VertexOutput {
    // The framebuffer position.
    @builtin(position) pos: vec4<f32>,
    // The color to draw.
    @location(0) color: vec4<f32>,
    // The starting index into the alpha mask buffer. If this is 0xffff, the
    // draw is a fill.
    @location(1) alpha_idx: u32,
    // An alpha mask vector to be applied to the columns.
    @location(2) column_mask: vec4<f32>,
    // The draw command origin x-coordinate in pixels.
    @location(3) x: u32,
    // The draw command origin y-coordinate in pixels.
    @location(4) y: u32,
}

struct Config {
    // The image width in pixels.
    width: u32,
    // The image height in pixels.
    height: u32,
}

@group(0) @binding(0)
var<uniform> config: Config;

@vertex
fn vs(
    @builtin(vertex_index) idx: u32,
    instance: Instance,
) -> VertexOutput {
    let x0 = -1 + 2 * f32(instance.x) / f32(config.width);
    let x1 = -1 + 2 * f32(instance.x + instance.width) / f32(config.width);
    let y0 = 1 - 2 * f32(instance.y) / f32(config.height);
    let y1 = 1 - 2 * f32(instance.y + TILE_HEIGHT) / f32(config.height);
    let vertex = array(
        vec2(x0, y0),
        vec2(x1, y0),
        vec2(x0, y1),
        vec2(x1, y1),
    );

    var output: VertexOutput;
    output.pos = vec4(vertex[idx], 0.0, 1.0);
    output.color = unpack4x8unorm(instance.color);
    output.alpha_idx = instance.alpha_idx;
    output.x = instance.x;
    output.y = instance.y;
    output.column_mask = instance.column_mask;
    return output;
}

@group(0) @binding(1)
var<uniform> alpha_masks: array<vec4<u32>, 1024>;

struct FragOut {
    @location(0) color: vec4<f32>,
}

@fragment
fn fs(in: VertexOutput) -> FragOut {
    let in_x = floor(in.pos.x);
    let in_y = floor(in.pos.y);

    var output: FragOut;
    let alpha_idx = in.alpha_idx + (u32(in_x) - in.x) / TILE_WIDTH;
    let alpha_mask = unpack4x8unorm(alpha_masks[alpha_idx][u32(in_x) % TILE_WIDTH]);
    output.color = in.color
        * in.column_mask[u32(in_y) % TILE_HEIGHT]
        * (f32(in.alpha_idx == 0xffff) + f32(in.alpha_idx != 0xffff) * alpha_mask[u32(in_y) - in.y]);
    return output;
}
