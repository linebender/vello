#version 300 es

precision highp float;
precision highp int;

uniform highp sampler2D _group_0_binding_2_fs;

flat in uint v0;
in vec2 v1;
in vec2 v2;
flat in uint v3;
flat in uint v4;
flat in uint v5;
layout(location = 0) out vec4 out_color;

void main() {
    uint a = v0;
    vec2 b = v1;
    vec2 c = v2;
    uint d = v3;
    uint e = v4;
    uint f = v5;
    vec4 g = gl_FragCoord;
    uint mode = (a >> 29u) & 3u;
    float sink = 0.0;
    sink += b.x * 0.0;
    sink += c.y * 0.0;
    sink += float(d & 1u) * 0.0;
    sink += float(f & 1u) * 0.0;

    if (mode == 1u || mode == 2u) {
        ivec2 p = ivec2(int(g.x), int(g.y) + int(e & 1u));
        out_color = texelFetch(_group_0_binding_2_fs, p, 0);
        out_color.x += sink;
    } else {
        out_color = vec4(1.0, 0.0, 0.0, 1.0);
        out_color.x += sink;
    }
}
