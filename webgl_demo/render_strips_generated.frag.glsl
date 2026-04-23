#version 300 es

precision highp float;
precision highp int;

struct Data {
    uint a;
    vec2 b;
    vec2 c;
    uint d;
    uint e;
    uint f;
    vec4 g;
};

uniform highp sampler2D _group_0_binding_2_fs;

flat in uint v0;
in vec2 v1;
in vec2 v2;
flat in uint v3;
flat in uint v4;
flat in uint v5;
layout(location = 0) out vec4 out_color;

void main() {
    Data x = Data(v0, v1, v2, v3, v4, v5, gl_FragCoord);
    uint mode = (x.a >> 29u) & 3u;
    float sink = 0.0;
    sink += x.b.x * 0.0;
    sink += x.c.y * 0.0;
    sink += float(x.d & 1u) * 0.0;
    sink += float(x.f & 1u) * 0.0;

    if (mode == 1u || mode == 2u) {
        ivec2 p = ivec2(int(x.g.x), int(x.g.y) + int(x.e & 1u));
        out_color = texelFetch(_group_0_binding_2_fs, p, 0);
        out_color.x += sink;
    } else {
        out_color = vec4(1.0, 0.0, 0.0, 1.0);
        out_color.x += sink;
    }
}
