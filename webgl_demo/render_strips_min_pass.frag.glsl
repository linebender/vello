#version 300 es

precision highp float;
precision highp int;

struct Pair {
    uint f0;
    vec4 f1;
};

uniform highp sampler2D _group_0_binding_2_fs;

flat in uint v0;
layout(location = 0) out vec4 out_color;

void main() {
    Pair p = Pair(v0, gl_FragCoord);
    uint mode = v0 >> 29;

    if (mode == 1u) {
        out_color = texelFetch(
            _group_0_binding_2_fs,
            ivec2(10, 20),
            0
        );
    } else {
        out_color = vec4(1.0, 0.0, 0.0, 1.0);
    }
}
