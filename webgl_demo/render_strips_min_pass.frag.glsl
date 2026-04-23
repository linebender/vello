#version 300 es

precision highp float;
precision highp int;

uniform highp sampler2D _group_0_binding_2_fs;

flat in uint v0;
layout(location = 0) out vec4 out_color;

void main() {
    uint mode = (v0 >> 29u) & 3u;

    if (mode == 1u) {
        out_color = texelFetch(
            _group_0_binding_2_fs,
            ivec2(int(gl_FragCoord.x), int(gl_FragCoord.y)),
            0
        );
    } else {
        out_color = vec4(1.0, 0.0, 0.0, 1.0);
    }
}
