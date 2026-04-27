#version 300 es

precision highp float;
precision highp int;

flat in uint v0;
layout(location = 0) out vec4 out_color;

void main() {
    uint mode = v0 >> 29;

    if (mode == 1u) {
        out_color = vec4(0.0, 1.0, 0.0, 1.0);
    } else {
        out_color = vec4(1.0, 0.0, 0.0, 1.0);
    }
}
