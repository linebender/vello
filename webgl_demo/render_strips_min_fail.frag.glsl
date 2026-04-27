#version 300 es

precision highp float;
precision highp int;

struct Pair {
    uint f0;
};

layout(location = 0) out vec4 out_color;

void main() {
    Pair p = Pair(0x00010000u);
    uint mode = p.f0 >> 16;
    float green = (mode == 1u) ? 1.0 : 0.0;
    float red = 1.0 - green;
    out_color = vec4(red, green, 0.0, 1.0);
}
