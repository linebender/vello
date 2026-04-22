#version 300 es

precision highp float;
precision highp int;

struct Config {
    uint slot_width;
    uint slot_height;
    uint texture_height;
    uint _padding;
};
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec4 position = gl_FragCoord;
    _fs2p_location0 = vec4(0.0, 0.0, 0.0, 0.0);
    return;
}

