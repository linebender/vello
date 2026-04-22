#version 300 es

precision highp float;
precision highp int;

struct Config {
    uint slot_width;
    uint slot_height;
    uint texture_height;
    uint _padding;
};
layout(std140) uniform Config_block_0Vertex { Config _group_0_binding_0_vs; };

layout(location = 0) in uint _p2vs_location0;

void main() {
    uint vertex_index = uint(gl_VertexID);
    uint index = _p2vs_location0;
    float x = float((vertex_index & 1u));
    float y = float((vertex_index >> 1u));
    uint _e10 = _group_0_binding_0_vs.slot_height;
    float slot_y_offset = float((index * _e10));
    uint _e15 = _group_0_binding_0_vs.slot_width;
    float pix_x = (x * float(_e15));
    uint _e20 = _group_0_binding_0_vs.slot_height;
    float pix_y = (slot_y_offset + (y * float(_e20)));
    uint _e28 = _group_0_binding_0_vs.slot_width;
    float ndc_x = (((pix_x * 2.0) / float(_e28)) - 1.0);
    uint _e37 = _group_0_binding_0_vs.texture_height;
    float ndc_y = (1.0 - ((pix_y * 2.0) / float(_e37)));
    gl_Position = vec4(ndc_x, ndc_y, 0.0, 1.0);
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

