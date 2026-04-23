#version 300 es

precision highp float;
precision highp int;

struct Config {
    uint width;
    uint height;
    uint strip_height;
    uint alphas_tex_width_bits;
    uint encoded_paints_tex_width_bits;
    int strip_offset_x;
    int strip_offset_y;
    uint ndc_y_negate;
};

struct TinyOutput2 {
    uint paint_and_rect_flag;
    vec4 position;
};

const uint COLOR_SOURCE_SLOT = 1u;
const uint COLOR_SOURCE_BLEND = 2u;

layout(std140) uniform Config_block_0Fragment { Config _group_0_binding_1_fs; };
uniform highp sampler2D _group_0_binding_2_fs;

flat in uint _vs2fs_location0;
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    TinyOutput2 in_ = TinyOutput2(_vs2fs_location0, gl_FragCoord);
    uint color_source = (in_.paint_and_rect_flag >> 29u) & 3u;

    if (color_source == COLOR_SOURCE_SLOT || color_source == COLOR_SOURCE_BLEND) {
        int x = (int(in_.position.x) - _group_0_binding_1_fs.strip_offset_x) & 255;
        int y = (int(in_.position.y) - _group_0_binding_1_fs.strip_offset_y);
        _fs2p_location0 = texelFetch(_group_0_binding_2_fs, ivec2(x, y), 0);
    } else {
        _fs2p_location0 = vec4(1.0, 0.0, 0.0, 1.0);
    }
}
