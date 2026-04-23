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

struct VertexOutput {
    uint paint_and_rect_flag;
    vec2 tex_coord;
    vec2 sample_xy;
    uint dense_end_or_rect_size;
    uint payload;
    uint rect_frac;
    vec4 position;
};

const uint COLOR_SOURCE_PAYLOAD = 0u;
const uint COLOR_SOURCE_SLOT = 1u;
const uint COLOR_SOURCE_BLEND = 2u;

layout(std140) uniform Config_block_0Fragment { Config _group_0_binding_1_fs; };

uniform highp sampler2D _group_0_binding_2_fs;

flat in uint _vs2fs_location0;
smooth in vec2 _vs2fs_location1;
smooth in vec2 _vs2fs_location2;
flat in uint _vs2fs_location3;
flat in uint _vs2fs_location4;
flat in uint _vs2fs_location5;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    VertexOutput in_ = VertexOutput(
        _vs2fs_location0,
        _vs2fs_location1,
        _vs2fs_location2,
        _vs2fs_location3,
        _vs2fs_location4,
        _vs2fs_location5,
        gl_FragCoord
    );

    uint color_source = (in_.paint_and_rect_flag >> 29u) & 3u;
    vec4 final_color = vec4(0.0);

    if (color_source == COLOR_SOURCE_PAYLOAD) {
        uint packed = in_.payload;
        final_color = vec4(
            packed & 0xFFu,
            (packed >> 8u) & 0xFFu,
            (packed >> 16u) & 0xFFu,
            (packed >> 24u) & 0xFFu
        ) / 255.0;
    } else if (color_source == COLOR_SOURCE_SLOT) {
        float sample_y = (_group_0_binding_1_fs.ndc_y_negate != 0u)
            ? (float(_group_0_binding_1_fs.height) - in_.position.y)
            : in_.position.y;
        uint clip_x = uint(int(in_.position.x) - _group_0_binding_1_fs.strip_offset_x) & 255u;
        uint clip_y = (
            uint(int(sample_y) - _group_0_binding_1_fs.strip_offset_y) & 3u
        ) + (in_.payload * _group_0_binding_1_fs.strip_height);
        final_color = texelFetch(_group_0_binding_2_fs, ivec2(uvec2(clip_x, clip_y)), 0);
    } else if (color_source == COLOR_SOURCE_BLEND) {
        float sample_y = (_group_0_binding_1_fs.ndc_y_negate != 0u)
            ? (float(_group_0_binding_1_fs.height) - in_.position.y)
            : in_.position.y;
        uint src_slot = in_.payload & 65535u;
        uint dest_slot = (in_.payload >> 16u) & 65535u;
        uint clip_x = uint(int(in_.position.x) - _group_0_binding_1_fs.strip_offset_x) & 255u;
        uint clip_y_in_strip = uint(int(sample_y) - _group_0_binding_1_fs.strip_offset_y) & 3u;
        uint src_y = clip_y_in_strip + (src_slot * _group_0_binding_1_fs.strip_height);
        uint dest_y = clip_y_in_strip + (dest_slot * _group_0_binding_1_fs.strip_height);
        vec4 src_color = texelFetch(_group_0_binding_2_fs, ivec2(uvec2(clip_x, src_y)), 0);
        vec4 dest_color = texelFetch(_group_0_binding_2_fs, ivec2(uvec2(clip_x, dest_y)), 0);
        final_color = mix(dest_color, src_color, 0.5);
    }

    _fs2p_location0 = final_color;
}
