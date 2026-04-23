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
const uint RECT_STRIP_FLAG = 2147483648u;

layout(std140) uniform Config_block_0Fragment { Config _group_0_binding_1_fs; };

uniform highp usampler2D _group_0_binding_0_fs;
uniform highp sampler2D _group_0_binding_2_fs;

flat in uint _vs2fs_location0;
smooth in vec2 _vs2fs_location1;
smooth in vec2 _vs2fs_location2;
flat in uint _vs2fs_location3;
flat in uint _vs2fs_location4;
flat in uint _vs2fs_location5;

layout(location = 0) out vec4 _fs2p_location0;

uint unpack_alphas_from_channel(uvec4 rgba, uint channel_index) {
    switch (channel_index) {
        case 0u: return rgba.x;
        case 1u: return rgba.y;
        case 2u: return rgba.z;
        case 3u: return rgba.w;
        default: return rgba.x;
    }
}

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

    float alpha = 1.0;
    bool is_rect = ((in_.paint_and_rect_flag & RECT_STRIP_FLAG) != 0u);

    if (is_rect && in_.rect_frac != 0u) {
        uint packed = in_.rect_frac;
        vec4 frac = vec4(
            packed & 0xFFu,
            (packed >> 8u) & 0xFFu,
            (packed >> 16u) & 0xFFu,
            (packed >> 24u) & 0xFFu
        ) / 255.0;
        vec2 rect_size = vec2(
            float(in_.dense_end_or_rect_size & 65535u),
            float(in_.dense_end_or_rect_size >> 16u)
        );
        vec2 tc = in_.tex_coord;
        vec2 bottom_and_right = min(tc + vec2(0.5), rect_size - frac.zw);
        vec2 top_and_left = max(tc - vec2(0.5), frac.xy);
        vec2 coverage = clamp(bottom_and_right - top_and_left, vec2(0.0), vec2(1.0));
        alpha = coverage.x * coverage.y;
    } else if (!is_rect && in_.dense_end_or_rect_size != 0u) {
        uint alphas_index = uint(floor(in_.tex_coord.x));
        uint y = uint(floor(in_.tex_coord.y));
        uint alphas_tex_width = uint(textureSize(_group_0_binding_0_fs, 0).x);
        uint texel_index = alphas_index / 4u;
        uint channel_index = alphas_index % 4u;
        uint tex_x = texel_index & (alphas_tex_width - 1u);
        uint tex_y = texel_index >> _group_0_binding_1_fs.alphas_tex_width_bits;
        uvec4 rgba_values = texelFetch(_group_0_binding_0_fs, ivec2(uvec2(tex_x, tex_y)), 0);
        uint packed_alpha = unpack_alphas_from_channel(rgba_values, channel_index);
        alpha = float((packed_alpha >> (y * 8u)) & 255u) * 0.003921569;
    }

    uint color_source = (in_.paint_and_rect_flag >> 29u) & 3u;
    vec4 final_color = vec4(0.0);

    if (color_source == COLOR_SOURCE_PAYLOAD) {
        uint packed = in_.payload;
        final_color = alpha * (vec4(
            packed & 0xFFu,
            (packed >> 8u) & 0xFFu,
            (packed >> 16u) & 0xFFu,
            (packed >> 24u) & 0xFFu
        ) / 255.0);
    } else if (color_source == COLOR_SOURCE_SLOT) {
        float sample_y = (_group_0_binding_1_fs.ndc_y_negate != 0u)
            ? (float(_group_0_binding_1_fs.height) - in_.position.y)
            : in_.position.y;
        uint clip_x = uint(int(in_.position.x) - _group_0_binding_1_fs.strip_offset_x) & 255u;
        uint clip_y = (
            uint(int(sample_y) - _group_0_binding_1_fs.strip_offset_y) & 3u
        ) + (in_.payload * _group_0_binding_1_fs.strip_height);
        vec4 clip_in_color = texelFetch(_group_0_binding_2_fs, ivec2(uvec2(clip_x, clip_y)), 0);
        float opacity = float(in_.paint_and_rect_flag & 255u) * 0.003921569;
        final_color = alpha * opacity * clip_in_color;
    } else if (color_source == COLOR_SOURCE_BLEND) {
        float sample_y = (_group_0_binding_1_fs.ndc_y_negate != 0u)
            ? (float(_group_0_binding_1_fs.height) - in_.position.y)
            : in_.position.y;
        float opacity = float((in_.paint_and_rect_flag >> 16u) & 255u) * 0.003921569;
        uint src_slot = in_.payload & 65535u;
        uint dest_slot = (in_.payload >> 16u) & 65535u;
        uint clip_x = uint(int(in_.position.x) - _group_0_binding_1_fs.strip_offset_x) & 255u;
        uint clip_y_in_strip = uint(int(sample_y) - _group_0_binding_1_fs.strip_offset_y) & 3u;
        uint src_y = clip_y_in_strip + (src_slot * _group_0_binding_1_fs.strip_height);
        uint dest_y = clip_y_in_strip + (dest_slot * _group_0_binding_1_fs.strip_height);
        vec4 src_color = texelFetch(_group_0_binding_2_fs, ivec2(uvec2(clip_x, src_y)), 0);
        vec4 dest_color = texelFetch(_group_0_binding_2_fs, ivec2(uvec2(clip_x, dest_y)), 0);
        final_color = mix(dest_color, src_color * alpha, opacity);
    }

    _fs2p_location0 = final_color;
}
