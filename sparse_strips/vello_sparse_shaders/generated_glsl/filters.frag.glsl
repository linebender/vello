#version 300 es

precision highp float;
precision highp int;

struct GpuFilterData {
    uint data[12];
};
struct OffsetFilter {
    float dx;
    float dy;
};
struct FloodFilter {
    uint color;
};
struct BlurParams {
    uint n_linear_taps;
    float center_weight;
    float linear_weights[3];
    float linear_offsets[3];
};
struct DropShadowFilter {
    float dx;
    float dy;
    uint color;
};
struct FilterInstanceData {
    uvec2 src_offset;
    uvec2 src_size;
    uvec2 dest_offset;
    uvec2 dest_size;
    uvec2 dest_atlas_size;
    uint filter_offset;
    uvec2 original_offset;
    uvec2 original_size;
    uint pass_kind;
};
struct FilterVertexOutput {
    vec4 position;
    uint filter_offset;
    uvec2 src_offset;
    uvec2 src_size;
    uvec2 dest_offset;
    uvec2 dest_size;
    uvec2 dest_atlas_size;
    uvec2 original_offset;
    uvec2 original_size;
    uint pass_kind;
};
const uint FILTER_SIZE_BYTES = 48u;
const uint FILTER_SIZE_U32_ = 12u;
const uint TEXELS_PER_FILTER = 3u;
const uint FILTER_TYPE_OFFSET = 0u;
const uint FILTER_TYPE_FLOOD = 1u;
const uint FILTER_TYPE_GAUSSIAN_BLUR = 2u;
const uint FILTER_TYPE_DROP_SHADOW = 3u;
const uint PASS_COPY = 0u;
const uint PASS_FLOOD = 1u;
const uint PASS_OFFSET = 2u;
const uint PASS_DOWNSCALE = 3u;
const uint PASS_BLUR_H = 4u;
const uint PASS_BLUR_V = 5u;
const uint PASS_UPSCALE = 6u;
const uint PASS_COMPOSITE_DROP_SHADOW = 7u;
const uint MAX_TAPS_PER_SIDE = 3u;
const vec2 HORIZONTAL = vec2(1.0, 0.0);
const vec2 VERTICAL = vec2(0.0, 1.0);

uniform highp usampler2D _group_0_binding_0_fs;

uniform highp sampler2D _group_1_binding_0_fs;

uniform highp sampler2D _group_2_binding_0_fs;

flat in uint _vs2fs_location0;
flat in uvec2 _vs2fs_location1;
flat in uvec2 _vs2fs_location2;
flat in uvec2 _vs2fs_location3;
flat in uvec2 _vs2fs_location4;
flat in uvec2 _vs2fs_location5;
flat in uvec2 _vs2fs_location6;
flat in uvec2 _vs2fs_location7;
flat in uint _vs2fs_location8;
layout(location = 0) out vec4 _fs2p_location0;

uint unpack_filter_type(GpuFilterData data) {
    return (data.data[0] & 31u);
}

uint unpack_header_n_linear_taps(uint header) {
    return ((header >> 11u) & 3u);
}

OffsetFilter unpack_offset_filter(GpuFilterData data_1) {
    return OffsetFilter(uintBitsToFloat(data_1.data[1]), uintBitsToFloat(data_1.data[2]));
}

FloodFilter unpack_flood_filter(GpuFilterData data_2) {
    return FloodFilter(data_2.data[1]);
}

BlurParams unpack_blur_params(GpuFilterData data_3) {
    float weights[3] = float[3](0.0, 0.0, 0.0);
    float offsets[3] = float[3](0.0, 0.0, 0.0);
    uint i = 0u;
    uint _e3 = unpack_header_n_linear_taps(data_3.data[0]);
    float center_weight_1 = uintBitsToFloat(data_3.data[1]);
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e31 = i;
            i = (_e31 + 1u);
        }
        loop_init = false;
        uint _e11 = i;
        if ((_e11 < MAX_TAPS_PER_SIDE)) {
        } else {
            break;
        }
        {
            uint _e14 = i;
            uint _e18 = i;
            weights[_e14] = uintBitsToFloat(data_3.data[(2u + _e18)]);
            uint _e22 = i;
            uint _e26 = i;
            offsets[_e22] = uintBitsToFloat(data_3.data[(5u + _e26)]);
        }
    }
    float _e33[3] = weights;
    float _e34[3] = offsets;
    return BlurParams(_e3, center_weight_1, _e33, _e34);
}

DropShadowFilter unpack_drop_shadow_filter(GpuFilterData data_4) {
    return DropShadowFilter(uintBitsToFloat(data_4.data[8]), uintBitsToFloat(data_4.data[9]), data_4.data[10]);
}

GpuFilterData load_filter_data(uint texel_offset) {
    uint w = uvec2(textureSize(_group_0_binding_0_fs, 0).xy).x;
    uvec4 t0_ = texelFetch(_group_0_binding_0_fs, ivec2(uvec2((texel_offset % w), (texel_offset / w))), 0);
    uvec4 t1_ = texelFetch(_group_0_binding_0_fs, ivec2(uvec2(((texel_offset + 1u) % w), ((texel_offset + 1u) / w))), 0);
    uvec4 t2_ = texelFetch(_group_0_binding_0_fs, ivec2(uvec2(((texel_offset + 2u) % w), ((texel_offset + 2u) / w))), 0);
    return GpuFilterData(uint[12](t0_.x, t0_.y, t0_.z, t0_.w, t1_.x, t1_.y, t1_.z, t1_.w, t2_.x, t2_.y, t2_.z, t2_.w));
}

vec4 sample_original(FilterVertexOutput in_1, vec2 rel_coord) {
    uvec2 src_coord = uvec2((ivec2(in_1.original_offset) + ivec2(rel_coord)));
    vec4 _e9 = texelFetch(_group_2_binding_0_fs, ivec2(src_coord), 0);
    return _e9;
}

vec4 sample_input(FilterVertexOutput in_2, vec2 rel_coord_1) {
    uvec2 src_coord_1 = uvec2((ivec2(in_2.src_offset) + ivec2(rel_coord_1)));
    vec4 _e9 = texelFetch(_group_1_binding_0_fs, ivec2(src_coord_1), 0);
    return _e9;
}

vec4 sample_input_checked(FilterVertexOutput in_3, vec2 rel_coord_2) {
    bool local_1 = false;
    bool local_2 = false;
    bool local_3 = false;
    if (!((rel_coord_2.x < 0.0))) {
        local_1 = (rel_coord_2.x >= float(in_3.src_size.x));
    } else {
        local_1 = true;
    }
    bool _e14 = local_1;
    if (!(_e14)) {
        local_2 = (rel_coord_2.y < 0.0);
    } else {
        local_2 = true;
    }
    bool _e22 = local_2;
    if (!(_e22)) {
        local_3 = (rel_coord_2.y >= float(in_3.src_size.y));
    } else {
        local_3 = true;
    }
    bool _e32 = local_3;
    if (_e32) {
        return vec4(0.0);
    }
    vec4 _e35 = sample_input(in_3, rel_coord_2);
    return _e35;
}

vec4 downscale(FilterVertexOutput in_4) {
    uvec2 frag_coord = uvec2(in_4.position.xy);
    ivec2 rel = ivec2((frag_coord - in_4.dest_offset));
    vec2 src_rel_1 = vec2((rel * 2));
    vec2 src_texel = (vec2(in_4.src_offset) + src_rel_1);
    vec2 tex_size = vec2(uvec2(textureSize(_group_1_binding_0_fs, 0).xy));
    vec2 lo = vec2(-0.25);
    vec2 hi = vec2(1.25);
    vec4 s00_ = textureLod(_group_1_binding_0_fs, vec2((((src_texel + vec2(lo.x, lo.y)) + vec2(0.5)) / tex_size)), 0.0);
    vec4 s01_ = textureLod(_group_1_binding_0_fs, vec2((((src_texel + vec2(lo.x, hi.y)) + vec2(0.5)) / tex_size)), 0.0);
    vec4 s10_ = textureLod(_group_1_binding_0_fs, vec2((((src_texel + vec2(hi.x, lo.y)) + vec2(0.5)) / tex_size)), 0.0);
    vec4 s11_ = textureLod(_group_1_binding_0_fs, vec2((((src_texel + vec2(hi.x, hi.y)) + vec2(0.5)) / tex_size)), 0.0);
    return ((((s00_ + s01_) + s10_) + s11_) * 0.25);
}

vec4 upscale(FilterVertexOutput in_5) {
    uvec2 frag_coord_1 = uvec2(in_5.position.xy);
    ivec2 rel_1 = ivec2((frag_coord_1 - in_5.dest_offset));
    vec2 src_base = vec2((rel_1 / ivec2(2)));
    vec2 phase = vec2((rel_1 % ivec2(2)));
    vec2 tex_size_1 = vec2(uvec2(textureSize(_group_1_binding_0_fs, 0).xy));
    vec2 sample_offset = mix(vec2(-0.25), vec2(0.25), equal(phase, vec2(1.0)));
    vec2 src_texel_1 = ((vec2(in_5.src_offset) + src_base) + sample_offset);
    vec4 _e37 = textureLod(_group_1_binding_0_fs, vec2(((src_texel_1 + vec2(0.5)) / tex_size_1)), 0.0);
    return _e37;
}

vec4 convolve(FilterVertexOutput in_6, vec2 src_rel, vec2 dir, uint n_linear_taps, float center_weight, float weights_1[3], float offsets_1[3]) {
    vec4 color = vec4(0.0);
    uint i_1 = 0u;
    vec2 src_texel_2 = (vec2(in_6.src_offset) + src_rel);
    vec2 tex_size_2 = vec2(uvec2(textureSize(_group_1_binding_0_fs, 0).xy));
    vec4 _e20 = textureLod(_group_1_binding_0_fs, vec2(((src_texel_2 + vec2(0.5)) / tex_size_2)), 0.0);
    color = (_e20 * center_weight);
    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
            uint _e57 = i_1;
            i_1 = (_e57 + 1u);
        }
        loop_init_1 = false;
        uint _e25 = i_1;
        if ((_e25 < n_linear_taps)) {
        } else {
            break;
        }
        {
            uint _e27 = i_1;
            float w_1 = weights_1[_e27];
            uint _e29 = i_1;
            vec2 d = (dir * offsets_1[_e29]);
            vec4 _e40 = textureLod(_group_1_binding_0_fs, vec2((((src_texel_2 + d) + vec2(0.5)) / tex_size_2)), 0.0);
            vec4 _e42 = color;
            color = (_e42 + (_e40 * w_1));
            vec4 _e52 = textureLod(_group_1_binding_0_fs, vec2((((src_texel_2 - d) + vec2(0.5)) / tex_size_2)), 0.0);
            vec4 _e54 = color;
            color = (_e54 + (_e52 * w_1));
        }
    }
    vec4 _e59 = color;
    return _e59;
}

void main() {
    FilterVertexOutput in_ = FilterVertexOutput(gl_FragCoord, _vs2fs_location0, _vs2fs_location1, _vs2fs_location2, _vs2fs_location3, _vs2fs_location4, _vs2fs_location5, _vs2fs_location6, _vs2fs_location7, _vs2fs_location8);
    bool local = false;
    vec2 dxdy = vec2(0.0);
    uvec2 frag_coord_2 = uvec2(in_.position.xy);
    vec2 rel_coord_3 = vec2((frag_coord_2 - in_.dest_offset));
    if (!((rel_coord_3.x >= float(in_.dest_size.x)))) {
        local = (rel_coord_3.y >= float(in_.dest_size.y));
    } else {
        local = true;
    }
    bool _e21 = local;
    if (_e21) {
        _fs2p_location0 = vec4(0.0);
        return;
    }
    switch(in_.pass_kind) {
        case 0u: {
            vec4 _e25 = sample_input(in_, rel_coord_3);
            _fs2p_location0 = _e25;
            return;
        }
        case 1u: {
            GpuFilterData _e27 = load_filter_data(in_.filter_offset);
            FloodFilter _e28 = unpack_flood_filter(_e27);
            uint _e29 = _e28.color;
            _fs2p_location0 = (vec4(_e29 & 0xFFu, _e29 >> 8 & 0xFFu, _e29 >> 16 & 0xFFu, _e29 >> 24) / 255.0);
            return;
        }
        case 2u: {
            GpuFilterData _e32 = load_filter_data(in_.filter_offset);
            uint _e33 = unpack_filter_type(_e32);
            if ((_e33 == FILTER_TYPE_DROP_SHADOW)) {
                DropShadowFilter _e37 = unpack_drop_shadow_filter(_e32);
                dxdy = vec2(_e37.dx, _e37.dy);
            } else {
                OffsetFilter _e41 = unpack_offset_filter(_e32);
                dxdy = vec2(_e41.dx, _e41.dy);
            }
            vec2 _e45 = dxdy;
            vec4 _e51 = sample_input_checked(in_, (rel_coord_3 - floor((_e45 + vec2(0.5)))));
            _fs2p_location0 = _e51;
            return;
        }
        case 3u: {
            vec4 _e52 = downscale(in_);
            _fs2p_location0 = _e52;
            return;
        }
        case 4u: {
            GpuFilterData _e54 = load_filter_data(in_.filter_offset);
            BlurParams _e55 = unpack_blur_params(_e54);
            vec4 _e61 = convolve(in_, rel_coord_3, HORIZONTAL, _e55.n_linear_taps, _e55.center_weight, _e55.linear_weights, _e55.linear_offsets);
            _fs2p_location0 = _e61;
            return;
        }
        case 5u: {
            GpuFilterData _e63 = load_filter_data(in_.filter_offset);
            BlurParams _e64 = unpack_blur_params(_e63);
            vec4 _e70 = convolve(in_, rel_coord_3, VERTICAL, _e64.n_linear_taps, _e64.center_weight, _e64.linear_weights, _e64.linear_offsets);
            _fs2p_location0 = _e70;
            return;
        }
        case 6u: {
            vec4 _e71 = upscale(in_);
            _fs2p_location0 = _e71;
            return;
        }
        case 7u: {
            GpuFilterData _e73 = load_filter_data(in_.filter_offset);
            DropShadowFilter _e74 = unpack_drop_shadow_filter(_e73);
            vec4 _e75 = sample_input(in_, rel_coord_3);
            uint _e76 = _e74.color;
            vec4 shadow_color = (vec4(_e76 & 0xFFu, _e76 >> 8 & 0xFFu, _e76 >> 16 & 0xFFu, _e76 >> 24) / 255.0);
            vec4 shadow_result = (shadow_color * _e75.w);
            vec4 _e80 = sample_original(in_, rel_coord_3);
            _fs2p_location0 = (_e80 + (shadow_result * (1.0 - _e80.w)));
            return;
        }
        default: {
            _fs2p_location0 = vec4(0.0);
            return;
        }
    }
}

