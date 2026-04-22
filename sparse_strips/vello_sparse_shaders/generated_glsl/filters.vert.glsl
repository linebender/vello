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

layout(location = 0) in uvec2 _p2vs_location0;
layout(location = 1) in uvec2 _p2vs_location1;
layout(location = 2) in uvec2 _p2vs_location2;
layout(location = 3) in uvec2 _p2vs_location3;
layout(location = 4) in uvec2 _p2vs_location4;
layout(location = 5) in uint _p2vs_location5;
layout(location = 6) in uvec2 _p2vs_location6;
layout(location = 7) in uvec2 _p2vs_location7;
layout(location = 8) in uint _p2vs_location8;
flat out uint _vs2fs_location0;
flat out uvec2 _vs2fs_location1;
flat out uvec2 _vs2fs_location2;
flat out uvec2 _vs2fs_location3;
flat out uvec2 _vs2fs_location4;
flat out uvec2 _vs2fs_location5;
flat out uvec2 _vs2fs_location6;
flat out uvec2 _vs2fs_location7;
flat out uint _vs2fs_location8;

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

void main() {
    uint vertex_index = uint(gl_VertexID);
    FilterInstanceData instance = FilterInstanceData(_p2vs_location0, _p2vs_location1, _p2vs_location2, _p2vs_location3, _p2vs_location4, _p2vs_location5, _p2vs_location6, _p2vs_location7, _p2vs_location8);
    FilterVertexOutput out_ = FilterVertexOutput(vec4(0.0), 0u, uvec2(0u), uvec2(0u), uvec2(0u), uvec2(0u), uvec2(0u), uvec2(0u), uvec2(0u), 0u);
    uint quad_vertex = (vertex_index % 4u);
    float x = float((quad_vertex & 1u));
    float y = float((quad_vertex >> 1u));
    float pix_x = (float(instance.dest_offset.x) + (x * float(instance.original_size.x)));
    float pix_y = (float(instance.dest_offset.y) + (y * float(instance.original_size.y)));
    vec2 atlas_size = vec2(instance.dest_atlas_size);
    float ndc_x = (((pix_x * 2.0) / atlas_size.x) - 1.0);
    float ndc_y = (1.0 - ((pix_y * 2.0) / atlas_size.y));
    out_.position = vec4(ndc_x, ndc_y, 0.0, 1.0);
    out_.filter_offset = instance.filter_offset;
    out_.src_offset = instance.src_offset;
    out_.src_size = instance.src_size;
    out_.dest_offset = instance.dest_offset;
    out_.dest_size = instance.dest_size;
    out_.dest_atlas_size = instance.dest_atlas_size;
    out_.original_offset = instance.original_offset;
    out_.original_size = instance.original_size;
    out_.pass_kind = instance.pass_kind;
    FilterVertexOutput _e63 = out_;
    gl_Position = _e63.position;
    _vs2fs_location0 = _e63.filter_offset;
    _vs2fs_location1 = _e63.src_offset;
    _vs2fs_location2 = _e63.src_size;
    _vs2fs_location3 = _e63.dest_offset;
    _vs2fs_location4 = _e63.dest_size;
    _vs2fs_location5 = _e63.dest_atlas_size;
    _vs2fs_location6 = _e63.original_offset;
    _vs2fs_location7 = _e63.original_size;
    _vs2fs_location8 = _e63.pass_kind;
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

