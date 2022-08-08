struct Alloc
{
    uint offset;
};

struct DrawMonoid
{
    uint path_ix;
    uint clip_ix;
    uint scene_offset;
    uint info_offset;
};

struct Config
{
    uint mem_size;
    uint n_elements;
    uint n_pathseg;
    uint width_in_tiles;
    uint height_in_tiles;
    Alloc tile_alloc;
    Alloc bin_alloc;
    Alloc ptcl_alloc;
    Alloc pathseg_alloc;
    Alloc anno_alloc;
    Alloc path_bbox_alloc;
    Alloc drawmonoid_alloc;
    Alloc clip_alloc;
    Alloc clip_bic_alloc;
    Alloc clip_stack_alloc;
    Alloc clip_bbox_alloc;
    Alloc draw_bbox_alloc;
    Alloc drawinfo_alloc;
    uint n_trans;
    uint n_path;
    uint n_clip;
    uint trans_offset;
    uint linewidth_offset;
    uint pathtag_offset;
    uint pathseg_offset;
    uint drawtag_offset;
    uint drawdata_offset;
};

static const uint3 gl_WorkGroupSize = uint3(256u, 1u, 1u);

RWByteAddressBuffer _57 : register(u0, space0);
ByteAddressBuffer _101 : register(t1, space0);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
};

groupshared uint bitmaps[8][256];
groupshared uint count[8][256];
groupshared uint sh_chunk_offset[256];

DrawMonoid load_draw_monoid(uint element_ix)
{
    uint base = (_101.Load(44) >> uint(2)) + (4u * element_ix);
    uint path_ix = _57.Load(base * 4 + 12);
    uint clip_ix = _57.Load((base + 1u) * 4 + 12);
    uint scene_offset = _57.Load((base + 2u) * 4 + 12);
    uint info_offset = _57.Load((base + 3u) * 4 + 12);
    DrawMonoid _136 = { path_ix, clip_ix, scene_offset, info_offset };
    return _136;
}

float4 load_clip_bbox(uint clip_ix)
{
    uint base = (_101.Load(60) >> uint(2)) + (4u * clip_ix);
    float x0 = asfloat(_57.Load(base * 4 + 12));
    float y0 = asfloat(_57.Load((base + 1u) * 4 + 12));
    float x1 = asfloat(_57.Load((base + 2u) * 4 + 12));
    float y1 = asfloat(_57.Load((base + 3u) * 4 + 12));
    float4 bbox = float4(x0, y0, x1, y1);
    return bbox;
}

float4 load_path_bbox(uint path_ix)
{
    uint base = (_101.Load(40) >> uint(2)) + (6u * path_ix);
    float bbox_l = float(_57.Load(base * 4 + 12)) - 32768.0f;
    float bbox_t = float(_57.Load((base + 1u) * 4 + 12)) - 32768.0f;
    float bbox_r = float(_57.Load((base + 2u) * 4 + 12)) - 32768.0f;
    float bbox_b = float(_57.Load((base + 3u) * 4 + 12)) - 32768.0f;
    float4 bbox = float4(bbox_l, bbox_t, bbox_r, bbox_b);
    return bbox;
}

float4 bbox_intersect(float4 a, float4 b)
{
    return float4(max(a.xy, b.xy), min(a.zw, b.zw));
}

void store_draw_bbox(uint draw_ix, float4 bbox)
{
    uint base = (_101.Load(64) >> uint(2)) + (4u * draw_ix);
    _57.Store(base * 4 + 12, asuint(bbox.x));
    _57.Store((base + 1u) * 4 + 12, asuint(bbox.y));
    _57.Store((base + 2u) * 4 + 12, asuint(bbox.z));
    _57.Store((base + 3u) * 4 + 12, asuint(bbox.w));
}

uint malloc_stage(uint size, uint mem_size, uint stage)
{
    uint _65;
    _57.InterlockedAdd(0, size, _65);
    uint offset = _65;
    if ((offset + size) > mem_size)
    {
        uint _76;
        _57.InterlockedOr(4, stage, _76);
        offset = 0u;
    }
    return offset;
}

bool touch_mem(Alloc alloc, uint offset)
{
    return true;
}

void write_mem(Alloc alloc, uint offset, uint val)
{
    Alloc param = alloc;
    uint param_1 = offset;
    if (!touch_mem(param, param_1))
    {
        return;
    }
    _57.Store(offset * 4 + 12, val);
}

void comp_main()
{
    uint my_partition = gl_WorkGroupID.x;
    for (uint i = 0u; i < 8u; i++)
    {
        bitmaps[i][gl_LocalInvocationID.x] = 0u;
    }
    uint element_ix = (my_partition * 256u) + gl_LocalInvocationID.x;
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
    if (element_ix < _101.Load(4))
    {
        uint param = element_ix;
        DrawMonoid draw_monoid = load_draw_monoid(param);
        uint path_ix = draw_monoid.path_ix;
        float4 clip_bbox = float4(-1000000000.0f, -1000000000.0f, 1000000000.0f, 1000000000.0f);
        uint clip_ix = draw_monoid.clip_ix;
        if (clip_ix > 0u)
        {
            uint param_1 = clip_ix - 1u;
            clip_bbox = load_clip_bbox(param_1);
        }
        uint param_2 = path_ix;
        float4 path_bbox = load_path_bbox(param_2);
        float4 param_3 = path_bbox;
        float4 param_4 = clip_bbox;
        float4 bbox = bbox_intersect(param_3, param_4);
        float4 _354 = bbox;
        float4 _356 = bbox;
        float2 _358 = max(_354.xy, _356.zw);
        bbox.z = _358.x;
        bbox.w = _358.y;
        uint param_5 = element_ix;
        float4 param_6 = bbox;
        store_draw_bbox(param_5, param_6);
        x0 = int(floor(bbox.x * 0.00390625f));
        y0 = int(floor(bbox.y * 0.00390625f));
        x1 = int(ceil(bbox.z * 0.00390625f));
        y1 = int(ceil(bbox.w * 0.00390625f));
    }
    uint width_in_bins = ((_101.Load(12) + 16u) - 1u) / 16u;
    uint height_in_bins = ((_101.Load(16) + 16u) - 1u) / 16u;
    x0 = clamp(x0, 0, int(width_in_bins));
    x1 = clamp(x1, x0, int(width_in_bins));
    y0 = clamp(y0, 0, int(height_in_bins));
    y1 = clamp(y1, y0, int(height_in_bins));
    if (x0 == x1)
    {
        y1 = y0;
    }
    int x = x0;
    int y = y0;
    uint my_slice = gl_LocalInvocationID.x / 32u;
    uint my_mask = 1u << (gl_LocalInvocationID.x & 31u);
    while (y < y1)
    {
        uint _460;
        InterlockedOr(bitmaps[my_slice][(uint(y) * width_in_bins) + uint(x)], my_mask, _460);
        x++;
        if (x == x1)
        {
            x = x0;
            y++;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint element_count = 0u;
    for (uint i_1 = 0u; i_1 < 8u; i_1++)
    {
        element_count += uint(int(countbits(bitmaps[i_1][gl_LocalInvocationID.x])));
        count[i_1][gl_LocalInvocationID.x] = element_count;
    }
    uint chunk_offset = 0u;
    if (element_count != 0u)
    {
        uint param_7 = element_count * 4u;
        uint param_8 = _101.Load(0);
        uint param_9 = 1u;
        uint _510 = malloc_stage(param_7, param_8, param_9);
        chunk_offset = _510;
        sh_chunk_offset[gl_LocalInvocationID.x] = chunk_offset;
    }
    uint out_ix = (_101.Load(24) >> uint(2)) + (((my_partition * 256u) + gl_LocalInvocationID.x) * 2u);
    Alloc _532;
    _532.offset = _101.Load(24);
    Alloc param_10;
    param_10.offset = _532.offset;
    uint param_11 = out_ix;
    uint param_12 = element_count;
    write_mem(param_10, param_11, param_12);
    Alloc _544;
    _544.offset = _101.Load(24);
    Alloc param_13;
    param_13.offset = _544.offset;
    uint param_14 = out_ix + 1u;
    uint param_15 = chunk_offset;
    write_mem(param_13, param_14, param_15);
    GroupMemoryBarrierWithGroupSync();
    x = x0;
    y = y0;
    while (y < y1)
    {
        uint bin_ix = (uint(y) * width_in_bins) + uint(x);
        uint out_mask = bitmaps[my_slice][bin_ix];
        if ((out_mask & my_mask) != 0u)
        {
            uint idx = uint(int(countbits(out_mask & (my_mask - 1u))));
            if (my_slice > 0u)
            {
                idx += count[my_slice - 1u][bin_ix];
            }
            uint chunk_offset_1 = sh_chunk_offset[bin_ix];
            if (chunk_offset_1 != 0u)
            {
                _57.Store(((chunk_offset_1 >> uint(2)) + idx) * 4 + 12, element_ix);
            }
        }
        x++;
        if (x == x1)
        {
            x = x0;
            y++;
        }
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    comp_main();
}
