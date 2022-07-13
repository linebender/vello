struct Alloc
{
    uint offset;
};

struct MallocResult
{
    Alloc alloc;
    bool failed;
};

struct BinInstanceRef
{
    uint offset;
};

struct BinInstance
{
    uint element_ix;
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
    uint n_elements;
    uint n_pathseg;
    uint width_in_tiles;
    uint height_in_tiles;
    Alloc tile_alloc;
    Alloc bin_alloc;
    Alloc ptcl_alloc;
    Alloc pathseg_alloc;
    Alloc anno_alloc;
    Alloc trans_alloc;
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

RWByteAddressBuffer _81 : register(u0, space0);
ByteAddressBuffer _156 : register(t1, space0);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
};

groupshared uint bitmaps[8][256];
groupshared bool sh_alloc_failed;
groupshared uint count[8][256];
groupshared Alloc sh_chunk_alloc[256];

DrawMonoid load_draw_monoid(uint element_ix)
{
    uint base = (_156.Load(44) >> uint(2)) + (4u * element_ix);
    uint path_ix = _81.Load(base * 4 + 8);
    uint clip_ix = _81.Load((base + 1u) * 4 + 8);
    uint scene_offset = _81.Load((base + 2u) * 4 + 8);
    uint info_offset = _81.Load((base + 3u) * 4 + 8);
    DrawMonoid _190 = { path_ix, clip_ix, scene_offset, info_offset };
    return _190;
}

float4 load_clip_bbox(uint clip_ix)
{
    uint base = (_156.Load(60) >> uint(2)) + (4u * clip_ix);
    float x0 = asfloat(_81.Load(base * 4 + 8));
    float y0 = asfloat(_81.Load((base + 1u) * 4 + 8));
    float x1 = asfloat(_81.Load((base + 2u) * 4 + 8));
    float y1 = asfloat(_81.Load((base + 3u) * 4 + 8));
    float4 bbox = float4(x0, y0, x1, y1);
    return bbox;
}

float4 load_path_bbox(uint path_ix)
{
    uint base = (_156.Load(40) >> uint(2)) + (6u * path_ix);
    float bbox_l = float(_81.Load(base * 4 + 8)) - 32768.0f;
    float bbox_t = float(_81.Load((base + 1u) * 4 + 8)) - 32768.0f;
    float bbox_r = float(_81.Load((base + 2u) * 4 + 8)) - 32768.0f;
    float bbox_b = float(_81.Load((base + 3u) * 4 + 8)) - 32768.0f;
    float4 bbox = float4(bbox_l, bbox_t, bbox_r, bbox_b);
    return bbox;
}

float4 bbox_intersect(float4 a, float4 b)
{
    return float4(max(a.xy, b.xy), min(a.zw, b.zw));
}

void store_draw_bbox(uint draw_ix, float4 bbox)
{
    uint base = (_156.Load(64) >> uint(2)) + (4u * draw_ix);
    _81.Store(base * 4 + 8, asuint(bbox.x));
    _81.Store((base + 1u) * 4 + 8, asuint(bbox.y));
    _81.Store((base + 2u) * 4 + 8, asuint(bbox.z));
    _81.Store((base + 3u) * 4 + 8, asuint(bbox.w));
}

Alloc new_alloc(uint offset, uint size, bool mem_ok)
{
    Alloc a;
    a.offset = offset;
    return a;
}

MallocResult malloc(uint size)
{
    uint _87;
    _81.InterlockedAdd(0, size, _87);
    uint offset = _87;
    uint _94;
    _81.GetDimensions(_94);
    _94 = (_94 - 8) / 4;
    MallocResult r;
    r.failed = (offset + size) > uint(int(_94) * 4);
    uint param = offset;
    uint param_1 = size;
    bool param_2 = !r.failed;
    r.alloc = new_alloc(param, param_1, param_2);
    if (r.failed)
    {
        uint _116;
        _81.InterlockedMax(4, 1u, _116);
        return r;
    }
    return r;
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
    _81.Store(offset * 4 + 8, val);
}

void BinInstance_write(Alloc a, BinInstanceRef ref, BinInstance s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = s.element_ix;
    write_mem(param, param_1, param_2);
}

void comp_main()
{
    uint my_partition = gl_WorkGroupID.x;
    for (uint i = 0u; i < 8u; i++)
    {
        bitmaps[i][gl_LocalInvocationID.x] = 0u;
    }
    if (gl_LocalInvocationID.x == 0u)
    {
        sh_alloc_failed = false;
    }
    GroupMemoryBarrierWithGroupSync();
    uint element_ix = (my_partition * 256u) + gl_LocalInvocationID.x;
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
    if (element_ix < _156.Load(0))
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
        float4 _417 = bbox;
        float4 _419 = bbox;
        float2 _421 = max(_417.xy, _419.zw);
        bbox.z = _421.x;
        bbox.w = _421.y;
        uint param_5 = element_ix;
        float4 param_6 = bbox;
        store_draw_bbox(param_5, param_6);
        x0 = int(floor(bbox.x * 0.00390625f));
        y0 = int(floor(bbox.y * 0.00390625f));
        x1 = int(ceil(bbox.z * 0.00390625f));
        y1 = int(ceil(bbox.w * 0.00390625f));
    }
    uint width_in_bins = ((_156.Load(8) + 16u) - 1u) / 16u;
    uint height_in_bins = ((_156.Load(12) + 16u) - 1u) / 16u;
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
        uint _523;
        InterlockedOr(bitmaps[my_slice][(uint(y) * width_in_bins) + uint(x)], my_mask, _523);
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
    uint param_7 = 0u;
    uint param_8 = 0u;
    bool param_9 = true;
    Alloc chunk_alloc = new_alloc(param_7, param_8, param_9);
    if (element_count != 0u)
    {
        uint param_10 = element_count * 4u;
        MallocResult _573 = malloc(param_10);
        MallocResult chunk = _573;
        chunk_alloc = chunk.alloc;
        sh_chunk_alloc[gl_LocalInvocationID.x] = chunk_alloc;
        if (chunk.failed)
        {
            sh_alloc_failed = true;
        }
    }
    uint out_ix = (_156.Load(20) >> uint(2)) + (((my_partition * 256u) + gl_LocalInvocationID.x) * 2u);
    Alloc _603;
    _603.offset = _156.Load(20);
    Alloc param_11;
    param_11.offset = _603.offset;
    uint param_12 = out_ix;
    uint param_13 = element_count;
    write_mem(param_11, param_12, param_13);
    Alloc _615;
    _615.offset = _156.Load(20);
    Alloc param_14;
    param_14.offset = _615.offset;
    uint param_15 = out_ix + 1u;
    uint param_16 = chunk_alloc.offset;
    write_mem(param_14, param_15, param_16);
    GroupMemoryBarrierWithGroupSync();
    bool _630;
    if (!sh_alloc_failed)
    {
        _630 = _81.Load(4) != 0u;
    }
    else
    {
        _630 = sh_alloc_failed;
    }
    if (_630)
    {
        return;
    }
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
            Alloc out_alloc = sh_chunk_alloc[bin_ix];
            uint out_offset = out_alloc.offset + (idx * 4u);
            BinInstanceRef _692 = { out_offset };
            BinInstance _694 = { element_ix };
            Alloc param_17 = out_alloc;
            BinInstanceRef param_18 = _692;
            BinInstance param_19 = _694;
            BinInstance_write(param_17, param_18, param_19);
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
