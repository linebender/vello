struct Alloc
{
    uint offset;
};

struct MallocResult
{
    Alloc alloc;
    bool failed;
};

struct AnnotatedRef
{
    uint offset;
};

struct AnnotatedTag
{
    uint tag;
    uint flags;
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
    Alloc bbox_alloc;
    Alloc drawmonoid_alloc;
    Alloc clip_alloc;
    Alloc clip_bic_alloc;
    Alloc clip_stack_alloc;
    Alloc clip_bbox_alloc;
    uint n_trans;
    uint n_path;
    uint n_clip;
    uint trans_offset;
    uint linewidth_offset;
    uint pathtag_offset;
    uint pathseg_offset;
};

static const uint3 gl_WorkGroupSize = uint3(256u, 1u, 1u);

RWByteAddressBuffer _94 : register(u0, space0);
ByteAddressBuffer _202 : register(t1, space0);

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

bool touch_mem(Alloc alloc, uint offset)
{
    return true;
}

uint read_mem(Alloc alloc, uint offset)
{
    Alloc param = alloc;
    uint param_1 = offset;
    if (!touch_mem(param, param_1))
    {
        return 0u;
    }
    uint v = _94.Load(offset * 4 + 8);
    return v;
}

AnnotatedTag Annotated_tag(Alloc a, AnnotatedRef ref)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint tag_and_flags = read_mem(param, param_1);
    AnnotatedTag _181 = { tag_and_flags & 65535u, tag_and_flags >> uint(16) };
    return _181;
}

DrawMonoid load_draw_monoid(uint element_ix)
{
    uint base = (_202.Load(44) >> uint(2)) + (2u * element_ix);
    uint path_ix = _94.Load(base * 4 + 8);
    uint clip_ix = _94.Load((base + 1u) * 4 + 8);
    DrawMonoid _222 = { path_ix, clip_ix };
    return _222;
}

float4 load_clip_bbox(uint clip_ix)
{
    uint base = (_202.Load(60) >> uint(2)) + (4u * clip_ix);
    float x0 = asfloat(_94.Load(base * 4 + 8));
    float y0 = asfloat(_94.Load((base + 1u) * 4 + 8));
    float x1 = asfloat(_94.Load((base + 2u) * 4 + 8));
    float y1 = asfloat(_94.Load((base + 3u) * 4 + 8));
    float4 bbox = float4(x0, y0, x1, y1);
    return bbox;
}

float4 load_path_bbox(uint path_ix)
{
    uint base = (_202.Load(40) >> uint(2)) + (6u * path_ix);
    float bbox_l = float(_94.Load(base * 4 + 8)) - 32768.0f;
    float bbox_t = float(_94.Load((base + 1u) * 4 + 8)) - 32768.0f;
    float bbox_r = float(_94.Load((base + 2u) * 4 + 8)) - 32768.0f;
    float bbox_b = float(_94.Load((base + 3u) * 4 + 8)) - 32768.0f;
    float4 bbox = float4(bbox_l, bbox_t, bbox_r, bbox_b);
    return bbox;
}

float4 bbox_intersect(float4 a, float4 b)
{
    return float4(max(a.xy, b.xy), min(a.zw, b.zw));
}

void store_path_bbox(AnnotatedRef ref, float4 bbox)
{
    uint ix = ref.offset >> uint(2);
    _94.Store((ix + 1u) * 4 + 8, asuint(bbox.x));
    _94.Store((ix + 2u) * 4 + 8, asuint(bbox.y));
    _94.Store((ix + 3u) * 4 + 8, asuint(bbox.z));
    _94.Store((ix + 4u) * 4 + 8, asuint(bbox.w));
}

Alloc new_alloc(uint offset, uint size, bool mem_ok)
{
    Alloc a;
    a.offset = offset;
    return a;
}

MallocResult malloc(uint size)
{
    uint _100;
    _94.InterlockedAdd(0, size, _100);
    uint offset = _100;
    uint _107;
    _94.GetDimensions(_107);
    _107 = (_107 - 8) / 4;
    MallocResult r;
    r.failed = (offset + size) > uint(int(_107) * 4);
    uint param = offset;
    uint param_1 = size;
    bool param_2 = !r.failed;
    r.alloc = new_alloc(param, param_1, param_2);
    if (r.failed)
    {
        uint _129;
        _94.InterlockedMax(4, 1u, _129);
        return r;
    }
    return r;
}

void write_mem(Alloc alloc, uint offset, uint val)
{
    Alloc param = alloc;
    uint param_1 = offset;
    if (!touch_mem(param, param_1))
    {
        return;
    }
    _94.Store(offset * 4 + 8, val);
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
    uint my_n_elements = _202.Load(0);
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
    AnnotatedRef _415 = { _202.Load(32) + (element_ix * 40u) };
    AnnotatedRef ref = _415;
    uint tag = 0u;
    if (element_ix < my_n_elements)
    {
        Alloc _425;
        _425.offset = _202.Load(32);
        Alloc param;
        param.offset = _425.offset;
        AnnotatedRef param_1 = ref;
        tag = Annotated_tag(param, param_1).tag;
    }
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
    switch (tag)
    {
        case 1u:
        case 2u:
        case 3u:
        case 4u:
        case 5u:
        {
            uint param_2 = element_ix;
            DrawMonoid draw_monoid = load_draw_monoid(param_2);
            uint path_ix = draw_monoid.path_ix;
            float4 clip_bbox = float4(-1000000000.0f, -1000000000.0f, 1000000000.0f, 1000000000.0f);
            uint clip_ix = draw_monoid.clip_ix;
            if (clip_ix > 0u)
            {
                uint param_3 = clip_ix - 1u;
                clip_bbox = load_clip_bbox(param_3);
            }
            uint param_4 = path_ix;
            float4 path_bbox = load_path_bbox(param_4);
            float4 param_5 = path_bbox;
            float4 param_6 = clip_bbox;
            float4 bbox = bbox_intersect(param_5, param_6);
            float4 _473 = bbox;
            float4 _475 = bbox;
            float2 _477 = max(_473.xy, _475.zw);
            bbox.z = _477.x;
            bbox.w = _477.y;
            AnnotatedRef param_7 = ref;
            float4 param_8 = bbox;
            store_path_bbox(param_7, param_8);
            x0 = int(floor(bbox.x * 0.00390625f));
            y0 = int(floor(bbox.y * 0.00390625f));
            x1 = int(ceil(bbox.z * 0.00390625f));
            y1 = int(ceil(bbox.w * 0.00390625f));
            break;
        }
    }
    uint width_in_bins = ((_202.Load(8) + 16u) - 1u) / 16u;
    uint height_in_bins = ((_202.Load(12) + 16u) - 1u) / 16u;
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
        uint _581;
        InterlockedOr(bitmaps[my_slice][(uint(y) * width_in_bins) + uint(x)], my_mask, _581);
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
    uint param_9 = 0u;
    uint param_10 = 0u;
    bool param_11 = true;
    Alloc chunk_alloc = new_alloc(param_9, param_10, param_11);
    if (element_count != 0u)
    {
        uint param_12 = element_count * 4u;
        MallocResult _631 = malloc(param_12);
        MallocResult chunk = _631;
        chunk_alloc = chunk.alloc;
        sh_chunk_alloc[gl_LocalInvocationID.x] = chunk_alloc;
        if (chunk.failed)
        {
            sh_alloc_failed = true;
        }
    }
    uint out_ix = (_202.Load(20) >> uint(2)) + (((my_partition * 256u) + gl_LocalInvocationID.x) * 2u);
    Alloc _660;
    _660.offset = _202.Load(20);
    Alloc param_13;
    param_13.offset = _660.offset;
    uint param_14 = out_ix;
    uint param_15 = element_count;
    write_mem(param_13, param_14, param_15);
    Alloc _672;
    _672.offset = _202.Load(20);
    Alloc param_16;
    param_16.offset = _672.offset;
    uint param_17 = out_ix + 1u;
    uint param_18 = chunk_alloc.offset;
    write_mem(param_16, param_17, param_18);
    GroupMemoryBarrierWithGroupSync();
    bool _687;
    if (!sh_alloc_failed)
    {
        _687 = _94.Load(4) != 0u;
    }
    else
    {
        _687 = sh_alloc_failed;
    }
    if (_687)
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
            BinInstanceRef _749 = { out_offset };
            BinInstance _751 = { element_ix };
            Alloc param_19 = out_alloc;
            BinInstanceRef param_20 = _749;
            BinInstance param_21 = _751;
            BinInstance_write(param_19, param_20, param_21);
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
