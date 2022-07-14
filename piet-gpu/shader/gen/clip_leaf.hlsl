struct Bic
{
    uint a;
    uint b;
};

struct ClipEl
{
    uint parent_ix;
    float4 bbox;
};

struct Alloc
{
    uint offset;
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

static const Bic _394 = { 0u, 0u };

ByteAddressBuffer _80 : register(t1, space0);
RWByteAddressBuffer _96 : register(u0, space0);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared Bic sh_bic[510];
groupshared uint sh_stack[256];
groupshared float4 sh_stack_bbox[256];
groupshared uint sh_link[256];
groupshared float4 sh_bbox[256];

Bic load_bic(uint ix)
{
    uint base = (_80.Load(56) >> uint(2)) + (2u * ix);
    Bic _287 = { _96.Load(base * 4 + 12), _96.Load((base + 1u) * 4 + 12) };
    return _287;
}

Bic bic_combine(Bic x, Bic y)
{
    uint m = min(x.b, y.a);
    Bic _72 = { (x.a + y.a) - m, (x.b + y.b) - m };
    return _72;
}

ClipEl load_clip_el(uint ix)
{
    uint base = (_80.Load(60) >> uint(2)) + (5u * ix);
    uint parent_ix = _96.Load(base * 4 + 12);
    float x0 = asfloat(_96.Load((base + 1u) * 4 + 12));
    float y0 = asfloat(_96.Load((base + 2u) * 4 + 12));
    float x1 = asfloat(_96.Load((base + 3u) * 4 + 12));
    float y1 = asfloat(_96.Load((base + 4u) * 4 + 12));
    float4 bbox = float4(x0, y0, x1, y1);
    ClipEl _336 = { parent_ix, bbox };
    return _336;
}

float4 bbox_intersect(float4 a, float4 b)
{
    return float4(max(a.xy, b.xy), min(a.zw, b.zw));
}

uint load_path_ix(uint ix)
{
    if (ix < _80.Load(84))
    {
        return _96.Load(((_80.Load(52) >> uint(2)) + ix) * 4 + 12);
    }
    else
    {
        return 2147483648u;
    }
}

float4 load_path_bbox(uint path_ix)
{
    uint base = (_80.Load(44) >> uint(2)) + (6u * path_ix);
    float bbox_l = float(_96.Load(base * 4 + 12)) - 32768.0f;
    float bbox_t = float(_96.Load((base + 1u) * 4 + 12)) - 32768.0f;
    float bbox_r = float(_96.Load((base + 2u) * 4 + 12)) - 32768.0f;
    float bbox_b = float(_96.Load((base + 3u) * 4 + 12)) - 32768.0f;
    float4 bbox = float4(bbox_l, bbox_t, bbox_r, bbox_b);
    return bbox;
}

uint search_link(inout Bic bic)
{
    uint ix = gl_LocalInvocationID.x;
    uint j = 0u;
    while (j < 8u)
    {
        uint base = 512u - (2u << (8u - j));
        if (((ix >> j) & 1u) != 0u)
        {
            Bic param = sh_bic[(base + (ix >> j)) - 1u];
            Bic param_1 = bic;
            Bic test = bic_combine(param, param_1);
            if (test.b > 0u)
            {
                break;
            }
            bic = test;
            ix -= (1u << j);
        }
        j++;
    }
    if (ix > 0u)
    {
        while (j > 0u)
        {
            j--;
            uint base_1 = 512u - (2u << (8u - j));
            Bic param_2 = sh_bic[(base_1 + (ix >> j)) - 1u];
            Bic param_3 = bic;
            Bic test_1 = bic_combine(param_2, param_3);
            if (test_1.b == 0u)
            {
                bic = test_1;
                ix -= (1u << j);
            }
        }
    }
    if (ix > 0u)
    {
        return ix - 1u;
    }
    else
    {
        return 4294967295u - bic.a;
    }
}

void store_clip_bbox(uint ix, float4 bbox)
{
    uint base = (_80.Load(64) >> uint(2)) + (4u * ix);
    _96.Store(base * 4 + 12, asuint(bbox.x));
    _96.Store((base + 1u) * 4 + 12, asuint(bbox.y));
    _96.Store((base + 2u) * 4 + 12, asuint(bbox.z));
    _96.Store((base + 3u) * 4 + 12, asuint(bbox.w));
}

void comp_main()
{
    uint th = gl_LocalInvocationID.x;
    Bic bic = _394;
    if (th < gl_WorkGroupID.x)
    {
        uint param = th;
        bic = load_bic(param);
    }
    sh_bic[th] = bic;
    for (uint i = 0u; i < 8u; i++)
    {
        GroupMemoryBarrierWithGroupSync();
        if ((th + (1u << i)) < 256u)
        {
            Bic other = sh_bic[th + (1u << i)];
            Bic param_1 = bic;
            Bic param_2 = other;
            bic = bic_combine(param_1, param_2);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_bic[th] = bic;
    }
    GroupMemoryBarrierWithGroupSync();
    uint stack_size = sh_bic[0].b;
    uint sp = 255u - th;
    uint ix = 0u;
    for (uint i_1 = 0u; i_1 < 8u; i_1++)
    {
        uint probe = ix + (128u >> i_1);
        if (sp < sh_bic[probe].b)
        {
            ix = probe;
        }
    }
    uint b = sh_bic[ix].b;
    float4 bbox = float4(-1000000000.0f, -1000000000.0f, 1000000000.0f, 1000000000.0f);
    if (sp < b)
    {
        uint param_3 = (((ix * 256u) + b) - sp) - 1u;
        ClipEl el = load_clip_el(param_3);
        sh_stack[th] = el.parent_ix;
        bbox = el.bbox;
    }
    for (uint i_2 = 0u; i_2 < 8u; i_2++)
    {
        sh_stack_bbox[th] = bbox;
        GroupMemoryBarrierWithGroupSync();
        if (th >= (1u << i_2))
        {
            float4 param_4 = sh_stack_bbox[th - (1u << i_2)];
            float4 param_5 = bbox;
            bbox = bbox_intersect(param_4, param_5);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    sh_stack_bbox[th] = bbox;
    uint param_6 = gl_GlobalInvocationID.x;
    uint inp = load_path_ix(param_6);
    bool is_push = int(inp) >= 0;
    Bic _560 = { 1u - uint(is_push), uint(is_push) };
    bic = _560;
    sh_bic[th] = bic;
    if (is_push)
    {
        uint param_7 = inp;
        bbox = load_path_bbox(param_7);
    }
    else
    {
        bbox = float4(-1000000000.0f, -1000000000.0f, 1000000000.0f, 1000000000.0f);
    }
    uint inbase = 0u;
    for (uint i_3 = 0u; i_3 < 7u; i_3++)
    {
        uint outbase = 512u - (1u << (8u - i_3));
        GroupMemoryBarrierWithGroupSync();
        if (th < (1u << (7u - i_3)))
        {
            Bic param_8 = sh_bic[inbase + (th * 2u)];
            Bic param_9 = sh_bic[(inbase + (th * 2u)) + 1u];
            sh_bic[outbase + th] = bic_combine(param_8, param_9);
        }
        inbase = outbase;
    }
    GroupMemoryBarrierWithGroupSync();
    bic = _394;
    Bic param_10 = bic;
    uint _619 = search_link(param_10);
    bic = param_10;
    uint link = _619;
    sh_link[th] = link;
    GroupMemoryBarrierWithGroupSync();
    uint grandparent;
    if (int(link) >= 0)
    {
        grandparent = sh_link[link];
    }
    else
    {
        grandparent = link - 1u;
    }
    uint parent;
    if (int(link) >= 0)
    {
        parent = (gl_WorkGroupID.x * 256u) + link;
    }
    else
    {
        if (int(link + stack_size) >= 0)
        {
            parent = sh_stack[256u + link];
        }
        else
        {
            parent = 4294967295u;
        }
    }
    for (uint i_4 = 0u; i_4 < 8u; i_4++)
    {
        if (i_4 != 0u)
        {
            sh_link[th] = link;
        }
        sh_bbox[th] = bbox;
        GroupMemoryBarrierWithGroupSync();
        if (int(link) >= 0)
        {
            float4 param_11 = sh_bbox[link];
            float4 param_12 = bbox;
            bbox = bbox_intersect(param_11, param_12);
            link = sh_link[link];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (int(link + stack_size) >= 0)
    {
        float4 param_13 = sh_stack_bbox[256u + link];
        float4 param_14 = bbox;
        bbox = bbox_intersect(param_13, param_14);
    }
    sh_bbox[th] = bbox;
    GroupMemoryBarrierWithGroupSync();
    uint path_ix = inp;
    bool _718 = !is_push;
    bool _726;
    if (_718)
    {
        _726 = gl_GlobalInvocationID.x < _80.Load(84);
    }
    else
    {
        _726 = _718;
    }
    if (_726)
    {
        uint param_15 = parent;
        path_ix = load_path_ix(param_15);
        uint drawmonoid_out_base = (_80.Load(48) >> uint(2)) + (4u * (~inp));
        _96.Store(drawmonoid_out_base * 4 + 12, path_ix);
        if (int(grandparent) >= 0)
        {
            bbox = sh_bbox[grandparent];
        }
        else
        {
            if (int(grandparent + stack_size) >= 0)
            {
                bbox = sh_stack_bbox[256u + grandparent];
            }
            else
            {
                bbox = float4(-1000000000.0f, -1000000000.0f, 1000000000.0f, 1000000000.0f);
            }
        }
    }
    uint param_16 = gl_GlobalInvocationID.x;
    float4 param_17 = bbox;
    store_clip_bbox(param_16, param_17);
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
