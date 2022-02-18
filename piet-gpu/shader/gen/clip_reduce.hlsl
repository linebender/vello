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

static const Bic _267 = { 0u, 0u };

ByteAddressBuffer _64 : register(t1, space0);
RWByteAddressBuffer _80 : register(u0, space0);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared Bic sh_bic[256];
groupshared uint sh_parent[256];
groupshared uint sh_path_ix[256];
groupshared float4 sh_bbox[256];

Bic bic_combine(Bic x, Bic y)
{
    uint m = min(x.b, y.a);
    Bic _56 = { (x.a + y.a) - m, (x.b + y.b) - m };
    return _56;
}

void store_bic(uint ix, Bic bic)
{
    uint base = (_64.Load(52) >> uint(2)) + (2u * ix);
    _80.Store(base * 4 + 8, bic.a);
    _80.Store((base + 1u) * 4 + 8, bic.b);
}

float4 load_path_bbox(uint path_ix)
{
    uint base = (_64.Load(40) >> uint(2)) + (6u * path_ix);
    float bbox_l = float(_80.Load(base * 4 + 8)) - 32768.0f;
    float bbox_t = float(_80.Load((base + 1u) * 4 + 8)) - 32768.0f;
    float bbox_r = float(_80.Load((base + 2u) * 4 + 8)) - 32768.0f;
    float bbox_b = float(_80.Load((base + 3u) * 4 + 8)) - 32768.0f;
    float4 bbox = float4(bbox_l, bbox_t, bbox_r, bbox_b);
    return bbox;
}

void store_clip_el(uint ix, ClipEl el)
{
    uint base = (_64.Load(56) >> uint(2)) + (5u * ix);
    _80.Store(base * 4 + 8, el.parent_ix);
    _80.Store((base + 1u) * 4 + 8, asuint(el.bbox.x));
    _80.Store((base + 2u) * 4 + 8, asuint(el.bbox.y));
    _80.Store((base + 3u) * 4 + 8, asuint(el.bbox.z));
    _80.Store((base + 4u) * 4 + 8, asuint(el.bbox.w));
}

void comp_main()
{
    uint th = gl_LocalInvocationID.x;
    uint inp = _80.Load(((_64.Load(48) >> uint(2)) + gl_GlobalInvocationID.x) * 4 + 8);
    bool is_push = int(inp) >= 0;
    Bic _207 = { 1u - uint(is_push), uint(is_push) };
    Bic bic = _207;
    sh_bic[gl_LocalInvocationID.x] = bic;
    for (uint i = 0u; i < 8u; i++)
    {
        GroupMemoryBarrierWithGroupSync();
        if ((th + (1u << i)) < 256u)
        {
            Bic other = sh_bic[gl_LocalInvocationID.x + (1u << i)];
            Bic param = bic;
            Bic param_1 = other;
            bic = bic_combine(param, param_1);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_bic[th] = bic;
    }
    if (th == 0u)
    {
        uint param_2 = gl_WorkGroupID.x;
        Bic param_3 = bic;
        store_bic(param_2, param_3);
    }
    GroupMemoryBarrierWithGroupSync();
    uint size = sh_bic[0].b;
    bic = _267;
    if ((th + 1u) < 256u)
    {
        bic = sh_bic[th + 1u];
    }
    bool _283;
    if (is_push)
    {
        _283 = bic.a == 0u;
    }
    else
    {
        _283 = is_push;
    }
    if (_283)
    {
        uint local_ix = (size - bic.b) - 1u;
        sh_parent[local_ix] = th;
        sh_path_ix[local_ix] = inp;
    }
    GroupMemoryBarrierWithGroupSync();
    float4 bbox;
    if (th < size)
    {
        uint path_ix = sh_path_ix[th];
        uint param_4 = path_ix;
        bbox = load_path_bbox(param_4);
    }
    if (th < size)
    {
        uint parent_ix = sh_parent[th] + (gl_WorkGroupID.x * 256u);
        ClipEl _331 = { parent_ix, bbox };
        ClipEl el = _331;
        uint param_5 = gl_GlobalInvocationID.x;
        ClipEl param_6 = el;
        store_clip_el(param_5, param_6);
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
