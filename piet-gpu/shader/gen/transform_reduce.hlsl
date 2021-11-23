struct TransformRef
{
    uint offset;
};

struct Transform
{
    float4 mat;
    float2 translate;
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
    uint n_trans;
    uint trans_offset;
};

static const uint3 gl_WorkGroupSize = uint3(512u, 1u, 1u);

ByteAddressBuffer _49 : register(t2);
ByteAddressBuffer _161 : register(t1);
RWByteAddressBuffer _251 : register(u3);
RWByteAddressBuffer _267 : register(u0);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared Transform sh_scratch[512];

Transform Transform_read(TransformRef ref)
{
    uint ix = ref.offset >> uint(2);
    uint raw0 = _49.Load((ix + 0u) * 4 + 0);
    uint raw1 = _49.Load((ix + 1u) * 4 + 0);
    uint raw2 = _49.Load((ix + 2u) * 4 + 0);
    uint raw3 = _49.Load((ix + 3u) * 4 + 0);
    uint raw4 = _49.Load((ix + 4u) * 4 + 0);
    uint raw5 = _49.Load((ix + 5u) * 4 + 0);
    Transform s;
    s.mat = float4(asfloat(raw0), asfloat(raw1), asfloat(raw2), asfloat(raw3));
    s.translate = float2(asfloat(raw4), asfloat(raw5));
    return s;
}

TransformRef Transform_index(TransformRef ref, uint index)
{
    TransformRef _37 = { ref.offset + (index * 24u) };
    return _37;
}

Transform combine_monoid(Transform a, Transform b)
{
    Transform c;
    c.mat = (a.mat.xyxy * b.mat.xxzz) + (a.mat.zwzw * b.mat.yyww);
    c.translate = ((a.mat.xy * b.translate.x) + (a.mat.zw * b.translate.y)) + a.translate;
    return c;
}

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x * 8u;
    TransformRef _168 = { _161.Load(44) + (ix * 24u) };
    TransformRef ref = _168;
    TransformRef param = ref;
    Transform agg = Transform_read(param);
    for (uint i = 1u; i < 8u; i++)
    {
        TransformRef param_1 = ref;
        uint param_2 = i;
        TransformRef param_3 = Transform_index(param_1, param_2);
        Transform param_4 = agg;
        Transform param_5 = Transform_read(param_3);
        agg = combine_monoid(param_4, param_5);
    }
    sh_scratch[gl_LocalInvocationID.x] = agg;
    for (uint i_1 = 0u; i_1 < 9u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if ((gl_LocalInvocationID.x + (1u << i_1)) < 512u)
        {
            Transform other = sh_scratch[gl_LocalInvocationID.x + (1u << i_1)];
            Transform param_6 = agg;
            Transform param_7 = other;
            agg = combine_monoid(param_6, param_7);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[gl_LocalInvocationID.x] = agg;
    }
    if (gl_LocalInvocationID.x == 0u)
    {
        _251.Store4(gl_WorkGroupID.x * 32 + 0, asuint(agg.mat));
        _251.Store2(gl_WorkGroupID.x * 32 + 16, asuint(agg.translate));
    }
}

[numthreads(512, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
