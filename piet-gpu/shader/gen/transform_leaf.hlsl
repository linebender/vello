struct Alloc
{
    uint offset;
};

struct TransformRef
{
    uint offset;
};

struct Transform
{
    float4 mat;
    float2 translate;
};

struct TransformSegRef
{
    uint offset;
};

struct TransformSeg
{
    float4 mat;
    float2 translate;
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

static const Transform _224 = { float4(1.0f, 0.0f, 0.0f, 1.0f), 0.0f.xx };

RWByteAddressBuffer _71 : register(u0);
ByteAddressBuffer _96 : register(t2);
ByteAddressBuffer _278 : register(t1);
ByteAddressBuffer _377 : register(t3);

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
    uint raw0 = _96.Load((ix + 0u) * 4 + 0);
    uint raw1 = _96.Load((ix + 1u) * 4 + 0);
    uint raw2 = _96.Load((ix + 2u) * 4 + 0);
    uint raw3 = _96.Load((ix + 3u) * 4 + 0);
    uint raw4 = _96.Load((ix + 4u) * 4 + 0);
    uint raw5 = _96.Load((ix + 5u) * 4 + 0);
    Transform s;
    s.mat = float4(asfloat(raw0), asfloat(raw1), asfloat(raw2), asfloat(raw3));
    s.translate = float2(asfloat(raw4), asfloat(raw5));
    return s;
}

TransformRef Transform_index(TransformRef ref, uint index)
{
    TransformRef _85 = { ref.offset + (index * 24u) };
    return _85;
}

Transform combine_monoid(Transform a, Transform b)
{
    Transform c;
    c.mat = (a.mat.xyxy * b.mat.xxzz) + (a.mat.zwzw * b.mat.yyww);
    c.translate = ((a.mat.xy * b.translate.x) + (a.mat.zw * b.translate.y)) + a.translate;
    return c;
}

Transform monoid_identity()
{
    return _224;
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
    _71.Store(offset * 4 + 8, val);
}

void TransformSeg_write(Alloc a, TransformSegRef ref, TransformSeg s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = asuint(s.mat.x);
    write_mem(param, param_1, param_2);
    Alloc param_3 = a;
    uint param_4 = ix + 1u;
    uint param_5 = asuint(s.mat.y);
    write_mem(param_3, param_4, param_5);
    Alloc param_6 = a;
    uint param_7 = ix + 2u;
    uint param_8 = asuint(s.mat.z);
    write_mem(param_6, param_7, param_8);
    Alloc param_9 = a;
    uint param_10 = ix + 3u;
    uint param_11 = asuint(s.mat.w);
    write_mem(param_9, param_10, param_11);
    Alloc param_12 = a;
    uint param_13 = ix + 4u;
    uint param_14 = asuint(s.translate.x);
    write_mem(param_12, param_13, param_14);
    Alloc param_15 = a;
    uint param_16 = ix + 5u;
    uint param_17 = asuint(s.translate.y);
    write_mem(param_15, param_16, param_17);
}

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x * 8u;
    TransformRef _285 = { _278.Load(44) + (ix * 24u) };
    TransformRef ref = _285;
    TransformRef param = ref;
    Transform agg = Transform_read(param);
    Transform local[8];
    local[0] = agg;
    for (uint i = 1u; i < 8u; i++)
    {
        TransformRef param_1 = ref;
        uint param_2 = i;
        TransformRef param_3 = Transform_index(param_1, param_2);
        Transform param_4 = agg;
        Transform param_5 = Transform_read(param_3);
        agg = combine_monoid(param_4, param_5);
        local[i] = agg;
    }
    sh_scratch[gl_LocalInvocationID.x] = agg;
    for (uint i_1 = 0u; i_1 < 9u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gl_LocalInvocationID.x >= (1u << i_1))
        {
            Transform other = sh_scratch[gl_LocalInvocationID.x - (1u << i_1)];
            Transform param_6 = other;
            Transform param_7 = agg;
            agg = combine_monoid(param_6, param_7);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[gl_LocalInvocationID.x] = agg;
    }
    GroupMemoryBarrierWithGroupSync();
    Transform row = monoid_identity();
    if (gl_WorkGroupID.x > 0u)
    {
        Transform _383;
        _383.mat = asfloat(_377.Load4((gl_WorkGroupID.x - 1u) * 32 + 0));
        _383.translate = asfloat(_377.Load2((gl_WorkGroupID.x - 1u) * 32 + 16));
        row.mat = _383.mat;
        row.translate = _383.translate;
    }
    if (gl_LocalInvocationID.x > 0u)
    {
        Transform param_8 = row;
        Transform param_9 = sh_scratch[gl_LocalInvocationID.x - 1u];
        row = combine_monoid(param_8, param_9);
    }
    Alloc param_12;
    for (uint i_2 = 0u; i_2 < 8u; i_2++)
    {
        Transform param_10 = row;
        Transform param_11 = local[i_2];
        Transform m = combine_monoid(param_10, param_11);
        TransformSeg _423 = { m.mat, m.translate };
        TransformSeg transform = _423;
        TransformSegRef _433 = { _278.Load(36) + ((ix + i_2) * 24u) };
        TransformSegRef trans_ref = _433;
        Alloc _437;
        _437.offset = _278.Load(36);
        param_12.offset = _437.offset;
        TransformSegRef param_13 = trans_ref;
        TransformSeg param_14 = transform;
        TransformSeg_write(param_12, param_13, param_14);
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
