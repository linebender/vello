struct Transform
{
    float4 mat;
    float2 translate;
};

static const uint3 gl_WorkGroupSize = uint3(256u, 1u, 1u);

static const Transform _23 = { float4(1.0f, 0.0f, 0.0f, 1.0f), 0.0f.xx };

RWByteAddressBuffer _89 : register(u0, space0);

static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared Transform sh_scratch[256];

Transform combine_monoid(Transform a, Transform b)
{
    Transform c;
    c.mat = (a.mat.xyxy * b.mat.xxzz) + (a.mat.zwzw * b.mat.yyww);
    c.translate = ((a.mat.xy * b.translate.x) + (a.mat.zw * b.translate.y)) + a.translate;
    return c;
}

Transform monoid_identity()
{
    return _23;
}

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x * 8u;
    Transform _93;
    _93.mat = asfloat(_89.Load4(ix * 32 + 0));
    _93.translate = asfloat(_89.Load2(ix * 32 + 16));
    Transform local[8];
    local[0].mat = _93.mat;
    local[0].translate = _93.translate;
    Transform param_1;
    for (uint i = 1u; i < 8u; i++)
    {
        Transform param = local[i - 1u];
        Transform _119;
        _119.mat = asfloat(_89.Load4((ix + i) * 32 + 0));
        _119.translate = asfloat(_89.Load2((ix + i) * 32 + 16));
        param_1.mat = _119.mat;
        param_1.translate = _119.translate;
        local[i] = combine_monoid(param, param_1);
    }
    Transform agg = local[7];
    sh_scratch[gl_LocalInvocationID.x] = agg;
    for (uint i_1 = 0u; i_1 < 8u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gl_LocalInvocationID.x >= (1u << i_1))
        {
            Transform other = sh_scratch[gl_LocalInvocationID.x - (1u << i_1)];
            Transform param_2 = other;
            Transform param_3 = agg;
            agg = combine_monoid(param_2, param_3);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[gl_LocalInvocationID.x] = agg;
    }
    GroupMemoryBarrierWithGroupSync();
    Transform row = monoid_identity();
    if (gl_LocalInvocationID.x > 0u)
    {
        row = sh_scratch[gl_LocalInvocationID.x - 1u];
    }
    for (uint i_2 = 0u; i_2 < 8u; i_2++)
    {
        Transform param_4 = row;
        Transform param_5 = local[i_2];
        Transform m = combine_monoid(param_4, param_5);
        uint _208 = ix + i_2;
        _89.Store4(_208 * 32 + 0, asuint(m.mat));
        _89.Store2(_208 * 32 + 16, asuint(m.translate));
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
