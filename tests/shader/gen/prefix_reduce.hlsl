struct Monoid
{
    uint element;
};

static const uint3 gl_WorkGroupSize = uint3(512u, 1u, 1u);

ByteAddressBuffer _40 : register(t0);
RWByteAddressBuffer _127 : register(u1);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared Monoid sh_scratch[512];

Monoid combine_monoid(Monoid a, Monoid b)
{
    Monoid _22 = { a.element + b.element };
    return _22;
}

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x * 8u;
    Monoid _44;
    _44.element = _40.Load(ix * 4 + 0);
    Monoid agg;
    agg.element = _44.element;
    Monoid param_1;
    for (uint i = 1u; i < 8u; i++)
    {
        Monoid param = agg;
        Monoid _64;
        _64.element = _40.Load((ix + i) * 4 + 0);
        param_1.element = _64.element;
        agg = combine_monoid(param, param_1);
    }
    sh_scratch[gl_LocalInvocationID.x] = agg;
    for (uint i_1 = 0u; i_1 < 9u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if ((gl_LocalInvocationID.x + (1u << i_1)) < 512u)
        {
            Monoid other = sh_scratch[gl_LocalInvocationID.x + (1u << i_1)];
            Monoid param_2 = agg;
            Monoid param_3 = other;
            agg = combine_monoid(param_2, param_3);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[gl_LocalInvocationID.x] = agg;
    }
    if (gl_LocalInvocationID.x == 0u)
    {
        _127.Store(gl_WorkGroupID.x * 4 + 0, agg.element);
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
