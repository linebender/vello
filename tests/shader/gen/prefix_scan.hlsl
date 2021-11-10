struct Monoid
{
    uint element;
};

static const uint3 gl_WorkGroupSize = uint3(512u, 1u, 1u);

static const Monoid _133 = { 0u };

RWByteAddressBuffer _42 : register(u0);
ByteAddressBuffer _143 : register(t1);

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
    Monoid _46;
    _46.element = _42.Load(ix * 4 + 0);
    Monoid local[8];
    local[0].element = _46.element;
    Monoid param_1;
    for (uint i = 1u; i < 8u; i++)
    {
        Monoid param = local[i - 1u];
        Monoid _71;
        _71.element = _42.Load((ix + i) * 4 + 0);
        param_1.element = _71.element;
        local[i] = combine_monoid(param, param_1);
    }
    Monoid agg = local[7];
    sh_scratch[gl_LocalInvocationID.x] = agg;
    for (uint i_1 = 0u; i_1 < 9u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gl_LocalInvocationID.x >= uint(1 << int(i_1)))
        {
            Monoid other = sh_scratch[gl_LocalInvocationID.x - uint(1 << int(i_1))];
            Monoid param_2 = other;
            Monoid param_3 = agg;
            agg = combine_monoid(param_2, param_3);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[gl_LocalInvocationID.x] = agg;
    }
    GroupMemoryBarrierWithGroupSync();
    Monoid row = _133;
    if (gl_WorkGroupID.x > 0u)
    {
        Monoid _148;
        _148.element = _143.Load((gl_WorkGroupID.x - 1u) * 4 + 0);
        row.element = _148.element;
    }
    if (gl_LocalInvocationID.x > 0u)
    {
        Monoid param_4 = row;
        Monoid param_5 = sh_scratch[gl_LocalInvocationID.x - 1u];
        row = combine_monoid(param_4, param_5);
    }
    for (uint i_2 = 0u; i_2 < 8u; i_2++)
    {
        Monoid param_6 = row;
        Monoid param_7 = local[i_2];
        Monoid m = combine_monoid(param_6, param_7);
        _42.Store((ix + i_2) * 4 + 0, m.element);
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
