struct Bic
{
    uint a;
    uint b;
};

static const uint3 gl_WorkGroupSize = uint3(512u, 1u, 1u);

static const Bic _174 = { 0u, 0u };

ByteAddressBuffer _48 : register(t0);
RWByteAddressBuffer _159 : register(u1);
RWByteAddressBuffer _221 : register(u2);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared Bic sh_bic[512];

Bic bic_combine(Bic x, Bic y)
{
    uint m = min(x.b, y.a);
    Bic _38 = { (x.a + y.a) - m, (x.b + y.b) - m };
    return _38;
}

void comp_main()
{
    uint inp[1];
    inp[0] = _48.Load((gl_GlobalInvocationID.x * 1u) * 4 + 0);
    Bic _67 = { 1u - inp[0], inp[0] };
    Bic bic = _67;
    for (uint i = 1u; i < 1u; i++)
    {
        inp[i] = _48.Load(((gl_GlobalInvocationID.x * 1u) + i) * 4 + 0);
        Bic _94 = { 1u - inp[i], inp[i] };
        Bic other = _94;
        Bic param = bic;
        Bic param_1 = other;
        bic = bic_combine(param, param_1);
    }
    sh_bic[gl_LocalInvocationID.x] = bic;
    for (uint i_1 = 0u; i_1 < 9u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if ((gl_LocalInvocationID.x + (1u << i_1)) < 512u)
        {
            Bic other_1 = sh_bic[gl_LocalInvocationID.x + (1u << i_1)];
            Bic param_2 = bic;
            Bic param_3 = other_1;
            bic = bic_combine(param_2, param_3);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_bic[gl_LocalInvocationID.x] = bic;
    }
    if (gl_LocalInvocationID.x == 0u)
    {
        _159.Store(gl_WorkGroupID.x * 8 + 0, bic.a);
        _159.Store(gl_WorkGroupID.x * 8 + 4, bic.b);
    }
    GroupMemoryBarrierWithGroupSync();
    uint size = sh_bic[0].b;
    bic = _174;
    if ((gl_LocalInvocationID.x + 1u) < 512u)
    {
        bic = sh_bic[gl_LocalInvocationID.x + 1u];
    }
    uint out_ix = ((gl_WorkGroupID.x * 512u) + size) - bic.b;
    for (uint i_2 = 1u; i_2 > 0u; i_2--)
    {
        bool _207 = inp[i_2 - 1u] == 1u;
        bool _213;
        if (_207)
        {
            _213 = bic.a == 0u;
        }
        else
        {
            _213 = _207;
        }
        if (_213)
        {
            out_ix--;
            _221.Store(out_ix * 4 + 0, ((gl_GlobalInvocationID.x * 1u) + i_2) - 1u);
        }
        Bic _240 = { 1u - inp[i_2 - 1u], inp[i_2 - 1u] };
        Bic other_2 = _240;
        Bic param_4 = other_2;
        Bic param_5 = bic;
        bic = bic_combine(param_4, param_5);
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
