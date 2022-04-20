struct Bic
{
    uint a;
    uint b;
};

static const uint3 gl_WorkGroupSize = uint3(64u, 1u, 1u);

static const Bic _175 = { 0u, 0u };

ByteAddressBuffer _48 : register(t0);
RWByteAddressBuffer _160 : register(u1);
RWByteAddressBuffer _223 : register(u2);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared Bic sh_bic[64];

Bic bic_combine(Bic x, Bic y)
{
    uint m = min(x.b, y.a);
    Bic _38 = { (x.a + y.a) - m, (x.b + y.b) - m };
    return _38;
}

void comp_main()
{
    uint inp[8];
    inp[0] = _48.Load((gl_GlobalInvocationID.x * 8u) * 4 + 0);
    Bic _68 = { 1u - inp[0], inp[0] };
    Bic bic = _68;
    for (uint i = 1u; i < 8u; i++)
    {
        inp[i] = _48.Load(((gl_GlobalInvocationID.x * 8u) + i) * 4 + 0);
        Bic _95 = { 1u - inp[i], inp[i] };
        Bic other = _95;
        Bic param = bic;
        Bic param_1 = other;
        bic = bic_combine(param, param_1);
    }
    sh_bic[gl_LocalInvocationID.x] = bic;
    for (uint i_1 = 0u; i_1 < 6u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if ((gl_LocalInvocationID.x + (1u << i_1)) < 64u)
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
        _160.Store(gl_WorkGroupID.x * 8 + 0, bic.a);
        _160.Store(gl_WorkGroupID.x * 8 + 4, bic.b);
    }
    GroupMemoryBarrierWithGroupSync();
    uint size = sh_bic[0].b;
    bic = _175;
    if ((gl_LocalInvocationID.x + 1u) < 64u)
    {
        bic = sh_bic[gl_LocalInvocationID.x + 1u];
    }
    uint out_ix = ((gl_WorkGroupID.x * 512u) + size) - bic.b;
    for (uint i_2 = 8u; i_2 > 0u; i_2--)
    {
        bool _209 = inp[i_2 - 1u] == 1u;
        bool _215;
        if (_209)
        {
            _215 = bic.a == 0u;
        }
        else
        {
            _215 = _209;
        }
        if (_215)
        {
            out_ix--;
            _223.Store(out_ix * 4 + 0, ((gl_GlobalInvocationID.x * 8u) + i_2) - 1u);
        }
        Bic _242 = { 1u - inp[i_2 - 1u], inp[i_2 - 1u] };
        Bic other_2 = _242;
        Bic param_4 = other_2;
        Bic param_5 = bic;
        bic = bic_combine(param_4, param_5);
    }
}

[numthreads(64, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
