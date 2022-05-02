struct Bic
{
    uint a;
    uint b;
};

static const uint3 gl_WorkGroupSize = uint3(512u, 1u, 1u);

static const Bic _157 = { 0u, 0u };

ByteAddressBuffer _170 : register(t1);
ByteAddressBuffer _298 : register(t2);
ByteAddressBuffer _314 : register(t0);
RWByteAddressBuffer _399 : register(u3);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared Bic sh_bic[1022];
groupshared uint sh_stack[512];

Bic bic_combine(Bic x, Bic y)
{
    uint m = min(x.b, y.a);
    Bic _42 = { (x.a + y.a) - m, (x.b + y.b) - m };
    return _42;
}

uint search_link(inout Bic bic)
{
    uint ix = gl_LocalInvocationID.x;
    uint j = 0u;
    while (j < 9u)
    {
        uint base = 1024u - (2u << (9u - j));
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
            uint base_1 = 1024u - (2u << (9u - j));
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
    return ix;
}

void comp_main()
{
    uint th = gl_LocalInvocationID.x;
    Bic bic = _157;
    if ((th * 1u) < gl_WorkGroupID.x)
    {
        Bic _175;
        _175.a = _170.Load((th * 1u) * 8 + 0);
        _175.b = _170.Load((th * 1u) * 8 + 4);
        bic.a = _175.a;
        bic.b = _175.b;
    }
    Bic other;
    for (uint i = 1u; i < 1u; i++)
    {
        if (((th * 1u) + i) < gl_WorkGroupID.x)
        {
            Bic _203;
            _203.a = _170.Load(((th * 1u) + i) * 8 + 0);
            _203.b = _170.Load(((th * 1u) + i) * 8 + 4);
            other.a = _203.a;
            other.b = _203.b;
            Bic param = bic;
            Bic param_1 = other;
            bic = bic_combine(param, param_1);
        }
    }
    sh_bic[th] = bic;
    for (uint i_1 = 0u; i_1 < 9u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if ((th + (1u << i_1)) < 512u)
        {
            Bic other_1 = sh_bic[th + (1u << i_1)];
            Bic param_2 = bic;
            Bic param_3 = other_1;
            bic = bic_combine(param_2, param_3);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_bic[th] = bic;
    }
    GroupMemoryBarrierWithGroupSync();
    uint sp = 511u - th;
    uint ix = 0u;
    for (uint i_2 = 0u; i_2 < 9u; i_2++)
    {
        uint probe = ix + (256u >> i_2);
        if (sp < sh_bic[probe].b)
        {
            ix = probe;
        }
    }
    uint b = sh_bic[ix].b;
    if (sp < b)
    {
        sh_stack[th] = _298.Load(((((ix * 512u) + b) - sp) - 1u) * 4 + 0);
    }
    GroupMemoryBarrierWithGroupSync();
    uint inp = _314.Load((((gl_GlobalInvocationID.x * 1u) + 1u) - 1u) * 4 + 0);
    Bic _326 = { 1u - inp, inp };
    bic = _326;
    sh_bic[th] = bic;
    uint inbase = 0u;
    for (uint i_3 = 0u; i_3 < 8u; i_3++)
    {
        uint outbase = 1024u - (1u << (9u - i_3));
        GroupMemoryBarrierWithGroupSync();
        if (th < (1u << (8u - i_3)))
        {
            Bic param_4 = sh_bic[inbase + (th * 2u)];
            Bic param_5 = sh_bic[(inbase + (th * 2u)) + 1u];
            sh_bic[outbase + th] = bic_combine(param_4, param_5);
        }
        inbase = outbase;
    }
    GroupMemoryBarrierWithGroupSync();
    bic = _157;
    Bic param_6 = bic;
    uint _377 = search_link(param_6);
    bic = param_6;
    ix = _377;
    uint outp;
    if (ix > 0u)
    {
        outp = ((gl_WorkGroupID.x * 512u) + ix) - 1u;
    }
    else
    {
        outp = sh_stack[511u - bic.a];
    }
    _399.Store(gl_GlobalInvocationID.x * 4 + 0, outp);
}

[numthreads(512, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
