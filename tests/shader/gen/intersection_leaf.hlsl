struct Bic
{
    uint a;
    uint b;
};

struct Node
{
    uint node_type;
    uint pad1;
    uint pad2;
    uint pad3;
    float4 bbox;
};

static const uint3 gl_WorkGroupSize = uint3(512u, 1u, 1u);

static const Bic _76 = { 0u, 0u };

ByteAddressBuffer _89 : register(t1);
ByteAddressBuffer _167 : register(t2);
ByteAddressBuffer _285 : register(t0);
RWByteAddressBuffer _520 : register(u3);

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
groupshared float4 sh_bbox[512];
groupshared float4 sh_stack[512];
groupshared uint sh_link[512];

Bic bic_combine(Bic x, Bic y)
{
    uint m = min(x.b, y.a);
    Bic _46 = { (x.a + y.a) - m, (x.b + y.b) - m };
    return _46;
}

float4 bbox_intersect(float4 a, float4 b)
{
    return float4(max(a.xy, b.xy), min(a.zw, b.zw));
}

void comp_main()
{
    uint th = gl_LocalInvocationID.x;
    Bic bic = _76;
    if (th < gl_WorkGroupID.x)
    {
        Bic _93;
        _93.a = _89.Load(th * 8 + 0);
        _93.b = _89.Load(th * 8 + 4);
        bic.a = _93.a;
        bic.b = _93.b;
    }
    sh_bic[th] = bic;
    for (uint i = 0u; i < 9u; i++)
    {
        GroupMemoryBarrierWithGroupSync();
        uint other_ix = th + (1u << i);
        if (other_ix < 512u)
        {
            Bic param = bic;
            Bic param_1 = sh_bic[other_ix];
            bic = bic_combine(param, param_1);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_bic[th] = bic;
    }
    GroupMemoryBarrierWithGroupSync();
    uint size = sh_bic[0].b;
    uint bic_next_b = 0u;
    if ((th + 1u) < 512u)
    {
        bic_next_b = sh_bic[th + 1u].b;
    }
    float4 bbox = float4(-1000000000.0f, -1000000000.0f, 1000000000.0f, 1000000000.0f);
    if (bic.b > bic_next_b)
    {
        bbox = asfloat(_167.Load4(((((th * 512u) + bic.b) - bic_next_b) - 1u) * 16 + 0));
    }
    for (uint i_1 = 0u; i_1 < 9u; i_1++)
    {
        sh_bbox[th] = bbox;
        GroupMemoryBarrierWithGroupSync();
        if (th >= (1u << i_1))
        {
            float4 param_2 = sh_bbox[th - (1u << i_1)];
            float4 param_3 = bbox;
            bbox = bbox_intersect(param_2, param_3);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    sh_bbox[th] = bbox;
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
        bbox = asfloat(_167.Load4(((((ix * 512u) + b) - sp) - 1u) * 16 + 0));
        if (ix > 0u)
        {
            float4 param_4 = sh_bbox[ix - 1u];
            float4 param_5 = bbox;
            bbox = bbox_intersect(param_4, param_5);
        }
        sh_stack[th] = bbox;
    }
    GroupMemoryBarrierWithGroupSync();
    Node _291;
    _291.node_type = _285.Load(gl_GlobalInvocationID.x * 32 + 0);
    _291.pad1 = _285.Load(gl_GlobalInvocationID.x * 32 + 4);
    _291.pad2 = _285.Load(gl_GlobalInvocationID.x * 32 + 8);
    _291.pad3 = _285.Load(gl_GlobalInvocationID.x * 32 + 12);
    _291.bbox = asfloat(_285.Load4(gl_GlobalInvocationID.x * 32 + 16));
    Node inp;
    inp.node_type = _291.node_type;
    inp.pad1 = _291.pad1;
    inp.pad2 = _291.pad2;
    inp.pad3 = _291.pad3;
    inp.bbox = _291.bbox;
    uint node_type = inp.node_type;
    Bic _314 = { uint(node_type == 1u), uint(node_type == 0u) };
    bic = _314;
    sh_bic[th] = bic;
    uint inbase = 0u;
    for (uint i_3 = 0u; i_3 < 8u; i_3++)
    {
        uint outbase = 1024u - (1u << (9u - i_3));
        GroupMemoryBarrierWithGroupSync();
        if (th < (1u << (8u - i_3)))
        {
            Bic param_6 = sh_bic[inbase + (th * 2u)];
            Bic param_7 = sh_bic[(inbase + (th * 2u)) + 1u];
            sh_bic[outbase + th] = bic_combine(param_6, param_7);
        }
        inbase = outbase;
    }
    GroupMemoryBarrierWithGroupSync();
    ix = th;
    bic = _76;
    uint j = 0u;
    while (j < 9u)
    {
        uint base = 1024u - (2u << (9u - j));
        if (((ix >> j) & 1u) != 0u)
        {
            Bic param_8 = sh_bic[(base + (ix >> j)) - 1u];
            Bic param_9 = bic;
            Bic test = bic_combine(param_8, param_9);
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
            Bic param_10 = sh_bic[(base_1 + (ix >> j)) - 1u];
            Bic param_11 = bic;
            Bic test_1 = bic_combine(param_10, param_11);
            if (test_1.b == 0u)
            {
                bic = test_1;
                ix -= (1u << j);
            }
        }
    }
    uint _455;
    if (ix > 0u)
    {
        _455 = ix - 1u;
    }
    else
    {
        _455 = 4294967295u - bic.a;
    }
    uint link = _455;
    bbox = inp.bbox;
    for (uint i_4 = 0u; i_4 < 9u; i_4++)
    {
        sh_link[th] = link;
        sh_bbox[th] = bbox;
        GroupMemoryBarrierWithGroupSync();
        if (int(link) >= 0)
        {
            float4 param_12 = sh_bbox[link];
            float4 param_13 = bbox;
            bbox = bbox_intersect(param_12, param_13);
            link = sh_link[link];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (int(link + size) >= 0)
    {
        float4 param_14 = sh_stack[512u + link];
        float4 param_15 = bbox;
        bbox = bbox_intersect(param_14, param_15);
    }
    _520.Store4(gl_GlobalInvocationID.x * 16 + 0, asuint(bbox));
}

[numthreads(512, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
