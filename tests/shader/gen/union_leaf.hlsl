struct Bic
{
    uint a;
    uint b;
};

struct BicBbox
{
    Bic bic;
    uint pad2;
    uint pad3;
    float4 bbox;
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

ByteAddressBuffer _94 : register(t1);
ByteAddressBuffer _213 : register(t2);
ByteAddressBuffer _249 : register(t0);
RWByteAddressBuffer _492 : register(u3);

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
groupshared float4 sh_bbox[1022];
groupshared float4 sh_stack[512];

Bic bic_combine(Bic x, Bic y)
{
    uint m = min(x.b, y.a);
    Bic _46 = { (x.a + y.a) - m, (x.b + y.b) - m };
    return _46;
}

float4 bbox_union(float4 a, float4 b)
{
    return float4(min(a.xy, b.xy), max(a.zw, b.zw));
}

void comp_main()
{
    uint th = gl_LocalInvocationID.x;
    Bic bic = _76;
    float4 bbox = float4(1000000000.0f, 1000000000.0f, -1000000000.0f, -1000000000.0f);
    if (th < gl_WorkGroupID.x)
    {
        Bic _98;
        _98.a = _94.Load(th * 32 + 0);
        _98.b = _94.Load(th * 32 + 4);
        bic.a = _98.a;
        bic.b = _98.b;
        bbox = asfloat(_94.Load4(th * 32 + 16));
    }
    sh_bic[th] = bic;
    sh_bbox[th] = bbox;
    for (uint i = 0u; i < 9u; i++)
    {
        GroupMemoryBarrierWithGroupSync();
        uint other_ix = th + (1u << i);
        if (other_ix < 512u)
        {
            Bic param = bic;
            Bic param_1 = sh_bic[other_ix];
            bic = bic_combine(param, param_1);
            float4 param_2 = bbox;
            float4 param_3 = sh_bbox[other_ix];
            bbox = bbox_union(param_2, param_3);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_bic[th] = bic;
        sh_bbox[th] = bbox;
    }
    GroupMemoryBarrierWithGroupSync();
    uint size = sh_bic[0].b;
    uint sp = 511u - th;
    uint ix = 0u;
    for (uint i_1 = 0u; i_1 < 9u; i_1++)
    {
        uint probe = ix + (256u >> i_1);
        if (sp < sh_bic[probe].b)
        {
            ix = probe;
        }
    }
    uint b = sh_bic[ix].b;
    if (sp < b)
    {
        float4 bbox_1 = asfloat(_213.Load4(((((ix * 512u) + b) - sp) - 1u) * 16 + 0));
        if ((ix + 1u) < 512u)
        {
            float4 param_4 = bbox_1;
            float4 param_5 = sh_bbox[ix + 1u];
            bbox_1 = bbox_union(param_4, param_5);
        }
        sh_stack[th] = bbox_1;
    }
    GroupMemoryBarrierWithGroupSync();
    Node _255;
    _255.node_type = _249.Load(gl_GlobalInvocationID.x * 32 + 0);
    _255.pad1 = _249.Load(gl_GlobalInvocationID.x * 32 + 4);
    _255.pad2 = _249.Load(gl_GlobalInvocationID.x * 32 + 8);
    _255.pad3 = _249.Load(gl_GlobalInvocationID.x * 32 + 12);
    _255.bbox = asfloat(_249.Load4(gl_GlobalInvocationID.x * 32 + 16));
    Node inp;
    inp.node_type = _255.node_type;
    inp.pad1 = _255.pad1;
    inp.pad2 = _255.pad2;
    inp.pad3 = _255.pad3;
    inp.bbox = _255.bbox;
    uint node_type = inp.node_type;
    Bic _277 = { uint(node_type == 1u), uint(node_type == 0u) };
    bic = _277;
    sh_bic[th] = bic;
    sh_bbox[th] = inp.bbox;
    uint inbase = 0u;
    for (uint i_2 = 0u; i_2 < 8u; i_2++)
    {
        uint outbase = 1024u - (1u << (9u - i_2));
        GroupMemoryBarrierWithGroupSync();
        if (th < (1u << (8u - i_2)))
        {
            Bic param_6 = sh_bic[inbase + (th * 2u)];
            Bic param_7 = sh_bic[(inbase + (th * 2u)) + 1u];
            sh_bic[outbase + th] = bic_combine(param_6, param_7);
            float4 param_8 = sh_bbox[inbase + (th * 2u)];
            float4 param_9 = sh_bbox[(inbase + (th * 2u)) + 1u];
            sh_bbox[outbase + th] = bbox_union(param_8, param_9);
        }
        inbase = outbase;
    }
    GroupMemoryBarrierWithGroupSync();
    ix = th;
    bbox = inp.bbox;
    bic = _76;
    if (node_type == 1u)
    {
        uint j = 0u;
        while (j < 9u)
        {
            uint base = 1024u - (2u << (9u - j));
            if (((ix >> j) & 1u) != 0u)
            {
                Bic param_10 = sh_bic[(base + (ix >> j)) - 1u];
                Bic param_11 = bic;
                Bic test = bic_combine(param_10, param_11);
                if (test.b > 0u)
                {
                    break;
                }
                bic = test;
                float4 param_12 = sh_bbox[(base + (ix >> j)) - 1u];
                float4 param_13 = bbox;
                bbox = bbox_union(param_12, param_13);
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
                Bic param_14 = sh_bic[(base_1 + (ix >> j)) - 1u];
                Bic param_15 = bic;
                Bic test_1 = bic_combine(param_14, param_15);
                if (test_1.b == 0u)
                {
                    bic = test_1;
                    float4 param_16 = sh_bbox[(base_1 + (ix >> j)) - 1u];
                    float4 param_17 = bbox;
                    bbox = bbox_union(param_16, param_17);
                    ix -= (1u << j);
                }
            }
        }
        bool _470 = ix == 0u;
        bool _477;
        if (_470)
        {
            _477 = bic.a < size;
        }
        else
        {
            _477 = _470;
        }
        if (_477)
        {
            float4 param_18 = sh_stack[511u - bic.a];
            float4 param_19 = bbox;
            bbox = bbox_union(param_18, param_19);
        }
    }
    _492.Store4(gl_GlobalInvocationID.x * 16 + 0, asuint(bbox));
}

[numthreads(512, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
