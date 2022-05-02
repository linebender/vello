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

struct BicBbox
{
    Bic bic;
    uint pad2;
    uint pad3;
    float4 bbox;
};

static const uint3 gl_WorkGroupSize = uint3(512u, 1u, 1u);

static const Bic _219 = { 0u, 0u };

ByteAddressBuffer _74 : register(t0);
RWByteAddressBuffer _193 : register(u1);
RWByteAddressBuffer _255 : register(u2);

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
groupshared float4 sh_bbox[512];

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
    Node _84;
    _84.node_type = _74.Load(gl_GlobalInvocationID.x * 32 + 0);
    _84.pad1 = _74.Load(gl_GlobalInvocationID.x * 32 + 4);
    _84.pad2 = _74.Load(gl_GlobalInvocationID.x * 32 + 8);
    _84.pad3 = _74.Load(gl_GlobalInvocationID.x * 32 + 12);
    _84.bbox = asfloat(_74.Load4(gl_GlobalInvocationID.x * 32 + 16));
    Node inp;
    inp.node_type = _84.node_type;
    inp.pad1 = _84.pad1;
    inp.pad2 = _84.pad2;
    inp.pad3 = _84.pad3;
    inp.bbox = _84.bbox;
    uint node_type = inp.node_type;
    float4 bbox = inp.bbox;
    Bic _113 = { uint(node_type == 1u), uint(node_type == 0u) };
    Bic bic = _113;
    sh_bic[gl_LocalInvocationID.x] = bic;
    sh_bbox[gl_LocalInvocationID.x] = bbox;
    for (uint i = 0u; i < 9u; i++)
    {
        GroupMemoryBarrierWithGroupSync();
        uint other_ix = gl_LocalInvocationID.x + (1u << i);
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
        sh_bic[gl_LocalInvocationID.x] = bic;
        sh_bbox[gl_LocalInvocationID.x] = bbox;
    }
    if (gl_LocalInvocationID.x == 0u)
    {
        BicBbox _187 = { bic, 0u, 0u, bbox };
        BicBbox bic_bbox = _187;
        _193.Store(gl_WorkGroupID.x * 32 + 0, bic_bbox.bic.a);
        _193.Store(gl_WorkGroupID.x * 32 + 4, bic_bbox.bic.b);
        _193.Store(gl_WorkGroupID.x * 32 + 8, bic_bbox.pad2);
        _193.Store(gl_WorkGroupID.x * 32 + 12, bic_bbox.pad3);
        _193.Store4(gl_WorkGroupID.x * 32 + 16, asuint(bic_bbox.bbox));
    }
    GroupMemoryBarrierWithGroupSync();
    uint size = sh_bic[0].b;
    bic = _219;
    if ((gl_LocalInvocationID.x + 1u) < 512u)
    {
        bic = sh_bic[gl_LocalInvocationID.x + 1u];
    }
    bool _233 = inp.node_type == 0u;
    bool _239;
    if (_233)
    {
        _239 = bic.a == 0u;
    }
    else
    {
        _239 = _233;
    }
    if (_239)
    {
        uint out_ix = (((gl_WorkGroupID.x * 512u) + size) - bic.b) - 1u;
        _255.Store4(out_ix * 16 + 0, asuint(sh_bbox[gl_LocalInvocationID.x]));
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
