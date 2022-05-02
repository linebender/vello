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

static const Bic _181 = { 0u, 0u };

ByteAddressBuffer _82 : register(t0);
RWByteAddressBuffer _165 : register(u1);
RWByteAddressBuffer _273 : register(u2);

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

float4 bbox_intersect(float4 a, float4 b)
{
    return float4(max(a.xy, b.xy), min(a.zw, b.zw));
}

void comp_main()
{
    uint th = gl_LocalInvocationID.x;
    Node _88;
    _88.node_type = _82.Load(gl_GlobalInvocationID.x * 32 + 0);
    _88.pad1 = _82.Load(gl_GlobalInvocationID.x * 32 + 4);
    _88.pad2 = _82.Load(gl_GlobalInvocationID.x * 32 + 8);
    _88.pad3 = _82.Load(gl_GlobalInvocationID.x * 32 + 12);
    _88.bbox = asfloat(_82.Load4(gl_GlobalInvocationID.x * 32 + 16));
    Node inp;
    inp.node_type = _88.node_type;
    inp.pad1 = _88.pad1;
    inp.pad2 = _88.pad2;
    inp.pad3 = _88.pad3;
    inp.bbox = _88.bbox;
    uint node_type = inp.node_type;
    Bic _114 = { uint(node_type == 1u), uint(node_type == 0u) };
    Bic bic = _114;
    sh_bic[gl_LocalInvocationID.x] = bic;
    for (uint i = 0u; i < 9u; i++)
    {
        GroupMemoryBarrierWithGroupSync();
        uint other_ix = gl_LocalInvocationID.x + (1u << i);
        if (other_ix < 512u)
        {
            Bic param = bic;
            Bic param_1 = sh_bic[other_ix];
            bic = bic_combine(param, param_1);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_bic[th] = bic;
    }
    if (th == 0u)
    {
        _165.Store(gl_WorkGroupID.x * 8 + 0, bic.a);
        _165.Store(gl_WorkGroupID.x * 8 + 4, bic.b);
    }
    GroupMemoryBarrierWithGroupSync();
    uint size = sh_bic[0].b;
    bic = _181;
    if ((th + 1u) < 512u)
    {
        bic = sh_bic[th + 1u];
    }
    bool _193 = inp.node_type == 0u;
    bool _199;
    if (_193)
    {
        _199 = bic.a == 0u;
    }
    else
    {
        _199 = _193;
    }
    if (_199)
    {
        uint out_ix = (size - bic.b) - 1u;
        sh_bbox[out_ix] = inp.bbox;
    }
    GroupMemoryBarrierWithGroupSync();
    float4 bbox;
    if (th < size)
    {
        bbox = sh_bbox[th];
    }
    for (uint i_1 = 0u; i_1 < 9u; i_1++)
    {
        bool _235 = th < size;
        bool _242;
        if (_235)
        {
            _242 = th >= (1u << i_1);
        }
        else
        {
            _242 = _235;
        }
        if (_242)
        {
            float4 param_2 = sh_bbox[th - (1u << i_1)];
            float4 param_3 = bbox;
            bbox = bbox_intersect(param_2, param_3);
        }
        GroupMemoryBarrierWithGroupSync();
        if (th < size)
        {
            sh_bbox[th] = bbox;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (th < size)
    {
        _273.Store4(gl_GlobalInvocationID.x * 16 + 0, asuint(bbox));
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
