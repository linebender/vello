struct DrawMonoid
{
    uint path_ix;
    uint clip_ix;
};

static const uint3 gl_WorkGroupSize = uint3(512u, 1u, 1u);

static const DrawMonoid _18 = { 0u, 0u };

RWByteAddressBuffer _57 : register(u0);

static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared DrawMonoid sh_scratch[512];

DrawMonoid combine_tag_monoid(DrawMonoid a, DrawMonoid b)
{
    DrawMonoid c;
    c.path_ix = a.path_ix + b.path_ix;
    c.clip_ix = a.clip_ix + b.clip_ix;
    return c;
}

DrawMonoid tag_monoid_identity()
{
    return _18;
}

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x * 8u;
    DrawMonoid _61;
    _61.path_ix = _57.Load(ix * 8 + 0);
    _61.clip_ix = _57.Load(ix * 8 + 4);
    DrawMonoid local[8];
    local[0].path_ix = _61.path_ix;
    local[0].clip_ix = _61.clip_ix;
    DrawMonoid param_1;
    for (uint i = 1u; i < 8u; i++)
    {
        DrawMonoid param = local[i - 1u];
        DrawMonoid _88;
        _88.path_ix = _57.Load((ix + i) * 8 + 0);
        _88.clip_ix = _57.Load((ix + i) * 8 + 4);
        param_1.path_ix = _88.path_ix;
        param_1.clip_ix = _88.clip_ix;
        local[i] = combine_tag_monoid(param, param_1);
    }
    DrawMonoid agg = local[7];
    sh_scratch[gl_LocalInvocationID.x] = agg;
    for (uint i_1 = 0u; i_1 < 9u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gl_LocalInvocationID.x >= (1u << i_1))
        {
            DrawMonoid other = sh_scratch[gl_LocalInvocationID.x - (1u << i_1)];
            DrawMonoid param_2 = other;
            DrawMonoid param_3 = agg;
            agg = combine_tag_monoid(param_2, param_3);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[gl_LocalInvocationID.x] = agg;
    }
    GroupMemoryBarrierWithGroupSync();
    DrawMonoid row = tag_monoid_identity();
    if (gl_LocalInvocationID.x > 0u)
    {
        row = sh_scratch[gl_LocalInvocationID.x - 1u];
    }
    for (uint i_2 = 0u; i_2 < 8u; i_2++)
    {
        DrawMonoid param_4 = row;
        DrawMonoid param_5 = local[i_2];
        DrawMonoid m = combine_tag_monoid(param_4, param_5);
        uint _178 = ix + i_2;
        _57.Store(_178 * 8 + 0, m.path_ix);
        _57.Store(_178 * 8 + 4, m.clip_ix);
    }
}

[numthreads(512, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
