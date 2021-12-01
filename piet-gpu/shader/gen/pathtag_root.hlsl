struct TagMonoid
{
    uint trans_ix;
    uint linewidth_ix;
    uint pathseg_ix;
    uint path_ix;
    uint pathseg_offset;
};

static const uint3 gl_WorkGroupSize = uint3(512u, 1u, 1u);

static const TagMonoid _18 = { 0u, 0u, 0u, 0u, 0u };

RWByteAddressBuffer _78 : register(u0);

static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared TagMonoid sh_scratch[512];

TagMonoid combine_tag_monoid(TagMonoid a, TagMonoid b)
{
    TagMonoid c;
    c.trans_ix = a.trans_ix + b.trans_ix;
    c.linewidth_ix = a.linewidth_ix + b.linewidth_ix;
    c.pathseg_ix = a.pathseg_ix + b.pathseg_ix;
    c.path_ix = a.path_ix + b.path_ix;
    c.pathseg_offset = a.pathseg_offset + b.pathseg_offset;
    return c;
}

TagMonoid tag_monoid_identity()
{
    return _18;
}

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x * 8u;
    TagMonoid _82;
    _82.trans_ix = _78.Load(ix * 20 + 0);
    _82.linewidth_ix = _78.Load(ix * 20 + 4);
    _82.pathseg_ix = _78.Load(ix * 20 + 8);
    _82.path_ix = _78.Load(ix * 20 + 12);
    _82.pathseg_offset = _78.Load(ix * 20 + 16);
    TagMonoid local[8];
    local[0].trans_ix = _82.trans_ix;
    local[0].linewidth_ix = _82.linewidth_ix;
    local[0].pathseg_ix = _82.pathseg_ix;
    local[0].path_ix = _82.path_ix;
    local[0].pathseg_offset = _82.pathseg_offset;
    TagMonoid param_1;
    for (uint i = 1u; i < 8u; i++)
    {
        TagMonoid param = local[i - 1u];
        TagMonoid _115;
        _115.trans_ix = _78.Load((ix + i) * 20 + 0);
        _115.linewidth_ix = _78.Load((ix + i) * 20 + 4);
        _115.pathseg_ix = _78.Load((ix + i) * 20 + 8);
        _115.path_ix = _78.Load((ix + i) * 20 + 12);
        _115.pathseg_offset = _78.Load((ix + i) * 20 + 16);
        param_1.trans_ix = _115.trans_ix;
        param_1.linewidth_ix = _115.linewidth_ix;
        param_1.pathseg_ix = _115.pathseg_ix;
        param_1.path_ix = _115.path_ix;
        param_1.pathseg_offset = _115.pathseg_offset;
        local[i] = combine_tag_monoid(param, param_1);
    }
    TagMonoid agg = local[7];
    sh_scratch[gl_LocalInvocationID.x] = agg;
    for (uint i_1 = 0u; i_1 < 9u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gl_LocalInvocationID.x >= (1u << i_1))
        {
            TagMonoid other = sh_scratch[gl_LocalInvocationID.x - (1u << i_1)];
            TagMonoid param_2 = other;
            TagMonoid param_3 = agg;
            agg = combine_tag_monoid(param_2, param_3);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[gl_LocalInvocationID.x] = agg;
    }
    GroupMemoryBarrierWithGroupSync();
    TagMonoid row = tag_monoid_identity();
    if (gl_LocalInvocationID.x > 0u)
    {
        row = sh_scratch[gl_LocalInvocationID.x - 1u];
    }
    for (uint i_2 = 0u; i_2 < 8u; i_2++)
    {
        TagMonoid param_4 = row;
        TagMonoid param_5 = local[i_2];
        TagMonoid m = combine_tag_monoid(param_4, param_5);
        uint _211 = ix + i_2;
        _78.Store(_211 * 20 + 0, m.trans_ix);
        _78.Store(_211 * 20 + 4, m.linewidth_ix);
        _78.Store(_211 * 20 + 8, m.pathseg_ix);
        _78.Store(_211 * 20 + 12, m.path_ix);
        _78.Store(_211 * 20 + 16, m.pathseg_offset);
    }
}

[numthreads(512, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
