struct DrawMonoid
{
    uint path_ix;
    uint clip_ix;
    uint scene_offset;
    uint info_offset;
};

static const uint3 gl_WorkGroupSize = uint3(256u, 1u, 1u);

static const DrawMonoid _18 = { 0u, 0u, 0u, 0u };

RWByteAddressBuffer _71 : register(u0, space0);

static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared DrawMonoid sh_scratch[256];

DrawMonoid combine_draw_monoid(DrawMonoid a, DrawMonoid b)
{
    DrawMonoid c;
    c.path_ix = a.path_ix + b.path_ix;
    c.clip_ix = a.clip_ix + b.clip_ix;
    c.scene_offset = a.scene_offset + b.scene_offset;
    c.info_offset = a.info_offset + b.info_offset;
    return c;
}

DrawMonoid draw_monoid_identity()
{
    return _18;
}

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x * 8u;
    DrawMonoid _75;
    _75.path_ix = _71.Load(ix * 16 + 0);
    _75.clip_ix = _71.Load(ix * 16 + 4);
    _75.scene_offset = _71.Load(ix * 16 + 8);
    _75.info_offset = _71.Load(ix * 16 + 12);
    DrawMonoid local[8];
    local[0].path_ix = _75.path_ix;
    local[0].clip_ix = _75.clip_ix;
    local[0].scene_offset = _75.scene_offset;
    local[0].info_offset = _75.info_offset;
    DrawMonoid param_1;
    for (uint i = 1u; i < 8u; i++)
    {
        DrawMonoid param = local[i - 1u];
        DrawMonoid _106;
        _106.path_ix = _71.Load((ix + i) * 16 + 0);
        _106.clip_ix = _71.Load((ix + i) * 16 + 4);
        _106.scene_offset = _71.Load((ix + i) * 16 + 8);
        _106.info_offset = _71.Load((ix + i) * 16 + 12);
        param_1.path_ix = _106.path_ix;
        param_1.clip_ix = _106.clip_ix;
        param_1.scene_offset = _106.scene_offset;
        param_1.info_offset = _106.info_offset;
        local[i] = combine_draw_monoid(param, param_1);
    }
    DrawMonoid agg = local[7];
    sh_scratch[gl_LocalInvocationID.x] = agg;
    for (uint i_1 = 0u; i_1 < 8u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gl_LocalInvocationID.x >= (1u << i_1))
        {
            DrawMonoid other = sh_scratch[gl_LocalInvocationID.x - (1u << i_1)];
            DrawMonoid param_2 = other;
            DrawMonoid param_3 = agg;
            agg = combine_draw_monoid(param_2, param_3);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[gl_LocalInvocationID.x] = agg;
    }
    GroupMemoryBarrierWithGroupSync();
    DrawMonoid row = draw_monoid_identity();
    if (gl_LocalInvocationID.x > 0u)
    {
        row = sh_scratch[gl_LocalInvocationID.x - 1u];
    }
    for (uint i_2 = 0u; i_2 < 8u; i_2++)
    {
        DrawMonoid param_4 = row;
        DrawMonoid param_5 = local[i_2];
        DrawMonoid m = combine_draw_monoid(param_4, param_5);
        uint _199 = ix + i_2;
        _71.Store(_199 * 16 + 0, m.path_ix);
        _71.Store(_199 * 16 + 4, m.clip_ix);
        _71.Store(_199 * 16 + 8, m.scene_offset);
        _71.Store(_199 * 16 + 12, m.info_offset);
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
