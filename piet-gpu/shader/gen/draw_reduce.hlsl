struct DrawMonoid
{
    uint path_ix;
    uint clip_ix;
    uint scene_offset;
    uint info_offset;
};

struct Alloc
{
    uint offset;
};

struct Config
{
    uint mem_size;
    uint n_elements;
    uint n_pathseg;
    uint width_in_tiles;
    uint height_in_tiles;
    Alloc tile_alloc;
    Alloc bin_alloc;
    Alloc ptcl_alloc;
    Alloc pathseg_alloc;
    Alloc anno_alloc;
    Alloc path_bbox_alloc;
    Alloc drawmonoid_alloc;
    Alloc clip_alloc;
    Alloc clip_bic_alloc;
    Alloc clip_stack_alloc;
    Alloc clip_bbox_alloc;
    Alloc draw_bbox_alloc;
    Alloc drawinfo_alloc;
    uint n_trans;
    uint n_path;
    uint n_clip;
    uint trans_offset;
    uint linewidth_offset;
    uint pathtag_offset;
    uint pathseg_offset;
    uint drawtag_offset;
    uint drawdata_offset;
};

static const uint3 gl_WorkGroupSize = uint3(256u, 1u, 1u);

ByteAddressBuffer _87 : register(t1, space0);
ByteAddressBuffer _97 : register(t2, space0);
RWByteAddressBuffer _188 : register(u3, space0);
RWByteAddressBuffer _206 : register(u0, space0);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared DrawMonoid sh_scratch[256];

DrawMonoid map_tag(uint tag_word)
{
    uint has_path = uint(tag_word != 0u);
    DrawMonoid _70 = { has_path, tag_word & 1u, tag_word & 28u, (tag_word >> uint(4)) & 60u };
    return _70;
}

DrawMonoid combine_draw_monoid(DrawMonoid a, DrawMonoid b)
{
    DrawMonoid c;
    c.path_ix = a.path_ix + b.path_ix;
    c.clip_ix = a.clip_ix + b.clip_ix;
    c.scene_offset = a.scene_offset + b.scene_offset;
    c.info_offset = a.info_offset + b.info_offset;
    return c;
}

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x * 8u;
    uint drawtag_base = _87.Load(100) >> uint(2);
    uint tag_word = _97.Load((drawtag_base + ix) * 4 + 0);
    uint param = tag_word;
    DrawMonoid agg = map_tag(param);
    for (uint i = 1u; i < 8u; i++)
    {
        uint tag_word_1 = _97.Load(((drawtag_base + ix) + i) * 4 + 0);
        uint param_1 = tag_word_1;
        DrawMonoid param_2 = agg;
        DrawMonoid param_3 = map_tag(param_1);
        agg = combine_draw_monoid(param_2, param_3);
    }
    sh_scratch[gl_LocalInvocationID.x] = agg;
    for (uint i_1 = 0u; i_1 < 8u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if ((gl_LocalInvocationID.x + (1u << i_1)) < 256u)
        {
            DrawMonoid other = sh_scratch[gl_LocalInvocationID.x + (1u << i_1)];
            DrawMonoid param_4 = agg;
            DrawMonoid param_5 = other;
            agg = combine_draw_monoid(param_4, param_5);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[gl_LocalInvocationID.x] = agg;
    }
    if (gl_LocalInvocationID.x == 0u)
    {
        _188.Store(gl_WorkGroupID.x * 16 + 0, agg.path_ix);
        _188.Store(gl_WorkGroupID.x * 16 + 4, agg.clip_ix);
        _188.Store(gl_WorkGroupID.x * 16 + 8, agg.scene_offset);
        _188.Store(gl_WorkGroupID.x * 16 + 12, agg.info_offset);
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
