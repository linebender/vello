struct TagMonoid
{
    uint trans_ix;
    uint linewidth_ix;
    uint pathseg_ix;
    uint path_ix;
    uint pathseg_offset;
};

struct Alloc
{
    uint offset;
};

struct Config
{
    uint n_elements;
    uint n_pathseg;
    uint width_in_tiles;
    uint height_in_tiles;
    Alloc tile_alloc;
    Alloc bin_alloc;
    Alloc ptcl_alloc;
    Alloc pathseg_alloc;
    Alloc anno_alloc;
    Alloc trans_alloc;
    Alloc bbox_alloc;
    Alloc drawmonoid_alloc;
    uint n_trans;
    uint n_path;
    uint trans_offset;
    uint linewidth_offset;
    uint pathtag_offset;
    uint pathseg_offset;
};

static const uint3 gl_WorkGroupSize = uint3(128u, 1u, 1u);

ByteAddressBuffer _139 : register(t1, space0);
ByteAddressBuffer _150 : register(t2, space0);
RWByteAddressBuffer _238 : register(u3, space0);
RWByteAddressBuffer _258 : register(u0, space0);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared TagMonoid sh_scratch[128];

TagMonoid reduce_tag(uint tag_word)
{
    uint point_count = tag_word & 50529027u;
    TagMonoid c;
    c.pathseg_ix = uint(int(countbits((point_count * 7u) & 67372036u)));
    c.linewidth_ix = uint(int(countbits(tag_word & 1077952576u)));
    c.path_ix = uint(int(countbits(tag_word & 269488144u)));
    c.trans_ix = uint(int(countbits(tag_word & 538976288u)));
    uint n_points = point_count + ((tag_word >> uint(2)) & 16843009u);
    uint a = n_points + (n_points & (((tag_word >> uint(3)) & 16843009u) * 15u));
    a += (a >> uint(8));
    a += (a >> uint(16));
    c.pathseg_offset = a & 255u;
    return c;
}

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

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x * 4u;
    uint scene_ix = (_139.Load(64) >> uint(2)) + ix;
    uint tag_word = _150.Load(scene_ix * 4 + 0);
    uint param = tag_word;
    TagMonoid agg = reduce_tag(param);
    for (uint i = 1u; i < 4u; i++)
    {
        tag_word = _150.Load((scene_ix + i) * 4 + 0);
        uint param_1 = tag_word;
        TagMonoid param_2 = agg;
        TagMonoid param_3 = reduce_tag(param_1);
        agg = combine_tag_monoid(param_2, param_3);
    }
    sh_scratch[gl_LocalInvocationID.x] = agg;
    for (uint i_1 = 0u; i_1 < 7u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if ((gl_LocalInvocationID.x + (1u << i_1)) < 128u)
        {
            TagMonoid other = sh_scratch[gl_LocalInvocationID.x + (1u << i_1)];
            TagMonoid param_4 = agg;
            TagMonoid param_5 = other;
            agg = combine_tag_monoid(param_4, param_5);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[gl_LocalInvocationID.x] = agg;
    }
    if (gl_LocalInvocationID.x == 0u)
    {
        _238.Store(gl_WorkGroupID.x * 20 + 0, agg.trans_ix);
        _238.Store(gl_WorkGroupID.x * 20 + 4, agg.linewidth_ix);
        _238.Store(gl_WorkGroupID.x * 20 + 8, agg.pathseg_ix);
        _238.Store(gl_WorkGroupID.x * 20 + 12, agg.path_ix);
        _238.Store(gl_WorkGroupID.x * 20 + 16, agg.pathseg_offset);
    }
}

[numthreads(128, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
