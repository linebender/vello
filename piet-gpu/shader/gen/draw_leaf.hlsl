struct ElementRef
{
    uint offset;
};

struct ElementTag
{
    uint tag;
    uint flags;
};

struct DrawMonoid
{
    uint path_ix;
    uint clip_ix;
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
    uint trans_offset;
    uint pathtag_offset;
    uint linewidth_offset;
    uint pathseg_offset;
};

static const uint3 gl_WorkGroupSize = uint3(512u, 1u, 1u);

static const DrawMonoid _67 = { 0u, 0u };
static const DrawMonoid _94 = { 1u, 0u };
static const DrawMonoid _96 = { 1u, 1u };
static const DrawMonoid _98 = { 0u, 1u };

ByteAddressBuffer _49 : register(t2);
ByteAddressBuffer _218 : register(t3);
ByteAddressBuffer _248 : register(t1);
RWByteAddressBuffer _277 : register(u0);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared DrawMonoid sh_scratch[512];

ElementTag Element_tag(ElementRef ref)
{
    uint tag_and_flags = _49.Load((ref.offset >> uint(2)) * 4 + 0);
    ElementTag _63 = { tag_and_flags & 65535u, tag_and_flags >> uint(16) };
    return _63;
}

DrawMonoid map_tag(uint tag_word)
{
    switch (tag_word)
    {
        case 4u:
        case 5u:
        case 6u:
        {
            return _94;
        }
        case 9u:
        {
            return _96;
        }
        case 10u:
        {
            return _98;
        }
        default:
        {
            return _67;
        }
    }
}

ElementRef Element_index(ElementRef ref, uint index)
{
    ElementRef _42 = { ref.offset + (index * 36u) };
    return _42;
}

DrawMonoid combine_tag_monoid(DrawMonoid a, DrawMonoid b)
{
    DrawMonoid c;
    c.path_ix = a.path_ix + b.path_ix;
    c.clip_ix = a.clip_ix + b.clip_ix;
    return c;
}

DrawMonoid tag_monoid_identity()
{
    return _67;
}

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x * 8u;
    ElementRef _115 = { ix * 36u };
    ElementRef ref = _115;
    ElementRef param = ref;
    uint tag_word = Element_tag(param).tag;
    uint param_1 = tag_word;
    DrawMonoid agg = map_tag(param_1);
    DrawMonoid local[8];
    local[0] = agg;
    for (uint i = 1u; i < 8u; i++)
    {
        ElementRef param_2 = ref;
        uint param_3 = i;
        ElementRef param_4 = Element_index(param_2, param_3);
        tag_word = Element_tag(param_4).tag;
        uint param_5 = tag_word;
        DrawMonoid param_6 = agg;
        DrawMonoid param_7 = map_tag(param_5);
        agg = combine_tag_monoid(param_6, param_7);
        local[i] = agg;
    }
    sh_scratch[gl_LocalInvocationID.x] = agg;
    for (uint i_1 = 0u; i_1 < 9u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gl_LocalInvocationID.x >= (1u << i_1))
        {
            DrawMonoid other = sh_scratch[gl_LocalInvocationID.x - (1u << i_1)];
            DrawMonoid param_8 = other;
            DrawMonoid param_9 = agg;
            agg = combine_tag_monoid(param_8, param_9);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[gl_LocalInvocationID.x] = agg;
    }
    GroupMemoryBarrierWithGroupSync();
    DrawMonoid row = tag_monoid_identity();
    if (gl_WorkGroupID.x > 0u)
    {
        DrawMonoid _224;
        _224.path_ix = _218.Load((gl_WorkGroupID.x - 1u) * 8 + 0);
        _224.clip_ix = _218.Load((gl_WorkGroupID.x - 1u) * 8 + 4);
        row.path_ix = _224.path_ix;
        row.clip_ix = _224.clip_ix;
    }
    if (gl_LocalInvocationID.x > 0u)
    {
        DrawMonoid param_10 = row;
        DrawMonoid param_11 = sh_scratch[gl_LocalInvocationID.x - 1u];
        row = combine_tag_monoid(param_10, param_11);
    }
    uint out_base = (_248.Load(44) >> uint(2)) + ((gl_GlobalInvocationID.x * 2u) * 8u);
    for (uint i_2 = 0u; i_2 < 8u; i_2++)
    {
        DrawMonoid param_12 = row;
        DrawMonoid param_13 = local[i_2];
        DrawMonoid m = combine_tag_monoid(param_12, param_13);
        _277.Store((out_base + (i_2 * 2u)) * 4 + 8, m.path_ix);
        _277.Store(((out_base + (i_2 * 2u)) + 1u) * 4 + 8, m.clip_ix);
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
