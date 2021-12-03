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

static const uint3 gl_WorkGroupSize = uint3(512u, 1u, 1u);

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

static const DrawMonoid _88 = { 1u, 0u };
static const DrawMonoid _90 = { 1u, 1u };
static const DrawMonoid _92 = { 0u, 1u };
static const DrawMonoid _94 = { 0u, 0u };

ByteAddressBuffer _46 : register(t2);
RWByteAddressBuffer _203 : register(u3);
RWByteAddressBuffer _217 : register(u0);
ByteAddressBuffer _223 : register(t1);

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
    uint tag_and_flags = _46.Load((ref.offset >> uint(2)) * 4 + 0);
    ElementTag _60 = { tag_and_flags & 65535u, tag_and_flags >> uint(16) };
    return _60;
}

DrawMonoid map_tag(uint tag_word)
{
    switch (tag_word)
    {
        case 4u:
        case 5u:
        case 6u:
        {
            return _88;
        }
        case 9u:
        {
            return _90;
        }
        case 10u:
        {
            return _92;
        }
        default:
        {
            return _94;
        }
    }
}

ElementRef Element_index(ElementRef ref, uint index)
{
    ElementRef _39 = { ref.offset + (index * 36u) };
    return _39;
}

DrawMonoid combine_tag_monoid(DrawMonoid a, DrawMonoid b)
{
    DrawMonoid c;
    c.path_ix = a.path_ix + b.path_ix;
    c.clip_ix = a.clip_ix + b.clip_ix;
    return c;
}

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x * 8u;
    ElementRef _110 = { ix * 36u };
    ElementRef ref = _110;
    ElementRef param = ref;
    uint tag_word = Element_tag(param).tag;
    uint param_1 = tag_word;
    DrawMonoid agg = map_tag(param_1);
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
    }
    sh_scratch[gl_LocalInvocationID.x] = agg;
    for (uint i_1 = 0u; i_1 < 9u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if ((gl_LocalInvocationID.x + (1u << i_1)) < 512u)
        {
            DrawMonoid other = sh_scratch[gl_LocalInvocationID.x + (1u << i_1)];
            DrawMonoid param_8 = agg;
            DrawMonoid param_9 = other;
            agg = combine_tag_monoid(param_8, param_9);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[gl_LocalInvocationID.x] = agg;
    }
    if (gl_LocalInvocationID.x == 0u)
    {
        _203.Store(gl_WorkGroupID.x * 8 + 0, agg.path_ix);
        _203.Store(gl_WorkGroupID.x * 8 + 4, agg.clip_ix);
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
