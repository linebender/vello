struct Alloc
{
    uint offset;
};

struct MallocResult
{
    Alloc alloc;
    bool failed;
};

struct AnnoEndClipRef
{
    uint offset;
};

struct AnnoEndClip
{
    float4 bbox;
};

struct AnnotatedRef
{
    uint offset;
};

struct AnnotatedTag
{
    uint tag;
    uint flags;
};

struct PathRef
{
    uint offset;
};

struct TileRef
{
    uint offset;
};

struct Path
{
    uint4 bbox;
    TileRef tiles;
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
    Alloc clip_alloc;
    Alloc clip_bic_alloc;
    Alloc clip_stack_alloc;
    Alloc clip_bbox_alloc;
    uint n_trans;
    uint n_path;
    uint n_clip;
    uint trans_offset;
    uint linewidth_offset;
    uint pathtag_offset;
    uint pathseg_offset;
};

static const uint3 gl_WorkGroupSize = uint3(256u, 1u, 1u);

RWByteAddressBuffer _92 : register(u0, space0);
ByteAddressBuffer _305 : register(t1, space0);

static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared uint sh_tile_count[256];
groupshared MallocResult sh_tile_alloc;

bool touch_mem(Alloc alloc, uint offset)
{
    return true;
}

uint read_mem(Alloc alloc, uint offset)
{
    Alloc param = alloc;
    uint param_1 = offset;
    if (!touch_mem(param, param_1))
    {
        return 0u;
    }
    uint v = _92.Load(offset * 4 + 8);
    return v;
}

AnnotatedTag Annotated_tag(Alloc a, AnnotatedRef ref)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint tag_and_flags = read_mem(param, param_1);
    AnnotatedTag _236 = { tag_and_flags & 65535u, tag_and_flags >> uint(16) };
    return _236;
}

AnnoEndClip AnnoEndClip_read(Alloc a, AnnoEndClipRef ref)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint raw0 = read_mem(param, param_1);
    Alloc param_2 = a;
    uint param_3 = ix + 1u;
    uint raw1 = read_mem(param_2, param_3);
    Alloc param_4 = a;
    uint param_5 = ix + 2u;
    uint raw2 = read_mem(param_4, param_5);
    Alloc param_6 = a;
    uint param_7 = ix + 3u;
    uint raw3 = read_mem(param_6, param_7);
    AnnoEndClip s;
    s.bbox = float4(asfloat(raw0), asfloat(raw1), asfloat(raw2), asfloat(raw3));
    return s;
}

AnnoEndClip Annotated_EndClip_read(Alloc a, AnnotatedRef ref)
{
    AnnoEndClipRef _243 = { ref.offset + 4u };
    Alloc param = a;
    AnnoEndClipRef param_1 = _243;
    return AnnoEndClip_read(param, param_1);
}

Alloc new_alloc(uint offset, uint size, bool mem_ok)
{
    Alloc a;
    a.offset = offset;
    return a;
}

MallocResult malloc(uint size)
{
    uint _98;
    _92.InterlockedAdd(0, size, _98);
    uint offset = _98;
    uint _105;
    _92.GetDimensions(_105);
    _105 = (_105 - 8) / 4;
    MallocResult r;
    r.failed = (offset + size) > uint(int(_105) * 4);
    uint param = offset;
    uint param_1 = size;
    bool param_2 = !r.failed;
    r.alloc = new_alloc(param, param_1, param_2);
    if (r.failed)
    {
        uint _127;
        _92.InterlockedMax(4, 1u, _127);
        return r;
    }
    return r;
}

Alloc slice_mem(Alloc a, uint offset, uint size)
{
    Alloc _169 = { a.offset + offset };
    return _169;
}

void write_mem(Alloc alloc, uint offset, uint val)
{
    Alloc param = alloc;
    uint param_1 = offset;
    if (!touch_mem(param, param_1))
    {
        return;
    }
    _92.Store(offset * 4 + 8, val);
}

void Path_write(Alloc a, PathRef ref, Path s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = s.bbox.x | (s.bbox.y << uint(16));
    write_mem(param, param_1, param_2);
    Alloc param_3 = a;
    uint param_4 = ix + 1u;
    uint param_5 = s.bbox.z | (s.bbox.w << uint(16));
    write_mem(param_3, param_4, param_5);
    Alloc param_6 = a;
    uint param_7 = ix + 2u;
    uint param_8 = s.tiles.offset;
    write_mem(param_6, param_7, param_8);
}

void comp_main()
{
    uint th_ix = gl_LocalInvocationID.x;
    uint element_ix = gl_GlobalInvocationID.x;
    PathRef _312 = { _305.Load(16) + (element_ix * 12u) };
    PathRef path_ref = _312;
    AnnotatedRef _321 = { _305.Load(32) + (element_ix * 40u) };
    AnnotatedRef ref = _321;
    uint tag = 0u;
    if (element_ix < _305.Load(0))
    {
        Alloc _332;
        _332.offset = _305.Load(32);
        Alloc param;
        param.offset = _332.offset;
        AnnotatedRef param_1 = ref;
        tag = Annotated_tag(param, param_1).tag;
    }
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
    switch (tag)
    {
        case 1u:
        case 2u:
        case 3u:
        case 4u:
        case 5u:
        {
            Alloc _350;
            _350.offset = _305.Load(32);
            Alloc param_2;
            param_2.offset = _350.offset;
            AnnotatedRef param_3 = ref;
            AnnoEndClip clip = Annotated_EndClip_read(param_2, param_3);
            x0 = int(floor(clip.bbox.x * 0.0625f));
            y0 = int(floor(clip.bbox.y * 0.0625f));
            x1 = int(ceil(clip.bbox.z * 0.0625f));
            y1 = int(ceil(clip.bbox.w * 0.0625f));
            break;
        }
    }
    x0 = clamp(x0, 0, int(_305.Load(8)));
    y0 = clamp(y0, 0, int(_305.Load(12)));
    x1 = clamp(x1, 0, int(_305.Load(8)));
    y1 = clamp(y1, 0, int(_305.Load(12)));
    Path path;
    path.bbox = uint4(uint(x0), uint(y0), uint(x1), uint(y1));
    uint tile_count = uint((x1 - x0) * (y1 - y0));
    if (tag == 5u)
    {
        tile_count = 0u;
    }
    sh_tile_count[th_ix] = tile_count;
    uint total_tile_count = tile_count;
    for (uint i = 0u; i < 8u; i++)
    {
        GroupMemoryBarrierWithGroupSync();
        if (th_ix >= (1u << i))
        {
            total_tile_count += sh_tile_count[th_ix - (1u << i)];
        }
        GroupMemoryBarrierWithGroupSync();
        sh_tile_count[th_ix] = total_tile_count;
    }
    if (th_ix == 255u)
    {
        uint param_4 = total_tile_count * 8u;
        MallocResult _476 = malloc(param_4);
        sh_tile_alloc = _476;
    }
    GroupMemoryBarrierWithGroupSync();
    MallocResult alloc_start = sh_tile_alloc;
    bool _487;
    if (!alloc_start.failed)
    {
        _487 = _92.Load(4) != 0u;
    }
    else
    {
        _487 = alloc_start.failed;
    }
    if (_487)
    {
        return;
    }
    if (element_ix < _305.Load(0))
    {
        uint _500;
        if (th_ix > 0u)
        {
            _500 = sh_tile_count[th_ix - 1u];
        }
        else
        {
            _500 = 0u;
        }
        uint tile_subix = _500;
        Alloc param_5 = alloc_start.alloc;
        uint param_6 = 8u * tile_subix;
        uint param_7 = 8u * tile_count;
        Alloc tiles_alloc = slice_mem(param_5, param_6, param_7);
        TileRef _522 = { tiles_alloc.offset };
        path.tiles = _522;
        Alloc _527;
        _527.offset = _305.Load(16);
        Alloc param_8;
        param_8.offset = _527.offset;
        PathRef param_9 = path_ref;
        Path param_10 = path;
        Path_write(param_8, param_9, param_10);
    }
    uint total_count = sh_tile_count[255] * 2u;
    uint start_ix = alloc_start.alloc.offset >> uint(2);
    for (uint i_1 = th_ix; i_1 < total_count; i_1 += 256u)
    {
        Alloc param_11 = alloc_start.alloc;
        uint param_12 = start_ix + i_1;
        uint param_13 = 0u;
        write_mem(param_11, param_12, param_13);
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
