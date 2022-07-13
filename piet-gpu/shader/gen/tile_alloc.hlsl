struct Alloc
{
    uint offset;
};

struct MallocResult
{
    Alloc alloc;
    bool failed;
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

RWByteAddressBuffer _70 : register(u0, space0);
ByteAddressBuffer _181 : register(t1, space0);
ByteAddressBuffer _257 : register(t2, space0);

static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared uint sh_tile_count[256];
groupshared MallocResult sh_tile_alloc;

float4 load_draw_bbox(uint draw_ix)
{
    uint base = (_181.Load(64) >> uint(2)) + (4u * draw_ix);
    float x0 = asfloat(_70.Load(base * 4 + 8));
    float y0 = asfloat(_70.Load((base + 1u) * 4 + 8));
    float x1 = asfloat(_70.Load((base + 2u) * 4 + 8));
    float y1 = asfloat(_70.Load((base + 3u) * 4 + 8));
    float4 bbox = float4(x0, y0, x1, y1);
    return bbox;
}

Alloc new_alloc(uint offset, uint size, bool mem_ok)
{
    Alloc a;
    a.offset = offset;
    return a;
}

MallocResult malloc(uint size)
{
    uint _76;
    _70.InterlockedAdd(0, size, _76);
    uint offset = _76;
    uint _83;
    _70.GetDimensions(_83);
    _83 = (_83 - 8) / 4;
    MallocResult r;
    r.failed = (offset + size) > uint(int(_83) * 4);
    uint param = offset;
    uint param_1 = size;
    bool param_2 = !r.failed;
    r.alloc = new_alloc(param, param_1, param_2);
    if (r.failed)
    {
        uint _105;
        _70.InterlockedMax(4, 1u, _105);
        return r;
    }
    return r;
}

Alloc slice_mem(Alloc a, uint offset, uint size)
{
    Alloc _131 = { a.offset + offset };
    return _131;
}

bool touch_mem(Alloc alloc, uint offset)
{
    return true;
}

void write_mem(Alloc alloc, uint offset, uint val)
{
    Alloc param = alloc;
    uint param_1 = offset;
    if (!touch_mem(param, param_1))
    {
        return;
    }
    _70.Store(offset * 4 + 8, val);
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
    PathRef _241 = { _181.Load(16) + (element_ix * 12u) };
    PathRef path_ref = _241;
    uint drawtag_base = _181.Load(100) >> uint(2);
    uint drawtag = 0u;
    if (element_ix < _181.Load(0))
    {
        drawtag = _257.Load((drawtag_base + element_ix) * 4 + 0);
    }
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
    if ((drawtag != 0u) && (drawtag != 37u))
    {
        uint param = element_ix;
        float4 bbox = load_draw_bbox(param);
        x0 = int(floor(bbox.x * 0.0625f));
        y0 = int(floor(bbox.y * 0.0625f));
        x1 = int(ceil(bbox.z * 0.0625f));
        y1 = int(ceil(bbox.w * 0.0625f));
    }
    x0 = clamp(x0, 0, int(_181.Load(8)));
    y0 = clamp(y0, 0, int(_181.Load(12)));
    x1 = clamp(x1, 0, int(_181.Load(8)));
    y1 = clamp(y1, 0, int(_181.Load(12)));
    Path path;
    path.bbox = uint4(uint(x0), uint(y0), uint(x1), uint(y1));
    uint tile_count = uint((x1 - x0) * (y1 - y0));
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
        uint param_1 = total_tile_count * 8u;
        MallocResult _392 = malloc(param_1);
        sh_tile_alloc = _392;
    }
    GroupMemoryBarrierWithGroupSync();
    MallocResult alloc_start = sh_tile_alloc;
    bool _403;
    if (!alloc_start.failed)
    {
        _403 = _70.Load(4) != 0u;
    }
    else
    {
        _403 = alloc_start.failed;
    }
    if (_403)
    {
        return;
    }
    if (element_ix < _181.Load(0))
    {
        uint _416;
        if (th_ix > 0u)
        {
            _416 = sh_tile_count[th_ix - 1u];
        }
        else
        {
            _416 = 0u;
        }
        uint tile_subix = _416;
        Alloc param_2 = alloc_start.alloc;
        uint param_3 = 8u * tile_subix;
        uint param_4 = 8u * tile_count;
        Alloc tiles_alloc = slice_mem(param_2, param_3, param_4);
        TileRef _438 = { tiles_alloc.offset };
        path.tiles = _438;
        Alloc _444;
        _444.offset = _181.Load(16);
        Alloc param_5;
        param_5.offset = _444.offset;
        PathRef param_6 = path_ref;
        Path param_7 = path;
        Path_write(param_5, param_6, param_7);
    }
    uint total_count = sh_tile_count[255] * 2u;
    uint start_ix = alloc_start.alloc.offset >> uint(2);
    for (uint i_1 = th_ix; i_1 < total_count; i_1 += 256u)
    {
        Alloc param_8 = alloc_start.alloc;
        uint param_9 = start_ix + i_1;
        uint param_10 = 0u;
        write_mem(param_8, param_9, param_10);
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
