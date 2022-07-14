struct Alloc
{
    uint offset;
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

RWByteAddressBuffer _53 : register(u0, space0);
ByteAddressBuffer _148 : register(t1, space0);
ByteAddressBuffer _232 : register(t2, space0);

static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared uint sh_tile_count[256];
groupshared uint sh_tile_offset;

bool check_deps(uint dep_stage)
{
    uint _60;
    _53.InterlockedOr(4, 0u, _60);
    return (_60 & dep_stage) == 0u;
}

float4 load_draw_bbox(uint draw_ix)
{
    uint base = (_148.Load(68) >> uint(2)) + (4u * draw_ix);
    float x0 = asfloat(_53.Load(base * 4 + 12));
    float y0 = asfloat(_53.Load((base + 1u) * 4 + 12));
    float x1 = asfloat(_53.Load((base + 2u) * 4 + 12));
    float y1 = asfloat(_53.Load((base + 3u) * 4 + 12));
    float4 bbox = float4(x0, y0, x1, y1);
    return bbox;
}

uint malloc_stage(uint size, uint mem_size, uint stage)
{
    uint _70;
    _53.InterlockedAdd(0, size, _70);
    uint offset = _70;
    if ((offset + size) > mem_size)
    {
        uint _80;
        _53.InterlockedOr(4, stage, _80);
        offset = 0u;
    }
    return offset;
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
    _53.Store(offset * 4 + 12, val);
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
    uint param = 1u;
    bool _192 = check_deps(param);
    if (!_192)
    {
        return;
    }
    uint th_ix = gl_LocalInvocationID.x;
    uint element_ix = gl_GlobalInvocationID.x;
    PathRef _216 = { _148.Load(20) + (element_ix * 12u) };
    PathRef path_ref = _216;
    uint drawtag_base = _148.Load(104) >> uint(2);
    uint drawtag = 0u;
    if (element_ix < _148.Load(4))
    {
        drawtag = _232.Load((drawtag_base + element_ix) * 4 + 0);
    }
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
    if ((drawtag != 0u) && (drawtag != 37u))
    {
        uint param_1 = element_ix;
        float4 bbox = load_draw_bbox(param_1);
        x0 = int(floor(bbox.x * 0.0625f));
        y0 = int(floor(bbox.y * 0.0625f));
        x1 = int(ceil(bbox.z * 0.0625f));
        y1 = int(ceil(bbox.w * 0.0625f));
    }
    x0 = clamp(x0, 0, int(_148.Load(12)));
    y0 = clamp(y0, 0, int(_148.Load(16)));
    x1 = clamp(x1, 0, int(_148.Load(12)));
    y1 = clamp(y1, 0, int(_148.Load(16)));
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
        uint param_2 = total_tile_count * 8u;
        uint param_3 = _148.Load(0);
        uint param_4 = 2u;
        uint _370 = malloc_stage(param_2, param_3, param_4);
        sh_tile_offset = _370;
    }
    GroupMemoryBarrierWithGroupSync();
    uint offset_start = sh_tile_offset;
    if (offset_start == 0u)
    {
        return;
    }
    if (element_ix < _148.Load(4))
    {
        uint _387;
        if (th_ix > 0u)
        {
            _387 = sh_tile_count[th_ix - 1u];
        }
        else
        {
            _387 = 0u;
        }
        uint tile_subix = _387;
        TileRef _400 = { offset_start + (8u * tile_subix) };
        path.tiles = _400;
        Alloc _406;
        _406.offset = _148.Load(20);
        Alloc param_5;
        param_5.offset = _406.offset;
        PathRef param_6 = path_ref;
        Path param_7 = path;
        Path_write(param_5, param_6, param_7);
    }
    uint total_count = sh_tile_count[255] * 2u;
    uint start_ix = offset_start >> uint(2);
    for (uint i_1 = th_ix; i_1 < total_count; i_1 += 256u)
    {
        _53.Store((start_ix + i_1) * 4 + 12, 0u);
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
