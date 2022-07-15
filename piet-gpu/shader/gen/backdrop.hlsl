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

RWByteAddressBuffer _59 : register(u0, space0);
ByteAddressBuffer _181 : register(t1, space0);

static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
static uint gl_LocalInvocationIndex;
struct SPIRV_Cross_Input
{
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
    uint gl_LocalInvocationIndex : SV_GroupIndex;
};

groupshared uint sh_row_width[256];
groupshared Alloc sh_row_alloc[256];
groupshared uint sh_row_count[256];

bool check_deps(uint dep_stage)
{
    uint _65;
    _59.InterlockedOr(4, 0u, _65);
    return (_65 & dep_stage) == 0u;
}

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
    uint v = _59.Load(offset * 4 + 12);
    return v;
}

Path Path_read(Alloc a, PathRef ref)
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
    Path s;
    s.bbox = uint4(raw0 & 65535u, raw0 >> uint(16), raw1 & 65535u, raw1 >> uint(16));
    TileRef _146 = { raw2 };
    s.tiles = _146;
    return s;
}

Alloc new_alloc(uint offset, uint size, bool mem_ok)
{
    Alloc a;
    a.offset = offset;
    return a;
}

void write_mem(Alloc alloc, uint offset, uint val)
{
    Alloc param = alloc;
    uint param_1 = offset;
    if (!touch_mem(param, param_1))
    {
        return;
    }
    _59.Store(offset * 4 + 12, val);
}

void comp_main()
{
    uint param = 7u;
    bool _154 = check_deps(param);
    if (!_154)
    {
        return;
    }
    uint th_ix = gl_LocalInvocationIndex;
    uint element_ix = gl_GlobalInvocationID.x;
    uint row_count = 0u;
    if (gl_LocalInvocationID.y == 0u)
    {
        if (element_ix < _181.Load(4))
        {
            PathRef _195 = { _181.Load(20) + (element_ix * 12u) };
            PathRef path_ref = _195;
            Alloc _200;
            _200.offset = _181.Load(20);
            Alloc param_1;
            param_1.offset = _200.offset;
            PathRef param_2 = path_ref;
            Path path = Path_read(param_1, param_2);
            sh_row_width[th_ix] = path.bbox.z - path.bbox.x;
            row_count = path.bbox.w - path.bbox.y;
            bool _225 = row_count == 1u;
            bool _231;
            if (_225)
            {
                _231 = path.bbox.y > 0u;
            }
            else
            {
                _231 = _225;
            }
            if (_231)
            {
                row_count = 0u;
            }
            uint param_3 = path.tiles.offset;
            uint param_4 = ((path.bbox.z - path.bbox.x) * (path.bbox.w - path.bbox.y)) * 8u;
            bool param_5 = true;
            Alloc path_alloc = new_alloc(param_3, param_4, param_5);
            sh_row_alloc[th_ix] = path_alloc;
        }
        sh_row_count[th_ix] = row_count;
    }
    for (uint i = 0u; i < 8u; i++)
    {
        GroupMemoryBarrierWithGroupSync();
        bool _276 = gl_LocalInvocationID.y == 0u;
        bool _283;
        if (_276)
        {
            _283 = th_ix >= (1u << i);
        }
        else
        {
            _283 = _276;
        }
        if (_283)
        {
            row_count += sh_row_count[th_ix - (1u << i)];
        }
        GroupMemoryBarrierWithGroupSync();
        if (gl_LocalInvocationID.y == 0u)
        {
            sh_row_count[th_ix] = row_count;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint total_rows = sh_row_count[255];
    uint _360;
    for (uint row = th_ix; row < total_rows; row += 256u)
    {
        uint el_ix = 0u;
        for (uint i_1 = 0u; i_1 < 8u; i_1++)
        {
            uint probe = el_ix + (128u >> i_1);
            if (row >= sh_row_count[probe - 1u])
            {
                el_ix = probe;
            }
        }
        uint width = sh_row_width[el_ix];
        if (width > 0u)
        {
            Alloc tiles_alloc = sh_row_alloc[el_ix];
            if (el_ix > 0u)
            {
                _360 = sh_row_count[el_ix - 1u];
            }
            else
            {
                _360 = 0u;
            }
            uint seq_ix = row - _360;
            uint tile_el_ix = ((tiles_alloc.offset >> uint(2)) + 1u) + ((seq_ix * 2u) * width);
            Alloc param_6 = tiles_alloc;
            uint param_7 = tile_el_ix;
            uint sum = read_mem(param_6, param_7);
            for (uint x = 1u; x < width; x++)
            {
                tile_el_ix += 2u;
                Alloc param_8 = tiles_alloc;
                uint param_9 = tile_el_ix;
                sum += read_mem(param_8, param_9);
                Alloc param_10 = tiles_alloc;
                uint param_11 = tile_el_ix;
                uint param_12 = sum;
                write_mem(param_10, param_11, param_12);
            }
        }
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    gl_LocalInvocationIndex = stage_input.gl_LocalInvocationIndex;
    comp_main();
}
