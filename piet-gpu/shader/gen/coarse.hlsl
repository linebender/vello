struct Alloc
{
    uint offset;
};

struct MallocResult
{
    Alloc alloc;
    bool failed;
};

struct BinInstanceRef
{
    uint offset;
};

struct BinInstance
{
    uint element_ix;
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

struct TileSegRef
{
    uint offset;
};

struct Tile
{
    TileSegRef tile;
    int backdrop;
};

struct CmdStrokeRef
{
    uint offset;
};

struct CmdStroke
{
    uint tile_ref;
    float half_width;
};

struct CmdFillRef
{
    uint offset;
};

struct CmdFill
{
    uint tile_ref;
    int backdrop;
};

struct CmdColorRef
{
    uint offset;
};

struct CmdColor
{
    uint rgba_color;
};

struct CmdLinGradRef
{
    uint offset;
};

struct CmdLinGrad
{
    uint index;
    float line_x;
    float line_y;
    float line_c;
};

struct CmdImageRef
{
    uint offset;
};

struct CmdImage
{
    uint index;
    int2 offset;
};

struct CmdEndClipRef
{
    uint offset;
};

struct CmdEndClip
{
    uint blend;
};

struct CmdJumpRef
{
    uint offset;
};

struct CmdJump
{
    uint new_ref;
};

struct CmdRef
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

RWByteAddressBuffer _242 : register(u0, space0);
ByteAddressBuffer _854 : register(t1, space0);
ByteAddressBuffer _1222 : register(t2, space0);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
};

groupshared uint sh_bitmaps[8][256];
groupshared Alloc sh_part_elements[256];
groupshared uint sh_part_count[256];
groupshared uint sh_elements[256];
groupshared uint sh_tile_stride[256];
groupshared uint sh_tile_width[256];
groupshared uint sh_tile_x0[256];
groupshared uint sh_tile_y0[256];
groupshared uint sh_tile_base[256];
groupshared uint sh_tile_count[256];

Alloc slice_mem(Alloc a, uint offset, uint size)
{
    Alloc _319 = { a.offset + offset };
    return _319;
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
    uint v = _242.Load(offset * 4 + 8);
    return v;
}

Alloc new_alloc(uint offset, uint size, bool mem_ok)
{
    Alloc a;
    a.offset = offset;
    return a;
}

BinInstanceRef BinInstance_index(BinInstanceRef ref, uint index)
{
    BinInstanceRef _328 = { ref.offset + (index * 4u) };
    return _328;
}

BinInstance BinInstance_read(Alloc a, BinInstanceRef ref)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint raw0 = read_mem(param, param_1);
    BinInstance s;
    s.element_ix = raw0;
    return s;
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
    TileRef _391 = { raw2 };
    s.tiles = _391;
    return s;
}

void write_tile_alloc(uint el_ix, Alloc a)
{
}

Alloc read_tile_alloc(uint el_ix, bool mem_ok)
{
    uint _741;
    _242.GetDimensions(_741);
    _741 = (_741 - 8) / 4;
    uint param = 0u;
    uint param_1 = uint(int(_741) * 4);
    bool param_2 = mem_ok;
    return new_alloc(param, param_1, param_2);
}

Tile Tile_read(Alloc a, TileRef ref)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint raw0 = read_mem(param, param_1);
    Alloc param_2 = a;
    uint param_3 = ix + 1u;
    uint raw1 = read_mem(param_2, param_3);
    TileSegRef _416 = { raw0 };
    Tile s;
    s.tile = _416;
    s.backdrop = int(raw1);
    return s;
}

MallocResult malloc(uint size)
{
    uint _248;
    _242.InterlockedAdd(0, size, _248);
    uint offset = _248;
    uint _255;
    _242.GetDimensions(_255);
    _255 = (_255 - 8) / 4;
    MallocResult r;
    r.failed = (offset + size) > uint(int(_255) * 4);
    uint param = offset;
    uint param_1 = size;
    bool param_2 = !r.failed;
    r.alloc = new_alloc(param, param_1, param_2);
    if (r.failed)
    {
        uint _277;
        _242.InterlockedMax(4, 1u, _277);
        return r;
    }
    return r;
}

void write_mem(Alloc alloc, uint offset, uint val)
{
    Alloc param = alloc;
    uint param_1 = offset;
    if (!touch_mem(param, param_1))
    {
        return;
    }
    _242.Store(offset * 4 + 8, val);
}

void CmdJump_write(Alloc a, CmdJumpRef ref, CmdJump s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = s.new_ref;
    write_mem(param, param_1, param_2);
}

void Cmd_Jump_write(Alloc a, CmdRef ref, CmdJump s)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = 10u;
    write_mem(param, param_1, param_2);
    CmdJumpRef _734 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdJumpRef param_4 = _734;
    CmdJump param_5 = s;
    CmdJump_write(param_3, param_4, param_5);
}

bool alloc_cmd(inout Alloc cmd_alloc, inout CmdRef cmd_ref, inout uint cmd_limit)
{
    if (cmd_ref.offset < cmd_limit)
    {
        return true;
    }
    uint param = 1024u;
    MallocResult _762 = malloc(param);
    MallocResult new_cmd = _762;
    if (new_cmd.failed)
    {
        return false;
    }
    CmdJump _772 = { new_cmd.alloc.offset };
    CmdJump jump = _772;
    Alloc param_1 = cmd_alloc;
    CmdRef param_2 = cmd_ref;
    CmdJump param_3 = jump;
    Cmd_Jump_write(param_1, param_2, param_3);
    cmd_alloc = new_cmd.alloc;
    CmdRef _784 = { cmd_alloc.offset };
    cmd_ref = _784;
    cmd_limit = (cmd_alloc.offset + 1024u) - 60u;
    return true;
}

void CmdFill_write(Alloc a, CmdFillRef ref, CmdFill s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = s.tile_ref;
    write_mem(param, param_1, param_2);
    Alloc param_3 = a;
    uint param_4 = ix + 1u;
    uint param_5 = uint(s.backdrop);
    write_mem(param_3, param_4, param_5);
}

void Cmd_Fill_write(Alloc a, CmdRef ref, CmdFill s)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = 1u;
    write_mem(param, param_1, param_2);
    CmdFillRef _604 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdFillRef param_4 = _604;
    CmdFill param_5 = s;
    CmdFill_write(param_3, param_4, param_5);
}

void Cmd_Solid_write(Alloc a, CmdRef ref)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = 3u;
    write_mem(param, param_1, param_2);
}

void CmdStroke_write(Alloc a, CmdStrokeRef ref, CmdStroke s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = s.tile_ref;
    write_mem(param, param_1, param_2);
    Alloc param_3 = a;
    uint param_4 = ix + 1u;
    uint param_5 = asuint(s.half_width);
    write_mem(param_3, param_4, param_5);
}

void Cmd_Stroke_write(Alloc a, CmdRef ref, CmdStroke s)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = 2u;
    write_mem(param, param_1, param_2);
    CmdStrokeRef _622 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdStrokeRef param_4 = _622;
    CmdStroke param_5 = s;
    CmdStroke_write(param_3, param_4, param_5);
}

void write_fill(Alloc alloc, inout CmdRef cmd_ref, Tile tile, float linewidth)
{
    if (linewidth < 0.0f)
    {
        if (tile.tile.offset != 0u)
        {
            CmdFill _807 = { tile.tile.offset, tile.backdrop };
            CmdFill cmd_fill = _807;
            Alloc param = alloc;
            CmdRef param_1 = cmd_ref;
            CmdFill param_2 = cmd_fill;
            Cmd_Fill_write(param, param_1, param_2);
            cmd_ref.offset += 12u;
        }
        else
        {
            Alloc param_3 = alloc;
            CmdRef param_4 = cmd_ref;
            Cmd_Solid_write(param_3, param_4);
            cmd_ref.offset += 4u;
        }
    }
    else
    {
        CmdStroke _837 = { tile.tile.offset, 0.5f * linewidth };
        CmdStroke cmd_stroke = _837;
        Alloc param_5 = alloc;
        CmdRef param_6 = cmd_ref;
        CmdStroke param_7 = cmd_stroke;
        Cmd_Stroke_write(param_5, param_6, param_7);
        cmd_ref.offset += 12u;
    }
}

void CmdColor_write(Alloc a, CmdColorRef ref, CmdColor s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = s.rgba_color;
    write_mem(param, param_1, param_2);
}

void Cmd_Color_write(Alloc a, CmdRef ref, CmdColor s)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = 5u;
    write_mem(param, param_1, param_2);
    CmdColorRef _649 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdColorRef param_4 = _649;
    CmdColor param_5 = s;
    CmdColor_write(param_3, param_4, param_5);
}

void CmdLinGrad_write(Alloc a, CmdLinGradRef ref, CmdLinGrad s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = s.index;
    write_mem(param, param_1, param_2);
    Alloc param_3 = a;
    uint param_4 = ix + 1u;
    uint param_5 = asuint(s.line_x);
    write_mem(param_3, param_4, param_5);
    Alloc param_6 = a;
    uint param_7 = ix + 2u;
    uint param_8 = asuint(s.line_y);
    write_mem(param_6, param_7, param_8);
    Alloc param_9 = a;
    uint param_10 = ix + 3u;
    uint param_11 = asuint(s.line_c);
    write_mem(param_9, param_10, param_11);
}

void Cmd_LinGrad_write(Alloc a, CmdRef ref, CmdLinGrad s)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = 6u;
    write_mem(param, param_1, param_2);
    CmdLinGradRef _668 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdLinGradRef param_4 = _668;
    CmdLinGrad param_5 = s;
    CmdLinGrad_write(param_3, param_4, param_5);
}

void CmdImage_write(Alloc a, CmdImageRef ref, CmdImage s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = s.index;
    write_mem(param, param_1, param_2);
    Alloc param_3 = a;
    uint param_4 = ix + 1u;
    uint param_5 = (uint(s.offset.x) & 65535u) | (uint(s.offset.y) << uint(16));
    write_mem(param_3, param_4, param_5);
}

void Cmd_Image_write(Alloc a, CmdRef ref, CmdImage s)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = 7u;
    write_mem(param, param_1, param_2);
    CmdImageRef _687 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdImageRef param_4 = _687;
    CmdImage param_5 = s;
    CmdImage_write(param_3, param_4, param_5);
}

void Cmd_BeginClip_write(Alloc a, CmdRef ref)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = 8u;
    write_mem(param, param_1, param_2);
}

void CmdEndClip_write(Alloc a, CmdEndClipRef ref, CmdEndClip s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = s.blend;
    write_mem(param, param_1, param_2);
}

void Cmd_EndClip_write(Alloc a, CmdRef ref, CmdEndClip s)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = 9u;
    write_mem(param, param_1, param_2);
    CmdEndClipRef _715 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdEndClipRef param_4 = _715;
    CmdEndClip param_5 = s;
    CmdEndClip_write(param_3, param_4, param_5);
}

void Cmd_End_write(Alloc a, CmdRef ref)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = 0u;
    write_mem(param, param_1, param_2);
}

void comp_main()
{
    uint width_in_bins = ((_854.Load(8) + 16u) - 1u) / 16u;
    uint bin_ix = (width_in_bins * gl_WorkGroupID.y) + gl_WorkGroupID.x;
    uint partition_ix = 0u;
    uint n_partitions = ((_854.Load(0) + 256u) - 1u) / 256u;
    uint th_ix = gl_LocalInvocationID.x;
    uint bin_tile_x = 16u * gl_WorkGroupID.x;
    uint bin_tile_y = 16u * gl_WorkGroupID.y;
    uint tile_x = gl_LocalInvocationID.x % 16u;
    uint tile_y = gl_LocalInvocationID.x / 16u;
    uint this_tile_ix = (((bin_tile_y + tile_y) * _854.Load(8)) + bin_tile_x) + tile_x;
    Alloc _919;
    _919.offset = _854.Load(24);
    Alloc param;
    param.offset = _919.offset;
    uint param_1 = this_tile_ix * 1024u;
    uint param_2 = 1024u;
    Alloc cmd_alloc = slice_mem(param, param_1, param_2);
    CmdRef _928 = { cmd_alloc.offset };
    CmdRef cmd_ref = _928;
    uint cmd_limit = (cmd_ref.offset + 1024u) - 60u;
    uint clip_depth = 0u;
    uint clip_zero_depth = 0u;
    uint rd_ix = 0u;
    uint wr_ix = 0u;
    uint part_start_ix = 0u;
    uint ready_ix = 0u;
    uint drawmonoid_start = _854.Load(44) >> uint(2);
    uint drawtag_start = _854.Load(100) >> uint(2);
    uint drawdata_start = _854.Load(104) >> uint(2);
    uint drawinfo_start = _854.Load(68) >> uint(2);
    bool mem_ok = _242.Load(4) == 0u;
    Alloc param_3;
    Alloc param_5;
    uint _1154;
    uint element_ix;
    Alloc param_14;
    uint tile_count;
    uint _1453;
    float linewidth;
    CmdLinGrad cmd_lin;
    while (true)
    {
        for (uint i = 0u; i < 8u; i++)
        {
            sh_bitmaps[i][th_ix] = 0u;
        }
        bool _1206;
        for (;;)
        {
            if ((ready_ix == wr_ix) && (partition_ix < n_partitions))
            {
                part_start_ix = ready_ix;
                uint count = 0u;
                bool _1003 = th_ix < 256u;
                bool _1011;
                if (_1003)
                {
                    _1011 = (partition_ix + th_ix) < n_partitions;
                }
                else
                {
                    _1011 = _1003;
                }
                if (_1011)
                {
                    uint in_ix = (_854.Load(20) >> uint(2)) + ((((partition_ix + th_ix) * 256u) + bin_ix) * 2u);
                    Alloc _1029;
                    _1029.offset = _854.Load(20);
                    param_3.offset = _1029.offset;
                    uint param_4 = in_ix;
                    count = read_mem(param_3, param_4);
                    Alloc _1040;
                    _1040.offset = _854.Load(20);
                    param_5.offset = _1040.offset;
                    uint param_6 = in_ix + 1u;
                    uint offset = read_mem(param_5, param_6);
                    uint param_7 = offset;
                    uint param_8 = count * 4u;
                    bool param_9 = mem_ok;
                    sh_part_elements[th_ix] = new_alloc(param_7, param_8, param_9);
                }
                for (uint i_1 = 0u; i_1 < 8u; i_1++)
                {
                    if (th_ix < 256u)
                    {
                        sh_part_count[th_ix] = count;
                    }
                    GroupMemoryBarrierWithGroupSync();
                    if (th_ix < 256u)
                    {
                        if (th_ix >= (1u << i_1))
                        {
                            count += sh_part_count[th_ix - (1u << i_1)];
                        }
                    }
                    GroupMemoryBarrierWithGroupSync();
                }
                if (th_ix < 256u)
                {
                    sh_part_count[th_ix] = part_start_ix + count;
                }
                GroupMemoryBarrierWithGroupSync();
                ready_ix = sh_part_count[255];
                partition_ix += 256u;
            }
            uint ix = rd_ix + th_ix;
            if (((ix >= wr_ix) && (ix < ready_ix)) && mem_ok)
            {
                uint part_ix = 0u;
                for (uint i_2 = 0u; i_2 < 8u; i_2++)
                {
                    uint probe = part_ix + (128u >> i_2);
                    if (ix >= sh_part_count[probe - 1u])
                    {
                        part_ix = probe;
                    }
                }
                if (part_ix > 0u)
                {
                    _1154 = sh_part_count[part_ix - 1u];
                }
                else
                {
                    _1154 = part_start_ix;
                }
                ix -= _1154;
                Alloc bin_alloc = sh_part_elements[part_ix];
                BinInstanceRef _1173 = { bin_alloc.offset };
                BinInstanceRef inst_ref = _1173;
                BinInstanceRef param_10 = inst_ref;
                uint param_11 = ix;
                Alloc param_12 = bin_alloc;
                BinInstanceRef param_13 = BinInstance_index(param_10, param_11);
                BinInstance inst = BinInstance_read(param_12, param_13);
                sh_elements[th_ix] = inst.element_ix;
            }
            GroupMemoryBarrierWithGroupSync();
            wr_ix = min((rd_ix + 256u), ready_ix);
            bool _1196 = (wr_ix - rd_ix) < 256u;
            if (_1196)
            {
                _1206 = (wr_ix < ready_ix) || (partition_ix < n_partitions);
            }
            else
            {
                _1206 = _1196;
            }
            if (_1206)
            {
                continue;
            }
            else
            {
                break;
            }
        }
        uint tag = 0u;
        if ((th_ix + rd_ix) < wr_ix)
        {
            element_ix = sh_elements[th_ix];
            tag = _1222.Load((drawtag_start + element_ix) * 4 + 0);
        }
        switch (tag)
        {
            case 68u:
            case 72u:
            case 276u:
            case 5u:
            case 37u:
            {
                uint drawmonoid_base = drawmonoid_start + (4u * element_ix);
                uint path_ix = _242.Load(drawmonoid_base * 4 + 8);
                PathRef _1247 = { _854.Load(16) + (path_ix * 12u) };
                Alloc _1250;
                _1250.offset = _854.Load(16);
                param_14.offset = _1250.offset;
                PathRef param_15 = _1247;
                Path path = Path_read(param_14, param_15);
                uint stride = path.bbox.z - path.bbox.x;
                sh_tile_stride[th_ix] = stride;
                int dx = int(path.bbox.x) - int(bin_tile_x);
                int dy = int(path.bbox.y) - int(bin_tile_y);
                int x0 = clamp(dx, 0, 16);
                int y0 = clamp(dy, 0, 16);
                int x1 = clamp(int(path.bbox.z) - int(bin_tile_x), 0, 16);
                int y1 = clamp(int(path.bbox.w) - int(bin_tile_y), 0, 16);
                sh_tile_width[th_ix] = uint(x1 - x0);
                sh_tile_x0[th_ix] = uint(x0);
                sh_tile_y0[th_ix] = uint(y0);
                tile_count = uint(x1 - x0) * uint(y1 - y0);
                uint base = path.tiles.offset - (((uint(dy) * stride) + uint(dx)) * 8u);
                sh_tile_base[th_ix] = base;
                uint param_16 = path.tiles.offset;
                uint param_17 = ((path.bbox.z - path.bbox.x) * (path.bbox.w - path.bbox.y)) * 8u;
                bool param_18 = mem_ok;
                Alloc path_alloc = new_alloc(param_16, param_17, param_18);
                uint param_19 = th_ix;
                Alloc param_20 = path_alloc;
                write_tile_alloc(param_19, param_20);
                break;
            }
            default:
            {
                tile_count = 0u;
                break;
            }
        }
        sh_tile_count[th_ix] = tile_count;
        for (uint i_3 = 0u; i_3 < 8u; i_3++)
        {
            GroupMemoryBarrierWithGroupSync();
            if (th_ix >= (1u << i_3))
            {
                tile_count += sh_tile_count[th_ix - (1u << i_3)];
            }
            GroupMemoryBarrierWithGroupSync();
            sh_tile_count[th_ix] = tile_count;
        }
        GroupMemoryBarrierWithGroupSync();
        uint total_tile_count = sh_tile_count[255];
        for (uint ix_1 = th_ix; ix_1 < total_tile_count; ix_1 += 256u)
        {
            uint el_ix = 0u;
            for (uint i_4 = 0u; i_4 < 8u; i_4++)
            {
                uint probe_1 = el_ix + (128u >> i_4);
                if (ix_1 >= sh_tile_count[probe_1 - 1u])
                {
                    el_ix = probe_1;
                }
            }
            uint tag_1 = _1222.Load((drawtag_start + sh_elements[el_ix]) * 4 + 0);
            if (el_ix > 0u)
            {
                _1453 = sh_tile_count[el_ix - 1u];
            }
            else
            {
                _1453 = 0u;
            }
            uint seq_ix = ix_1 - _1453;
            uint width = sh_tile_width[el_ix];
            uint x = sh_tile_x0[el_ix] + (seq_ix % width);
            uint y = sh_tile_y0[el_ix] + (seq_ix / width);
            bool include_tile = false;
            if (mem_ok)
            {
                uint param_21 = el_ix;
                bool param_22 = mem_ok;
                TileRef _1505 = { sh_tile_base[el_ix] + (((sh_tile_stride[el_ix] * y) + x) * 8u) };
                Alloc param_23 = read_tile_alloc(param_21, param_22);
                TileRef param_24 = _1505;
                Tile tile = Tile_read(param_23, param_24);
                bool is_clip = (tag_1 & 1u) != 0u;
                bool is_blend = false;
                bool _1516 = tile.tile.offset != 0u;
                bool _1525;
                if (!_1516)
                {
                    _1525 = (tile.backdrop == 0) == is_clip;
                }
                else
                {
                    _1525 = _1516;
                }
                bool _1532;
                if (!_1525)
                {
                    _1532 = is_clip && is_blend;
                }
                else
                {
                    _1532 = _1525;
                }
                include_tile = _1532;
            }
            if (include_tile)
            {
                uint el_slice = el_ix / 32u;
                uint el_mask = 1u << (el_ix & 31u);
                uint _1552;
                InterlockedOr(sh_bitmaps[el_slice][(y * 16u) + x], el_mask, _1552);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        uint slice_ix = 0u;
        uint bitmap = sh_bitmaps[0][th_ix];
        while (mem_ok)
        {
            if (bitmap == 0u)
            {
                slice_ix++;
                if (slice_ix == 8u)
                {
                    break;
                }
                bitmap = sh_bitmaps[slice_ix][th_ix];
                if (bitmap == 0u)
                {
                    continue;
                }
            }
            uint element_ref_ix = (slice_ix * 32u) + uint(int(firstbitlow(bitmap)));
            uint element_ix_1 = sh_elements[element_ref_ix];
            bitmap &= (bitmap - 1u);
            uint drawtag = _1222.Load((drawtag_start + element_ix_1) * 4 + 0);
            if (clip_zero_depth == 0u)
            {
                uint param_25 = element_ref_ix;
                bool param_26 = mem_ok;
                TileRef _1629 = { sh_tile_base[element_ref_ix] + (((sh_tile_stride[element_ref_ix] * tile_y) + tile_x) * 8u) };
                Alloc param_27 = read_tile_alloc(param_25, param_26);
                TileRef param_28 = _1629;
                Tile tile_1 = Tile_read(param_27, param_28);
                uint drawmonoid_base_1 = drawmonoid_start + (4u * element_ix_1);
                uint scene_offset = _242.Load((drawmonoid_base_1 + 2u) * 4 + 8);
                uint info_offset = _242.Load((drawmonoid_base_1 + 3u) * 4 + 8);
                uint dd = drawdata_start + (scene_offset >> uint(2));
                uint di = drawinfo_start + (info_offset >> uint(2));
                switch (drawtag)
                {
                    case 68u:
                    {
                        linewidth = asfloat(_242.Load(di * 4 + 8));
                        Alloc param_29 = cmd_alloc;
                        CmdRef param_30 = cmd_ref;
                        uint param_31 = cmd_limit;
                        bool _1676 = alloc_cmd(param_29, param_30, param_31);
                        cmd_alloc = param_29;
                        cmd_ref = param_30;
                        cmd_limit = param_31;
                        if (!_1676)
                        {
                            break;
                        }
                        Alloc param_32 = cmd_alloc;
                        CmdRef param_33 = cmd_ref;
                        Tile param_34 = tile_1;
                        float param_35 = linewidth;
                        write_fill(param_32, param_33, param_34, param_35);
                        cmd_ref = param_33;
                        uint rgba = _1222.Load(dd * 4 + 0);
                        CmdColor _1699 = { rgba };
                        Alloc param_36 = cmd_alloc;
                        CmdRef param_37 = cmd_ref;
                        CmdColor param_38 = _1699;
                        Cmd_Color_write(param_36, param_37, param_38);
                        cmd_ref.offset += 8u;
                        break;
                    }
                    case 276u:
                    {
                        Alloc param_39 = cmd_alloc;
                        CmdRef param_40 = cmd_ref;
                        uint param_41 = cmd_limit;
                        bool _1717 = alloc_cmd(param_39, param_40, param_41);
                        cmd_alloc = param_39;
                        cmd_ref = param_40;
                        cmd_limit = param_41;
                        if (!_1717)
                        {
                            break;
                        }
                        linewidth = asfloat(_242.Load(di * 4 + 8));
                        Alloc param_42 = cmd_alloc;
                        CmdRef param_43 = cmd_ref;
                        Tile param_44 = tile_1;
                        float param_45 = linewidth;
                        write_fill(param_42, param_43, param_44, param_45);
                        cmd_ref = param_43;
                        cmd_lin.index = _1222.Load(dd * 4 + 0);
                        cmd_lin.line_x = asfloat(_242.Load((di + 1u) * 4 + 8));
                        cmd_lin.line_y = asfloat(_242.Load((di + 2u) * 4 + 8));
                        cmd_lin.line_c = asfloat(_242.Load((di + 3u) * 4 + 8));
                        Alloc param_46 = cmd_alloc;
                        CmdRef param_47 = cmd_ref;
                        CmdLinGrad param_48 = cmd_lin;
                        Cmd_LinGrad_write(param_46, param_47, param_48);
                        cmd_ref.offset += 20u;
                        break;
                    }
                    case 72u:
                    {
                        linewidth = asfloat(_242.Load(di * 4 + 8));
                        Alloc param_49 = cmd_alloc;
                        CmdRef param_50 = cmd_ref;
                        uint param_51 = cmd_limit;
                        bool _1785 = alloc_cmd(param_49, param_50, param_51);
                        cmd_alloc = param_49;
                        cmd_ref = param_50;
                        cmd_limit = param_51;
                        if (!_1785)
                        {
                            break;
                        }
                        Alloc param_52 = cmd_alloc;
                        CmdRef param_53 = cmd_ref;
                        Tile param_54 = tile_1;
                        float param_55 = linewidth;
                        write_fill(param_52, param_53, param_54, param_55);
                        cmd_ref = param_53;
                        uint index = _1222.Load(dd * 4 + 0);
                        uint raw1 = _1222.Load((dd + 1u) * 4 + 0);
                        int2 offset_1 = int2(int(raw1 << uint(16)) >> 16, int(raw1) >> 16);
                        CmdImage _1824 = { index, offset_1 };
                        Alloc param_56 = cmd_alloc;
                        CmdRef param_57 = cmd_ref;
                        CmdImage param_58 = _1824;
                        Cmd_Image_write(param_56, param_57, param_58);
                        cmd_ref.offset += 12u;
                        break;
                    }
                    case 5u:
                    {
                        bool _1838 = tile_1.tile.offset == 0u;
                        bool _1844;
                        if (_1838)
                        {
                            _1844 = tile_1.backdrop == 0;
                        }
                        else
                        {
                            _1844 = _1838;
                        }
                        if (_1844)
                        {
                            clip_zero_depth = clip_depth + 1u;
                        }
                        else
                        {
                            Alloc param_59 = cmd_alloc;
                            CmdRef param_60 = cmd_ref;
                            uint param_61 = cmd_limit;
                            bool _1856 = alloc_cmd(param_59, param_60, param_61);
                            cmd_alloc = param_59;
                            cmd_ref = param_60;
                            cmd_limit = param_61;
                            if (!_1856)
                            {
                                break;
                            }
                            Alloc param_62 = cmd_alloc;
                            CmdRef param_63 = cmd_ref;
                            Cmd_BeginClip_write(param_62, param_63);
                            cmd_ref.offset += 4u;
                        }
                        clip_depth++;
                        break;
                    }
                    case 37u:
                    {
                        clip_depth--;
                        Alloc param_64 = cmd_alloc;
                        CmdRef param_65 = cmd_ref;
                        uint param_66 = cmd_limit;
                        bool _1884 = alloc_cmd(param_64, param_65, param_66);
                        cmd_alloc = param_64;
                        cmd_ref = param_65;
                        cmd_limit = param_66;
                        if (!_1884)
                        {
                            break;
                        }
                        Alloc param_67 = cmd_alloc;
                        CmdRef param_68 = cmd_ref;
                        Tile param_69 = tile_1;
                        float param_70 = -1.0f;
                        write_fill(param_67, param_68, param_69, param_70);
                        cmd_ref = param_68;
                        uint blend = _1222.Load(dd * 4 + 0);
                        CmdEndClip _1907 = { blend };
                        Alloc param_71 = cmd_alloc;
                        CmdRef param_72 = cmd_ref;
                        CmdEndClip param_73 = _1907;
                        Cmd_EndClip_write(param_71, param_72, param_73);
                        cmd_ref.offset += 8u;
                        break;
                    }
                }
            }
            else
            {
                switch (drawtag)
                {
                    case 5u:
                    {
                        clip_depth++;
                        break;
                    }
                    case 37u:
                    {
                        if (clip_depth == clip_zero_depth)
                        {
                            clip_zero_depth = 0u;
                        }
                        clip_depth--;
                        break;
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        rd_ix += 256u;
        if ((rd_ix >= ready_ix) && (partition_ix >= n_partitions))
        {
            break;
        }
    }
    bool _1954 = (bin_tile_x + tile_x) < _854.Load(8);
    bool _1963;
    if (_1954)
    {
        _1963 = (bin_tile_y + tile_y) < _854.Load(12);
    }
    else
    {
        _1963 = _1954;
    }
    if (_1963)
    {
        Alloc param_74 = cmd_alloc;
        CmdRef param_75 = cmd_ref;
        Cmd_End_write(param_74, param_75);
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    comp_main();
}
