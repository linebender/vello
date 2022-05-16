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

struct CmdRadGradRef
{
    uint offset;
};

struct CmdRadGrad
{
    uint index;
    float4 mat;
    float2 xlat;
    float2 c1;
    float ra;
    float roff;
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

RWByteAddressBuffer _260 : register(u0, space0);
ByteAddressBuffer _1005 : register(t1, space0);
ByteAddressBuffer _1378 : register(t2, space0);

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
    Alloc _337 = { a.offset + offset };
    return _337;
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
    uint v = _260.Load(offset * 4 + 8);
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
    BinInstanceRef _346 = { ref.offset + (index * 4u) };
    return _346;
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
    TileRef _409 = { raw2 };
    s.tiles = _409;
    return s;
}

void write_tile_alloc(uint el_ix, Alloc a)
{
}

Alloc read_tile_alloc(uint el_ix, bool mem_ok)
{
    uint _892;
    _260.GetDimensions(_892);
    _892 = (_892 - 8) / 4;
    uint param = 0u;
    uint param_1 = uint(int(_892) * 4);
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
    TileSegRef _434 = { raw0 };
    Tile s;
    s.tile = _434;
    s.backdrop = int(raw1);
    return s;
}

MallocResult malloc(uint size)
{
    uint _266;
    _260.InterlockedAdd(0, size, _266);
    uint offset = _266;
    uint _273;
    _260.GetDimensions(_273);
    _273 = (_273 - 8) / 4;
    MallocResult r;
    r.failed = (offset + size) > uint(int(_273) * 4);
    uint param = offset;
    uint param_1 = size;
    bool param_2 = !r.failed;
    r.alloc = new_alloc(param, param_1, param_2);
    if (r.failed)
    {
        uint _295;
        _260.InterlockedMax(4, 1u, _295);
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
    _260.Store(offset * 4 + 8, val);
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
    uint param_2 = 11u;
    write_mem(param, param_1, param_2);
    CmdJumpRef _885 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdJumpRef param_4 = _885;
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
    MallocResult _913 = malloc(param);
    MallocResult new_cmd = _913;
    if (new_cmd.failed)
    {
        return false;
    }
    CmdJump _923 = { new_cmd.alloc.offset };
    CmdJump jump = _923;
    Alloc param_1 = cmd_alloc;
    CmdRef param_2 = cmd_ref;
    CmdJump param_3 = jump;
    Cmd_Jump_write(param_1, param_2, param_3);
    cmd_alloc = new_cmd.alloc;
    CmdRef _935 = { cmd_alloc.offset };
    cmd_ref = _935;
    cmd_limit = (cmd_alloc.offset + 1024u) - 144u;
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
    CmdFillRef _742 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdFillRef param_4 = _742;
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
    CmdStrokeRef _760 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdStrokeRef param_4 = _760;
    CmdStroke param_5 = s;
    CmdStroke_write(param_3, param_4, param_5);
}

void write_fill(Alloc alloc, inout CmdRef cmd_ref, Tile tile, float linewidth)
{
    if (linewidth < 0.0f)
    {
        if (tile.tile.offset != 0u)
        {
            CmdFill _958 = { tile.tile.offset, tile.backdrop };
            CmdFill cmd_fill = _958;
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
        CmdStroke _988 = { tile.tile.offset, 0.5f * linewidth };
        CmdStroke cmd_stroke = _988;
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
    CmdColorRef _786 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdColorRef param_4 = _786;
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
    CmdLinGradRef _804 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdLinGradRef param_4 = _804;
    CmdLinGrad param_5 = s;
    CmdLinGrad_write(param_3, param_4, param_5);
}

void CmdRadGrad_write(Alloc a, CmdRadGradRef ref, CmdRadGrad s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = s.index;
    write_mem(param, param_1, param_2);
    Alloc param_3 = a;
    uint param_4 = ix + 1u;
    uint param_5 = asuint(s.mat.x);
    write_mem(param_3, param_4, param_5);
    Alloc param_6 = a;
    uint param_7 = ix + 2u;
    uint param_8 = asuint(s.mat.y);
    write_mem(param_6, param_7, param_8);
    Alloc param_9 = a;
    uint param_10 = ix + 3u;
    uint param_11 = asuint(s.mat.z);
    write_mem(param_9, param_10, param_11);
    Alloc param_12 = a;
    uint param_13 = ix + 4u;
    uint param_14 = asuint(s.mat.w);
    write_mem(param_12, param_13, param_14);
    Alloc param_15 = a;
    uint param_16 = ix + 5u;
    uint param_17 = asuint(s.xlat.x);
    write_mem(param_15, param_16, param_17);
    Alloc param_18 = a;
    uint param_19 = ix + 6u;
    uint param_20 = asuint(s.xlat.y);
    write_mem(param_18, param_19, param_20);
    Alloc param_21 = a;
    uint param_22 = ix + 7u;
    uint param_23 = asuint(s.c1.x);
    write_mem(param_21, param_22, param_23);
    Alloc param_24 = a;
    uint param_25 = ix + 8u;
    uint param_26 = asuint(s.c1.y);
    write_mem(param_24, param_25, param_26);
    Alloc param_27 = a;
    uint param_28 = ix + 9u;
    uint param_29 = asuint(s.ra);
    write_mem(param_27, param_28, param_29);
    Alloc param_30 = a;
    uint param_31 = ix + 10u;
    uint param_32 = asuint(s.roff);
    write_mem(param_30, param_31, param_32);
}

void Cmd_RadGrad_write(Alloc a, CmdRef ref, CmdRadGrad s)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = 7u;
    write_mem(param, param_1, param_2);
    CmdRadGradRef _822 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdRadGradRef param_4 = _822;
    CmdRadGrad param_5 = s;
    CmdRadGrad_write(param_3, param_4, param_5);
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
    uint param_2 = 8u;
    write_mem(param, param_1, param_2);
    CmdImageRef _840 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdImageRef param_4 = _840;
    CmdImage param_5 = s;
    CmdImage_write(param_3, param_4, param_5);
}

void Cmd_BeginClip_write(Alloc a, CmdRef ref)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = 9u;
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
    uint param_2 = 10u;
    write_mem(param, param_1, param_2);
    CmdEndClipRef _866 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdEndClipRef param_4 = _866;
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
    uint width_in_bins = ((_1005.Load(8) + 16u) - 1u) / 16u;
    uint bin_ix = (width_in_bins * gl_WorkGroupID.y) + gl_WorkGroupID.x;
    uint partition_ix = 0u;
    uint n_partitions = ((_1005.Load(0) + 256u) - 1u) / 256u;
    uint th_ix = gl_LocalInvocationID.x;
    uint bin_tile_x = 16u * gl_WorkGroupID.x;
    uint bin_tile_y = 16u * gl_WorkGroupID.y;
    uint tile_x = gl_LocalInvocationID.x % 16u;
    uint tile_y = gl_LocalInvocationID.x / 16u;
    uint this_tile_ix = (((bin_tile_y + tile_y) * _1005.Load(8)) + bin_tile_x) + tile_x;
    Alloc _1070;
    _1070.offset = _1005.Load(24);
    Alloc param;
    param.offset = _1070.offset;
    uint param_1 = this_tile_ix * 1024u;
    uint param_2 = 1024u;
    Alloc cmd_alloc = slice_mem(param, param_1, param_2);
    CmdRef _1079 = { cmd_alloc.offset };
    CmdRef cmd_ref = _1079;
    uint cmd_limit = (cmd_ref.offset + 1024u) - 144u;
    uint clip_depth = 0u;
    uint clip_zero_depth = 0u;
    uint rd_ix = 0u;
    uint wr_ix = 0u;
    uint part_start_ix = 0u;
    uint ready_ix = 0u;
    cmd_ref.offset += 4u;
    uint render_blend_depth = 0u;
    uint max_blend_depth = 0u;
    uint drawmonoid_start = _1005.Load(44) >> uint(2);
    uint drawtag_start = _1005.Load(100) >> uint(2);
    uint drawdata_start = _1005.Load(104) >> uint(2);
    uint drawinfo_start = _1005.Load(68) >> uint(2);
    bool mem_ok = _260.Load(4) == 0u;
    Alloc param_3;
    Alloc param_5;
    uint _1310;
    uint element_ix;
    Alloc param_14;
    uint tile_count;
    uint _1611;
    float linewidth;
    CmdLinGrad cmd_lin;
    CmdRadGrad cmd_rad;
    while (true)
    {
        for (uint i = 0u; i < 8u; i++)
        {
            sh_bitmaps[i][th_ix] = 0u;
        }
        bool _1362;
        for (;;)
        {
            if ((ready_ix == wr_ix) && (partition_ix < n_partitions))
            {
                part_start_ix = ready_ix;
                uint count = 0u;
                bool _1160 = th_ix < 256u;
                bool _1168;
                if (_1160)
                {
                    _1168 = (partition_ix + th_ix) < n_partitions;
                }
                else
                {
                    _1168 = _1160;
                }
                if (_1168)
                {
                    uint in_ix = (_1005.Load(20) >> uint(2)) + ((((partition_ix + th_ix) * 256u) + bin_ix) * 2u);
                    Alloc _1185;
                    _1185.offset = _1005.Load(20);
                    param_3.offset = _1185.offset;
                    uint param_4 = in_ix;
                    count = read_mem(param_3, param_4);
                    Alloc _1196;
                    _1196.offset = _1005.Load(20);
                    param_5.offset = _1196.offset;
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
                    _1310 = sh_part_count[part_ix - 1u];
                }
                else
                {
                    _1310 = part_start_ix;
                }
                ix -= _1310;
                Alloc bin_alloc = sh_part_elements[part_ix];
                BinInstanceRef _1329 = { bin_alloc.offset };
                BinInstanceRef inst_ref = _1329;
                BinInstanceRef param_10 = inst_ref;
                uint param_11 = ix;
                Alloc param_12 = bin_alloc;
                BinInstanceRef param_13 = BinInstance_index(param_10, param_11);
                BinInstance inst = BinInstance_read(param_12, param_13);
                sh_elements[th_ix] = inst.element_ix;
            }
            GroupMemoryBarrierWithGroupSync();
            wr_ix = min((rd_ix + 256u), ready_ix);
            bool _1352 = (wr_ix - rd_ix) < 256u;
            if (_1352)
            {
                _1362 = (wr_ix < ready_ix) || (partition_ix < n_partitions);
            }
            else
            {
                _1362 = _1352;
            }
            if (_1362)
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
            tag = _1378.Load((drawtag_start + element_ix) * 4 + 0);
        }
        switch (tag)
        {
            case 68u:
            case 72u:
            case 276u:
            case 732u:
            case 5u:
            case 37u:
            {
                uint drawmonoid_base = drawmonoid_start + (4u * element_ix);
                uint path_ix = _260.Load(drawmonoid_base * 4 + 8);
                PathRef _1403 = { _1005.Load(16) + (path_ix * 12u) };
                Alloc _1406;
                _1406.offset = _1005.Load(16);
                param_14.offset = _1406.offset;
                PathRef param_15 = _1403;
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
            uint element_ix_1 = sh_elements[el_ix];
            uint tag_1 = _1378.Load((drawtag_start + element_ix_1) * 4 + 0);
            if (el_ix > 0u)
            {
                _1611 = sh_tile_count[el_ix - 1u];
            }
            else
            {
                _1611 = 0u;
            }
            uint seq_ix = ix_1 - _1611;
            uint width = sh_tile_width[el_ix];
            uint x = sh_tile_x0[el_ix] + (seq_ix % width);
            uint y = sh_tile_y0[el_ix] + (seq_ix / width);
            bool include_tile = false;
            if (mem_ok)
            {
                uint param_21 = el_ix;
                bool param_22 = mem_ok;
                TileRef _1663 = { sh_tile_base[el_ix] + (((sh_tile_stride[el_ix] * y) + x) * 8u) };
                Alloc param_23 = read_tile_alloc(param_21, param_22);
                TileRef param_24 = _1663;
                Tile tile = Tile_read(param_23, param_24);
                bool is_clip = (tag_1 & 1u) != 0u;
                bool is_blend = false;
                if (is_clip)
                {
                    uint drawmonoid_base_1 = drawmonoid_start + (4u * element_ix_1);
                    uint scene_offset = _260.Load((drawmonoid_base_1 + 2u) * 4 + 8);
                    uint dd = drawdata_start + (scene_offset >> uint(2));
                    uint blend = _1378.Load(dd * 4 + 0);
                    is_blend = blend != 3u;
                }
                bool _1698 = tile.tile.offset != 0u;
                bool _1707;
                if (!_1698)
                {
                    _1707 = (tile.backdrop == 0) == is_clip;
                }
                else
                {
                    _1707 = _1698;
                }
                include_tile = _1707 || is_blend;
            }
            if (include_tile)
            {
                uint el_slice = el_ix / 32u;
                uint el_mask = 1u << (el_ix & 31u);
                uint _1729;
                InterlockedOr(sh_bitmaps[el_slice][(y * 16u) + x], el_mask, _1729);
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
            uint element_ix_2 = sh_elements[element_ref_ix];
            bitmap &= (bitmap - 1u);
            uint drawtag = _1378.Load((drawtag_start + element_ix_2) * 4 + 0);
            if (clip_zero_depth == 0u)
            {
                uint param_25 = element_ref_ix;
                bool param_26 = mem_ok;
                TileRef _1806 = { sh_tile_base[element_ref_ix] + (((sh_tile_stride[element_ref_ix] * tile_y) + tile_x) * 8u) };
                Alloc param_27 = read_tile_alloc(param_25, param_26);
                TileRef param_28 = _1806;
                Tile tile_1 = Tile_read(param_27, param_28);
                uint drawmonoid_base_2 = drawmonoid_start + (4u * element_ix_2);
                uint scene_offset_1 = _260.Load((drawmonoid_base_2 + 2u) * 4 + 8);
                uint info_offset = _260.Load((drawmonoid_base_2 + 3u) * 4 + 8);
                uint dd_1 = drawdata_start + (scene_offset_1 >> uint(2));
                uint di = drawinfo_start + (info_offset >> uint(2));
                switch (drawtag)
                {
                    case 68u:
                    {
                        linewidth = asfloat(_260.Load(di * 4 + 8));
                        Alloc param_29 = cmd_alloc;
                        CmdRef param_30 = cmd_ref;
                        uint param_31 = cmd_limit;
                        bool _1854 = alloc_cmd(param_29, param_30, param_31);
                        cmd_alloc = param_29;
                        cmd_ref = param_30;
                        cmd_limit = param_31;
                        if (!_1854)
                        {
                            break;
                        }
                        Alloc param_32 = cmd_alloc;
                        CmdRef param_33 = cmd_ref;
                        Tile param_34 = tile_1;
                        float param_35 = linewidth;
                        write_fill(param_32, param_33, param_34, param_35);
                        cmd_ref = param_33;
                        uint rgba = _1378.Load(dd_1 * 4 + 0);
                        CmdColor _1877 = { rgba };
                        Alloc param_36 = cmd_alloc;
                        CmdRef param_37 = cmd_ref;
                        CmdColor param_38 = _1877;
                        Cmd_Color_write(param_36, param_37, param_38);
                        cmd_ref.offset += 8u;
                        break;
                    }
                    case 276u:
                    {
                        Alloc param_39 = cmd_alloc;
                        CmdRef param_40 = cmd_ref;
                        uint param_41 = cmd_limit;
                        bool _1895 = alloc_cmd(param_39, param_40, param_41);
                        cmd_alloc = param_39;
                        cmd_ref = param_40;
                        cmd_limit = param_41;
                        if (!_1895)
                        {
                            break;
                        }
                        linewidth = asfloat(_260.Load(di * 4 + 8));
                        Alloc param_42 = cmd_alloc;
                        CmdRef param_43 = cmd_ref;
                        Tile param_44 = tile_1;
                        float param_45 = linewidth;
                        write_fill(param_42, param_43, param_44, param_45);
                        cmd_ref = param_43;
                        cmd_lin.index = _1378.Load(dd_1 * 4 + 0);
                        cmd_lin.line_x = asfloat(_260.Load((di + 1u) * 4 + 8));
                        cmd_lin.line_y = asfloat(_260.Load((di + 2u) * 4 + 8));
                        cmd_lin.line_c = asfloat(_260.Load((di + 3u) * 4 + 8));
                        Alloc param_46 = cmd_alloc;
                        CmdRef param_47 = cmd_ref;
                        CmdLinGrad param_48 = cmd_lin;
                        Cmd_LinGrad_write(param_46, param_47, param_48);
                        cmd_ref.offset += 20u;
                        break;
                    }
                    case 732u:
                    {
                        Alloc param_49 = cmd_alloc;
                        CmdRef param_50 = cmd_ref;
                        uint param_51 = cmd_limit;
                        bool _1959 = alloc_cmd(param_49, param_50, param_51);
                        cmd_alloc = param_49;
                        cmd_ref = param_50;
                        cmd_limit = param_51;
                        if (!_1959)
                        {
                            break;
                        }
                        linewidth = asfloat(_260.Load(di * 4 + 8));
                        Alloc param_52 = cmd_alloc;
                        CmdRef param_53 = cmd_ref;
                        Tile param_54 = tile_1;
                        float param_55 = linewidth;
                        write_fill(param_52, param_53, param_54, param_55);
                        cmd_ref = param_53;
                        cmd_rad.index = _1378.Load(dd_1 * 4 + 0);
                        cmd_rad.mat = asfloat(uint4(_260.Load((di + 1u) * 4 + 8), _260.Load((di + 2u) * 4 + 8), _260.Load((di + 3u) * 4 + 8), _260.Load((di + 4u) * 4 + 8)));
                        cmd_rad.xlat = asfloat(uint2(_260.Load((di + 5u) * 4 + 8), _260.Load((di + 6u) * 4 + 8)));
                        cmd_rad.c1 = asfloat(uint2(_260.Load((di + 7u) * 4 + 8), _260.Load((di + 8u) * 4 + 8)));
                        cmd_rad.ra = asfloat(_260.Load((di + 9u) * 4 + 8));
                        cmd_rad.roff = asfloat(_260.Load((di + 10u) * 4 + 8));
                        Alloc param_56 = cmd_alloc;
                        CmdRef param_57 = cmd_ref;
                        CmdRadGrad param_58 = cmd_rad;
                        Cmd_RadGrad_write(param_56, param_57, param_58);
                        cmd_ref.offset += 48u;
                        break;
                    }
                    case 72u:
                    {
                        linewidth = asfloat(_260.Load(di * 4 + 8));
                        Alloc param_59 = cmd_alloc;
                        CmdRef param_60 = cmd_ref;
                        uint param_61 = cmd_limit;
                        bool _2065 = alloc_cmd(param_59, param_60, param_61);
                        cmd_alloc = param_59;
                        cmd_ref = param_60;
                        cmd_limit = param_61;
                        if (!_2065)
                        {
                            break;
                        }
                        Alloc param_62 = cmd_alloc;
                        CmdRef param_63 = cmd_ref;
                        Tile param_64 = tile_1;
                        float param_65 = linewidth;
                        write_fill(param_62, param_63, param_64, param_65);
                        cmd_ref = param_63;
                        uint index = _1378.Load(dd_1 * 4 + 0);
                        uint raw1 = _1378.Load((dd_1 + 1u) * 4 + 0);
                        int2 offset_1 = int2(int(raw1 << uint(16)) >> 16, int(raw1) >> 16);
                        CmdImage _2104 = { index, offset_1 };
                        Alloc param_66 = cmd_alloc;
                        CmdRef param_67 = cmd_ref;
                        CmdImage param_68 = _2104;
                        Cmd_Image_write(param_66, param_67, param_68);
                        cmd_ref.offset += 12u;
                        break;
                    }
                    case 5u:
                    {
                        bool _2118 = tile_1.tile.offset == 0u;
                        bool _2124;
                        if (_2118)
                        {
                            _2124 = tile_1.backdrop == 0;
                        }
                        else
                        {
                            _2124 = _2118;
                        }
                        if (_2124)
                        {
                            clip_zero_depth = clip_depth + 1u;
                        }
                        else
                        {
                            Alloc param_69 = cmd_alloc;
                            CmdRef param_70 = cmd_ref;
                            uint param_71 = cmd_limit;
                            bool _2136 = alloc_cmd(param_69, param_70, param_71);
                            cmd_alloc = param_69;
                            cmd_ref = param_70;
                            cmd_limit = param_71;
                            if (!_2136)
                            {
                                break;
                            }
                            Alloc param_72 = cmd_alloc;
                            CmdRef param_73 = cmd_ref;
                            Cmd_BeginClip_write(param_72, param_73);
                            cmd_ref.offset += 4u;
                            render_blend_depth++;
                            max_blend_depth = max(max_blend_depth, render_blend_depth);
                        }
                        clip_depth++;
                        break;
                    }
                    case 37u:
                    {
                        clip_depth--;
                        Alloc param_74 = cmd_alloc;
                        CmdRef param_75 = cmd_ref;
                        uint param_76 = cmd_limit;
                        bool _2169 = alloc_cmd(param_74, param_75, param_76);
                        cmd_alloc = param_74;
                        cmd_ref = param_75;
                        cmd_limit = param_76;
                        if (!_2169)
                        {
                            break;
                        }
                        Alloc param_77 = cmd_alloc;
                        CmdRef param_78 = cmd_ref;
                        Tile param_79 = tile_1;
                        float param_80 = -1.0f;
                        write_fill(param_77, param_78, param_79, param_80);
                        cmd_ref = param_78;
                        uint blend_1 = _1378.Load(dd_1 * 4 + 0);
                        CmdEndClip _2192 = { blend_1 };
                        Alloc param_81 = cmd_alloc;
                        CmdRef param_82 = cmd_ref;
                        CmdEndClip param_83 = _2192;
                        Cmd_EndClip_write(param_81, param_82, param_83);
                        cmd_ref.offset += 8u;
                        render_blend_depth--;
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
    bool _2241 = (bin_tile_x + tile_x) < _1005.Load(8);
    bool _2250;
    if (_2241)
    {
        _2250 = (bin_tile_y + tile_y) < _1005.Load(12);
    }
    else
    {
        _2250 = _2241;
    }
    if (_2250)
    {
        Alloc param_84 = cmd_alloc;
        CmdRef param_85 = cmd_ref;
        Cmd_End_write(param_84, param_85);
        if (max_blend_depth > 4u)
        {
        }
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    comp_main();
}
