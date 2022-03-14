struct Alloc
{
    uint offset;
};

struct MallocResult
{
    Alloc alloc;
    bool failed;
};

struct AnnoImageRef
{
    uint offset;
};

struct AnnoImage
{
    float4 bbox;
    float linewidth;
    uint index;
    int2 offset;
};

struct AnnoColorRef
{
    uint offset;
};

struct AnnoColor
{
    float4 bbox;
    float linewidth;
    uint rgba_color;
};

struct AnnoLinGradientRef
{
    uint offset;
};

struct AnnoLinGradient
{
    float4 bbox;
    float linewidth;
    uint index;
    float line_x;
    float line_y;
    float line_c;
};

struct AnnoEndClipRef
{
    uint offset;
};

struct AnnoEndClip
{
    float4 bbox;
    uint blend;
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

RWByteAddressBuffer _308 : register(u0, space0);
ByteAddressBuffer _1283 : register(t1, space0);

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
    Alloc _385 = { a.offset + offset };
    return _385;
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
    uint v = _308.Load(offset * 4 + 8);
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
    BinInstanceRef _765 = { ref.offset + (index * 4u) };
    return _765;
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

AnnotatedTag Annotated_tag(Alloc a, AnnotatedRef ref)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint tag_and_flags = read_mem(param, param_1);
    AnnotatedTag _717 = { tag_and_flags & 65535u, tag_and_flags >> uint(16) };
    return _717;
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
    TileRef _825 = { raw2 };
    s.tiles = _825;
    return s;
}

void write_tile_alloc(uint el_ix, Alloc a)
{
}

Alloc read_tile_alloc(uint el_ix, bool mem_ok)
{
    uint _1169;
    _308.GetDimensions(_1169);
    _1169 = (_1169 - 8) / 4;
    uint param = 0u;
    uint param_1 = uint(int(_1169) * 4);
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
    TileSegRef _850 = { raw0 };
    Tile s;
    s.tile = _850;
    s.backdrop = int(raw1);
    return s;
}

AnnoColor AnnoColor_read(Alloc a, AnnoColorRef ref)
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
    Alloc param_8 = a;
    uint param_9 = ix + 4u;
    uint raw4 = read_mem(param_8, param_9);
    Alloc param_10 = a;
    uint param_11 = ix + 5u;
    uint raw5 = read_mem(param_10, param_11);
    AnnoColor s;
    s.bbox = float4(asfloat(raw0), asfloat(raw1), asfloat(raw2), asfloat(raw3));
    s.linewidth = asfloat(raw4);
    s.rgba_color = raw5;
    return s;
}

AnnoColor Annotated_Color_read(Alloc a, AnnotatedRef ref)
{
    AnnoColorRef _723 = { ref.offset + 4u };
    Alloc param = a;
    AnnoColorRef param_1 = _723;
    return AnnoColor_read(param, param_1);
}

MallocResult malloc(uint size)
{
    uint _314;
    _308.InterlockedAdd(0, size, _314);
    uint offset = _314;
    uint _321;
    _308.GetDimensions(_321);
    _321 = (_321 - 8) / 4;
    MallocResult r;
    r.failed = (offset + size) > uint(int(_321) * 4);
    uint param = offset;
    uint param_1 = size;
    bool param_2 = !r.failed;
    r.alloc = new_alloc(param, param_1, param_2);
    if (r.failed)
    {
        uint _343;
        _308.InterlockedMax(4, 1u, _343);
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
    _308.Store(offset * 4 + 8, val);
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
    CmdJumpRef _1162 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdJumpRef param_4 = _1162;
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
    MallocResult _1190 = malloc(param);
    MallocResult new_cmd = _1190;
    if (new_cmd.failed)
    {
        return false;
    }
    CmdJump _1200 = { new_cmd.alloc.offset };
    CmdJump jump = _1200;
    Alloc param_1 = cmd_alloc;
    CmdRef param_2 = cmd_ref;
    CmdJump param_3 = jump;
    Cmd_Jump_write(param_1, param_2, param_3);
    cmd_alloc = new_cmd.alloc;
    CmdRef _1212 = { cmd_alloc.offset };
    cmd_ref = _1212;
    cmd_limit = (cmd_alloc.offset + 1024u) - 60u;
    return true;
}

uint fill_mode_from_flags(uint flags)
{
    return flags & 1u;
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
    CmdFillRef _1036 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdFillRef param_4 = _1036;
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
    CmdStrokeRef _1054 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdStrokeRef param_4 = _1054;
    CmdStroke param_5 = s;
    CmdStroke_write(param_3, param_4, param_5);
}

void write_fill(Alloc alloc, inout CmdRef cmd_ref, uint flags, Tile tile, float linewidth)
{
    uint param = flags;
    if (fill_mode_from_flags(param) == 0u)
    {
        if (tile.tile.offset != 0u)
        {
            CmdFill _1236 = { tile.tile.offset, tile.backdrop };
            CmdFill cmd_fill = _1236;
            Alloc param_1 = alloc;
            CmdRef param_2 = cmd_ref;
            CmdFill param_3 = cmd_fill;
            Cmd_Fill_write(param_1, param_2, param_3);
            cmd_ref.offset += 12u;
        }
        else
        {
            Alloc param_4 = alloc;
            CmdRef param_5 = cmd_ref;
            Cmd_Solid_write(param_4, param_5);
            cmd_ref.offset += 4u;
        }
    }
    else
    {
        CmdStroke _1266 = { tile.tile.offset, 0.5f * linewidth };
        CmdStroke cmd_stroke = _1266;
        Alloc param_6 = alloc;
        CmdRef param_7 = cmd_ref;
        CmdStroke param_8 = cmd_stroke;
        Cmd_Stroke_write(param_6, param_7, param_8);
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
    CmdColorRef _1080 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdColorRef param_4 = _1080;
    CmdColor param_5 = s;
    CmdColor_write(param_3, param_4, param_5);
}

AnnoLinGradient AnnoLinGradient_read(Alloc a, AnnoLinGradientRef ref)
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
    Alloc param_8 = a;
    uint param_9 = ix + 4u;
    uint raw4 = read_mem(param_8, param_9);
    Alloc param_10 = a;
    uint param_11 = ix + 5u;
    uint raw5 = read_mem(param_10, param_11);
    Alloc param_12 = a;
    uint param_13 = ix + 6u;
    uint raw6 = read_mem(param_12, param_13);
    Alloc param_14 = a;
    uint param_15 = ix + 7u;
    uint raw7 = read_mem(param_14, param_15);
    Alloc param_16 = a;
    uint param_17 = ix + 8u;
    uint raw8 = read_mem(param_16, param_17);
    AnnoLinGradient s;
    s.bbox = float4(asfloat(raw0), asfloat(raw1), asfloat(raw2), asfloat(raw3));
    s.linewidth = asfloat(raw4);
    s.index = raw5;
    s.line_x = asfloat(raw6);
    s.line_y = asfloat(raw7);
    s.line_c = asfloat(raw8);
    return s;
}

AnnoLinGradient Annotated_LinGradient_read(Alloc a, AnnotatedRef ref)
{
    AnnoLinGradientRef _733 = { ref.offset + 4u };
    Alloc param = a;
    AnnoLinGradientRef param_1 = _733;
    return AnnoLinGradient_read(param, param_1);
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
    CmdLinGradRef _1098 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdLinGradRef param_4 = _1098;
    CmdLinGrad param_5 = s;
    CmdLinGrad_write(param_3, param_4, param_5);
}

AnnoImage AnnoImage_read(Alloc a, AnnoImageRef ref)
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
    Alloc param_8 = a;
    uint param_9 = ix + 4u;
    uint raw4 = read_mem(param_8, param_9);
    Alloc param_10 = a;
    uint param_11 = ix + 5u;
    uint raw5 = read_mem(param_10, param_11);
    Alloc param_12 = a;
    uint param_13 = ix + 6u;
    uint raw6 = read_mem(param_12, param_13);
    AnnoImage s;
    s.bbox = float4(asfloat(raw0), asfloat(raw1), asfloat(raw2), asfloat(raw3));
    s.linewidth = asfloat(raw4);
    s.index = raw5;
    s.offset = int2(int(raw6 << uint(16)) >> 16, int(raw6) >> 16);
    return s;
}

AnnoImage Annotated_Image_read(Alloc a, AnnotatedRef ref)
{
    AnnoImageRef _743 = { ref.offset + 4u };
    Alloc param = a;
    AnnoImageRef param_1 = _743;
    return AnnoImage_read(param, param_1);
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
    CmdImageRef _1116 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdImageRef param_4 = _1116;
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
    Alloc param_8 = a;
    uint param_9 = ix + 4u;
    uint raw4 = read_mem(param_8, param_9);
    AnnoEndClip s;
    s.bbox = float4(asfloat(raw0), asfloat(raw1), asfloat(raw2), asfloat(raw3));
    s.blend = raw4;
    return s;
}

AnnoEndClip Annotated_EndClip_read(Alloc a, AnnotatedRef ref)
{
    AnnoEndClipRef _753 = { ref.offset + 4u };
    Alloc param = a;
    AnnoEndClipRef param_1 = _753;
    return AnnoEndClip_read(param, param_1);
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
    CmdEndClipRef _1143 = { ref.offset + 4u };
    Alloc param_3 = a;
    CmdEndClipRef param_4 = _1143;
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
    uint width_in_bins = ((_1283.Load(8) + 16u) - 1u) / 16u;
    uint bin_ix = (width_in_bins * gl_WorkGroupID.y) + gl_WorkGroupID.x;
    uint partition_ix = 0u;
    uint n_partitions = ((_1283.Load(0) + 256u) - 1u) / 256u;
    uint th_ix = gl_LocalInvocationID.x;
    uint bin_tile_x = 16u * gl_WorkGroupID.x;
    uint bin_tile_y = 16u * gl_WorkGroupID.y;
    uint tile_x = gl_LocalInvocationID.x % 16u;
    uint tile_y = gl_LocalInvocationID.x / 16u;
    uint this_tile_ix = (((bin_tile_y + tile_y) * _1283.Load(8)) + bin_tile_x) + tile_x;
    Alloc _1348;
    _1348.offset = _1283.Load(24);
    Alloc param;
    param.offset = _1348.offset;
    uint param_1 = this_tile_ix * 1024u;
    uint param_2 = 1024u;
    Alloc cmd_alloc = slice_mem(param, param_1, param_2);
    CmdRef _1357 = { cmd_alloc.offset };
    CmdRef cmd_ref = _1357;
    uint cmd_limit = (cmd_ref.offset + 1024u) - 60u;
    uint clip_depth = 0u;
    uint clip_zero_depth = 0u;
    uint rd_ix = 0u;
    uint wr_ix = 0u;
    uint part_start_ix = 0u;
    uint ready_ix = 0u;
    bool mem_ok = _308.Load(4) == 0u;
    Alloc param_3;
    Alloc param_5;
    uint _1562;
    uint element_ix;
    AnnotatedRef ref;
    Alloc param_14;
    Alloc param_16;
    uint tile_count;
    Alloc param_23;
    uint _1887;
    Alloc param_29;
    Tile tile_1;
    AnnoColor fill;
    Alloc param_35;
    Alloc param_52;
    CmdLinGrad cmd_lin;
    Alloc param_69;
    Alloc param_95;
    while (true)
    {
        for (uint i = 0u; i < 8u; i++)
        {
            sh_bitmaps[i][th_ix] = 0u;
        }
        bool _1614;
        for (;;)
        {
            if ((ready_ix == wr_ix) && (partition_ix < n_partitions))
            {
                part_start_ix = ready_ix;
                uint count = 0u;
                bool _1412 = th_ix < 256u;
                bool _1420;
                if (_1412)
                {
                    _1420 = (partition_ix + th_ix) < n_partitions;
                }
                else
                {
                    _1420 = _1412;
                }
                if (_1420)
                {
                    uint in_ix = (_1283.Load(20) >> uint(2)) + ((((partition_ix + th_ix) * 256u) + bin_ix) * 2u);
                    Alloc _1437;
                    _1437.offset = _1283.Load(20);
                    param_3.offset = _1437.offset;
                    uint param_4 = in_ix;
                    count = read_mem(param_3, param_4);
                    Alloc _1448;
                    _1448.offset = _1283.Load(20);
                    param_5.offset = _1448.offset;
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
                    _1562 = sh_part_count[part_ix - 1u];
                }
                else
                {
                    _1562 = part_start_ix;
                }
                ix -= _1562;
                Alloc bin_alloc = sh_part_elements[part_ix];
                BinInstanceRef _1581 = { bin_alloc.offset };
                BinInstanceRef inst_ref = _1581;
                BinInstanceRef param_10 = inst_ref;
                uint param_11 = ix;
                Alloc param_12 = bin_alloc;
                BinInstanceRef param_13 = BinInstance_index(param_10, param_11);
                BinInstance inst = BinInstance_read(param_12, param_13);
                sh_elements[th_ix] = inst.element_ix;
            }
            GroupMemoryBarrierWithGroupSync();
            wr_ix = min((rd_ix + 256u), ready_ix);
            bool _1604 = (wr_ix - rd_ix) < 256u;
            if (_1604)
            {
                _1614 = (wr_ix < ready_ix) || (partition_ix < n_partitions);
            }
            else
            {
                _1614 = _1604;
            }
            if (_1614)
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
            AnnotatedRef _1635 = { _1283.Load(32) + (element_ix * 40u) };
            ref = _1635;
            Alloc _1638;
            _1638.offset = _1283.Load(32);
            param_14.offset = _1638.offset;
            AnnotatedRef param_15 = ref;
            tag = Annotated_tag(param_14, param_15).tag;
        }
        switch (tag)
        {
            case 1u:
            case 3u:
            case 2u:
            case 4u:
            case 5u:
            {
                uint drawmonoid_base = (_1283.Load(44) >> uint(2)) + (2u * element_ix);
                uint path_ix = _308.Load(drawmonoid_base * 4 + 8);
                PathRef _1667 = { _1283.Load(16) + (path_ix * 12u) };
                Alloc _1670;
                _1670.offset = _1283.Load(16);
                param_16.offset = _1670.offset;
                PathRef param_17 = _1667;
                Path path = Path_read(param_16, param_17);
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
                uint param_18 = path.tiles.offset;
                uint param_19 = ((path.bbox.z - path.bbox.x) * (path.bbox.w - path.bbox.y)) * 8u;
                bool param_20 = mem_ok;
                Alloc path_alloc = new_alloc(param_18, param_19, param_20);
                uint param_21 = th_ix;
                Alloc param_22 = path_alloc;
                write_tile_alloc(param_21, param_22);
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
            AnnotatedRef _1869 = { _1283.Load(32) + (sh_elements[el_ix] * 40u) };
            AnnotatedRef ref_1 = _1869;
            Alloc _1874;
            _1874.offset = _1283.Load(32);
            param_23.offset = _1874.offset;
            AnnotatedRef param_24 = ref_1;
            AnnotatedTag anno_tag = Annotated_tag(param_23, param_24);
            uint tag_1 = anno_tag.tag;
            if (el_ix > 0u)
            {
                _1887 = sh_tile_count[el_ix - 1u];
            }
            else
            {
                _1887 = 0u;
            }
            uint seq_ix = ix_1 - _1887;
            uint width = sh_tile_width[el_ix];
            uint x = sh_tile_x0[el_ix] + (seq_ix % width);
            uint y = sh_tile_y0[el_ix] + (seq_ix / width);
            bool include_tile = false;
            if (mem_ok)
            {
                uint param_25 = el_ix;
                bool param_26 = mem_ok;
                TileRef _1939 = { sh_tile_base[el_ix] + (((sh_tile_stride[el_ix] * y) + x) * 8u) };
                Alloc param_27 = read_tile_alloc(param_25, param_26);
                TileRef param_28 = _1939;
                Tile tile = Tile_read(param_27, param_28);
                bool is_clip = (tag_1 == 4u) || (tag_1 == 5u);
                bool _1951 = tile.tile.offset != 0u;
                bool _1960;
                if (!_1951)
                {
                    _1960 = (tile.backdrop == 0) == is_clip;
                }
                else
                {
                    _1960 = _1951;
                }
                bool _1972;
                if (!_1960)
                {
                    bool _1971;
                    if (is_clip)
                    {
                        _1971 = (anno_tag.flags & 2u) != 0u;
                    }
                    else
                    {
                        _1971 = is_clip;
                    }
                    _1972 = _1971;
                }
                else
                {
                    _1972 = _1960;
                }
                include_tile = _1972;
            }
            if (include_tile)
            {
                uint el_slice = el_ix / 32u;
                uint el_mask = 1u << (el_ix & 31u);
                uint _1992;
                InterlockedOr(sh_bitmaps[el_slice][(y * 16u) + x], el_mask, _1992);
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
            AnnotatedRef _2046 = { _1283.Load(32) + (element_ix_1 * 40u) };
            ref = _2046;
            Alloc _2050;
            _2050.offset = _1283.Load(32);
            param_29.offset = _2050.offset;
            AnnotatedRef param_30 = ref;
            AnnotatedTag tag_2 = Annotated_tag(param_29, param_30);
            if (clip_zero_depth == 0u)
            {
                switch (tag_2.tag)
                {
                    case 1u:
                    {
                        uint param_31 = element_ref_ix;
                        bool param_32 = mem_ok;
                        TileRef _2086 = { sh_tile_base[element_ref_ix] + (((sh_tile_stride[element_ref_ix] * tile_y) + tile_x) * 8u) };
                        Alloc param_33 = read_tile_alloc(param_31, param_32);
                        TileRef param_34 = _2086;
                        tile_1 = Tile_read(param_33, param_34);
                        Alloc _2093;
                        _2093.offset = _1283.Load(32);
                        param_35.offset = _2093.offset;
                        AnnotatedRef param_36 = ref;
                        fill = Annotated_Color_read(param_35, param_36);
                        Alloc param_37 = cmd_alloc;
                        CmdRef param_38 = cmd_ref;
                        uint param_39 = cmd_limit;
                        bool _2105 = alloc_cmd(param_37, param_38, param_39);
                        cmd_alloc = param_37;
                        cmd_ref = param_38;
                        cmd_limit = param_39;
                        if (!_2105)
                        {
                            break;
                        }
                        Alloc param_40 = cmd_alloc;
                        CmdRef param_41 = cmd_ref;
                        uint param_42 = tag_2.flags;
                        Tile param_43 = tile_1;
                        float param_44 = fill.linewidth;
                        write_fill(param_40, param_41, param_42, param_43, param_44);
                        cmd_ref = param_41;
                        CmdColor _2129 = { fill.rgba_color };
                        Alloc param_45 = cmd_alloc;
                        CmdRef param_46 = cmd_ref;
                        CmdColor param_47 = _2129;
                        Cmd_Color_write(param_45, param_46, param_47);
                        cmd_ref.offset += 8u;
                        break;
                    }
                    case 2u:
                    {
                        uint param_48 = element_ref_ix;
                        bool param_49 = mem_ok;
                        TileRef _2158 = { sh_tile_base[element_ref_ix] + (((sh_tile_stride[element_ref_ix] * tile_y) + tile_x) * 8u) };
                        Alloc param_50 = read_tile_alloc(param_48, param_49);
                        TileRef param_51 = _2158;
                        tile_1 = Tile_read(param_50, param_51);
                        Alloc _2165;
                        _2165.offset = _1283.Load(32);
                        param_52.offset = _2165.offset;
                        AnnotatedRef param_53 = ref;
                        AnnoLinGradient lin = Annotated_LinGradient_read(param_52, param_53);
                        Alloc param_54 = cmd_alloc;
                        CmdRef param_55 = cmd_ref;
                        uint param_56 = cmd_limit;
                        bool _2177 = alloc_cmd(param_54, param_55, param_56);
                        cmd_alloc = param_54;
                        cmd_ref = param_55;
                        cmd_limit = param_56;
                        if (!_2177)
                        {
                            break;
                        }
                        Alloc param_57 = cmd_alloc;
                        CmdRef param_58 = cmd_ref;
                        uint param_59 = tag_2.flags;
                        Tile param_60 = tile_1;
                        float param_61 = fill.linewidth;
                        write_fill(param_57, param_58, param_59, param_60, param_61);
                        cmd_ref = param_58;
                        cmd_lin.index = lin.index;
                        cmd_lin.line_x = lin.line_x;
                        cmd_lin.line_y = lin.line_y;
                        cmd_lin.line_c = lin.line_c;
                        Alloc param_62 = cmd_alloc;
                        CmdRef param_63 = cmd_ref;
                        CmdLinGrad param_64 = cmd_lin;
                        Cmd_LinGrad_write(param_62, param_63, param_64);
                        cmd_ref.offset += 20u;
                        break;
                    }
                    case 3u:
                    {
                        uint param_65 = element_ref_ix;
                        bool param_66 = mem_ok;
                        TileRef _2242 = { sh_tile_base[element_ref_ix] + (((sh_tile_stride[element_ref_ix] * tile_y) + tile_x) * 8u) };
                        Alloc param_67 = read_tile_alloc(param_65, param_66);
                        TileRef param_68 = _2242;
                        tile_1 = Tile_read(param_67, param_68);
                        Alloc _2249;
                        _2249.offset = _1283.Load(32);
                        param_69.offset = _2249.offset;
                        AnnotatedRef param_70 = ref;
                        AnnoImage fill_img = Annotated_Image_read(param_69, param_70);
                        Alloc param_71 = cmd_alloc;
                        CmdRef param_72 = cmd_ref;
                        uint param_73 = cmd_limit;
                        bool _2261 = alloc_cmd(param_71, param_72, param_73);
                        cmd_alloc = param_71;
                        cmd_ref = param_72;
                        cmd_limit = param_73;
                        if (!_2261)
                        {
                            break;
                        }
                        Alloc param_74 = cmd_alloc;
                        CmdRef param_75 = cmd_ref;
                        uint param_76 = tag_2.flags;
                        Tile param_77 = tile_1;
                        float param_78 = fill_img.linewidth;
                        write_fill(param_74, param_75, param_76, param_77, param_78);
                        cmd_ref = param_75;
                        CmdImage _2287 = { fill_img.index, fill_img.offset };
                        Alloc param_79 = cmd_alloc;
                        CmdRef param_80 = cmd_ref;
                        CmdImage param_81 = _2287;
                        Cmd_Image_write(param_79, param_80, param_81);
                        cmd_ref.offset += 12u;
                        break;
                    }
                    case 4u:
                    {
                        uint param_82 = element_ref_ix;
                        bool param_83 = mem_ok;
                        TileRef _2316 = { sh_tile_base[element_ref_ix] + (((sh_tile_stride[element_ref_ix] * tile_y) + tile_x) * 8u) };
                        Alloc param_84 = read_tile_alloc(param_82, param_83);
                        TileRef param_85 = _2316;
                        tile_1 = Tile_read(param_84, param_85);
                        bool _2322 = tile_1.tile.offset == 0u;
                        bool _2328;
                        if (_2322)
                        {
                            _2328 = tile_1.backdrop == 0;
                        }
                        else
                        {
                            _2328 = _2322;
                        }
                        if (_2328)
                        {
                            clip_zero_depth = clip_depth + 1u;
                        }
                        else
                        {
                            Alloc param_86 = cmd_alloc;
                            CmdRef param_87 = cmd_ref;
                            uint param_88 = cmd_limit;
                            bool _2340 = alloc_cmd(param_86, param_87, param_88);
                            cmd_alloc = param_86;
                            cmd_ref = param_87;
                            cmd_limit = param_88;
                            if (!_2340)
                            {
                                break;
                            }
                            Alloc param_89 = cmd_alloc;
                            CmdRef param_90 = cmd_ref;
                            Cmd_BeginClip_write(param_89, param_90);
                            cmd_ref.offset += 4u;
                        }
                        clip_depth++;
                        break;
                    }
                    case 5u:
                    {
                        uint param_91 = element_ref_ix;
                        bool param_92 = mem_ok;
                        TileRef _2377 = { sh_tile_base[element_ref_ix] + (((sh_tile_stride[element_ref_ix] * tile_y) + tile_x) * 8u) };
                        Alloc param_93 = read_tile_alloc(param_91, param_92);
                        TileRef param_94 = _2377;
                        tile_1 = Tile_read(param_93, param_94);
                        Alloc _2384;
                        _2384.offset = _1283.Load(32);
                        param_95.offset = _2384.offset;
                        AnnotatedRef param_96 = ref;
                        AnnoEndClip end_clip = Annotated_EndClip_read(param_95, param_96);
                        clip_depth--;
                        Alloc param_97 = cmd_alloc;
                        CmdRef param_98 = cmd_ref;
                        uint param_99 = cmd_limit;
                        bool _2398 = alloc_cmd(param_97, param_98, param_99);
                        cmd_alloc = param_97;
                        cmd_ref = param_98;
                        cmd_limit = param_99;
                        if (!_2398)
                        {
                            break;
                        }
                        Alloc param_100 = cmd_alloc;
                        CmdRef param_101 = cmd_ref;
                        uint param_102 = 0u;
                        Tile param_103 = tile_1;
                        float param_104 = 0.0f;
                        write_fill(param_100, param_101, param_102, param_103, param_104);
                        cmd_ref = param_101;
                        CmdEndClip _2419 = { end_clip.blend };
                        Alloc param_105 = cmd_alloc;
                        CmdRef param_106 = cmd_ref;
                        CmdEndClip param_107 = _2419;
                        Cmd_EndClip_write(param_105, param_106, param_107);
                        cmd_ref.offset += 8u;
                        break;
                    }
                }
            }
            else
            {
                switch (tag_2.tag)
                {
                    case 4u:
                    {
                        clip_depth++;
                        break;
                    }
                    case 5u:
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
    bool _2467 = (bin_tile_x + tile_x) < _1283.Load(8);
    bool _2476;
    if (_2467)
    {
        _2476 = (bin_tile_y + tile_y) < _1283.Load(12);
    }
    else
    {
        _2476 = _2467;
    }
    if (_2476)
    {
        Alloc param_108 = cmd_alloc;
        CmdRef param_109 = cmd_ref;
        Cmd_End_write(param_108, param_109);
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    comp_main();
}
