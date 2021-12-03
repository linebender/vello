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

struct BinInstanceRef
{
    uint offset;
};

struct BinInstance
{
    uint element_ix;
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

static const uint3 gl_WorkGroupSize = uint3(256u, 1u, 1u);

RWByteAddressBuffer _84 : register(u0, space0);
ByteAddressBuffer _253 : register(t1, space0);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
};

groupshared uint bitmaps[8][256];
groupshared bool sh_alloc_failed;
groupshared uint count[8][256];
groupshared Alloc sh_chunk_alloc[256];

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
    uint v = _84.Load(offset * 4 + 8);
    return v;
}

AnnotatedTag Annotated_tag(Alloc a, AnnotatedRef ref)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint tag_and_flags = read_mem(param, param_1);
    AnnotatedTag _221 = { tag_and_flags & 65535u, tag_and_flags >> uint(16) };
    return _221;
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
    AnnoEndClipRef _228 = { ref.offset + 4u };
    Alloc param = a;
    AnnoEndClipRef param_1 = _228;
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
    uint _90;
    _84.InterlockedAdd(0, size, _90);
    uint offset = _90;
    uint _97;
    _84.GetDimensions(_97);
    _97 = (_97 - 8) / 4;
    MallocResult r;
    r.failed = (offset + size) > uint(int(_97) * 4);
    uint param = offset;
    uint param_1 = size;
    bool param_2 = !r.failed;
    r.alloc = new_alloc(param, param_1, param_2);
    if (r.failed)
    {
        uint _119;
        _84.InterlockedMax(4, 1u, _119);
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
    _84.Store(offset * 4 + 8, val);
}

void BinInstance_write(Alloc a, BinInstanceRef ref, BinInstance s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = s.element_ix;
    write_mem(param, param_1, param_2);
}

void comp_main()
{
    uint my_n_elements = _253.Load(0);
    uint my_partition = gl_WorkGroupID.x;
    for (uint i = 0u; i < 8u; i++)
    {
        bitmaps[i][gl_LocalInvocationID.x] = 0u;
    }
    if (gl_LocalInvocationID.x == 0u)
    {
        sh_alloc_failed = false;
    }
    GroupMemoryBarrierWithGroupSync();
    uint element_ix = (my_partition * 256u) + gl_LocalInvocationID.x;
    AnnotatedRef _308 = { _253.Load(32) + (element_ix * 40u) };
    AnnotatedRef ref = _308;
    uint tag = 0u;
    if (element_ix < my_n_elements)
    {
        Alloc _318;
        _318.offset = _253.Load(32);
        Alloc param;
        param.offset = _318.offset;
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
            Alloc _336;
            _336.offset = _253.Load(32);
            Alloc param_2;
            param_2.offset = _336.offset;
            AnnotatedRef param_3 = ref;
            AnnoEndClip clip = Annotated_EndClip_read(param_2, param_3);
            x0 = int(floor(clip.bbox.x * 0.00390625f));
            y0 = int(floor(clip.bbox.y * 0.00390625f));
            x1 = int(ceil(clip.bbox.z * 0.00390625f));
            y1 = int(ceil(clip.bbox.w * 0.00390625f));
            break;
        }
    }
    uint width_in_bins = ((_253.Load(8) + 16u) - 1u) / 16u;
    uint height_in_bins = ((_253.Load(12) + 16u) - 1u) / 16u;
    x0 = clamp(x0, 0, int(width_in_bins));
    x1 = clamp(x1, x0, int(width_in_bins));
    y0 = clamp(y0, 0, int(height_in_bins));
    y1 = clamp(y1, y0, int(height_in_bins));
    if (x0 == x1)
    {
        y1 = y0;
    }
    int x = x0;
    int y = y0;
    uint my_slice = gl_LocalInvocationID.x / 32u;
    uint my_mask = uint(1 << int(gl_LocalInvocationID.x & 31u));
    while (y < y1)
    {
        uint _438;
        InterlockedOr(bitmaps[my_slice][(uint(y) * width_in_bins) + uint(x)], my_mask, _438);
        x++;
        if (x == x1)
        {
            x = x0;
            y++;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint element_count = 0u;
    for (uint i_1 = 0u; i_1 < 8u; i_1++)
    {
        element_count += uint(int(countbits(bitmaps[i_1][gl_LocalInvocationID.x])));
        count[i_1][gl_LocalInvocationID.x] = element_count;
    }
    uint param_4 = 0u;
    uint param_5 = 0u;
    bool param_6 = true;
    Alloc chunk_alloc = new_alloc(param_4, param_5, param_6);
    if (element_count != 0u)
    {
        uint param_7 = element_count * 4u;
        MallocResult _488 = malloc(param_7);
        MallocResult chunk = _488;
        chunk_alloc = chunk.alloc;
        sh_chunk_alloc[gl_LocalInvocationID.x] = chunk_alloc;
        if (chunk.failed)
        {
            sh_alloc_failed = true;
        }
    }
    uint out_ix = (_253.Load(20) >> uint(2)) + (((my_partition * 256u) + gl_LocalInvocationID.x) * 2u);
    Alloc _517;
    _517.offset = _253.Load(20);
    Alloc param_8;
    param_8.offset = _517.offset;
    uint param_9 = out_ix;
    uint param_10 = element_count;
    write_mem(param_8, param_9, param_10);
    Alloc _529;
    _529.offset = _253.Load(20);
    Alloc param_11;
    param_11.offset = _529.offset;
    uint param_12 = out_ix + 1u;
    uint param_13 = chunk_alloc.offset;
    write_mem(param_11, param_12, param_13);
    GroupMemoryBarrierWithGroupSync();
    bool _544;
    if (!sh_alloc_failed)
    {
        _544 = _84.Load(4) != 0u;
    }
    else
    {
        _544 = sh_alloc_failed;
    }
    if (_544)
    {
        return;
    }
    x = x0;
    y = y0;
    while (y < y1)
    {
        uint bin_ix = (uint(y) * width_in_bins) + uint(x);
        uint out_mask = bitmaps[my_slice][bin_ix];
        if ((out_mask & my_mask) != 0u)
        {
            uint idx = uint(int(countbits(out_mask & (my_mask - 1u))));
            if (my_slice > 0u)
            {
                idx += count[my_slice - 1u][bin_ix];
            }
            Alloc out_alloc = sh_chunk_alloc[bin_ix];
            uint out_offset = out_alloc.offset + (idx * 4u);
            BinInstanceRef _606 = { out_offset };
            BinInstance _608 = { element_ix };
            Alloc param_14 = out_alloc;
            BinInstanceRef param_15 = _606;
            BinInstance param_16 = _608;
            BinInstance_write(param_14, param_15, param_16);
        }
        x++;
        if (x == x1)
        {
            x = x0;
            y++;
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
