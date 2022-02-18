struct Alloc
{
    uint offset;
};

struct ElementRef
{
    uint offset;
};

struct FillColorRef
{
    uint offset;
};

struct FillColor
{
    uint rgba_color;
};

struct FillLinGradientRef
{
    uint offset;
};

struct FillLinGradient
{
    uint index;
    float2 p0;
    float2 p1;
};

struct FillImageRef
{
    uint offset;
};

struct FillImage
{
    uint index;
    int2 offset;
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

struct AnnoBeginClipRef
{
    uint offset;
};

struct AnnoBeginClip
{
    float4 bbox;
    float linewidth;
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

static const DrawMonoid _348 = { 0u, 0u };
static const DrawMonoid _372 = { 1u, 0u };
static const DrawMonoid _374 = { 1u, 1u };

RWByteAddressBuffer _187 : register(u0, space0);
ByteAddressBuffer _211 : register(t2, space0);
ByteAddressBuffer _934 : register(t3, space0);
ByteAddressBuffer _968 : register(t1, space0);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared DrawMonoid sh_scratch[256];

ElementTag Element_tag(ElementRef ref)
{
    uint tag_and_flags = _211.Load((ref.offset >> uint(2)) * 4 + 0);
    ElementTag _321 = { tag_and_flags & 65535u, tag_and_flags >> uint(16) };
    return _321;
}

DrawMonoid map_tag(uint tag_word)
{
    switch (tag_word)
    {
        case 4u:
        case 5u:
        case 6u:
        {
            return _372;
        }
        case 9u:
        case 10u:
        {
            return _374;
        }
        default:
        {
            return _348;
        }
    }
}

ElementRef Element_index(ElementRef ref, uint index)
{
    ElementRef _200 = { ref.offset + (index * 36u) };
    return _200;
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
    return _348;
}

FillColor FillColor_read(FillColorRef ref)
{
    uint ix = ref.offset >> uint(2);
    uint raw0 = _211.Load((ix + 0u) * 4 + 0);
    FillColor s;
    s.rgba_color = raw0;
    return s;
}

FillColor Element_FillColor_read(ElementRef ref)
{
    FillColorRef _327 = { ref.offset + 4u };
    FillColorRef param = _327;
    return FillColor_read(param);
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
    _187.Store(offset * 4 + 8, val);
}

void AnnoColor_write(Alloc a, AnnoColorRef ref, AnnoColor s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = asuint(s.bbox.x);
    write_mem(param, param_1, param_2);
    Alloc param_3 = a;
    uint param_4 = ix + 1u;
    uint param_5 = asuint(s.bbox.y);
    write_mem(param_3, param_4, param_5);
    Alloc param_6 = a;
    uint param_7 = ix + 2u;
    uint param_8 = asuint(s.bbox.z);
    write_mem(param_6, param_7, param_8);
    Alloc param_9 = a;
    uint param_10 = ix + 3u;
    uint param_11 = asuint(s.bbox.w);
    write_mem(param_9, param_10, param_11);
    Alloc param_12 = a;
    uint param_13 = ix + 4u;
    uint param_14 = asuint(s.linewidth);
    write_mem(param_12, param_13, param_14);
    Alloc param_15 = a;
    uint param_16 = ix + 5u;
    uint param_17 = s.rgba_color;
    write_mem(param_15, param_16, param_17);
}

void Annotated_Color_write(Alloc a, AnnotatedRef ref, uint flags, AnnoColor s)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = (flags << uint(16)) | 1u;
    write_mem(param, param_1, param_2);
    AnnoColorRef _735 = { ref.offset + 4u };
    Alloc param_3 = a;
    AnnoColorRef param_4 = _735;
    AnnoColor param_5 = s;
    AnnoColor_write(param_3, param_4, param_5);
}

FillLinGradient FillLinGradient_read(FillLinGradientRef ref)
{
    uint ix = ref.offset >> uint(2);
    uint raw0 = _211.Load((ix + 0u) * 4 + 0);
    uint raw1 = _211.Load((ix + 1u) * 4 + 0);
    uint raw2 = _211.Load((ix + 2u) * 4 + 0);
    uint raw3 = _211.Load((ix + 3u) * 4 + 0);
    uint raw4 = _211.Load((ix + 4u) * 4 + 0);
    FillLinGradient s;
    s.index = raw0;
    s.p0 = float2(asfloat(raw1), asfloat(raw2));
    s.p1 = float2(asfloat(raw3), asfloat(raw4));
    return s;
}

FillLinGradient Element_FillLinGradient_read(ElementRef ref)
{
    FillLinGradientRef _335 = { ref.offset + 4u };
    FillLinGradientRef param = _335;
    return FillLinGradient_read(param);
}

void AnnoLinGradient_write(Alloc a, AnnoLinGradientRef ref, AnnoLinGradient s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = asuint(s.bbox.x);
    write_mem(param, param_1, param_2);
    Alloc param_3 = a;
    uint param_4 = ix + 1u;
    uint param_5 = asuint(s.bbox.y);
    write_mem(param_3, param_4, param_5);
    Alloc param_6 = a;
    uint param_7 = ix + 2u;
    uint param_8 = asuint(s.bbox.z);
    write_mem(param_6, param_7, param_8);
    Alloc param_9 = a;
    uint param_10 = ix + 3u;
    uint param_11 = asuint(s.bbox.w);
    write_mem(param_9, param_10, param_11);
    Alloc param_12 = a;
    uint param_13 = ix + 4u;
    uint param_14 = asuint(s.linewidth);
    write_mem(param_12, param_13, param_14);
    Alloc param_15 = a;
    uint param_16 = ix + 5u;
    uint param_17 = s.index;
    write_mem(param_15, param_16, param_17);
    Alloc param_18 = a;
    uint param_19 = ix + 6u;
    uint param_20 = asuint(s.line_x);
    write_mem(param_18, param_19, param_20);
    Alloc param_21 = a;
    uint param_22 = ix + 7u;
    uint param_23 = asuint(s.line_y);
    write_mem(param_21, param_22, param_23);
    Alloc param_24 = a;
    uint param_25 = ix + 8u;
    uint param_26 = asuint(s.line_c);
    write_mem(param_24, param_25, param_26);
}

void Annotated_LinGradient_write(Alloc a, AnnotatedRef ref, uint flags, AnnoLinGradient s)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = (flags << uint(16)) | 2u;
    write_mem(param, param_1, param_2);
    AnnoLinGradientRef _756 = { ref.offset + 4u };
    Alloc param_3 = a;
    AnnoLinGradientRef param_4 = _756;
    AnnoLinGradient param_5 = s;
    AnnoLinGradient_write(param_3, param_4, param_5);
}

FillImage FillImage_read(FillImageRef ref)
{
    uint ix = ref.offset >> uint(2);
    uint raw0 = _211.Load((ix + 0u) * 4 + 0);
    uint raw1 = _211.Load((ix + 1u) * 4 + 0);
    FillImage s;
    s.index = raw0;
    s.offset = int2(int(raw1 << uint(16)) >> 16, int(raw1) >> 16);
    return s;
}

FillImage Element_FillImage_read(ElementRef ref)
{
    FillImageRef _343 = { ref.offset + 4u };
    FillImageRef param = _343;
    return FillImage_read(param);
}

void AnnoImage_write(Alloc a, AnnoImageRef ref, AnnoImage s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = asuint(s.bbox.x);
    write_mem(param, param_1, param_2);
    Alloc param_3 = a;
    uint param_4 = ix + 1u;
    uint param_5 = asuint(s.bbox.y);
    write_mem(param_3, param_4, param_5);
    Alloc param_6 = a;
    uint param_7 = ix + 2u;
    uint param_8 = asuint(s.bbox.z);
    write_mem(param_6, param_7, param_8);
    Alloc param_9 = a;
    uint param_10 = ix + 3u;
    uint param_11 = asuint(s.bbox.w);
    write_mem(param_9, param_10, param_11);
    Alloc param_12 = a;
    uint param_13 = ix + 4u;
    uint param_14 = asuint(s.linewidth);
    write_mem(param_12, param_13, param_14);
    Alloc param_15 = a;
    uint param_16 = ix + 5u;
    uint param_17 = s.index;
    write_mem(param_15, param_16, param_17);
    Alloc param_18 = a;
    uint param_19 = ix + 6u;
    uint param_20 = (uint(s.offset.x) & 65535u) | (uint(s.offset.y) << uint(16));
    write_mem(param_18, param_19, param_20);
}

void Annotated_Image_write(Alloc a, AnnotatedRef ref, uint flags, AnnoImage s)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = (flags << uint(16)) | 3u;
    write_mem(param, param_1, param_2);
    AnnoImageRef _777 = { ref.offset + 4u };
    Alloc param_3 = a;
    AnnoImageRef param_4 = _777;
    AnnoImage param_5 = s;
    AnnoImage_write(param_3, param_4, param_5);
}

void AnnoBeginClip_write(Alloc a, AnnoBeginClipRef ref, AnnoBeginClip s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = asuint(s.bbox.x);
    write_mem(param, param_1, param_2);
    Alloc param_3 = a;
    uint param_4 = ix + 1u;
    uint param_5 = asuint(s.bbox.y);
    write_mem(param_3, param_4, param_5);
    Alloc param_6 = a;
    uint param_7 = ix + 2u;
    uint param_8 = asuint(s.bbox.z);
    write_mem(param_6, param_7, param_8);
    Alloc param_9 = a;
    uint param_10 = ix + 3u;
    uint param_11 = asuint(s.bbox.w);
    write_mem(param_9, param_10, param_11);
    Alloc param_12 = a;
    uint param_13 = ix + 4u;
    uint param_14 = asuint(s.linewidth);
    write_mem(param_12, param_13, param_14);
}

void Annotated_BeginClip_write(Alloc a, AnnotatedRef ref, uint flags, AnnoBeginClip s)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = (flags << uint(16)) | 4u;
    write_mem(param, param_1, param_2);
    AnnoBeginClipRef _798 = { ref.offset + 4u };
    Alloc param_3 = a;
    AnnoBeginClipRef param_4 = _798;
    AnnoBeginClip param_5 = s;
    AnnoBeginClip_write(param_3, param_4, param_5);
}

void AnnoEndClip_write(Alloc a, AnnoEndClipRef ref, AnnoEndClip s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = asuint(s.bbox.x);
    write_mem(param, param_1, param_2);
    Alloc param_3 = a;
    uint param_4 = ix + 1u;
    uint param_5 = asuint(s.bbox.y);
    write_mem(param_3, param_4, param_5);
    Alloc param_6 = a;
    uint param_7 = ix + 2u;
    uint param_8 = asuint(s.bbox.z);
    write_mem(param_6, param_7, param_8);
    Alloc param_9 = a;
    uint param_10 = ix + 3u;
    uint param_11 = asuint(s.bbox.w);
    write_mem(param_9, param_10, param_11);
}

void Annotated_EndClip_write(Alloc a, AnnotatedRef ref, AnnoEndClip s)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = 5u;
    write_mem(param, param_1, param_2);
    AnnoEndClipRef _816 = { ref.offset + 4u };
    Alloc param_3 = a;
    AnnoEndClipRef param_4 = _816;
    AnnoEndClip param_5 = s;
    AnnoEndClip_write(param_3, param_4, param_5);
}

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x * 8u;
    ElementRef _834 = { ix * 36u };
    ElementRef ref = _834;
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
    for (uint i_1 = 0u; i_1 < 8u; i_1++)
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
        DrawMonoid _940;
        _940.path_ix = _934.Load((gl_WorkGroupID.x - 1u) * 8 + 0);
        _940.clip_ix = _934.Load((gl_WorkGroupID.x - 1u) * 8 + 4);
        row.path_ix = _940.path_ix;
        row.clip_ix = _940.clip_ix;
    }
    if (gl_LocalInvocationID.x > 0u)
    {
        DrawMonoid param_10 = row;
        DrawMonoid param_11 = sh_scratch[gl_LocalInvocationID.x - 1u];
        row = combine_tag_monoid(param_10, param_11);
    }
    uint out_ix = gl_GlobalInvocationID.x * 8u;
    uint out_base = (_968.Load(44) >> uint(2)) + (out_ix * 2u);
    uint clip_out_base = _968.Load(48) >> uint(2);
    AnnotatedRef _989 = { _968.Load(32) + (out_ix * 40u) };
    AnnotatedRef out_ref = _989;
    float4 mat;
    float2 translate;
    AnnoColor anno_fill;
    Alloc param_18;
    AnnoLinGradient anno_lin;
    Alloc param_23;
    AnnoImage anno_img;
    Alloc param_28;
    AnnoBeginClip anno_begin_clip;
    Alloc param_32;
    AnnoEndClip anno_end_clip;
    Alloc param_36;
    for (uint i_2 = 0u; i_2 < 8u; i_2++)
    {
        DrawMonoid m = row;
        if (i_2 > 0u)
        {
            DrawMonoid param_12 = m;
            DrawMonoid param_13 = local[i_2 - 1u];
            m = combine_tag_monoid(param_12, param_13);
        }
        _187.Store((out_base + (i_2 * 2u)) * 4 + 8, m.path_ix);
        _187.Store(((out_base + (i_2 * 2u)) + 1u) * 4 + 8, m.clip_ix);
        ElementRef param_14 = ref;
        uint param_15 = i_2;
        ElementRef this_ref = Element_index(param_14, param_15);
        ElementRef param_16 = this_ref;
        tag_word = Element_tag(param_16).tag;
        if ((((tag_word == 4u) || (tag_word == 5u)) || (tag_word == 6u)) || (tag_word == 9u))
        {
            uint bbox_offset = (_968.Load(40) >> uint(2)) + (6u * m.path_ix);
            float bbox_l = float(_187.Load(bbox_offset * 4 + 8)) - 32768.0f;
            float bbox_t = float(_187.Load((bbox_offset + 1u) * 4 + 8)) - 32768.0f;
            float bbox_r = float(_187.Load((bbox_offset + 2u) * 4 + 8)) - 32768.0f;
            float bbox_b = float(_187.Load((bbox_offset + 3u) * 4 + 8)) - 32768.0f;
            float4 bbox = float4(bbox_l, bbox_t, bbox_r, bbox_b);
            float linewidth = asfloat(_187.Load((bbox_offset + 4u) * 4 + 8));
            uint fill_mode = uint(linewidth >= 0.0f);
            if ((linewidth >= 0.0f) || (tag_word == 5u))
            {
                uint trans_ix = _187.Load((bbox_offset + 5u) * 4 + 8);
                uint t = (_968.Load(36) >> uint(2)) + (6u * trans_ix);
                mat = asfloat(uint4(_187.Load(t * 4 + 8), _187.Load((t + 1u) * 4 + 8), _187.Load((t + 2u) * 4 + 8), _187.Load((t + 3u) * 4 + 8)));
                if (tag_word == 5u)
                {
                    translate = asfloat(uint2(_187.Load((t + 4u) * 4 + 8), _187.Load((t + 5u) * 4 + 8)));
                }
            }
            if (linewidth >= 0.0f)
            {
                linewidth *= sqrt(abs((mat.x * mat.w) - (mat.y * mat.z)));
            }
            linewidth = max(linewidth, 0.0f);
            switch (tag_word)
            {
                case 4u:
                {
                    ElementRef param_17 = this_ref;
                    FillColor fill = Element_FillColor_read(param_17);
                    anno_fill.bbox = bbox;
                    anno_fill.linewidth = linewidth;
                    anno_fill.rgba_color = fill.rgba_color;
                    Alloc _1203;
                    _1203.offset = _968.Load(32);
                    param_18.offset = _1203.offset;
                    AnnotatedRef param_19 = out_ref;
                    uint param_20 = fill_mode;
                    AnnoColor param_21 = anno_fill;
                    Annotated_Color_write(param_18, param_19, param_20, param_21);
                    break;
                }
                case 5u:
                {
                    ElementRef param_22 = this_ref;
                    FillLinGradient lin = Element_FillLinGradient_read(param_22);
                    anno_lin.bbox = bbox;
                    anno_lin.linewidth = linewidth;
                    anno_lin.index = lin.index;
                    float2 p0 = ((mat.xy * lin.p0.x) + (mat.zw * lin.p0.y)) + translate;
                    float2 p1 = ((mat.xy * lin.p1.x) + (mat.zw * lin.p1.y)) + translate;
                    float2 dxy = p1 - p0;
                    float scale = 1.0f / ((dxy.x * dxy.x) + (dxy.y * dxy.y));
                    float line_x = dxy.x * scale;
                    float line_y = dxy.y * scale;
                    anno_lin.line_x = line_x;
                    anno_lin.line_y = line_y;
                    anno_lin.line_c = -((p0.x * line_x) + (p0.y * line_y));
                    Alloc _1299;
                    _1299.offset = _968.Load(32);
                    param_23.offset = _1299.offset;
                    AnnotatedRef param_24 = out_ref;
                    uint param_25 = fill_mode;
                    AnnoLinGradient param_26 = anno_lin;
                    Annotated_LinGradient_write(param_23, param_24, param_25, param_26);
                    break;
                }
                case 6u:
                {
                    ElementRef param_27 = this_ref;
                    FillImage fill_img = Element_FillImage_read(param_27);
                    anno_img.bbox = bbox;
                    anno_img.linewidth = linewidth;
                    anno_img.index = fill_img.index;
                    anno_img.offset = fill_img.offset;
                    Alloc _1327;
                    _1327.offset = _968.Load(32);
                    param_28.offset = _1327.offset;
                    AnnotatedRef param_29 = out_ref;
                    uint param_30 = fill_mode;
                    AnnoImage param_31 = anno_img;
                    Annotated_Image_write(param_28, param_29, param_30, param_31);
                    break;
                }
                case 9u:
                {
                    anno_begin_clip.bbox = bbox;
                    anno_begin_clip.linewidth = 0.0f;
                    Alloc _1344;
                    _1344.offset = _968.Load(32);
                    param_32.offset = _1344.offset;
                    AnnotatedRef param_33 = out_ref;
                    uint param_34 = 0u;
                    AnnoBeginClip param_35 = anno_begin_clip;
                    Annotated_BeginClip_write(param_32, param_33, param_34, param_35);
                    break;
                }
            }
        }
        else
        {
            if (tag_word == 10u)
            {
                anno_end_clip.bbox = float4(-1000000000.0f, -1000000000.0f, 1000000000.0f, 1000000000.0f);
                Alloc _1368;
                _1368.offset = _968.Load(32);
                param_36.offset = _1368.offset;
                AnnotatedRef param_37 = out_ref;
                AnnoEndClip param_38 = anno_end_clip;
                Annotated_EndClip_write(param_36, param_37, param_38);
            }
        }
        if ((tag_word == 9u) || (tag_word == 10u))
        {
            uint path_ix = ~(out_ix + i_2);
            if (tag_word == 9u)
            {
                path_ix = m.path_ix;
            }
            _187.Store((clip_out_base + m.clip_ix) * 4 + 8, path_ix);
        }
        out_ref.offset += 40u;
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
