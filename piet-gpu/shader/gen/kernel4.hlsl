struct Alloc
{
    uint offset;
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

struct CmdAlphaRef
{
    uint offset;
};

struct CmdAlpha
{
    float alpha;
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

struct CmdTag
{
    uint tag;
    uint flags;
};

struct TileSegRef
{
    uint offset;
};

struct TileSeg
{
    float2 origin;
    float2 _vector;
    float y_edge;
    TileSegRef next;
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

static const uint3 gl_WorkGroupSize = uint3(8u, 4u, 1u);

RWByteAddressBuffer _291 : register(u0, space0);
ByteAddressBuffer _1666 : register(t1, space0);
RWTexture2D<unorm float4> image_atlas : register(u3, space0);
RWTexture2D<unorm float4> gradients : register(u4, space0);
RWTexture2D<unorm float4> image : register(u2, space0);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
};

uint spvPackUnorm4x8(float4 value)
{
    uint4 Packed = uint4(round(saturate(value) * 255.0));
    return Packed.x | (Packed.y << 8) | (Packed.z << 16) | (Packed.w << 24);
}

float4 spvUnpackUnorm4x8(uint value)
{
    uint4 Packed = uint4(value & 0xff, (value >> 8) & 0xff, (value >> 16) & 0xff, value >> 24);
    return float4(Packed) / 255.0;
}

Alloc slice_mem(Alloc a, uint offset, uint size)
{
    Alloc _304 = { a.offset + offset };
    return _304;
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
    uint v = _291.Load(offset * 4 + 8);
    return v;
}

CmdTag Cmd_tag(Alloc a, CmdRef ref)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint tag_and_flags = read_mem(param, param_1);
    CmdTag _663 = { tag_and_flags & 65535u, tag_and_flags >> uint(16) };
    return _663;
}

CmdStroke CmdStroke_read(Alloc a, CmdStrokeRef ref)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint raw0 = read_mem(param, param_1);
    Alloc param_2 = a;
    uint param_3 = ix + 1u;
    uint raw1 = read_mem(param_2, param_3);
    CmdStroke s;
    s.tile_ref = raw0;
    s.half_width = asfloat(raw1);
    return s;
}

CmdStroke Cmd_Stroke_read(Alloc a, CmdRef ref)
{
    CmdStrokeRef _679 = { ref.offset + 4u };
    Alloc param = a;
    CmdStrokeRef param_1 = _679;
    return CmdStroke_read(param, param_1);
}

Alloc new_alloc(uint offset, uint size, bool mem_ok)
{
    Alloc a;
    a.offset = offset;
    return a;
}

TileSeg TileSeg_read(Alloc a, TileSegRef ref)
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
    TileSeg s;
    s.origin = float2(asfloat(raw0), asfloat(raw1));
    s._vector = float2(asfloat(raw2), asfloat(raw3));
    s.y_edge = asfloat(raw4);
    TileSegRef _820 = { raw5 };
    s.next = _820;
    return s;
}

uint2 chunk_offset(uint i)
{
    return uint2((i % 2u) * 8u, (i / 2u) * 4u);
}

CmdFill CmdFill_read(Alloc a, CmdFillRef ref)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint raw0 = read_mem(param, param_1);
    Alloc param_2 = a;
    uint param_3 = ix + 1u;
    uint raw1 = read_mem(param_2, param_3);
    CmdFill s;
    s.tile_ref = raw0;
    s.backdrop = int(raw1);
    return s;
}

CmdFill Cmd_Fill_read(Alloc a, CmdRef ref)
{
    CmdFillRef _669 = { ref.offset + 4u };
    Alloc param = a;
    CmdFillRef param_1 = _669;
    return CmdFill_read(param, param_1);
}

CmdAlpha CmdAlpha_read(Alloc a, CmdAlphaRef ref)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint raw0 = read_mem(param, param_1);
    CmdAlpha s;
    s.alpha = asfloat(raw0);
    return s;
}

CmdAlpha Cmd_Alpha_read(Alloc a, CmdRef ref)
{
    CmdAlphaRef _689 = { ref.offset + 4u };
    Alloc param = a;
    CmdAlphaRef param_1 = _689;
    return CmdAlpha_read(param, param_1);
}

CmdColor CmdColor_read(Alloc a, CmdColorRef ref)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint raw0 = read_mem(param, param_1);
    CmdColor s;
    s.rgba_color = raw0;
    return s;
}

CmdColor Cmd_Color_read(Alloc a, CmdRef ref)
{
    CmdColorRef _699 = { ref.offset + 4u };
    Alloc param = a;
    CmdColorRef param_1 = _699;
    return CmdColor_read(param, param_1);
}

float3 fromsRGB(float3 srgb)
{
    bool3 cutoff = bool3(srgb.x >= 0.040449999272823333740234375f.xxx.x, srgb.y >= 0.040449999272823333740234375f.xxx.y, srgb.z >= 0.040449999272823333740234375f.xxx.z);
    float3 below = srgb / 12.9200000762939453125f.xxx;
    float3 above = pow((srgb + 0.054999999701976776123046875f.xxx) / 1.05499994754791259765625f.xxx, 2.400000095367431640625f.xxx);
    return float3(cutoff.x ? above.x : below.x, cutoff.y ? above.y : below.y, cutoff.z ? above.z : below.z);
}

float4 unpacksRGB(uint srgba)
{
    float4 color = spvUnpackUnorm4x8(srgba).wzyx;
    float3 param = color.xyz;
    return float4(fromsRGB(param), color.w);
}

CmdLinGrad CmdLinGrad_read(Alloc a, CmdLinGradRef ref)
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
    CmdLinGrad s;
    s.index = raw0;
    s.line_x = asfloat(raw1);
    s.line_y = asfloat(raw2);
    s.line_c = asfloat(raw3);
    return s;
}

CmdLinGrad Cmd_LinGrad_read(Alloc a, CmdRef ref)
{
    CmdLinGradRef _709 = { ref.offset + 4u };
    Alloc param = a;
    CmdLinGradRef param_1 = _709;
    return CmdLinGrad_read(param, param_1);
}

CmdRadGrad CmdRadGrad_read(Alloc a, CmdRadGradRef ref)
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
    Alloc param_18 = a;
    uint param_19 = ix + 9u;
    uint raw9 = read_mem(param_18, param_19);
    Alloc param_20 = a;
    uint param_21 = ix + 10u;
    uint raw10 = read_mem(param_20, param_21);
    CmdRadGrad s;
    s.index = raw0;
    s.mat = float4(asfloat(raw1), asfloat(raw2), asfloat(raw3), asfloat(raw4));
    s.xlat = float2(asfloat(raw5), asfloat(raw6));
    s.c1 = float2(asfloat(raw7), asfloat(raw8));
    s.ra = asfloat(raw9);
    s.roff = asfloat(raw10);
    return s;
}

CmdRadGrad Cmd_RadGrad_read(Alloc a, CmdRef ref)
{
    CmdRadGradRef _719 = { ref.offset + 4u };
    Alloc param = a;
    CmdRadGradRef param_1 = _719;
    return CmdRadGrad_read(param, param_1);
}

CmdImage CmdImage_read(Alloc a, CmdImageRef ref)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint raw0 = read_mem(param, param_1);
    Alloc param_2 = a;
    uint param_3 = ix + 1u;
    uint raw1 = read_mem(param_2, param_3);
    CmdImage s;
    s.index = raw0;
    s.offset = int2(int(raw1 << uint(16)) >> 16, int(raw1) >> 16);
    return s;
}

CmdImage Cmd_Image_read(Alloc a, CmdRef ref)
{
    CmdImageRef _729 = { ref.offset + 4u };
    Alloc param = a;
    CmdImageRef param_1 = _729;
    return CmdImage_read(param, param_1);
}

void fillImage(out float4 spvReturnValue[8], uint2 xy, CmdImage cmd_img)
{
    float4 rgba[8];
    for (uint i = 0u; i < 8u; i++)
    {
        uint param = i;
        int2 uv = int2(xy + chunk_offset(param)) + cmd_img.offset;
        float4 fg_rgba = image_atlas[uv];
        float3 param_1 = fg_rgba.xyz;
        float3 _1638 = fromsRGB(param_1);
        fg_rgba.x = _1638.x;
        fg_rgba.y = _1638.y;
        fg_rgba.z = _1638.z;
        rgba[i] = fg_rgba;
    }
    spvReturnValue = rgba;
}

float3 tosRGB(float3 rgb)
{
    bool3 cutoff = bool3(rgb.x >= 0.003130800090730190277099609375f.xxx.x, rgb.y >= 0.003130800090730190277099609375f.xxx.y, rgb.z >= 0.003130800090730190277099609375f.xxx.z);
    float3 below = 12.9200000762939453125f.xxx * rgb;
    float3 above = (1.05499994754791259765625f.xxx * pow(rgb, 0.416660010814666748046875f.xxx)) - 0.054999999701976776123046875f.xxx;
    return float3(cutoff.x ? above.x : below.x, cutoff.y ? above.y : below.y, cutoff.z ? above.z : below.z);
}

uint packsRGB(inout float4 rgba)
{
    float3 param = rgba.xyz;
    rgba = float4(tosRGB(param), rgba.w);
    return spvPackUnorm4x8(rgba.wzyx);
}

CmdEndClip CmdEndClip_read(Alloc a, CmdEndClipRef ref)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint raw0 = read_mem(param, param_1);
    CmdEndClip s;
    s.blend = raw0;
    return s;
}

CmdEndClip Cmd_EndClip_read(Alloc a, CmdRef ref)
{
    CmdEndClipRef _739 = { ref.offset + 4u };
    Alloc param = a;
    CmdEndClipRef param_1 = _739;
    return CmdEndClip_read(param, param_1);
}

float3 screen(float3 cb, float3 cs)
{
    return (cb + cs) - (cb * cs);
}

float3 hard_light(float3 cb, float3 cs)
{
    float3 param = cb;
    float3 param_1 = (cs * 2.0f) - 1.0f.xxx;
    return lerp(screen(param, param_1), (cb * 2.0f) * cs, float3(bool3(cs.x <= 0.5f.xxx.x, cs.y <= 0.5f.xxx.y, cs.z <= 0.5f.xxx.z)));
}

float color_dodge(float cb, float cs)
{
    if (cb == 0.0f)
    {
        return 0.0f;
    }
    else
    {
        if (cs == 1.0f)
        {
            return 1.0f;
        }
        else
        {
            return min(1.0f, cb / (1.0f - cs));
        }
    }
}

float color_burn(float cb, float cs)
{
    if (cb == 1.0f)
    {
        return 1.0f;
    }
    else
    {
        if (cs == 0.0f)
        {
            return 0.0f;
        }
        else
        {
            return 1.0f - min(1.0f, (1.0f - cb) / cs);
        }
    }
}

float3 soft_light(float3 cb, float3 cs)
{
    float3 d = lerp(sqrt(cb), ((((cb * 16.0f) - 12.0f.xxx) * cb) + 4.0f.xxx) * cb, float3(bool3(cb.x <= 0.25f.xxx.x, cb.y <= 0.25f.xxx.y, cb.z <= 0.25f.xxx.z)));
    return lerp(cb + (((cs * 2.0f) - 1.0f.xxx) * (d - cb)), cb - (((1.0f.xxx - (cs * 2.0f)) * cb) * (1.0f.xxx - cb)), float3(bool3(cs.x <= 0.5f.xxx.x, cs.y <= 0.5f.xxx.y, cs.z <= 0.5f.xxx.z)));
}

float sat(float3 c)
{
    return max(c.x, max(c.y, c.z)) - min(c.x, min(c.y, c.z));
}

void set_sat_inner(inout float cmin, inout float cmid, inout float cmax, float s)
{
    if (cmax > cmin)
    {
        cmid = ((cmid - cmin) * s) / (cmax - cmin);
        cmax = s;
    }
    else
    {
        cmid = 0.0f;
        cmax = 0.0f;
    }
    cmin = 0.0f;
}

float3 set_sat(inout float3 c, float s)
{
    if (c.x <= c.y)
    {
        if (c.y <= c.z)
        {
            float param = c.x;
            float param_1 = c.y;
            float param_2 = c.z;
            float param_3 = s;
            set_sat_inner(param, param_1, param_2, param_3);
            c.x = param;
            c.y = param_1;
            c.z = param_2;
        }
        else
        {
            if (c.x <= c.z)
            {
                float param_4 = c.x;
                float param_5 = c.z;
                float param_6 = c.y;
                float param_7 = s;
                set_sat_inner(param_4, param_5, param_6, param_7);
                c.x = param_4;
                c.z = param_5;
                c.y = param_6;
            }
            else
            {
                float param_8 = c.z;
                float param_9 = c.x;
                float param_10 = c.y;
                float param_11 = s;
                set_sat_inner(param_8, param_9, param_10, param_11);
                c.z = param_8;
                c.x = param_9;
                c.y = param_10;
            }
        }
    }
    else
    {
        if (c.x <= c.z)
        {
            float param_12 = c.y;
            float param_13 = c.x;
            float param_14 = c.z;
            float param_15 = s;
            set_sat_inner(param_12, param_13, param_14, param_15);
            c.y = param_12;
            c.x = param_13;
            c.z = param_14;
        }
        else
        {
            if (c.y <= c.z)
            {
                float param_16 = c.y;
                float param_17 = c.z;
                float param_18 = c.x;
                float param_19 = s;
                set_sat_inner(param_16, param_17, param_18, param_19);
                c.y = param_16;
                c.z = param_17;
                c.x = param_18;
            }
            else
            {
                float param_20 = c.z;
                float param_21 = c.y;
                float param_22 = c.x;
                float param_23 = s;
                set_sat_inner(param_20, param_21, param_22, param_23);
                c.z = param_20;
                c.y = param_21;
                c.x = param_22;
            }
        }
    }
    return c;
}

float lum(float3 c)
{
    float3 f = float3(0.300000011920928955078125f, 0.589999973773956298828125f, 0.10999999940395355224609375f);
    return dot(c, f);
}

float3 clip_color(inout float3 c)
{
    float3 param = c;
    float L = lum(param);
    float n = min(c.x, min(c.y, c.z));
    float x = max(c.x, max(c.y, c.z));
    if (n < 0.0f)
    {
        c = L.xxx + (((c - L.xxx) * L) / (L - n).xxx);
    }
    if (x > 1.0f)
    {
        c = L.xxx + (((c - L.xxx) * (1.0f - L)) / (x - L).xxx);
    }
    return c;
}

float3 set_lum(float3 c, float l)
{
    float3 param = c;
    float3 param_1 = c + (l - lum(param)).xxx;
    float3 _1046 = clip_color(param_1);
    return _1046;
}

float3 mix_blend(float3 cb, float3 cs, uint mode)
{
    float3 b = 0.0f.xxx;
    switch (mode)
    {
        case 1u:
        {
            b = cb * cs;
            break;
        }
        case 2u:
        {
            float3 param = cb;
            float3 param_1 = cs;
            b = screen(param, param_1);
            break;
        }
        case 3u:
        {
            float3 param_2 = cs;
            float3 param_3 = cb;
            b = hard_light(param_2, param_3);
            break;
        }
        case 4u:
        {
            b = min(cb, cs);
            break;
        }
        case 5u:
        {
            b = max(cb, cs);
            break;
        }
        case 6u:
        {
            float param_4 = cb.x;
            float param_5 = cs.x;
            float param_6 = cb.y;
            float param_7 = cs.y;
            float param_8 = cb.z;
            float param_9 = cs.z;
            b = float3(color_dodge(param_4, param_5), color_dodge(param_6, param_7), color_dodge(param_8, param_9));
            break;
        }
        case 7u:
        {
            float param_10 = cb.x;
            float param_11 = cs.x;
            float param_12 = cb.y;
            float param_13 = cs.y;
            float param_14 = cb.z;
            float param_15 = cs.z;
            b = float3(color_burn(param_10, param_11), color_burn(param_12, param_13), color_burn(param_14, param_15));
            break;
        }
        case 8u:
        {
            float3 param_16 = cb;
            float3 param_17 = cs;
            b = hard_light(param_16, param_17);
            break;
        }
        case 9u:
        {
            float3 param_18 = cb;
            float3 param_19 = cs;
            b = soft_light(param_18, param_19);
            break;
        }
        case 10u:
        {
            b = abs(cb - cs);
            break;
        }
        case 11u:
        {
            b = (cb + cs) - ((cb * 2.0f) * cs);
            break;
        }
        case 12u:
        {
            float3 param_20 = cb;
            float3 param_21 = cs;
            float param_22 = sat(param_20);
            float3 _1337 = set_sat(param_21, param_22);
            float3 param_23 = cb;
            float3 param_24 = _1337;
            float param_25 = lum(param_23);
            b = set_lum(param_24, param_25);
            break;
        }
        case 13u:
        {
            float3 param_26 = cs;
            float3 param_27 = cb;
            float param_28 = sat(param_26);
            float3 _1351 = set_sat(param_27, param_28);
            float3 param_29 = cb;
            float3 param_30 = _1351;
            float param_31 = lum(param_29);
            b = set_lum(param_30, param_31);
            break;
        }
        case 14u:
        {
            float3 param_32 = cb;
            float3 param_33 = cs;
            float param_34 = lum(param_32);
            b = set_lum(param_33, param_34);
            break;
        }
        case 15u:
        {
            float3 param_35 = cs;
            float3 param_36 = cb;
            float param_37 = lum(param_35);
            b = set_lum(param_36, param_37);
            break;
        }
        default:
        {
            b = cs;
            break;
        }
    }
    return b;
}

float4 mix_compose(float3 cb, float3 cs, float ab, float as, uint mode)
{
    float fa = 0.0f;
    float fb = 0.0f;
    switch (mode)
    {
        case 1u:
        {
            fa = 1.0f;
            fb = 0.0f;
            break;
        }
        case 2u:
        {
            fa = 0.0f;
            fb = 1.0f;
            break;
        }
        case 3u:
        {
            fa = 1.0f;
            fb = 1.0f - as;
            break;
        }
        case 4u:
        {
            fa = 1.0f - ab;
            fb = 1.0f;
            break;
        }
        case 5u:
        {
            fa = ab;
            fb = 0.0f;
            break;
        }
        case 6u:
        {
            fa = 0.0f;
            fb = as;
            break;
        }
        case 7u:
        {
            fa = 1.0f - ab;
            fb = 0.0f;
            break;
        }
        case 8u:
        {
            fa = 0.0f;
            fb = 1.0f - as;
            break;
        }
        case 9u:
        {
            fa = ab;
            fb = 1.0f - as;
            break;
        }
        case 10u:
        {
            fa = 1.0f - ab;
            fb = as;
            break;
        }
        case 11u:
        {
            fa = 1.0f - ab;
            fb = 1.0f - as;
            break;
        }
        case 12u:
        {
            fa = 1.0f;
            fb = 1.0f;
            break;
        }
        case 13u:
        {
            return float4(max(0.0f.xxxx, ((1.0f.xxxx - (float4(cs, as) * as)) + 1.0f.xxxx) - (float4(cb, ab) * ab)).xyz, max(0.0f, ((1.0f - as) + 1.0f) - ab));
        }
        case 14u:
        {
            return float4(min(1.0f.xxxx, (float4(cs, as) * as) + (float4(cb, ab) * ab)).xyz, min(1.0f, as + ab));
        }
        default:
        {
            break;
        }
    }
    return (float4(cs, as) * (as * fa)) + (float4(cb, ab) * (ab * fb));
}

CmdJump CmdJump_read(Alloc a, CmdJumpRef ref)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint raw0 = read_mem(param, param_1);
    CmdJump s;
    s.new_ref = raw0;
    return s;
}

CmdJump Cmd_Jump_read(Alloc a, CmdRef ref)
{
    CmdJumpRef _749 = { ref.offset + 4u };
    Alloc param = a;
    CmdJumpRef param_1 = _749;
    return CmdJump_read(param, param_1);
}

void comp_main()
{
    uint tile_ix = (gl_WorkGroupID.y * _1666.Load(8)) + gl_WorkGroupID.x;
    Alloc _1681;
    _1681.offset = _1666.Load(24);
    Alloc param;
    param.offset = _1681.offset;
    uint param_1 = tile_ix * 1024u;
    uint param_2 = 1024u;
    Alloc cmd_alloc = slice_mem(param, param_1, param_2);
    CmdRef _1690 = { cmd_alloc.offset };
    CmdRef cmd_ref = _1690;
    uint2 xy_uint = uint2(gl_LocalInvocationID.x + (16u * gl_WorkGroupID.x), gl_LocalInvocationID.y + (16u * gl_WorkGroupID.y));
    float2 xy = float2(xy_uint);
    float4 rgba[8];
    for (uint i = 0u; i < 8u; i++)
    {
        rgba[i] = 0.0f.xxxx;
    }
    uint clip_depth = 0u;
    bool mem_ok = _291.Load(4) == 0u;
    float df[8];
    TileSegRef tile_seg_ref;
    float area[8];
    uint blend_stack[128][8];
    while (mem_ok)
    {
        Alloc param_3 = cmd_alloc;
        CmdRef param_4 = cmd_ref;
        uint tag = Cmd_tag(param_3, param_4).tag;
        if (tag == 0u)
        {
            break;
        }
        switch (tag)
        {
            case 2u:
            {
                Alloc param_5 = cmd_alloc;
                CmdRef param_6 = cmd_ref;
                CmdStroke stroke = Cmd_Stroke_read(param_5, param_6);
                for (uint k = 0u; k < 8u; k++)
                {
                    df[k] = 1000000000.0f;
                }
                TileSegRef _1784 = { stroke.tile_ref };
                tile_seg_ref = _1784;
                do
                {
                    uint param_7 = tile_seg_ref.offset;
                    uint param_8 = 24u;
                    bool param_9 = mem_ok;
                    Alloc param_10 = new_alloc(param_7, param_8, param_9);
                    TileSegRef param_11 = tile_seg_ref;
                    TileSeg seg = TileSeg_read(param_10, param_11);
                    float2 line_vec = seg._vector;
                    for (uint k_1 = 0u; k_1 < 8u; k_1++)
                    {
                        float2 dpos = (xy + 0.5f.xx) - seg.origin;
                        uint param_12 = k_1;
                        dpos += float2(chunk_offset(param_12));
                        float t = clamp(dot(line_vec, dpos) / dot(line_vec, line_vec), 0.0f, 1.0f);
                        df[k_1] = min(df[k_1], length((line_vec * t) - dpos));
                    }
                    tile_seg_ref = seg.next;
                } while (tile_seg_ref.offset != 0u);
                for (uint k_2 = 0u; k_2 < 8u; k_2++)
                {
                    area[k_2] = clamp((stroke.half_width + 0.5f) - df[k_2], 0.0f, 1.0f);
                }
                cmd_ref.offset += 12u;
                break;
            }
            case 1u:
            {
                Alloc param_13 = cmd_alloc;
                CmdRef param_14 = cmd_ref;
                CmdFill fill = Cmd_Fill_read(param_13, param_14);
                for (uint k_3 = 0u; k_3 < 8u; k_3++)
                {
                    area[k_3] = float(fill.backdrop);
                }
                TileSegRef _1904 = { fill.tile_ref };
                tile_seg_ref = _1904;
                do
                {
                    uint param_15 = tile_seg_ref.offset;
                    uint param_16 = 24u;
                    bool param_17 = mem_ok;
                    Alloc param_18 = new_alloc(param_15, param_16, param_17);
                    TileSegRef param_19 = tile_seg_ref;
                    TileSeg seg_1 = TileSeg_read(param_18, param_19);
                    for (uint k_4 = 0u; k_4 < 8u; k_4++)
                    {
                        uint param_20 = k_4;
                        float2 my_xy = xy + float2(chunk_offset(param_20));
                        float2 start = seg_1.origin - my_xy;
                        float2 end = start + seg_1._vector;
                        float2 window = clamp(float2(start.y, end.y), 0.0f.xx, 1.0f.xx);
                        if (window.x != window.y)
                        {
                            float2 t_1 = (window - start.y.xx) / seg_1._vector.y.xx;
                            float2 xs = float2(lerp(start.x, end.x, t_1.x), lerp(start.x, end.x, t_1.y));
                            float xmin = min(min(xs.x, xs.y), 1.0f) - 9.9999999747524270787835121154785e-07f;
                            float xmax = max(xs.x, xs.y);
                            float b = min(xmax, 1.0f);
                            float c = max(b, 0.0f);
                            float d = max(xmin, 0.0f);
                            float a = ((b + (0.5f * ((d * d) - (c * c)))) - xmin) / (xmax - xmin);
                            area[k_4] += (a * (window.x - window.y));
                        }
                        area[k_4] += (sign(seg_1._vector.x) * clamp((my_xy.y - seg_1.y_edge) + 1.0f, 0.0f, 1.0f));
                    }
                    tile_seg_ref = seg_1.next;
                } while (tile_seg_ref.offset != 0u);
                for (uint k_5 = 0u; k_5 < 8u; k_5++)
                {
                    area[k_5] = min(abs(area[k_5]), 1.0f);
                }
                cmd_ref.offset += 12u;
                break;
            }
            case 3u:
            {
                for (uint k_6 = 0u; k_6 < 8u; k_6++)
                {
                    area[k_6] = 1.0f;
                }
                cmd_ref.offset += 4u;
                break;
            }
            case 4u:
            {
                Alloc param_21 = cmd_alloc;
                CmdRef param_22 = cmd_ref;
                CmdAlpha alpha = Cmd_Alpha_read(param_21, param_22);
                for (uint k_7 = 0u; k_7 < 8u; k_7++)
                {
                    area[k_7] = alpha.alpha;
                }
                cmd_ref.offset += 8u;
                break;
            }
            case 5u:
            {
                Alloc param_23 = cmd_alloc;
                CmdRef param_24 = cmd_ref;
                CmdColor color = Cmd_Color_read(param_23, param_24);
                uint param_25 = color.rgba_color;
                float4 fg = unpacksRGB(param_25);
                for (uint k_8 = 0u; k_8 < 8u; k_8++)
                {
                    float4 fg_k = fg * area[k_8];
                    rgba[k_8] = (rgba[k_8] * (1.0f - fg_k.w)) + fg_k;
                }
                cmd_ref.offset += 8u;
                break;
            }
            case 6u:
            {
                Alloc param_26 = cmd_alloc;
                CmdRef param_27 = cmd_ref;
                CmdLinGrad lin = Cmd_LinGrad_read(param_26, param_27);
                float d_1 = ((lin.line_x * xy.x) + (lin.line_y * xy.y)) + lin.line_c;
                for (uint k_9 = 0u; k_9 < 8u; k_9++)
                {
                    uint param_28 = k_9;
                    float2 chunk_xy = float2(chunk_offset(param_28));
                    float my_d = (d_1 + (lin.line_x * chunk_xy.x)) + (lin.line_y * chunk_xy.y);
                    int x = int(round(clamp(my_d, 0.0f, 1.0f) * 511.0f));
                    float4 fg_rgba = gradients[int2(x, int(lin.index))];
                    float3 param_29 = fg_rgba.xyz;
                    float3 _2238 = fromsRGB(param_29);
                    fg_rgba.x = _2238.x;
                    fg_rgba.y = _2238.y;
                    fg_rgba.z = _2238.z;
                    float4 fg_k_1 = fg_rgba * area[k_9];
                    rgba[k_9] = (rgba[k_9] * (1.0f - fg_k_1.w)) + fg_k_1;
                }
                cmd_ref.offset += 20u;
                break;
            }
            case 7u:
            {
                Alloc param_30 = cmd_alloc;
                CmdRef param_31 = cmd_ref;
                CmdRadGrad rad = Cmd_RadGrad_read(param_30, param_31);
                for (uint k_10 = 0u; k_10 < 8u; k_10++)
                {
                    uint param_32 = k_10;
                    float2 my_xy_1 = xy + float2(chunk_offset(param_32));
                    my_xy_1 = ((rad.mat.xz * my_xy_1.x) + (rad.mat.yw * my_xy_1.y)) - rad.xlat;
                    float ba = dot(my_xy_1, rad.c1);
                    float ca = rad.ra * dot(my_xy_1, my_xy_1);
                    float t_2 = (sqrt((ba * ba) + ca) - ba) - rad.roff;
                    int x_1 = int(round(clamp(t_2, 0.0f, 1.0f) * 511.0f));
                    float4 fg_rgba_1 = gradients[int2(x_1, int(rad.index))];
                    float3 param_33 = fg_rgba_1.xyz;
                    float3 _2348 = fromsRGB(param_33);
                    fg_rgba_1.x = _2348.x;
                    fg_rgba_1.y = _2348.y;
                    fg_rgba_1.z = _2348.z;
                    float4 fg_k_2 = fg_rgba_1 * area[k_10];
                    rgba[k_10] = (rgba[k_10] * (1.0f - fg_k_2.w)) + fg_k_2;
                }
                cmd_ref.offset += 48u;
                break;
            }
            case 8u:
            {
                Alloc param_34 = cmd_alloc;
                CmdRef param_35 = cmd_ref;
                CmdImage fill_img = Cmd_Image_read(param_34, param_35);
                uint2 param_36 = xy_uint;
                CmdImage param_37 = fill_img;
                float4 _2391[8];
                fillImage(_2391, param_36, param_37);
                float4 img[8] = _2391;
                for (uint k_11 = 0u; k_11 < 8u; k_11++)
                {
                    float4 fg_k_3 = img[k_11] * area[k_11];
                    rgba[k_11] = (rgba[k_11] * (1.0f - fg_k_3.w)) + fg_k_3;
                }
                cmd_ref.offset += 12u;
                break;
            }
            case 9u:
            {
                for (uint k_12 = 0u; k_12 < 8u; k_12++)
                {
                    uint d_2 = min(clip_depth, 127u);
                    float4 param_38 = float4(rgba[k_12]);
                    uint _2454 = packsRGB(param_38);
                    blend_stack[d_2][k_12] = _2454;
                    rgba[k_12] = 0.0f.xxxx;
                }
                clip_depth++;
                cmd_ref.offset += 4u;
                break;
            }
            case 10u:
            {
                Alloc param_39 = cmd_alloc;
                CmdRef param_40 = cmd_ref;
                CmdEndClip end_clip = Cmd_EndClip_read(param_39, param_40);
                uint blend_mode = end_clip.blend >> uint(8);
                uint comp_mode = end_clip.blend & 255u;
                clip_depth--;
                for (uint k_13 = 0u; k_13 < 8u; k_13++)
                {
                    uint d_3 = min(clip_depth, 127u);
                    uint param_41 = blend_stack[d_3][k_13];
                    float4 bg = unpacksRGB(param_41);
                    float4 fg_1 = rgba[k_13] * area[k_13];
                    float3 param_42 = bg.xyz;
                    float3 param_43 = fg_1.xyz;
                    uint param_44 = blend_mode;
                    float3 blend = mix_blend(param_42, param_43, param_44);
                    float4 _2521 = fg_1;
                    float _2525 = fg_1.w;
                    float3 _2532 = lerp(_2521.xyz, blend, float((_2525 * bg.w) > 0.0f).xxx);
                    fg_1.x = _2532.x;
                    fg_1.y = _2532.y;
                    fg_1.z = _2532.z;
                    float3 param_45 = bg.xyz;
                    float3 param_46 = fg_1.xyz;
                    float param_47 = bg.w;
                    float param_48 = fg_1.w;
                    uint param_49 = comp_mode;
                    rgba[k_13] = mix_compose(param_45, param_46, param_47, param_48, param_49);
                }
                cmd_ref.offset += 8u;
                break;
            }
            case 11u:
            {
                Alloc param_50 = cmd_alloc;
                CmdRef param_51 = cmd_ref;
                CmdRef _2569 = { Cmd_Jump_read(param_50, param_51).new_ref };
                cmd_ref = _2569;
                cmd_alloc.offset = cmd_ref.offset;
                break;
            }
        }
    }
    for (uint i_1 = 0u; i_1 < 8u; i_1++)
    {
        uint param_52 = i_1;
        float3 param_53 = rgba[i_1].xyz;
        image[int2(xy_uint + chunk_offset(param_52))] = float4(tosRGB(param_53), rgba[i_1].w);
    }
}

[numthreads(8, 4, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    comp_main();
}
