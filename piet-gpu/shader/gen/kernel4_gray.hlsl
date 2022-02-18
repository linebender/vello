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

static const uint3 gl_WorkGroupSize = uint3(8u, 4u, 1u);

RWByteAddressBuffer _202 : register(u0, space0);
ByteAddressBuffer _723 : register(t1, space0);
RWTexture2D<unorm float4> image_atlas : register(u3, space0);
RWTexture2D<unorm float4> gradients : register(u4, space0);
RWTexture2D<unorm float> image : register(u2, space0);

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
    Alloc _215 = { a.offset + offset };
    return _215;
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
    uint v = _202.Load(offset * 4 + 8);
    return v;
}

CmdTag Cmd_tag(Alloc a, CmdRef ref)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint tag_and_flags = read_mem(param, param_1);
    CmdTag _432 = { tag_and_flags & 65535u, tag_and_flags >> uint(16) };
    return _432;
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
    CmdStrokeRef _449 = { ref.offset + 4u };
    Alloc param = a;
    CmdStrokeRef param_1 = _449;
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
    TileSegRef _572 = { raw5 };
    s.next = _572;
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
    CmdFillRef _439 = { ref.offset + 4u };
    Alloc param = a;
    CmdFillRef param_1 = _439;
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
    CmdAlphaRef _459 = { ref.offset + 4u };
    Alloc param = a;
    CmdAlphaRef param_1 = _459;
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
    CmdColorRef _469 = { ref.offset + 4u };
    Alloc param = a;
    CmdColorRef param_1 = _469;
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
    CmdLinGradRef _479 = { ref.offset + 4u };
    Alloc param = a;
    CmdLinGradRef param_1 = _479;
    return CmdLinGrad_read(param, param_1);
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
    CmdImageRef _489 = { ref.offset + 4u };
    Alloc param = a;
    CmdImageRef param_1 = _489;
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
        float3 _695 = fromsRGB(param_1);
        fg_rgba.x = _695.x;
        fg_rgba.y = _695.y;
        fg_rgba.z = _695.z;
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
    CmdJumpRef _499 = { ref.offset + 4u };
    Alloc param = a;
    CmdJumpRef param_1 = _499;
    return CmdJump_read(param, param_1);
}

void comp_main()
{
    uint tile_ix = (gl_WorkGroupID.y * _723.Load(8)) + gl_WorkGroupID.x;
    Alloc _738;
    _738.offset = _723.Load(24);
    Alloc param;
    param.offset = _738.offset;
    uint param_1 = tile_ix * 1024u;
    uint param_2 = 1024u;
    Alloc cmd_alloc = slice_mem(param, param_1, param_2);
    CmdRef _747 = { cmd_alloc.offset };
    CmdRef cmd_ref = _747;
    uint2 xy_uint = uint2(gl_LocalInvocationID.x + (16u * gl_WorkGroupID.x), gl_LocalInvocationID.y + (16u * gl_WorkGroupID.y));
    float2 xy = float2(xy_uint);
    float4 rgba[8];
    for (uint i = 0u; i < 8u; i++)
    {
        rgba[i] = 0.0f.xxxx;
    }
    uint clip_depth = 0u;
    bool mem_ok = _202.Load(4) == 0u;
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
                TileSegRef _842 = { stroke.tile_ref };
                tile_seg_ref = _842;
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
                TileSegRef _964 = { fill.tile_ref };
                tile_seg_ref = _964;
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
                    float3 _1298 = fromsRGB(param_29);
                    fg_rgba.x = _1298.x;
                    fg_rgba.y = _1298.y;
                    fg_rgba.z = _1298.z;
                    rgba[k_9] = fg_rgba;
                }
                cmd_ref.offset += 20u;
                break;
            }
            case 7u:
            {
                Alloc param_30 = cmd_alloc;
                CmdRef param_31 = cmd_ref;
                CmdImage fill_img = Cmd_Image_read(param_30, param_31);
                uint2 param_32 = xy_uint;
                CmdImage param_33 = fill_img;
                float4 _1327[8];
                fillImage(_1327, param_32, param_33);
                float4 img[8] = _1327;
                for (uint k_10 = 0u; k_10 < 8u; k_10++)
                {
                    float4 fg_k_1 = img[k_10] * area[k_10];
                    rgba[k_10] = (rgba[k_10] * (1.0f - fg_k_1.w)) + fg_k_1;
                }
                cmd_ref.offset += 12u;
                break;
            }
            case 8u:
            {
                for (uint k_11 = 0u; k_11 < 8u; k_11++)
                {
                    uint d_2 = min(clip_depth, 127u);
                    float4 param_34 = float4(rgba[k_11]);
                    uint _1390 = packsRGB(param_34);
                    blend_stack[d_2][k_11] = _1390;
                    rgba[k_11] = 0.0f.xxxx;
                }
                clip_depth++;
                cmd_ref.offset += 4u;
                break;
            }
            case 9u:
            {
                clip_depth--;
                for (uint k_12 = 0u; k_12 < 8u; k_12++)
                {
                    uint d_3 = min(clip_depth, 127u);
                    uint param_35 = blend_stack[d_3][k_12];
                    float4 bg = unpacksRGB(param_35);
                    float4 fg_1 = rgba[k_12] * area[k_12];
                    rgba[k_12] = (bg * (1.0f - fg_1.w)) + fg_1;
                }
                cmd_ref.offset += 4u;
                break;
            }
            case 10u:
            {
                Alloc param_36 = cmd_alloc;
                CmdRef param_37 = cmd_ref;
                CmdRef _1453 = { Cmd_Jump_read(param_36, param_37).new_ref };
                cmd_ref = _1453;
                cmd_alloc.offset = cmd_ref.offset;
                break;
            }
        }
    }
    for (uint i_1 = 0u; i_1 < 8u; i_1++)
    {
        uint param_38 = i_1;
        image[int2(xy_uint + chunk_offset(param_38))] = rgba[i_1].w.x;
    }
}

[numthreads(8, 4, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    comp_main();
}
