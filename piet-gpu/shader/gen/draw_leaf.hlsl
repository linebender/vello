struct DrawMonoid
{
    uint path_ix;
    uint clip_ix;
    uint scene_offset;
    uint info_offset;
};

struct Alloc
{
    uint offset;
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

static const DrawMonoid _23 = { 0u, 0u, 0u, 0u };

ByteAddressBuffer _93 : register(t1, space0);
ByteAddressBuffer _103 : register(t2, space0);
ByteAddressBuffer _203 : register(t3, space0);
RWByteAddressBuffer _285 : register(u0, space0);

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

DrawMonoid map_tag(uint tag_word)
{
    uint has_path = uint(tag_word != 0u);
    DrawMonoid _76 = { has_path, tag_word & 1u, tag_word & 28u, (tag_word >> uint(4)) & 60u };
    return _76;
}

DrawMonoid combine_draw_monoid(DrawMonoid a, DrawMonoid b)
{
    DrawMonoid c;
    c.path_ix = a.path_ix + b.path_ix;
    c.clip_ix = a.clip_ix + b.clip_ix;
    c.scene_offset = a.scene_offset + b.scene_offset;
    c.info_offset = a.info_offset + b.info_offset;
    return c;
}

DrawMonoid draw_monoid_identity()
{
    return _23;
}

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x * 8u;
    uint drawtag_base = _93.Load(100) >> uint(2);
    uint tag_word = _103.Load((drawtag_base + ix) * 4 + 0);
    uint param = tag_word;
    DrawMonoid agg = map_tag(param);
    DrawMonoid local[8];
    local[0] = agg;
    for (uint i = 1u; i < 8u; i++)
    {
        tag_word = _103.Load(((drawtag_base + ix) + i) * 4 + 0);
        uint param_1 = tag_word;
        DrawMonoid param_2 = agg;
        DrawMonoid param_3 = map_tag(param_1);
        agg = combine_draw_monoid(param_2, param_3);
        local[i] = agg;
    }
    sh_scratch[gl_LocalInvocationID.x] = agg;
    for (uint i_1 = 0u; i_1 < 8u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gl_LocalInvocationID.x >= (1u << i_1))
        {
            DrawMonoid other = sh_scratch[gl_LocalInvocationID.x - (1u << i_1)];
            DrawMonoid param_4 = other;
            DrawMonoid param_5 = agg;
            agg = combine_draw_monoid(param_4, param_5);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[gl_LocalInvocationID.x] = agg;
    }
    GroupMemoryBarrierWithGroupSync();
    DrawMonoid row = draw_monoid_identity();
    if (gl_WorkGroupID.x > 0u)
    {
        DrawMonoid _209;
        _209.path_ix = _203.Load((gl_WorkGroupID.x - 1u) * 16 + 0);
        _209.clip_ix = _203.Load((gl_WorkGroupID.x - 1u) * 16 + 4);
        _209.scene_offset = _203.Load((gl_WorkGroupID.x - 1u) * 16 + 8);
        _209.info_offset = _203.Load((gl_WorkGroupID.x - 1u) * 16 + 12);
        row.path_ix = _209.path_ix;
        row.clip_ix = _209.clip_ix;
        row.scene_offset = _209.scene_offset;
        row.info_offset = _209.info_offset;
    }
    if (gl_LocalInvocationID.x > 0u)
    {
        DrawMonoid param_6 = row;
        DrawMonoid param_7 = sh_scratch[gl_LocalInvocationID.x - 1u];
        row = combine_draw_monoid(param_6, param_7);
    }
    uint drawdata_base = _93.Load(104) >> uint(2);
    uint drawinfo_base = _93.Load(68) >> uint(2);
    uint out_ix = gl_GlobalInvocationID.x * 8u;
    uint out_base = (_93.Load(44) >> uint(2)) + (out_ix * 4u);
    uint clip_out_base = _93.Load(48) >> uint(2);
    float4 mat;
    float2 translate;
    float2 p0;
    float2 p1;
    for (uint i_2 = 0u; i_2 < 8u; i_2++)
    {
        DrawMonoid m = row;
        if (i_2 > 0u)
        {
            DrawMonoid param_8 = m;
            DrawMonoid param_9 = local[i_2 - 1u];
            m = combine_draw_monoid(param_8, param_9);
        }
        _285.Store((out_base + (i_2 * 4u)) * 4 + 12, m.path_ix);
        _285.Store(((out_base + (i_2 * 4u)) + 1u) * 4 + 12, m.clip_ix);
        _285.Store(((out_base + (i_2 * 4u)) + 2u) * 4 + 12, m.scene_offset);
        _285.Store(((out_base + (i_2 * 4u)) + 3u) * 4 + 12, m.info_offset);
        uint dd = drawdata_base + (m.scene_offset >> uint(2));
        uint di = drawinfo_base + (m.info_offset >> uint(2));
        tag_word = _103.Load(((drawtag_base + ix) + i_2) * 4 + 0);
        if (((((tag_word == 68u) || (tag_word == 276u)) || (tag_word == 732u)) || (tag_word == 72u)) || (tag_word == 5u))
        {
            uint bbox_offset = (_93.Load(40) >> uint(2)) + (6u * m.path_ix);
            float bbox_l = float(_285.Load(bbox_offset * 4 + 12)) - 32768.0f;
            float bbox_t = float(_285.Load((bbox_offset + 1u) * 4 + 12)) - 32768.0f;
            float bbox_r = float(_285.Load((bbox_offset + 2u) * 4 + 12)) - 32768.0f;
            float bbox_b = float(_285.Load((bbox_offset + 3u) * 4 + 12)) - 32768.0f;
            float4 bbox = float4(bbox_l, bbox_t, bbox_r, bbox_b);
            float linewidth = asfloat(_285.Load((bbox_offset + 4u) * 4 + 12));
            uint fill_mode = uint(linewidth >= 0.0f);
            if (((linewidth >= 0.0f) || (tag_word == 276u)) || (tag_word == 732u))
            {
                uint trans_ix = _285.Load((bbox_offset + 5u) * 4 + 12);
                uint t = (_93.Load(84) >> uint(2)) + (trans_ix * 6u);
                mat = asfloat(uint4(_103.Load(t * 4 + 0), _103.Load((t + 1u) * 4 + 0), _103.Load((t + 2u) * 4 + 0), _103.Load((t + 3u) * 4 + 0)));
                if ((tag_word == 276u) || (tag_word == 732u))
                {
                    translate = asfloat(uint2(_103.Load((t + 4u) * 4 + 0), _103.Load((t + 5u) * 4 + 0)));
                }
            }
            if (linewidth >= 0.0f)
            {
                linewidth *= sqrt(abs((mat.x * mat.w) - (mat.y * mat.z)));
            }
            switch (tag_word)
            {
                case 68u:
                case 72u:
                {
                    _285.Store(di * 4 + 12, asuint(linewidth));
                    break;
                }
                case 276u:
                {
                    _285.Store(di * 4 + 12, asuint(linewidth));
                    p0 = asfloat(uint2(_103.Load((dd + 1u) * 4 + 0), _103.Load((dd + 2u) * 4 + 0)));
                    p1 = asfloat(uint2(_103.Load((dd + 3u) * 4 + 0), _103.Load((dd + 4u) * 4 + 0)));
                    p0 = ((mat.xy * p0.x) + (mat.zw * p0.y)) + translate;
                    p1 = ((mat.xy * p1.x) + (mat.zw * p1.y)) + translate;
                    float2 dxy = p1 - p0;
                    float scale = 1.0f / ((dxy.x * dxy.x) + (dxy.y * dxy.y));
                    float line_x = dxy.x * scale;
                    float line_y = dxy.y * scale;
                    float line_c = -((p0.x * line_x) + (p0.y * line_y));
                    _285.Store((di + 1u) * 4 + 12, asuint(line_x));
                    _285.Store((di + 2u) * 4 + 12, asuint(line_y));
                    _285.Store((di + 3u) * 4 + 12, asuint(line_c));
                    break;
                }
                case 732u:
                {
                    p0 = asfloat(uint2(_103.Load((dd + 1u) * 4 + 0), _103.Load((dd + 2u) * 4 + 0)));
                    p1 = asfloat(uint2(_103.Load((dd + 3u) * 4 + 0), _103.Load((dd + 4u) * 4 + 0)));
                    float r0 = asfloat(_103.Load((dd + 5u) * 4 + 0));
                    float r1 = asfloat(_103.Load((dd + 6u) * 4 + 0));
                    float inv_det = 1.0f / ((mat.x * mat.w) - (mat.y * mat.z));
                    float4 inv_mat = float4(mat.w, -mat.y, -mat.z, mat.x) * inv_det;
                    float2 inv_tr = (inv_mat.xz * translate.x) + (inv_mat.yw * translate.y);
                    inv_tr += p0;
                    float2 center1 = p1 - p0;
                    float rr = r1 / (r1 - r0);
                    float rainv = rr / ((r1 * r1) - dot(center1, center1));
                    float2 c1 = center1 * rainv;
                    float ra = rr * rainv;
                    float roff = rr - 1.0f;
                    _285.Store(di * 4 + 12, asuint(linewidth));
                    _285.Store((di + 1u) * 4 + 12, asuint(inv_mat.x));
                    _285.Store((di + 2u) * 4 + 12, asuint(inv_mat.y));
                    _285.Store((di + 3u) * 4 + 12, asuint(inv_mat.z));
                    _285.Store((di + 4u) * 4 + 12, asuint(inv_mat.w));
                    _285.Store((di + 5u) * 4 + 12, asuint(inv_tr.x));
                    _285.Store((di + 6u) * 4 + 12, asuint(inv_tr.y));
                    _285.Store((di + 7u) * 4 + 12, asuint(c1.x));
                    _285.Store((di + 8u) * 4 + 12, asuint(c1.y));
                    _285.Store((di + 9u) * 4 + 12, asuint(ra));
                    _285.Store((di + 10u) * 4 + 12, asuint(roff));
                    break;
                }
                case 5u:
                {
                    break;
                }
            }
        }
        if ((tag_word == 5u) || (tag_word == 37u))
        {
            uint path_ix = ~(out_ix + i_2);
            if (tag_word == 5u)
            {
                path_ix = m.path_ix;
            }
            _285.Store((clip_out_base + m.clip_ix) * 4 + 12, path_ix);
        }
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
