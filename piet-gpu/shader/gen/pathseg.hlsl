struct Alloc
{
    uint offset;
};

struct TagMonoid
{
    uint trans_ix;
    uint linewidth_ix;
    uint pathseg_ix;
    uint path_ix;
    uint pathseg_offset;
};

struct TransformSegRef
{
    uint offset;
};

struct TransformSeg
{
    float4 mat;
    float2 translate;
};

struct PathCubicRef
{
    uint offset;
};

struct PathCubic
{
    float2 p0;
    float2 p1;
    float2 p2;
    float2 p3;
    uint path_ix;
    uint trans_ix;
    float2 stroke;
};

struct PathSegRef
{
    uint offset;
};

struct Monoid
{
    float4 bbox;
    uint flags;
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
    uint n_trans;
    uint trans_offset;
    uint pathtag_offset;
    uint linewidth_offset;
    uint pathseg_offset;
};

static const uint3 gl_WorkGroupSize = uint3(512u, 1u, 1u);

static const TagMonoid _135 = { 0u, 0u, 0u, 0u, 0u };
static const Monoid _567 = { 0.0f.xxxx, 0u };

RWByteAddressBuffer _111 : register(u0);
ByteAddressBuffer _574 : register(t2);
ByteAddressBuffer _639 : register(t1);
ByteAddressBuffer _710 : register(t3);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared TagMonoid sh_tag[512];
groupshared Monoid sh_scratch[512];

TagMonoid reduce_tag(uint tag_word)
{
    uint point_count = tag_word & 50529027u;
    TagMonoid c;
    c.pathseg_ix = uint(int(countbits((point_count * 7u) & 67372036u)));
    c.linewidth_ix = uint(int(countbits(tag_word & 1077952576u)));
    c.path_ix = uint(int(countbits(tag_word & 269488144u)));
    c.trans_ix = uint(int(countbits(tag_word & 538976288u)));
    uint n_points = point_count + ((tag_word >> uint(2)) & 16843009u);
    uint a = n_points + (n_points & (((tag_word >> uint(3)) & 16843009u) * 15u));
    a += (a >> uint(8));
    a += (a >> uint(16));
    c.pathseg_offset = a & 255u;
    return c;
}

TagMonoid combine_tag_monoid(TagMonoid a, TagMonoid b)
{
    TagMonoid c;
    c.trans_ix = a.trans_ix + b.trans_ix;
    c.linewidth_ix = a.linewidth_ix + b.linewidth_ix;
    c.pathseg_ix = a.pathseg_ix + b.pathseg_ix;
    c.path_ix = a.path_ix + b.path_ix;
    c.pathseg_offset = a.pathseg_offset + b.pathseg_offset;
    return c;
}

TagMonoid tag_monoid_identity()
{
    return _135;
}

float2 read_f32_point(uint ix)
{
    float x = asfloat(_574.Load(ix * 4 + 0));
    float y = asfloat(_574.Load((ix + 1u) * 4 + 0));
    return float2(x, y);
}

float2 read_i16_point(uint ix)
{
    uint raw = _574.Load(ix * 4 + 0);
    float x = float(int(raw << uint(16)) >> 16);
    float y = float(int(raw) >> 16);
    return float2(x, y);
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
    uint v = _111.Load(offset * 4 + 8);
    return v;
}

TransformSeg TransformSeg_read(Alloc a, TransformSegRef ref)
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
    TransformSeg s;
    s.mat = float4(asfloat(raw0), asfloat(raw1), asfloat(raw2), asfloat(raw3));
    s.translate = float2(asfloat(raw4), asfloat(raw5));
    return s;
}

void write_mem(Alloc alloc, uint offset, uint val)
{
    Alloc param = alloc;
    uint param_1 = offset;
    if (!touch_mem(param, param_1))
    {
        return;
    }
    _111.Store(offset * 4 + 8, val);
}

void PathCubic_write(Alloc a, PathCubicRef ref, PathCubic s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = asuint(s.p0.x);
    write_mem(param, param_1, param_2);
    Alloc param_3 = a;
    uint param_4 = ix + 1u;
    uint param_5 = asuint(s.p0.y);
    write_mem(param_3, param_4, param_5);
    Alloc param_6 = a;
    uint param_7 = ix + 2u;
    uint param_8 = asuint(s.p1.x);
    write_mem(param_6, param_7, param_8);
    Alloc param_9 = a;
    uint param_10 = ix + 3u;
    uint param_11 = asuint(s.p1.y);
    write_mem(param_9, param_10, param_11);
    Alloc param_12 = a;
    uint param_13 = ix + 4u;
    uint param_14 = asuint(s.p2.x);
    write_mem(param_12, param_13, param_14);
    Alloc param_15 = a;
    uint param_16 = ix + 5u;
    uint param_17 = asuint(s.p2.y);
    write_mem(param_15, param_16, param_17);
    Alloc param_18 = a;
    uint param_19 = ix + 6u;
    uint param_20 = asuint(s.p3.x);
    write_mem(param_18, param_19, param_20);
    Alloc param_21 = a;
    uint param_22 = ix + 7u;
    uint param_23 = asuint(s.p3.y);
    write_mem(param_21, param_22, param_23);
    Alloc param_24 = a;
    uint param_25 = ix + 8u;
    uint param_26 = s.path_ix;
    write_mem(param_24, param_25, param_26);
    Alloc param_27 = a;
    uint param_28 = ix + 9u;
    uint param_29 = s.trans_ix;
    write_mem(param_27, param_28, param_29);
    Alloc param_30 = a;
    uint param_31 = ix + 10u;
    uint param_32 = asuint(s.stroke.x);
    write_mem(param_30, param_31, param_32);
    Alloc param_33 = a;
    uint param_34 = ix + 11u;
    uint param_35 = asuint(s.stroke.y);
    write_mem(param_33, param_34, param_35);
}

void PathSeg_Cubic_write(Alloc a, PathSegRef ref, uint flags, PathCubic s)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint param_2 = (flags << uint(16)) | 1u;
    write_mem(param, param_1, param_2);
    PathCubicRef _458 = { ref.offset + 4u };
    Alloc param_3 = a;
    PathCubicRef param_4 = _458;
    PathCubic param_5 = s;
    PathCubic_write(param_3, param_4, param_5);
}

Monoid combine_monoid(Monoid a, Monoid b)
{
    Monoid c;
    c.bbox = b.bbox;
    bool _472 = (a.flags & 1u) == 0u;
    bool _480;
    if (_472)
    {
        _480 = b.bbox.z <= b.bbox.x;
    }
    else
    {
        _480 = _472;
    }
    bool _488;
    if (_480)
    {
        _488 = b.bbox.w <= b.bbox.y;
    }
    else
    {
        _488 = _480;
    }
    if (_488)
    {
        c.bbox = a.bbox;
    }
    else
    {
        bool _498 = (a.flags & 1u) == 0u;
        bool _505;
        if (_498)
        {
            _505 = (b.flags & 2u) == 0u;
        }
        else
        {
            _505 = _498;
        }
        bool _522;
        if (_505)
        {
            bool _512 = a.bbox.z > a.bbox.x;
            bool _521;
            if (!_512)
            {
                _521 = a.bbox.w > a.bbox.y;
            }
            else
            {
                _521 = _512;
            }
            _522 = _521;
        }
        else
        {
            _522 = _505;
        }
        if (_522)
        {
            float4 _529 = c.bbox;
            float2 _531 = min(a.bbox.xy, _529.xy);
            c.bbox.x = _531.x;
            c.bbox.y = _531.y;
            float4 _540 = c.bbox;
            float2 _542 = max(a.bbox.zw, _540.zw);
            c.bbox.z = _542.x;
            c.bbox.w = _542.y;
        }
    }
    c.flags = (a.flags & 2u) | b.flags;
    c.flags |= ((a.flags & 1u) << uint(1));
    return c;
}

Monoid monoid_identity()
{
    return _567;
}

uint round_down(float x)
{
    return uint(max(0.0f, floor(x) + 32768.0f));
}

uint round_up(float x)
{
    return uint(min(65535.0f, ceil(x) + 32768.0f));
}

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x * 4u;
    uint tag_word = _574.Load(((_639.Load(52) >> uint(2)) + (ix >> uint(2))) * 4 + 0);
    uint param = tag_word;
    TagMonoid local_tm = reduce_tag(param);
    sh_tag[gl_LocalInvocationID.x] = local_tm;
    for (uint i = 0u; i < 9u; i++)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gl_LocalInvocationID.x >= (1u << i))
        {
            TagMonoid other = sh_tag[gl_LocalInvocationID.x - (1u << i)];
            TagMonoid param_1 = other;
            TagMonoid param_2 = local_tm;
            local_tm = combine_tag_monoid(param_1, param_2);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_tag[gl_LocalInvocationID.x] = local_tm;
    }
    GroupMemoryBarrierWithGroupSync();
    TagMonoid tm = tag_monoid_identity();
    if (gl_WorkGroupID.x > 0u)
    {
        TagMonoid _716;
        _716.trans_ix = _710.Load((gl_WorkGroupID.x - 1u) * 20 + 0);
        _716.linewidth_ix = _710.Load((gl_WorkGroupID.x - 1u) * 20 + 4);
        _716.pathseg_ix = _710.Load((gl_WorkGroupID.x - 1u) * 20 + 8);
        _716.path_ix = _710.Load((gl_WorkGroupID.x - 1u) * 20 + 12);
        _716.pathseg_offset = _710.Load((gl_WorkGroupID.x - 1u) * 20 + 16);
        tm.trans_ix = _716.trans_ix;
        tm.linewidth_ix = _716.linewidth_ix;
        tm.pathseg_ix = _716.pathseg_ix;
        tm.path_ix = _716.path_ix;
        tm.pathseg_offset = _716.pathseg_offset;
    }
    if (gl_LocalInvocationID.x > 0u)
    {
        TagMonoid param_3 = tm;
        TagMonoid param_4 = sh_tag[gl_LocalInvocationID.x - 1u];
        tm = combine_tag_monoid(param_3, param_4);
    }
    uint ps_ix = (_639.Load(60) >> uint(2)) + tm.pathseg_offset;
    uint lw_ix = (_639.Load(56) >> uint(2)) + tm.linewidth_ix;
    uint save_path_ix = tm.path_ix;
    TransformSegRef _769 = { _639.Load(36) + (tm.trans_ix * 24u) };
    TransformSegRef trans_ref = _769;
    PathSegRef _779 = { _639.Load(28) + (tm.pathseg_ix * 52u) };
    PathSegRef ps_ref = _779;
    float2 p0;
    float2 p1;
    float2 p2;
    float2 p3;
    Alloc param_13;
    Monoid local[4];
    PathCubic cubic;
    Alloc param_15;
    for (uint i_1 = 0u; i_1 < 4u; i_1++)
    {
        uint tag_byte = tag_word >> (i_1 * 8u);
        uint seg_type = tag_byte & 3u;
        if (seg_type != 0u)
        {
            if ((tag_byte & 8u) != 0u)
            {
                uint param_5 = ps_ix;
                p0 = read_f32_point(param_5);
                uint param_6 = ps_ix + 2u;
                p1 = read_f32_point(param_6);
                if (seg_type >= 2u)
                {
                    uint param_7 = ps_ix + 4u;
                    p2 = read_f32_point(param_7);
                    if (seg_type == 3u)
                    {
                        uint param_8 = ps_ix + 6u;
                        p3 = read_f32_point(param_8);
                    }
                }
            }
            else
            {
                uint param_9 = ps_ix;
                p0 = read_i16_point(param_9);
                uint param_10 = ps_ix + 1u;
                p1 = read_i16_point(param_10);
                if (seg_type >= 2u)
                {
                    uint param_11 = ps_ix + 2u;
                    p2 = read_i16_point(param_11);
                    if (seg_type == 3u)
                    {
                        uint param_12 = ps_ix + 3u;
                        p3 = read_i16_point(param_12);
                    }
                }
            }
            float linewidth = asfloat(_574.Load(lw_ix * 4 + 0));
            Alloc _865;
            _865.offset = _639.Load(36);
            param_13.offset = _865.offset;
            TransformSegRef param_14 = trans_ref;
            TransformSeg transform = TransformSeg_read(param_13, param_14);
            p0 = ((transform.mat.xy * p0.x) + (transform.mat.zw * p0.y)) + transform.translate;
            p1 = ((transform.mat.xy * p1.x) + (transform.mat.zw * p1.y)) + transform.translate;
            float4 bbox = float4(min(p0, p1), max(p0, p1));
            if (seg_type >= 2u)
            {
                p2 = ((transform.mat.xy * p2.x) + (transform.mat.zw * p2.y)) + transform.translate;
                float4 _935 = bbox;
                float2 _938 = min(_935.xy, p2);
                bbox.x = _938.x;
                bbox.y = _938.y;
                float4 _943 = bbox;
                float2 _946 = max(_943.zw, p2);
                bbox.z = _946.x;
                bbox.w = _946.y;
                if (seg_type == 3u)
                {
                    p3 = ((transform.mat.xy * p3.x) + (transform.mat.zw * p3.y)) + transform.translate;
                    float4 _971 = bbox;
                    float2 _974 = min(_971.xy, p3);
                    bbox.x = _974.x;
                    bbox.y = _974.y;
                    float4 _979 = bbox;
                    float2 _982 = max(_979.zw, p3);
                    bbox.z = _982.x;
                    bbox.w = _982.y;
                }
                else
                {
                    p3 = p2;
                    p2 = lerp(p1, p2, 0.3333333432674407958984375f.xx);
                    p1 = lerp(p1, p0, 0.3333333432674407958984375f.xx);
                }
            }
            else
            {
                p3 = p1;
                p2 = lerp(p3, p0, 0.3333333432674407958984375f.xx);
                p1 = lerp(p0, p3, 0.3333333432674407958984375f.xx);
            }
            float2 stroke = 0.0f.xx;
            if (linewidth >= 0.0f)
            {
                stroke = float2(length(transform.mat.xz), length(transform.mat.yw)) * (0.5f * linewidth);
                bbox += float4(-stroke, stroke);
            }
            local[i_1].bbox = bbox;
            local[i_1].flags = 0u;
            cubic.p0 = p0;
            cubic.p1 = p1;
            cubic.p2 = p2;
            cubic.p3 = p3;
            cubic.path_ix = tm.path_ix;
            cubic.trans_ix = (gl_GlobalInvocationID.x * 4u) + i_1;
            cubic.stroke = stroke;
            uint fill_mode = uint(linewidth >= 0.0f);
            Alloc _1071;
            _1071.offset = _639.Load(28);
            param_15.offset = _1071.offset;
            PathSegRef param_16 = ps_ref;
            uint param_17 = fill_mode;
            PathCubic param_18 = cubic;
            PathSeg_Cubic_write(param_15, param_16, param_17, param_18);
            ps_ref.offset += 52u;
            uint n_points = (tag_byte & 3u) + ((tag_byte >> uint(2)) & 1u);
            uint n_words = n_points + (n_points & (((tag_byte >> uint(3)) & 1u) * 15u));
            ps_ix += n_words;
        }
        else
        {
            local[i_1].bbox = 0.0f.xxxx;
            uint is_path = (tag_byte >> uint(4)) & 1u;
            local[i_1].flags = is_path;
            tm.path_ix += is_path;
            trans_ref.offset += (((tag_byte >> uint(5)) & 1u) * 24u);
            lw_ix += ((tag_byte >> uint(6)) & 1u);
        }
    }
    Monoid agg = local[0];
    for (uint i_2 = 1u; i_2 < 4u; i_2++)
    {
        Monoid param_19 = agg;
        Monoid param_20 = local[i_2];
        agg = combine_monoid(param_19, param_20);
        local[i_2] = agg;
    }
    sh_scratch[gl_LocalInvocationID.x] = agg;
    for (uint i_3 = 0u; i_3 < 9u; i_3++)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gl_LocalInvocationID.x >= (1u << i_3))
        {
            Monoid other_1 = sh_scratch[gl_LocalInvocationID.x - (1u << i_3)];
            Monoid param_21 = other_1;
            Monoid param_22 = agg;
            agg = combine_monoid(param_21, param_22);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[gl_LocalInvocationID.x] = agg;
    }
    GroupMemoryBarrierWithGroupSync();
    uint path_ix = save_path_ix;
    uint bbox_out_ix = (_639.Load(40) >> uint(2)) + (path_ix * 4u);
    Monoid row = monoid_identity();
    if (gl_LocalInvocationID.x > 0u)
    {
        row = sh_scratch[gl_LocalInvocationID.x - 1u];
    }
    for (uint i_4 = 0u; i_4 < 4u; i_4++)
    {
        Monoid param_23 = row;
        Monoid param_24 = local[i_4];
        Monoid m = combine_monoid(param_23, param_24);
        bool do_atomic = false;
        bool _1241 = i_4 == 3u;
        bool _1248;
        if (_1241)
        {
            _1248 = gl_LocalInvocationID.x == 511u;
        }
        else
        {
            _1248 = _1241;
        }
        if (_1248)
        {
            do_atomic = true;
        }
        if ((m.flags & 1u) != 0u)
        {
            if ((m.flags & 2u) == 0u)
            {
                do_atomic = true;
            }
            else
            {
                float param_25 = m.bbox.x;
                _111.Store(bbox_out_ix * 4 + 8, round_down(param_25));
                float param_26 = m.bbox.y;
                _111.Store((bbox_out_ix + 1u) * 4 + 8, round_down(param_26));
                float param_27 = m.bbox.z;
                _111.Store((bbox_out_ix + 2u) * 4 + 8, round_up(param_27));
                float param_28 = m.bbox.w;
                _111.Store((bbox_out_ix + 3u) * 4 + 8, round_up(param_28));
                bbox_out_ix += 4u;
                do_atomic = false;
            }
        }
        if (do_atomic)
        {
            bool _1300 = m.bbox.z > m.bbox.x;
            bool _1309;
            if (!_1300)
            {
                _1309 = m.bbox.w > m.bbox.y;
            }
            else
            {
                _1309 = _1300;
            }
            if (_1309)
            {
                float param_29 = m.bbox.x;
                uint _1318;
                _111.InterlockedMin(bbox_out_ix * 4 + 8, round_down(param_29), _1318);
                float param_30 = m.bbox.y;
                uint _1326;
                _111.InterlockedMin((bbox_out_ix + 1u) * 4 + 8, round_down(param_30), _1326);
                float param_31 = m.bbox.z;
                uint _1334;
                _111.InterlockedMax((bbox_out_ix + 2u) * 4 + 8, round_up(param_31), _1334);
                float param_32 = m.bbox.w;
                uint _1342;
                _111.InterlockedMax((bbox_out_ix + 3u) * 4 + 8, round_up(param_32), _1342);
            }
            bbox_out_ix += 4u;
        }
    }
}

[numthreads(512, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
