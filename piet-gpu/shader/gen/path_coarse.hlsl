struct Alloc
{
    uint offset;
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

struct PathSegTag
{
    uint tag;
    uint flags;
};

struct TileRef
{
    uint offset;
};

struct PathRef
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

struct TileSeg
{
    float2 origin;
    float2 _vector;
    float y_edge;
    TileSegRef next;
};

struct SubdivResult
{
    float val;
    float a0;
    float a2;
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

static const uint3 gl_WorkGroupSize = uint3(32u, 1u, 1u);

static const PathSegTag _722 = { 0u, 0u };

RWByteAddressBuffer _143 : register(u0, space0);
ByteAddressBuffer _711 : register(t1, space0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

static bool mem_ok;

bool check_deps(uint dep_stage)
{
    uint _149;
    _143.InterlockedOr(4, 0u, _149);
    return (_149 & dep_stage) == 0u;
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
    uint v = _143.Load(offset * 4 + 12);
    return v;
}

PathSegTag PathSeg_tag(Alloc a, PathSegRef ref)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint tag_and_flags = read_mem(param, param_1);
    PathSegTag _362 = { tag_and_flags & 65535u, tag_and_flags >> uint(16) };
    return _362;
}

PathCubic PathCubic_read(Alloc a, PathCubicRef ref)
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
    Alloc param_22 = a;
    uint param_23 = ix + 11u;
    uint raw11 = read_mem(param_22, param_23);
    PathCubic s;
    s.p0 = float2(asfloat(raw0), asfloat(raw1));
    s.p1 = float2(asfloat(raw2), asfloat(raw3));
    s.p2 = float2(asfloat(raw4), asfloat(raw5));
    s.p3 = float2(asfloat(raw6), asfloat(raw7));
    s.path_ix = raw8;
    s.trans_ix = raw9;
    s.stroke = float2(asfloat(raw10), asfloat(raw11));
    return s;
}

PathCubic PathSeg_Cubic_read(Alloc a, PathSegRef ref)
{
    PathCubicRef _368 = { ref.offset + 4u };
    Alloc param = a;
    PathCubicRef param_1 = _368;
    return PathCubic_read(param, param_1);
}

float2 eval_cubic(float2 p0, float2 p1, float2 p2, float2 p3, float t)
{
    float mt = 1.0f - t;
    return (p0 * ((mt * mt) * mt)) + (((p1 * ((mt * mt) * 3.0f)) + (((p2 * (mt * 3.0f)) + (p3 * t)) * t)) * t);
}

float approx_parabola_integral(float x)
{
    return x * rsqrt(sqrt(0.3300000131130218505859375f + (0.201511204242706298828125f + ((0.25f * x) * x))));
}

SubdivResult estimate_subdiv(float2 p0, float2 p1, float2 p2, float sqrt_tol)
{
    float2 d01 = p1 - p0;
    float2 d12 = p2 - p1;
    float2 dd = d01 - d12;
    float _cross = ((p2.x - p0.x) * dd.y) - ((p2.y - p0.y) * dd.x);
    float x0 = ((d01.x * dd.x) + (d01.y * dd.y)) / _cross;
    float x2 = ((d12.x * dd.x) + (d12.y * dd.y)) / _cross;
    float scale = abs(_cross / (length(dd) * (x2 - x0)));
    float param = x0;
    float a0 = approx_parabola_integral(param);
    float param_1 = x2;
    float a2 = approx_parabola_integral(param_1);
    float val = 0.0f;
    if (scale < 1000000000.0f)
    {
        float da = abs(a2 - a0);
        float sqrt_scale = sqrt(scale);
        if (sign(x0) == sign(x2))
        {
            val = da * sqrt_scale;
        }
        else
        {
            float xmin = sqrt_tol / sqrt_scale;
            float param_2 = xmin;
            val = (sqrt_tol * da) / approx_parabola_integral(param_2);
        }
    }
    SubdivResult _690 = { val, a0, a2 };
    return _690;
}

uint fill_mode_from_flags(uint flags)
{
    return flags & 1u;
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
    TileRef _422 = { raw2 };
    s.tiles = _422;
    return s;
}

Alloc new_alloc(uint offset, uint size, bool mem_ok_1)
{
    Alloc a;
    a.offset = offset;
    return a;
}

float approx_parabola_inv_integral(float x)
{
    return x * sqrt(0.61000001430511474609375f + (0.1520999968051910400390625f + ((0.25f * x) * x)));
}

float2 eval_quad(float2 p0, float2 p1, float2 p2, float t)
{
    float mt = 1.0f - t;
    return (p0 * (mt * mt)) + (((p1 * (mt * 2.0f)) + (p2 * t)) * t);
}

uint malloc_stage(uint size, uint mem_size, uint stage)
{
    uint _158;
    _143.InterlockedAdd(0, size, _158);
    uint offset = _158;
    if ((offset + size) > mem_size)
    {
        uint _168;
        _143.InterlockedOr(4, stage, _168);
        offset = 0u;
    }
    return offset;
}

TileRef Tile_index(TileRef ref, uint index)
{
    TileRef _380 = { ref.offset + (index * 8u) };
    return _380;
}

void write_mem(Alloc alloc, uint offset, uint val)
{
    Alloc param = alloc;
    uint param_1 = offset;
    if (!touch_mem(param, param_1))
    {
        return;
    }
    _143.Store(offset * 4 + 12, val);
}

void TileSeg_write(Alloc a, TileSegRef ref, TileSeg s)
{
    uint ix = ref.offset >> uint(2);
    Alloc param = a;
    uint param_1 = ix + 0u;
    uint param_2 = asuint(s.origin.x);
    write_mem(param, param_1, param_2);
    Alloc param_3 = a;
    uint param_4 = ix + 1u;
    uint param_5 = asuint(s.origin.y);
    write_mem(param_3, param_4, param_5);
    Alloc param_6 = a;
    uint param_7 = ix + 2u;
    uint param_8 = asuint(s._vector.x);
    write_mem(param_6, param_7, param_8);
    Alloc param_9 = a;
    uint param_10 = ix + 3u;
    uint param_11 = asuint(s._vector.y);
    write_mem(param_9, param_10, param_11);
    Alloc param_12 = a;
    uint param_13 = ix + 4u;
    uint param_14 = asuint(s.y_edge);
    write_mem(param_12, param_13, param_14);
    Alloc param_15 = a;
    uint param_16 = ix + 5u;
    uint param_17 = s.next.offset;
    write_mem(param_15, param_16, param_17);
}

void comp_main()
{
    mem_ok = true;
    uint param = 7u;
    bool _694 = check_deps(param);
    if (!_694)
    {
        return;
    }
    uint element_ix = gl_GlobalInvocationID.x;
    PathSegRef _719 = { _711.Load(32) + (element_ix * 52u) };
    PathSegRef ref = _719;
    PathSegTag tag = _722;
    if (element_ix < _711.Load(8))
    {
        Alloc _732;
        _732.offset = _711.Load(32);
        Alloc param_1;
        param_1.offset = _732.offset;
        PathSegRef param_2 = ref;
        tag = PathSeg_tag(param_1, param_2);
    }
    switch (tag.tag)
    {
        case 1u:
        {
            Alloc _745;
            _745.offset = _711.Load(32);
            Alloc param_3;
            param_3.offset = _745.offset;
            PathSegRef param_4 = ref;
            PathCubic cubic = PathSeg_Cubic_read(param_3, param_4);
            float2 err_v = (((cubic.p2 - cubic.p1) * 3.0f) + cubic.p0) - cubic.p3;
            float err = (err_v.x * err_v.x) + (err_v.y * err_v.y);
            uint n_quads = max(uint(ceil(pow(err * 3.7037036418914794921875f, 0.16666667163372039794921875f))), 1u);
            n_quads = min(n_quads, 16u);
            float val = 0.0f;
            float2 qp0 = cubic.p0;
            float _step = 1.0f / float(n_quads);
            SubdivResult keep_params[16];
            for (uint i = 0u; i < n_quads; i++)
            {
                float t = float(i + 1u) * _step;
                float2 param_5 = cubic.p0;
                float2 param_6 = cubic.p1;
                float2 param_7 = cubic.p2;
                float2 param_8 = cubic.p3;
                float param_9 = t;
                float2 qp2 = eval_cubic(param_5, param_6, param_7, param_8, param_9);
                float2 param_10 = cubic.p0;
                float2 param_11 = cubic.p1;
                float2 param_12 = cubic.p2;
                float2 param_13 = cubic.p3;
                float param_14 = t - (0.5f * _step);
                float2 qp1 = eval_cubic(param_10, param_11, param_12, param_13, param_14);
                qp1 = (qp1 * 2.0f) - ((qp0 + qp2) * 0.5f);
                float2 param_15 = qp0;
                float2 param_16 = qp1;
                float2 param_17 = qp2;
                float param_18 = 0.4743416607379913330078125f;
                SubdivResult params = estimate_subdiv(param_15, param_16, param_17, param_18);
                keep_params[i] = params;
                val += params.val;
                qp0 = qp2;
            }
            uint n = max(uint(ceil((val * 0.5f) / 0.4743416607379913330078125f)), 1u);
            uint param_19 = tag.flags;
            bool is_stroke = fill_mode_from_flags(param_19) == 1u;
            uint path_ix = cubic.path_ix;
            PathRef _901 = { _711.Load(20) + (path_ix * 12u) };
            Alloc _904;
            _904.offset = _711.Load(20);
            Alloc param_20;
            param_20.offset = _904.offset;
            PathRef param_21 = _901;
            Path path = Path_read(param_20, param_21);
            uint param_22 = path.tiles.offset;
            uint param_23 = ((path.bbox.z - path.bbox.x) * (path.bbox.w - path.bbox.y)) * 8u;
            bool param_24 = true;
            Alloc path_alloc = new_alloc(param_22, param_23, param_24);
            int4 bbox = int4(path.bbox);
            float2 p0 = cubic.p0;
            qp0 = cubic.p0;
            float v_step = val / float(n);
            int n_out = 1;
            float val_sum = 0.0f;
            float2 p1;
            float _1143;
            TileSeg tile_seg;
            for (uint i_1 = 0u; i_1 < n_quads; i_1++)
            {
                float t_1 = float(i_1 + 1u) * _step;
                float2 param_25 = cubic.p0;
                float2 param_26 = cubic.p1;
                float2 param_27 = cubic.p2;
                float2 param_28 = cubic.p3;
                float param_29 = t_1;
                float2 qp2_1 = eval_cubic(param_25, param_26, param_27, param_28, param_29);
                float2 param_30 = cubic.p0;
                float2 param_31 = cubic.p1;
                float2 param_32 = cubic.p2;
                float2 param_33 = cubic.p3;
                float param_34 = t_1 - (0.5f * _step);
                float2 qp1_1 = eval_cubic(param_30, param_31, param_32, param_33, param_34);
                qp1_1 = (qp1_1 * 2.0f) - ((qp0 + qp2_1) * 0.5f);
                SubdivResult params_1 = keep_params[i_1];
                float param_35 = params_1.a0;
                float u0 = approx_parabola_inv_integral(param_35);
                float param_36 = params_1.a2;
                float u2 = approx_parabola_inv_integral(param_36);
                float uscale = 1.0f / (u2 - u0);
                float target = float(n_out) * v_step;
                for (;;)
                {
                    bool _1036 = uint(n_out) == n;
                    bool _1046;
                    if (!_1036)
                    {
                        _1046 = target < (val_sum + params_1.val);
                    }
                    else
                    {
                        _1046 = _1036;
                    }
                    if (_1046)
                    {
                        if (uint(n_out) == n)
                        {
                            p1 = cubic.p3;
                        }
                        else
                        {
                            float u = (target - val_sum) / params_1.val;
                            float a = lerp(params_1.a0, params_1.a2, u);
                            float param_37 = a;
                            float au = approx_parabola_inv_integral(param_37);
                            float t_2 = (au - u0) * uscale;
                            float2 param_38 = qp0;
                            float2 param_39 = qp1_1;
                            float2 param_40 = qp2_1;
                            float param_41 = t_2;
                            p1 = eval_quad(param_38, param_39, param_40, param_41);
                        }
                        float xmin = min(p0.x, p1.x) - cubic.stroke.x;
                        float xmax = max(p0.x, p1.x) + cubic.stroke.x;
                        float ymin = min(p0.y, p1.y) - cubic.stroke.y;
                        float ymax = max(p0.y, p1.y) + cubic.stroke.y;
                        float dx = p1.x - p0.x;
                        float dy = p1.y - p0.y;
                        if (abs(dy) < 9.999999717180685365747194737196e-10f)
                        {
                            _1143 = 1000000000.0f;
                        }
                        else
                        {
                            _1143 = dx / dy;
                        }
                        float invslope = _1143;
                        float c = (cubic.stroke.x + (abs(invslope) * (8.0f + cubic.stroke.y))) * 0.0625f;
                        float b = invslope;
                        float a_1 = (p0.x - ((p0.y - 8.0f) * b)) * 0.0625f;
                        int x0 = int(floor(xmin * 0.0625f));
                        int x1 = int(floor(xmax * 0.0625f) + 1.0f);
                        int y0 = int(floor(ymin * 0.0625f));
                        int y1 = int(floor(ymax * 0.0625f) + 1.0f);
                        x0 = clamp(x0, bbox.x, bbox.z);
                        y0 = clamp(y0, bbox.y, bbox.w);
                        x1 = clamp(x1, bbox.x, bbox.z);
                        y1 = clamp(y1, bbox.y, bbox.w);
                        float xc = a_1 + (b * float(y0));
                        int stride = bbox.z - bbox.x;
                        int base = ((y0 - bbox.y) * stride) - bbox.x;
                        uint n_tile_alloc = uint((x1 - x0) * (y1 - y0));
                        uint malloc_size = n_tile_alloc * 24u;
                        uint param_42 = malloc_size;
                        uint param_43 = _711.Load(0);
                        uint param_44 = 4u;
                        uint _1265 = malloc_stage(param_42, param_43, param_44);
                        uint tile_offset = _1265;
                        if (tile_offset == 0u)
                        {
                            mem_ok = false;
                        }
                        uint param_45 = tile_offset;
                        uint param_46 = malloc_size;
                        bool param_47 = true;
                        Alloc tile_alloc = new_alloc(param_45, param_46, param_47);
                        int xray = int(floor(p0.x * 0.0625f));
                        int last_xray = int(floor(p1.x * 0.0625f));
                        if (p0.y > p1.y)
                        {
                            int tmp = xray;
                            xray = last_xray;
                            last_xray = tmp;
                        }
                        for (int y = y0; y < y1; y++)
                        {
                            float tile_y0 = float(y * 16);
                            int xbackdrop = max((xray + 1), bbox.x);
                            bool _1322 = !is_stroke;
                            bool _1332;
                            if (_1322)
                            {
                                _1332 = min(p0.y, p1.y) < tile_y0;
                            }
                            else
                            {
                                _1332 = _1322;
                            }
                            bool _1339;
                            if (_1332)
                            {
                                _1339 = xbackdrop < bbox.z;
                            }
                            else
                            {
                                _1339 = _1332;
                            }
                            if (_1339)
                            {
                                int backdrop = (p1.y < p0.y) ? 1 : (-1);
                                TileRef param_48 = path.tiles;
                                uint param_49 = uint(base + xbackdrop);
                                TileRef tile_ref = Tile_index(param_48, param_49);
                                uint tile_el = tile_ref.offset >> uint(2);
                                uint _1369;
                                _143.InterlockedAdd((tile_el + 1u) * 4 + 12, uint(backdrop), _1369);
                            }
                            int next_xray = last_xray;
                            if (y < (y1 - 1))
                            {
                                float tile_y1 = float((y + 1) * 16);
                                float x_edge = lerp(p0.x, p1.x, (tile_y1 - p0.y) / dy);
                                next_xray = int(floor(x_edge * 0.0625f));
                            }
                            int min_xray = min(xray, next_xray);
                            int max_xray = max(xray, next_xray);
                            int xx0 = min(int(floor(xc - c)), min_xray);
                            int xx1 = max(int(ceil(xc + c)), (max_xray + 1));
                            xx0 = clamp(xx0, x0, x1);
                            xx1 = clamp(xx1, x0, x1);
                            for (int x = xx0; x < xx1; x++)
                            {
                                float tile_x0 = float(x * 16);
                                TileRef _1449 = { path.tiles.offset };
                                TileRef param_50 = _1449;
                                uint param_51 = uint(base + x);
                                TileRef tile_ref_1 = Tile_index(param_50, param_51);
                                uint tile_el_1 = tile_ref_1.offset >> uint(2);
                                uint old = 0u;
                                uint _1465;
                                _143.InterlockedExchange(tile_el_1 * 4 + 12, tile_offset, _1465);
                                old = _1465;
                                tile_seg.origin = p0;
                                tile_seg._vector = p1 - p0;
                                float y_edge = 0.0f;
                                if (!is_stroke)
                                {
                                    y_edge = lerp(p0.y, p1.y, (tile_x0 - p0.x) / dx);
                                    if (min(p0.x, p1.x) < tile_x0)
                                    {
                                        float2 p = float2(tile_x0, y_edge);
                                        if (p0.x > p1.x)
                                        {
                                            tile_seg._vector = p - p0;
                                        }
                                        else
                                        {
                                            tile_seg.origin = p;
                                            tile_seg._vector = p1 - p;
                                        }
                                        if (tile_seg._vector.x == 0.0f)
                                        {
                                            tile_seg._vector.x = sign(p1.x - p0.x) * 9.999999717180685365747194737196e-10f;
                                        }
                                    }
                                    if ((x <= min_xray) || (max_xray < x))
                                    {
                                        y_edge = 1000000000.0f;
                                    }
                                }
                                tile_seg.y_edge = y_edge;
                                tile_seg.next.offset = old;
                                if (mem_ok)
                                {
                                    TileSegRef _1550 = { tile_offset };
                                    Alloc param_52 = tile_alloc;
                                    TileSegRef param_53 = _1550;
                                    TileSeg param_54 = tile_seg;
                                    TileSeg_write(param_52, param_53, param_54);
                                }
                                tile_offset += 24u;
                            }
                            xc += b;
                            base += stride;
                            xray = next_xray;
                        }
                        n_out++;
                        target += v_step;
                        p0 = p1;
                        continue;
                    }
                    else
                    {
                        break;
                    }
                }
                val_sum += params_1.val;
                qp0 = qp2_1;
            }
            break;
        }
    }
}

[numthreads(32, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
