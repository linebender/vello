struct Alloc
{
    uint offset;
};

struct MallocResult
{
    Alloc alloc;
    bool failed;
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

static const PathSegTag _721 = { 0u, 0u };

RWByteAddressBuffer _136 : register(u0, space0);
ByteAddressBuffer _710 : register(t1, space0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

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
    uint v = _136.Load(offset * 4 + 8);
    return v;
}

PathSegTag PathSeg_tag(Alloc a, PathSegRef ref)
{
    Alloc param = a;
    uint param_1 = ref.offset >> uint(2);
    uint tag_and_flags = read_mem(param, param_1);
    PathSegTag _367 = { tag_and_flags & 65535u, tag_and_flags >> uint(16) };
    return _367;
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
    PathCubicRef _373 = { ref.offset + 4u };
    Alloc param = a;
    PathCubicRef param_1 = _373;
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
    SubdivResult _695 = { val, a0, a2 };
    return _695;
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
    TileRef _427 = { raw2 };
    s.tiles = _427;
    return s;
}

Alloc new_alloc(uint offset, uint size, bool mem_ok)
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

MallocResult malloc(uint size)
{
    uint _142;
    _136.InterlockedAdd(0, size, _142);
    uint offset = _142;
    uint _149;
    _136.GetDimensions(_149);
    _149 = (_149 - 8) / 4;
    MallocResult r;
    r.failed = (offset + size) > uint(int(_149) * 4);
    uint param = offset;
    uint param_1 = size;
    bool param_2 = !r.failed;
    r.alloc = new_alloc(param, param_1, param_2);
    if (r.failed)
    {
        uint _171;
        _136.InterlockedMax(4, 1u, _171);
        return r;
    }
    return r;
}

TileRef Tile_index(TileRef ref, uint index)
{
    TileRef _385 = { ref.offset + (index * 8u) };
    return _385;
}

void write_mem(Alloc alloc, uint offset, uint val)
{
    Alloc param = alloc;
    uint param_1 = offset;
    if (!touch_mem(param, param_1))
    {
        return;
    }
    _136.Store(offset * 4 + 8, val);
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
    uint element_ix = gl_GlobalInvocationID.x;
    PathSegRef _718 = { _710.Load(28) + (element_ix * 52u) };
    PathSegRef ref = _718;
    PathSegTag tag = _721;
    if (element_ix < _710.Load(4))
    {
        Alloc _731;
        _731.offset = _710.Load(28);
        Alloc param;
        param.offset = _731.offset;
        PathSegRef param_1 = ref;
        tag = PathSeg_tag(param, param_1);
    }
    bool mem_ok = _136.Load(4) == 0u;
    switch (tag.tag)
    {
        case 1u:
        {
            Alloc _748;
            _748.offset = _710.Load(28);
            Alloc param_2;
            param_2.offset = _748.offset;
            PathSegRef param_3 = ref;
            PathCubic cubic = PathSeg_Cubic_read(param_2, param_3);
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
                float2 param_4 = cubic.p0;
                float2 param_5 = cubic.p1;
                float2 param_6 = cubic.p2;
                float2 param_7 = cubic.p3;
                float param_8 = t;
                float2 qp2 = eval_cubic(param_4, param_5, param_6, param_7, param_8);
                float2 param_9 = cubic.p0;
                float2 param_10 = cubic.p1;
                float2 param_11 = cubic.p2;
                float2 param_12 = cubic.p3;
                float param_13 = t - (0.5f * _step);
                float2 qp1 = eval_cubic(param_9, param_10, param_11, param_12, param_13);
                qp1 = (qp1 * 2.0f) - ((qp0 + qp2) * 0.5f);
                float2 param_14 = qp0;
                float2 param_15 = qp1;
                float2 param_16 = qp2;
                float param_17 = 0.4743416607379913330078125f;
                SubdivResult params = estimate_subdiv(param_14, param_15, param_16, param_17);
                keep_params[i] = params;
                val += params.val;
                qp0 = qp2;
            }
            uint n = max(uint(ceil((val * 0.5f) / 0.4743416607379913330078125f)), 1u);
            uint param_18 = tag.flags;
            bool is_stroke = fill_mode_from_flags(param_18) == 1u;
            uint path_ix = cubic.path_ix;
            PathRef _904 = { _710.Load(16) + (path_ix * 12u) };
            Alloc _907;
            _907.offset = _710.Load(16);
            Alloc param_19;
            param_19.offset = _907.offset;
            PathRef param_20 = _904;
            Path path = Path_read(param_19, param_20);
            uint param_21 = path.tiles.offset;
            uint param_22 = ((path.bbox.z - path.bbox.x) * (path.bbox.w - path.bbox.y)) * 8u;
            bool param_23 = mem_ok;
            Alloc path_alloc = new_alloc(param_21, param_22, param_23);
            int4 bbox = int4(path.bbox);
            float2 p0 = cubic.p0;
            qp0 = cubic.p0;
            float v_step = val / float(n);
            int n_out = 1;
            float val_sum = 0.0f;
            float2 p1;
            float _1147;
            TileSeg tile_seg;
            for (uint i_1 = 0u; i_1 < n_quads; i_1++)
            {
                float t_1 = float(i_1 + 1u) * _step;
                float2 param_24 = cubic.p0;
                float2 param_25 = cubic.p1;
                float2 param_26 = cubic.p2;
                float2 param_27 = cubic.p3;
                float param_28 = t_1;
                float2 qp2_1 = eval_cubic(param_24, param_25, param_26, param_27, param_28);
                float2 param_29 = cubic.p0;
                float2 param_30 = cubic.p1;
                float2 param_31 = cubic.p2;
                float2 param_32 = cubic.p3;
                float param_33 = t_1 - (0.5f * _step);
                float2 qp1_1 = eval_cubic(param_29, param_30, param_31, param_32, param_33);
                qp1_1 = (qp1_1 * 2.0f) - ((qp0 + qp2_1) * 0.5f);
                SubdivResult params_1 = keep_params[i_1];
                float param_34 = params_1.a0;
                float u0 = approx_parabola_inv_integral(param_34);
                float param_35 = params_1.a2;
                float u2 = approx_parabola_inv_integral(param_35);
                float uscale = 1.0f / (u2 - u0);
                float target = float(n_out) * v_step;
                for (;;)
                {
                    bool _1040 = uint(n_out) == n;
                    bool _1050;
                    if (!_1040)
                    {
                        _1050 = target < (val_sum + params_1.val);
                    }
                    else
                    {
                        _1050 = _1040;
                    }
                    if (_1050)
                    {
                        if (uint(n_out) == n)
                        {
                            p1 = cubic.p3;
                        }
                        else
                        {
                            float u = (target - val_sum) / params_1.val;
                            float a = lerp(params_1.a0, params_1.a2, u);
                            float param_36 = a;
                            float au = approx_parabola_inv_integral(param_36);
                            float t_2 = (au - u0) * uscale;
                            float2 param_37 = qp0;
                            float2 param_38 = qp1_1;
                            float2 param_39 = qp2_1;
                            float param_40 = t_2;
                            p1 = eval_quad(param_37, param_38, param_39, param_40);
                        }
                        float xmin = min(p0.x, p1.x) - cubic.stroke.x;
                        float xmax = max(p0.x, p1.x) + cubic.stroke.x;
                        float ymin = min(p0.y, p1.y) - cubic.stroke.y;
                        float ymax = max(p0.y, p1.y) + cubic.stroke.y;
                        float dx = p1.x - p0.x;
                        float dy = p1.y - p0.y;
                        if (abs(dy) < 9.999999717180685365747194737196e-10f)
                        {
                            _1147 = 1000000000.0f;
                        }
                        else
                        {
                            _1147 = dx / dy;
                        }
                        float invslope = _1147;
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
                        uint param_41 = n_tile_alloc * 24u;
                        MallocResult _1263 = malloc(param_41);
                        MallocResult tile_alloc = _1263;
                        if (tile_alloc.failed || (!mem_ok))
                        {
                            return;
                        }
                        uint tile_offset = tile_alloc.alloc.offset;
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
                            bool _1319 = !is_stroke;
                            bool _1329;
                            if (_1319)
                            {
                                _1329 = min(p0.y, p1.y) < tile_y0;
                            }
                            else
                            {
                                _1329 = _1319;
                            }
                            bool _1336;
                            if (_1329)
                            {
                                _1336 = xbackdrop < bbox.z;
                            }
                            else
                            {
                                _1336 = _1329;
                            }
                            if (_1336)
                            {
                                int backdrop = (p1.y < p0.y) ? 1 : (-1);
                                TileRef param_42 = path.tiles;
                                uint param_43 = uint(base + xbackdrop);
                                TileRef tile_ref = Tile_index(param_42, param_43);
                                uint tile_el = tile_ref.offset >> uint(2);
                                Alloc param_44 = path_alloc;
                                uint param_45 = tile_el + 1u;
                                if (touch_mem(param_44, param_45))
                                {
                                    uint _1374;
                                    _136.InterlockedAdd((tile_el + 1u) * 4 + 8, uint(backdrop), _1374);
                                }
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
                                TileRef _1454 = { path.tiles.offset };
                                TileRef param_46 = _1454;
                                uint param_47 = uint(base + x);
                                TileRef tile_ref_1 = Tile_index(param_46, param_47);
                                uint tile_el_1 = tile_ref_1.offset >> uint(2);
                                uint old = 0u;
                                Alloc param_48 = path_alloc;
                                uint param_49 = tile_el_1;
                                if (touch_mem(param_48, param_49))
                                {
                                    uint _1477;
                                    _136.InterlockedExchange(tile_el_1 * 4 + 8, tile_offset, _1477);
                                    old = _1477;
                                }
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
                                TileSegRef _1559 = { tile_offset };
                                Alloc param_50 = tile_alloc.alloc;
                                TileSegRef param_51 = _1559;
                                TileSeg param_52 = tile_seg;
                                TileSeg_write(param_50, param_51, param_52);
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
