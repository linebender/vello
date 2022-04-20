struct Bic
{
    uint a;
    uint b;
};

static const uint3 gl_WorkGroupSize = uint3(64u, 1u, 1u);

static const Bic _237 = { 0u, 0u };

ByteAddressBuffer _250 : register(t1);
ByteAddressBuffer _512 : register(t2);
ByteAddressBuffer _593 : register(t0);
RWByteAddressBuffer _751 : register(u3);

static uint3 gl_WorkGroupID;
static uint3 gl_LocalInvocationID;
static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

groupshared Bic sh_bic[126];
groupshared uint sh_bitmaps[64];
groupshared uint sh_stack[512];
groupshared uint sh_link[64];
groupshared uint sh_next[64];

Bic bic_combine(Bic x, Bic y)
{
    uint m = min(x.b, y.a);
    Bic _47 = { (x.a + y.a) - m, (x.b + y.b) - m };
    return _47;
}

uint search_bit_set(uint bitmask, uint ix)
{
    uint result = 0u;
    for (uint j = 0u; j < 5u; j++)
    {
        uint _step = 1u << (4u - j);
        if (uint(int(countbits(bitmask & ((1u << (result + _step)) - 1u)))) <= ix)
        {
            result += _step;
        }
    }
    return result;
}

uint search_link(inout Bic bic)
{
    uint ix = gl_LocalInvocationID.x;
    uint j = 0u;
    while (j < 6u)
    {
        uint base = 128u - (2u << (6u - j));
        if (((ix >> j) & 1u) != 0u)
        {
            Bic param = sh_bic[(base + (ix >> j)) - 1u];
            Bic param_1 = bic;
            Bic test = bic_combine(param, param_1);
            if (test.b > 0u)
            {
                break;
            }
            bic = test;
            ix -= (1u << j);
        }
        j++;
    }
    if (ix > 0u)
    {
        while (j > 0u)
        {
            j--;
            uint base_1 = 128u - (2u << (6u - j));
            Bic param_2 = sh_bic[(base_1 + (ix >> j)) - 1u];
            Bic param_3 = bic;
            Bic test_1 = bic_combine(param_2, param_3);
            if (test_1.b == 0u)
            {
                bic = test_1;
                ix -= (1u << j);
            }
        }
    }
    if (ix > 0u)
    {
        ix--;
        Bic param_4 = sh_bic[ix];
        Bic param_5 = bic;
        Bic test_2 = bic_combine(param_4, param_5);
        uint param_6 = sh_bitmaps[ix];
        uint param_7 = test_2.b - 1u;
        uint ix_in_chunk = search_bit_set(param_6, param_7);
        return (ix * 8u) + ix_in_chunk;
    }
    else
    {
        return 4294967295u - bic.a;
    }
}

void comp_main()
{
    uint th = gl_LocalInvocationID.x;
    Bic bic = _237;
    if ((th * 8u) < gl_WorkGroupID.x)
    {
        Bic _255;
        _255.a = _250.Load((th * 8u) * 8 + 0);
        _255.b = _250.Load((th * 8u) * 8 + 4);
        bic.a = _255.a;
        bic.b = _255.b;
    }
    Bic other;
    for (uint i = 1u; i < 8u; i++)
    {
        if (((th * 8u) + i) < gl_WorkGroupID.x)
        {
            Bic _283;
            _283.a = _250.Load(((th * 8u) + i) * 8 + 0);
            _283.b = _250.Load(((th * 8u) + i) * 8 + 4);
            other.a = _283.a;
            other.b = _283.b;
            Bic param = bic;
            Bic param_1 = other;
            bic = bic_combine(param, param_1);
        }
    }
    sh_bic[th] = bic;
    for (uint i_1 = 0u; i_1 < 6u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if ((th + (1u << i_1)) < 64u)
        {
            Bic other_1 = sh_bic[th + (1u << i_1)];
            Bic param_2 = bic;
            Bic param_3 = other_1;
            bic = bic_combine(param_2, param_3);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_bic[th] = bic;
    }
    GroupMemoryBarrierWithGroupSync();
    if (th == 63u)
    {
        bic = _237;
    }
    else
    {
        bic = sh_bic[th + 1u];
    }
    uint last_b = bic.b;
    uint bitmap = 0u;
    Bic param_4;
    for (uint i_2 = 0u; i_2 < 8u; i_2++)
    {
        uint this_ix = (((th * 8u) + 8u) - 1u) - i_2;
        if (this_ix < gl_WorkGroupID.x)
        {
            Bic _369;
            _369.a = _250.Load(this_ix * 8 + 0);
            _369.b = _250.Load(this_ix * 8 + 4);
            param_4.a = _369.a;
            param_4.b = _369.b;
            Bic param_5 = bic;
            bic = bic_combine(param_4, param_5);
        }
        sh_stack[this_ix] = bic.b;
        if (bic.b > last_b)
        {
            bitmap |= (1u << (7u - i_2));
        }
        last_b = bic.b;
    }
    sh_bitmaps[th] = bitmap;
    uint link = 0u;
    if (bitmap != 0u)
    {
        link = (th * 8u) + uint(int(firstbithigh(bitmap)));
    }
    sh_link[th] = link;
    for (uint i_3 = 0u; i_3 < 6u; i_3++)
    {
        GroupMemoryBarrierWithGroupSync();
        if (th >= (1u << i_3))
        {
            link = max(link, sh_link[th - (1u << i_3)]);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_link[th] = link;
    }
    GroupMemoryBarrierWithGroupSync();
    uint sp = 504u - (th * 8u);
    uint ix = 0u;
    for (uint i_4 = 0u; i_4 < 9u; i_4++)
    {
        uint probe = ix + (256u >> i_4);
        if (sp < sh_stack[probe])
        {
            ix = probe;
        }
    }
    uint b = sh_stack[ix];
    uint local_stack[8];
    for (uint i_5 = 0u; i_5 < 8u; i_5++)
    {
        local_stack[i_5] = 0u;
    }
    uint i_6 = 0u;
    while ((sp + i_6) < b)
    {
        local_stack[7u - i_6] = _512.Load(((((ix * 512u) + b) - (sp + i_6)) - 1u) * 4 + 0);
        i_6++;
        if (i_6 == 8u)
        {
            break;
        }
        if ((sp + i_6) == b)
        {
            uint bits = sh_bitmaps[ix / 8u] & ((1u << (ix % 8u)) - 1u);
            if (bits == 0u)
            {
                ix = sh_link[max((ix / 8u), 1u) - 1u];
            }
            else
            {
                ix = (ix & 4294967288u) + uint(int(firstbithigh(bits)));
            }
            b = sh_stack[ix];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint i_7 = 0u; i_7 < 8u; i_7++)
    {
        sh_stack[(th * 8u) + i_7] = local_stack[i_7];
    }
    uint inp = _593.Load((((gl_GlobalInvocationID.x * 8u) + 8u) - 1u) * 4 + 0);
    Bic _605 = { 1u - inp, inp };
    bic = _605;
    bitmap = inp << uint(7);
    for (uint i_8 = 7u; i_8 > 0u; i_8--)
    {
        inp = _593.Load((((gl_GlobalInvocationID.x * 8u) + i_8) - 1u) * 4 + 0);
        bool _626 = inp == 1u;
        bool _632;
        if (_626)
        {
            _632 = bic.a == 0u;
        }
        else
        {
            _632 = _626;
        }
        if (_632)
        {
            bitmap |= (1u << (i_8 - 1u));
        }
        Bic _644 = { 1u - inp, inp };
        Bic other_2 = _644;
        Bic param_6 = other_2;
        Bic param_7 = bic;
        bic = bic_combine(param_6, param_7);
    }
    sh_bitmaps[th] = bitmap;
    sh_bic[th] = bic;
    uint inbase = 0u;
    for (uint i_9 = 0u; i_9 < 5u; i_9++)
    {
        uint outbase = 128u - (1u << (6u - i_9));
        GroupMemoryBarrierWithGroupSync();
        if (th < (1u << (5u - i_9)))
        {
            Bic param_8 = sh_bic[inbase + (th * 2u)];
            Bic param_9 = sh_bic[(inbase + (th * 2u)) + 1u];
            sh_bic[outbase + th] = bic_combine(param_8, param_9);
        }
        inbase = outbase;
    }
    GroupMemoryBarrierWithGroupSync();
    bic.b = 0u;
    Bic param_10 = bic;
    uint _706 = search_link(param_10);
    bic = param_10;
    sh_link[th] = _706;
    bic = _237;
    Bic param_11 = bic;
    uint _711 = search_link(param_11);
    bic = param_11;
    ix = _711;
    uint loc_sp = 0u;
    uint outp;
    uint loc_stack[8];
    for (uint i_10 = 0u; i_10 < 8u; i_10++)
    {
        if (loc_sp > 0u)
        {
            outp = loc_stack[loc_sp - 1u];
        }
        else
        {
            if (int(ix) >= 0)
            {
                outp = (gl_WorkGroupID.x * 512u) + ix;
            }
            else
            {
                outp = sh_stack[512u + ix];
            }
        }
        _751.Store(((gl_GlobalInvocationID.x * 8u) + i_10) * 4 + 0, outp);
        inp = _593.Load(((gl_GlobalInvocationID.x * 8u) + i_10) * 4 + 0);
        if (inp == 1u)
        {
            loc_stack[loc_sp] = (gl_GlobalInvocationID.x * 8u) + i_10;
            loc_sp++;
        }
        else
        {
            if (inp == 0u)
            {
                if (loc_sp > 0u)
                {
                    loc_sp--;
                }
                else
                {
                    if (int(ix) >= 0)
                    {
                        uint bits_1 = sh_bitmaps[ix / 8u] & ((1u << (ix % 8u)) - 1u);
                        if (bits_1 == 0u)
                        {
                            ix = sh_link[ix / 8u];
                        }
                        else
                        {
                            ix = (ix & 4294967288u) + uint(int(firstbithigh(bits_1)));
                        }
                    }
                    else
                    {
                        ix--;
                    }
                }
            }
        }
    }
}

[numthreads(64, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
