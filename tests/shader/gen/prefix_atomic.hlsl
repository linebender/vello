struct Monoid
{
    uint element;
};

struct State
{
    uint flag;
    Monoid aggregate;
    Monoid prefix;
};

static const uint3 gl_WorkGroupSize = uint3(512u, 1u, 1u);

static const Monoid _183 = { 0u };

globallycoherent RWByteAddressBuffer _43 : register(u2);
ByteAddressBuffer _67 : register(t0);
RWByteAddressBuffer _367 : register(u1);

static uint3 gl_LocalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_LocalInvocationID : SV_GroupThreadID;
};

groupshared uint sh_part_ix;
groupshared Monoid sh_scratch[512];
groupshared uint sh_flag;
groupshared Monoid sh_prefix;

Monoid combine_monoid(Monoid a, Monoid b)
{
    Monoid _22 = { a.element + b.element };
    return _22;
}

void comp_main()
{
    if (gl_LocalInvocationID.x == 0u)
    {
        uint _47;
        _43.InterlockedAdd(0, 1u, _47);
        sh_part_ix = _47;
    }
    GroupMemoryBarrierWithGroupSync();
    uint part_ix = sh_part_ix;
    uint ix = (part_ix * 8192u) + (gl_LocalInvocationID.x * 16u);
    Monoid _71;
    _71.element = _67.Load(ix * 4 + 0);
    Monoid local[16];
    local[0].element = _71.element;
    Monoid param_1;
    for (uint i = 1u; i < 16u; i++)
    {
        Monoid param = local[i - 1u];
        Monoid _94;
        _94.element = _67.Load((ix + i) * 4 + 0);
        param_1.element = _94.element;
        local[i] = combine_monoid(param, param_1);
    }
    Monoid agg = local[15];
    sh_scratch[gl_LocalInvocationID.x] = agg;
    for (uint i_1 = 0u; i_1 < 9u; i_1++)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gl_LocalInvocationID.x >= (1u << i_1))
        {
            Monoid other = sh_scratch[gl_LocalInvocationID.x - (1u << i_1)];
            Monoid param_2 = other;
            Monoid param_3 = agg;
            agg = combine_monoid(param_2, param_3);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[gl_LocalInvocationID.x] = agg;
    }
    if (gl_LocalInvocationID.x == 511u)
    {
        uint _378;
        _43.InterlockedExchange(part_ix * 12 + 8, agg.element, _378);
        if (part_ix == 0u)
        {
            uint _379;
            _43.InterlockedExchange(12, agg.element, _379);
        }
    }
    DeviceMemoryBarrier();
    if (gl_LocalInvocationID.x == 511u)
    {
        uint flag = 1u;
        if (part_ix == 0u)
        {
            flag = 2u;
        }
        uint _380;
        _43.InterlockedExchange(part_ix * 12 + 4, flag, _380);
    }
    Monoid exclusive = _183;
    if (part_ix != 0u)
    {
        uint look_back_ix = part_ix - 1u;
        uint their_ix = 0u;
        Monoid their_agg;
        Monoid m;
        while (true)
        {
            if (gl_LocalInvocationID.x == 511u)
            {
                uint _206;
                _43.InterlockedAdd(look_back_ix * 12 + 4, 0, _206);
                sh_flag = _206;
            }
            GroupMemoryBarrierWithGroupSync();
            DeviceMemoryBarrier();
            uint flag_1 = sh_flag;
            if (flag_1 == 2u)
            {
                if (gl_LocalInvocationID.x == 511u)
                {
                    uint _221;
                    _43.InterlockedAdd(look_back_ix * 12 + 12, 0, _221);
                    Monoid _222 = { _221 };
                    Monoid their_prefix = _222;
                    Monoid param_4 = their_prefix;
                    Monoid param_5 = exclusive;
                    exclusive = combine_monoid(param_4, param_5);
                }
                break;
            }
            else
            {
                if (flag_1 == 1u)
                {
                    if (gl_LocalInvocationID.x == 511u)
                    {
                        uint _242;
                        _43.InterlockedAdd(look_back_ix * 12 + 8, 0, _242);
                        their_agg.element = _242;
                        Monoid param_6 = their_agg;
                        Monoid param_7 = exclusive;
                        exclusive = combine_monoid(param_6, param_7);
                    }
                    look_back_ix--;
                    their_ix = 0u;
                    continue;
                }
            }
            if (gl_LocalInvocationID.x == 511u)
            {
                Monoid _263;
                _263.element = _67.Load(((look_back_ix * 8192u) + their_ix) * 4 + 0);
                m.element = _263.element;
                if (their_ix == 0u)
                {
                    their_agg = m;
                }
                else
                {
                    Monoid param_8 = their_agg;
                    Monoid param_9 = m;
                    their_agg = combine_monoid(param_8, param_9);
                }
                their_ix++;
                if (their_ix == 8192u)
                {
                    Monoid param_10 = their_agg;
                    Monoid param_11 = exclusive;
                    exclusive = combine_monoid(param_10, param_11);
                    if (look_back_ix == 0u)
                    {
                        sh_flag = 2u;
                    }
                    else
                    {
                        look_back_ix--;
                        their_ix = 0u;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
            flag_1 = sh_flag;
            if (flag_1 == 2u)
            {
                break;
            }
        }
        if (gl_LocalInvocationID.x == 511u)
        {
            Monoid param_12 = exclusive;
            Monoid param_13 = agg;
            Monoid inclusive_prefix = combine_monoid(param_12, param_13);
            sh_prefix = exclusive;
            uint _381;
            _43.InterlockedExchange(part_ix * 12 + 12, inclusive_prefix.element, _381);
        }
        DeviceMemoryBarrier();
        if (gl_LocalInvocationID.x == 511u)
        {
            uint _382;
            _43.InterlockedExchange(part_ix * 12 + 4, 2u, _382);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (part_ix != 0u)
    {
        exclusive = sh_prefix;
    }
    Monoid row = exclusive;
    if (gl_LocalInvocationID.x > 0u)
    {
        Monoid other_1 = sh_scratch[gl_LocalInvocationID.x - 1u];
        Monoid param_14 = row;
        Monoid param_15 = other_1;
        row = combine_monoid(param_14, param_15);
    }
    for (uint i_2 = 0u; i_2 < 16u; i_2++)
    {
        Monoid param_16 = row;
        Monoid param_17 = local[i_2];
        Monoid m_1 = combine_monoid(param_16, param_17);
        _367.Store((ix + i_2) * 4 + 0, m_1.element);
    }
}

[numthreads(512, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_LocalInvocationID = stage_input.gl_LocalInvocationID;
    comp_main();
}
