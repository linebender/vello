static const uint3 gl_WorkGroupSize = uint3(256u, 1u, 1u);

RWByteAddressBuffer _20 : register(u0);
RWByteAddressBuffer _30 : register(u1);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x;
    uint _26;
    _20.InterlockedAdd(0, 1u, _26);
    uint now = _26;
    _30.Store((ix * 256u) * 4 + 0, now);
    uint _40;
    _20.InterlockedAdd(ix * 4 + 4, 0, _40);
    uint last_val = _40;
    uint out_ix = 1u;
    uint rng = gl_GlobalInvocationID.x + 1u;
    for (uint i = 0u; i < 4096u; i++)
    {
        uint _59;
        _20.InterlockedOr(ix * 4 + 4, 0u, _59);
        uint new_val = _59;
        if (new_val != last_val)
        {
            last_val = new_val;
            uint _67;
            _20.InterlockedAdd(0, 1u, _67);
            now = _67;
            if (out_ix < 255u)
            {
                _30.Store(((ix * 256u) + out_ix) * 4 + 0, now - new_val);
                out_ix++;
            }
        }
        rng ^= (rng << uint(13));
        rng ^= (rng >> uint(17));
        rng ^= (rng << uint(5));
        if (rng < 16777216u)
        {
            uint _104;
            _20.InterlockedAdd(0, 1u, _104);
            now = _104;
            uint target = rng % 65536u;
            uint _112;
            _20.InterlockedExchange(target * 4 + 4, now, _112);
        }
    }
    uint _116;
    _20.InterlockedAdd(0, 1u, _116);
    now = _116;
    _30.Store(((ix * 256u) + out_ix) * 4 + 0, now);
    if (out_ix < 255u)
    {
        _30.Store((((ix * 256u) + out_ix) + 1u) * 4 + 0, 4294967295u);
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
