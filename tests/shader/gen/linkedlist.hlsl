static const uint3 gl_WorkGroupSize = uint3(256u, 1u, 1u);

RWByteAddressBuffer _56 : register(u0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

void comp_main()
{
    uint rng = gl_GlobalInvocationID.x + 1u;
    for (uint i = 0u; i < 100u; i++)
    {
        rng ^= (rng << uint(13));
        rng ^= (rng >> uint(17));
        rng ^= (rng << uint(5));
        uint bucket = rng % 65536u;
        if (bucket != 0u)
        {
            uint _61;
            _56.InterlockedAdd(0, 2u, _61);
            uint alloc = _61 + 65536u;
            uint _67;
            _56.InterlockedExchange(bucket * 4 + 0, alloc, _67);
            uint old = _67;
            _56.Store(alloc * 4 + 0, old);
            _56.Store((alloc + 1u) * 4 + 0, gl_GlobalInvocationID.x);
        }
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
