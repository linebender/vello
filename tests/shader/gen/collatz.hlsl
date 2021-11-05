static const uint3 gl_WorkGroupSize = uint3(1u, 1u, 1u);

RWByteAddressBuffer _53 : register(u1);
ByteAddressBuffer _59 : register(t0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

uint collatz_iterations(inout uint n)
{
    uint i = 0u;
    while (n != 1u)
    {
        if ((n % 2u) == 0u)
        {
            n /= 2u;
        }
        else
        {
            n = (3u * n) + 1u;
        }
        i++;
    }
    return i;
}

void comp_main()
{
    uint index = gl_GlobalInvocationID.x;
    uint param = _59.Load(index * 4 + 0);
    uint _65 = collatz_iterations(param);
    _53.Store(index * 4 + 0, _65);
}

[numthreads(1, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
