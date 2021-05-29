static const uint3 gl_WorkGroupSize = uint3(1u, 1u, 1u);

RWByteAddressBuffer _57 : register(u0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

float mod(float x, float y)
{
    return x - y * floor(x / y);
}

float2 mod(float2 x, float2 y)
{
    return x - y * floor(x / y);
}

float3 mod(float3 x, float3 y)
{
    return x - y * floor(x / y);
}

float4 mod(float4 x, float4 y)
{
    return x - y * floor(x / y);
}

uint collatz_iterations(inout uint n)
{
    uint i = 0u;
    while (n != 1u)
    {
        if (mod(float(n), 2.0f) == 0.0f)
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
    uint param = _57.Load(index * 4 + 0);
    uint _65 = collatz_iterations(param);
    _57.Store(index * 4 + 0, _65);
}

[numthreads(1, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
