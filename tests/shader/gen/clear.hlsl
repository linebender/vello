static const uint3 gl_WorkGroupSize = uint3(256u, 1u, 1u);

ByteAddressBuffer _19 : register(t0);
RWByteAddressBuffer _32 : register(u1);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x;
    if (ix < _19.Load(0))
    {
        _32.Store(ix * 4 + 0, _19.Load(4));
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
