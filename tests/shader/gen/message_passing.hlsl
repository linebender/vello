struct Element
{
    uint data;
    uint flag;
};

static const uint3 gl_WorkGroupSize = uint3(256u, 1u, 1u);

globallycoherent RWByteAddressBuffer data_buf : register(u0);
RWByteAddressBuffer control_buf : register(u1);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

uint permute_flag_ix(uint data_ix)
{
    return (data_ix * 419u) & 65535u;
}

void comp_main()
{
    data_buf.Store(gl_GlobalInvocationID.x * 8 + 0, 1u);
    DeviceMemoryBarrier();
    uint param = gl_GlobalInvocationID.x;
    uint write_flag_ix = permute_flag_ix(param);
    uint _76;
    data_buf.InterlockedExchange(write_flag_ix * 8 + 4, 1u, _76);
    uint read_ix = (gl_GlobalInvocationID.x * 4099u) & 65535u;
    uint param_1 = read_ix;
    uint read_flag_ix = permute_flag_ix(param_1);
    uint _58;
    data_buf.InterlockedAdd(read_flag_ix * 8 + 4, 0, _58);
    uint flag = _58;
    DeviceMemoryBarrier();
    uint data = data_buf.Load(read_ix * 8 + 0);
    if (flag > data)
    {
        uint _73;
        control_buf.InterlockedAdd(0, 1u, _73);
    }
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
