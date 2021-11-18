struct Result
{
    uint r0;
    uint r1;
    uint r2;
    uint r3;
};

static const uint3 gl_WorkGroupSize = uint3(256u, 1u, 1u);

RWByteAddressBuffer data_buf : register(u0);
RWByteAddressBuffer out_buf : register(u1);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x;
    uint role_0_ix = (ix * 661u) & 65535u;
    uint role_1_ix = (ix * 1087u) & 65535u;
    uint role_2_ix = (ix * 2749u) & 65535u;
    uint role_3_ix = (ix * 3433u) & 65535u;
    uint _89;
    data_buf.InterlockedExchange(role_0_ix * 4 + 0, 1u, _89);
    uint _52;
    data_buf.InterlockedAdd(role_1_ix * 4 + 0, 0, _52);
    uint r0 = _52;
    uint _56;
    data_buf.InterlockedAdd(role_1_ix * 4 + 0, 0, _56);
    uint r1 = _56;
    uint _90;
    data_buf.InterlockedExchange(role_2_ix * 4 + 0, 2u, _90);
    uint _63;
    data_buf.InterlockedAdd(role_3_ix * 4 + 0, 0, _63);
    uint r2 = _63;
    uint _67;
    data_buf.InterlockedAdd(role_3_ix * 4 + 0, 0, _67);
    uint r3 = _67;
    out_buf.Store(role_1_ix * 16 + 0, r0);
    out_buf.Store(role_1_ix * 16 + 4, r1);
    out_buf.Store(role_3_ix * 16 + 8, r2);
    out_buf.Store(role_3_ix * 16 + 12, r3);
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
