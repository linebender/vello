struct Alloc
{
    uint offset;
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
    Alloc bbox_alloc;
    Alloc drawmonoid_alloc;
    uint n_trans;
    uint trans_offset;
    uint pathtag_offset;
    uint linewidth_offset;
    uint pathseg_offset;
};

static const uint3 gl_WorkGroupSize = uint3(512u, 1u, 1u);

ByteAddressBuffer _21 : register(t1);
RWByteAddressBuffer _44 : register(u0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

void comp_main()
{
    uint ix = gl_GlobalInvocationID.x;
    if (ix < _21.Load(0))
    {
        uint out_ix = (_21.Load(40) >> uint(2)) + (4u * ix);
        _44.Store(out_ix * 4 + 8, 65535u);
        _44.Store((out_ix + 1u) * 4 + 8, 65535u);
        _44.Store((out_ix + 2u) * 4 + 8, 0u);
        _44.Store((out_ix + 3u) * 4 + 8, 0u);
    }
}

[numthreads(512, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
