//! An example to exercise the dx12 backend, while it's being developed.
//! This will probably go away when it's fully implemented and we can
//! just use the hub.

use piet_gpu_hal::backend::{CmdBuf, Device};
use piet_gpu_hal::{dx12, BufferUsage, Error};

const SHADER_CODE: &str = r#"RWByteAddressBuffer _53 : register(u0, space0);

RWTexture2D<float4> textureOut : register(u1);

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
        if ((n & 1u) == 0u)
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
    uint param = _53.Load(index * 4 + 0);
    uint _61 = collatz_iterations(param);
    _53.Store(index * 4 + 0, _61);
    textureOut[uint2(index, 0)] = float4(1.0, 0.0, 0.0, 1.0);
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
"#;

fn toy() -> Result<(), Error> {
    let (instance, _surface) = dx12::Dx12Instance::new(None)?;
    let device = instance.device(None)?;
    let buf = device.create_buffer(
        1024,
        BufferUsage::MAP_READ
            | BufferUsage::MAP_WRITE
            | BufferUsage::COPY_SRC
            | BufferUsage::COPY_DST,
    )?;
    let dev_buf = device.create_buffer(
        1024,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
    )?;
    let img_readback_buf =
        device.create_buffer(1024, BufferUsage::MAP_READ | BufferUsage::COPY_DST)?;
    let data: Vec<u32> = (1..257).collect();
    let query_pool = device.create_query_pool(2)?;
    unsafe {
        let img = device.create_image2d(256, 1)?;
        device.write_buffer(&buf, data.as_ptr() as *const u8, 0, 1024)?;
        let pipeline = device.create_simple_compute_pipeline(SHADER_CODE, 1, 1)?;
        let ds = device.create_descriptor_set(&pipeline, &[&dev_buf], &[&img])?;
        let mut cmd_buf = device.create_cmd_buf()?;
        let mut fence = device.create_fence(false)?;
        cmd_buf.begin();
        cmd_buf.copy_buffer(&buf, &dev_buf);
        cmd_buf.memory_barrier();
        cmd_buf.write_timestamp(&query_pool, 0);
        cmd_buf.dispatch(&pipeline, &ds, (1, 1, 1), (256, 1, 1));
        cmd_buf.write_timestamp(&query_pool, 1);
        cmd_buf.memory_barrier();
        cmd_buf.copy_buffer(&dev_buf, &buf);
        cmd_buf.copy_image_to_buffer(&img, &img_readback_buf);
        cmd_buf.finish_timestamps(&query_pool);
        cmd_buf.host_barrier();
        cmd_buf.finish();
        device.run_cmd_bufs(&[&cmd_buf], &[], &[], Some(&mut fence))?;
        device.wait_and_reset(vec![&mut fence])?;
        let mut readback: Vec<u32> = vec![0u32; 256];
        device.read_buffer(&buf, readback.as_mut_ptr() as *mut u8, 0, 1024)?;
        println!("{:?}", readback);
        println!("{:?}", device.fetch_query_pool(&query_pool));
    }
    Ok(())
}

fn main() {
    toy().unwrap();
}
