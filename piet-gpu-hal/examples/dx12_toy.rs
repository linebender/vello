//! An example to exercise the dx12 backend, while it's being developed.
//! This will probably go away when it's fully implemented and we can
//! just use the hub.

use piet_gpu_hal::{dx12, CmdBuf, Device, Error, MemFlags};

const SHADER_CODE: &str = r#"RWByteAddressBuffer _53 : register(u0, space0);

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
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
"#;

fn toy() -> Result<(), Error> {
    let instance = dx12::Dx12Instance::new()?;
    let device = instance.device()?;
    let buf = device.create_buffer(1024, MemFlags::host_coherent())?;
    let dev_buf = device.create_buffer(1024, MemFlags::device_local())?;
    let data: Vec<u32> = (1..257).collect();
    unsafe {
        device.write_buffer(&buf, &data)?;
        let pipeline = device.create_simple_compute_pipeline(SHADER_CODE, 1, 0)?;
        let ds = device.create_descriptor_set(&pipeline, &[&dev_buf], &[])?;
        let mut cmd_buf = device.create_cmd_buf()?;
        let fence = device.create_fence(false)?;
        cmd_buf.begin();
        cmd_buf.copy_buffer(&buf, &dev_buf);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(&pipeline, &ds, (1, 1, 1));
        cmd_buf.memory_barrier();
        cmd_buf.copy_buffer(&dev_buf, &buf);
        cmd_buf.host_barrier();
        cmd_buf.finish();
        device.run_cmd_buf(&cmd_buf, &[], &[], Some(&fence))?;
        device.wait_and_reset(&[fence])?;
        let mut readback: Vec<u32> = Vec::new();
        device.read_buffer(&buf, &mut readback)?;
        println!("{:?}", readback);
    }
    Ok(())
}

fn main() {
    toy().unwrap();
}
