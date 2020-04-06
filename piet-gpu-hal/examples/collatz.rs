use piet_gpu_hal::vulkan::VkInstance;
use piet_gpu_hal::{CmdBuf, Device, MemFlags};

fn main() {
    let instance = VkInstance::new().unwrap();
    unsafe {
        let device = instance.device().unwrap();
        let mem_flags = MemFlags::host_coherent();
        let src = (0..256).map(|x| x + 1).collect::<Vec<u32>>();
        let buffer = device
            .create_buffer(std::mem::size_of_val(&src[..]) as u64, mem_flags)
            .unwrap();
        device.write_buffer(&buffer, &src).unwrap();
        let code = include_bytes!("./shader/collatz.spv");
        let pipeline = device.create_simple_compute_pipeline(code, 1).unwrap();
        let descriptor_set = device.create_descriptor_set(&pipeline, &[&buffer]).unwrap();
        let mut cmd_buf = device.create_cmd_buf().unwrap();
        cmd_buf.begin();
        cmd_buf.dispatch(&pipeline, &descriptor_set);
        cmd_buf.finish();
        device.run_cmd_buf(&cmd_buf).unwrap();
        let mut dst: Vec<u32> = Default::default();
        device.read_buffer(&buffer, &mut dst).unwrap();
        for (i, val) in dst.iter().enumerate().take(16) {
            println!("{}: {}", i, val);
        }
    }
}
