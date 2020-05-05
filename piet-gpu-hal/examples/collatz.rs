use piet_gpu_hal::vulkan::VkInstance;
use piet_gpu_hal::{CmdBuf, Device, MemFlags};

fn main() {
    let (instance, _) = VkInstance::new(None).unwrap();
    unsafe {
        let device = instance.device(None).unwrap();
        let mem_flags = MemFlags::host_coherent();
        let src = (0..256).map(|x| x + 1).collect::<Vec<u32>>();
        let buffer = device
            .create_buffer(std::mem::size_of_val(&src[..]) as u64, mem_flags)
            .unwrap();
        device.write_buffer(&buffer, &src).unwrap();
        let code = include_bytes!("./shader/collatz.spv");
        let pipeline = device.create_simple_compute_pipeline(code, 1).unwrap();
        let descriptor_set = device.create_descriptor_set(&pipeline, &[&buffer]).unwrap();
        let query_pool = device.create_query_pool(2).unwrap();
        let mut cmd_buf = device.create_cmd_buf().unwrap();
        cmd_buf.begin();
        cmd_buf.reset_query_pool(&query_pool);
        cmd_buf.write_timestamp(&query_pool, 0);
        cmd_buf.dispatch(&pipeline, &descriptor_set, (256, 1, 1));
        cmd_buf.write_timestamp(&query_pool, 1);
        cmd_buf.finish();
        device.run_cmd_buf(&cmd_buf).unwrap();
        let timestamps = device.reap_query_pool(query_pool);
        let mut dst: Vec<u32> = Default::default();
        device.read_buffer(&buffer, &mut dst).unwrap();
        for (i, val) in dst.iter().enumerate().take(16) {
            println!("{}: {}", i, val);
        }
        println!("{:?}", timestamps);
    }
}
