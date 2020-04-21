use piet_gpu_hal::vulkan::VkInstance;
use piet_gpu_hal::{CmdBuf, Device, MemFlags};

const BLOCKSIZE: usize = 1024;

fn main() {
    let n = 64 * 1024 * 1024;
    let n_tiles = n / BLOCKSIZE;
    let instance = VkInstance::new().unwrap();
    unsafe {
        let device = instance.device().unwrap();
        let mem_flags = MemFlags::host_coherent();
        let device_local = MemFlags::device_local();
        let src = (0..n).map(|x| (x & 3) as u32).collect::<Vec<u32>>();
        let buffer = device
            .create_buffer(std::mem::size_of_val(&src[..]) as u64, mem_flags)
            .unwrap();
        let buffer_dev = device
            .create_buffer(std::mem::size_of_val(&src[..]) as u64, device_local)
            .unwrap();
        let dst_buffer = device
            .create_buffer(std::mem::size_of_val(&src[..]) as u64, device_local)
            .unwrap();
        let work_buffer = device
            .create_buffer((n_tiles * 16) as u64, device_local)
            .unwrap();
        device.write_buffer(&buffer, &src).unwrap();
        let code = include_bytes!("./shader/prefix.spv");
        let pipeline = device.create_simple_compute_pipeline(code, 3).unwrap();
        let bufs = [&buffer_dev, &dst_buffer, &work_buffer];
        let descriptor_set = device.create_descriptor_set(&pipeline, &bufs).unwrap();
        let query_pool = device.create_query_pool(2).unwrap();
        let mut cmd_buf = device.create_cmd_buf().unwrap();
        cmd_buf.begin();
        cmd_buf.clear_buffer(&work_buffer);
        cmd_buf.copy_buffer(&buffer, &buffer_dev);
        cmd_buf.memory_barrier();
        cmd_buf.write_timestamp(&query_pool, 0);
        cmd_buf.dispatch(&pipeline, &descriptor_set, (n_tiles as u32, 1, 1));
        cmd_buf.write_timestamp(&query_pool, 1);
        cmd_buf.memory_barrier();
        cmd_buf.copy_buffer(&dst_buffer, &buffer);
        cmd_buf.finish();
        device.run_cmd_buf(&cmd_buf).unwrap();
        let timestamps = device.reap_query_pool(query_pool).unwrap();
        let mut dst: Vec<u32> = Default::default();
        device.read_buffer(&buffer, &mut dst).unwrap();
        for (i, val) in dst.iter().enumerate().take(16) {
            println!("{}: {}", i, val);
        }
        println!("{:?}ms", timestamps[0] * 1e3);
    }
}
