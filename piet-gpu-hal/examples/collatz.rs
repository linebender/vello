use piet_gpu_hal::hub;
use piet_gpu_hal::include_shader;
use piet_gpu_hal::mux::Instance;
use piet_gpu_hal::BufferUsage;

fn main() {
    let (instance, _) = Instance::new(None).unwrap();
    unsafe {
        let device = instance.device(None).unwrap();
        let session = hub::Session::new(device);
        let usage = BufferUsage::MAP_READ | BufferUsage::STORAGE;
        let src = (0..256).map(|x| x + 1).collect::<Vec<u32>>();
        let buffer = session.create_buffer_init(&src, usage).unwrap();
        let code = include_shader!(&session, "./shader/gen/collatz");
        let pipeline = session.create_simple_compute_pipeline(code, 1).unwrap();
        let descriptor_set = session
            .create_simple_descriptor_set(&pipeline, &[&buffer])
            .unwrap();
        let query_pool = session.create_query_pool(2).unwrap();
        let mut cmd_buf = session.cmd_buf().unwrap();
        cmd_buf.begin();
        cmd_buf.reset_query_pool(&query_pool);
        cmd_buf.write_timestamp(&query_pool, 0);
        cmd_buf.dispatch(&pipeline, &descriptor_set, (256, 1, 1), (1, 1, 1));
        cmd_buf.write_timestamp(&query_pool, 1);
        cmd_buf.host_barrier();
        cmd_buf.finish();
        let submitted = session.run_cmd_buf(cmd_buf, &[], &[]).unwrap();
        submitted.wait().unwrap();
        let timestamps = session.fetch_query_pool(&query_pool);
        let mut dst: Vec<u32> = Default::default();
        buffer.read(&mut dst).unwrap();
        for (i, val) in dst.iter().enumerate().take(16) {
            println!("{}: {}", i, val);
        }
        println!("{:?}", timestamps);
    }
}
