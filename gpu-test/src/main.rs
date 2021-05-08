use piet_gpu_hal::hub;
use piet_gpu_hal::vulkan::{VkDevice, VkInstance};
use piet_gpu_hal::{CmdBuf, Error, MemFlags};

const DATA_SIZE: u64 = 1 << 19;

struct Context {
    session: hub::Session,
    control_buf: hub::Buffer,
    data_buf: hub::Buffer,
}

impl Context {
    fn new(device: VkDevice) -> Context {
        let session = hub::Session::new(device);
        let host_mem = MemFlags::host_coherent();
        let dev_mem = MemFlags::device_local();
        let control_buf = session.create_buffer(12, host_mem).unwrap();
        let data_buf = session.create_buffer(DATA_SIZE, dev_mem).unwrap();

        Context {
            session,
            control_buf,
            data_buf,
        }
    }

    fn run_test(&mut self, spv: &[u8]) -> Result<(Vec<u32>, f64), Error> {
        unsafe {
            let control = vec![1024u32, 0u32, 0u32];
            self.control_buf.write(&control).unwrap();
            let pipeline = self.session.create_simple_compute_pipeline(spv, 2)?;
            let descriptor_set = self
                .session
                .create_simple_descriptor_set(&pipeline, &[&self.control_buf, &self.data_buf])
                .unwrap();
            let query_pool = self.session.create_query_pool(2)?;
            let mut cmd_buf = self.session.cmd_buf()?;
            cmd_buf.begin();
            cmd_buf.clear_buffer(self.data_buf.vk_buffer(), None);
            cmd_buf.reset_query_pool(&query_pool);
            cmd_buf.write_timestamp(&query_pool, 0);
            cmd_buf.dispatch(&pipeline, &descriptor_set, (256, 1, 1));
            cmd_buf.write_timestamp(&query_pool, 1);
            cmd_buf.host_barrier();
            cmd_buf.finish();
            let submitted = self.session.run_cmd_buf(cmd_buf, &[], &[])?;
            submitted.wait()?;
            let timestamps = self.session.fetch_query_pool(&query_pool)?;
            let mut dst: Vec<u32> = Default::default();
            self.control_buf.read(&mut dst).unwrap();
            Ok((dst, timestamps[0]))
        }
    }

    fn report_coherence(&mut self, spv: &[u8], desc: &str) {
        let (control, ts) = self.run_test(spv).unwrap();
        let status = if control[1] == 0 {
            format!("ok")
        } else {
            format!("{} failures", control[1])
        };
        println!("{}: {}, {:.1}ms", desc, status, ts * 1e3);
    }

    fn has_memory_model(&self) -> bool {
        self.session.gpu_info().has_memory_model
    }
}

fn main() {
    let (instance, _) = VkInstance::new(None).unwrap();
    unsafe {
        let device = instance.device(None).unwrap();
        let mut context = Context::new(device);
        // To warm up GPU
        let _ = context.run_test(include_bytes!("../shader/coherence_test.spv"));
        context.report_coherence(
            include_bytes!("../shader/coherence_test.spv"),
            "Raw array access",
        );
        context.report_coherence(
            include_bytes!("../shader/coherence_volatile.spv"),
            "Raw array access w/ volatile",
        );
        context.report_coherence(include_bytes!("../shader/coherence_atomic.spv"), "Atomics");
        if context.has_memory_model() {
            context.report_coherence(
                include_bytes!("../shader/coherence_vkmm.spv"),
                "Vulkan memory model",
            );
        }
    }
}
