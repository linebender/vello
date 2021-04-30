//! An example to exercise the dx12 backend, while it's being developed.
//! This will probably go away when it's fully implemented and we can
//! just use the hub.

use piet_gpu_hal::{dx12, Device, Error, MemFlags};

fn toy() -> Result<(), Error> {
    let instance = dx12::Dx12Instance::new()?;
    let device = instance.device()?;
    let buf = device.create_buffer(1024, MemFlags::host_coherent())?;
    let data: Vec<u32> = (0..256).collect();
    unsafe {
        device.write_buffer(&buf, &data)?;
        let mut readback: Vec<u32> = Vec::new();
        device.read_buffer(&buf, &mut readback)?;
        println!("{:?}", readback);
    }
    Ok(())
}

fn main() {
    toy().unwrap();
    println!("hello dx12");
}
