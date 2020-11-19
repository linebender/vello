//! An example to exercise the dx12 backend, while it's being developed.
//! This will probably go away when it's fully implemented and we can
//! just use the hub.

use piet_gpu_hal::dx12;
use piet_gpu_hal::Error;

fn toy() -> Result<(), Error> {
    let instance = dx12::Dx12Instance::new()?;
    let device = instance.device()?;
    Ok(())
}

fn main() {
    toy().unwrap();
    println!("hello dx12");
}
