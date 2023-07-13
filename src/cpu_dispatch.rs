//! Support for CPU implementations of compute shaders.

pub enum CpuResourceRef<'a> {
    Buffer(&'a mut [u8]),
    Texture(&'a mut CpuTexture),
}

impl<'a> CpuResourceRef<'a> {
    pub fn as_buf(&mut self) -> &mut [u8] {
        match self {
            CpuResourceRef::Buffer(b) => b,
            _ => panic!("resource type mismatch"),
        }
    }

    pub fn as_tex(&mut self) -> &mut CpuTexture {
        match self {
            CpuResourceRef::Texture(t) => t,
            _ => panic!("resource type mismatch"),
        }
    }
}

/// Structure used for binding textures to CPU shaders.
pub struct CpuTexture {
    pub width: usize,
    pub height: usize,
    // In RGBA format. May expand in the future.
    pub pixels: Vec<u32>,
}
